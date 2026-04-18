#include "gp.h"
#include "gp_utils.h"
#include "matplotlibcpp.h"
#include "recursive_gp.h"
#include "sparse_gp.h"
#include <Eigen/Dense>

#include <chrono>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "evaluate.h"

#include "cg.h"
#include "de.h"
#include "ga.h"
#include "lbfgs.h"
#include "pso.h"
#include "rprop.h"

namespace plt = matplotlibcpp;

using namespace libgp;

int main(int argc, char const *argv[]) {
  bool verbose = true;
  int n = 500, m = 2000;
  double tss = 0, tss_ind = 0, error, error_ind, f, f_ind, y, var, var_ind;
  double css = 0;
  size_t input_dim = 1; // number of input dimensions

  // covariance function with additive white noise.
  SparseGaussianProcess sgp(input_dim, "CovSum ( CovSEard, CovNoise)");

  // initialize hyper parameter vector
  Eigen::VectorXd params(sgp.covf().get_param_dim());
  params << std::log(1.0), std::log(1.0), std::log(0.1);

  // set parameters of covariance function
  sgp.covf().set_loghyper(params);

  // create training set
  std::vector<double> x_train_0(n), y_train(n);
  std::shared_ptr<SampleSet> train_set(new SampleSet(input_dim));
  for (int i = 0; i < n; ++i) {
    double x[] = {drand48() * 6 - 3};
    y = Utils::toy_function(x[0]) + Utils::randn() * 0.1;
    train_set->add(x, y);
    x_train_0.at(i) = x[0];
    y_train.at(i) = y;
  }
  sgp.set_sampleset(train_set);

  // create test set
  std::shared_ptr<SampleSet> test_set_ptr(new SampleSet(input_dim));
  for (int i = 0; i < m; ++i) {
    double xi = -3.0 + 12.0 * i / (m - 1);
    double x[] = {xi};
    y = Utils::toy_function(xi) + 0.2;
    test_set_ptr->add(x, y);
  }

  std::vector<Eigen::VectorXd> inducing_points;
  // Generate uniformly spaced inducing points in [-2, 2]
  int grid_size = 15;
  for (int i = 0; i < grid_size; ++i) {
    Eigen::VectorXd point(input_dim);
    point(0) = -3.0 + 6.0 * i / (grid_size - 1);
    inducing_points.push_back(point);
  }
  sgp.specify_inducingSet(inducing_points);

  size_t epoch = 0;
  libgp::CG optimizer; // 差分演化优化器
  size_t max_epochs = 10;
  std::vector<double> rmse(2), lml(2), mae(2), duration(2);

  std::vector<std::string> method_names = {"SVGP", "RSVGP"};
  GPEvaluation evaluator(method_names);
  evaluator.setTrainSet(train_set);
  evaluator.setTestSet(test_set_ptr);

  optimizer.maximize(&sgp, 100, 1);
  // sgp.check_gradient();

  while (epoch < max_epochs) {
    epoch++;
    std::cout << "\033[34m==================== Epoch " << epoch
              << " ====================\033[0m" << std::endl;
    auto epoch_t1 = std::chrono::high_resolution_clock::now();

    auto epoch_t2 = std::chrono::high_resolution_clock::now();
    duration[1] =
        std::chrono::duration<double, std::milli>(epoch_t2 - epoch_t1).count();
    Eigen::VectorXd mean_sgp, var_sgp;
    sgp.pred_diag(test_set_ptr, mean_sgp, var_sgp);
    evaluator.record_epoch_results("SVGP", mean_sgp, var_sgp,
                                   sgp.log_likelihood(), duration[1], verbose);
    sgp.exportModelToYAML("sgp_model.yaml");
    SparseGaussianProcess rgp("sgp_model.yaml");
    epoch_t1 = std::chrono::high_resolution_clock::now();

    epoch_t2 = std::chrono::high_resolution_clock::now();
    duration[2] =
        std::chrono::duration<double, std::milli>(epoch_t2 - epoch_t1).count();
    Eigen::VectorXd mean_rgp, var_rgp;
    std::cout << "begin batch update" << std::endl;
    rgp.storePosteriorPretrained();
    for (size_t i = 1000; i < 1400; ++i) {
      // rgp.sampleset->add(test_set_ptr->x(i), test_set_ptr->y(i) +
      // Utils::randn() * 0.2);

      rgp.add_pattern_batch(test_set_ptr->x(i),
                            test_set_ptr->y(i) + Utils::randn() * 0.2);
    }

    auto tick = std::chrono::high_resolution_clock::now();
    optimizer.maximize(&rgp, 100, 1);
    auto tock = std::chrono::high_resolution_clock::now();
    double optimization_time =
        std::chrono::duration<double, std::milli>(tock - tick).count();
    std::cout << "Optimization time (RGP): " << optimization_time << " ms"
              << std::endl;
    duration[2] += optimization_time;

    // check gradient
    // rgp.check_gradient();

    rgp.exportModelToYAML("rgp_model.yaml");

    rgp.storePosteriorPretrained();
    for (size_t i = 1500; i < test_set_ptr->size(); ++i) {
      rgp.add_pattern_batch(test_set_ptr->x(i),
                            test_set_ptr->y(i) + Utils::randn() * 0.2);
    }

    rgp.check_gradient();

    optimizer.maximize(&rgp, 100, 1);

    rgp.pred_diag(test_set_ptr, mean_rgp, var_rgp);

    evaluator.record_epoch_results("RSVGP", mean_rgp, var_rgp,
                                   rgp.log_likelihood(), duration[2], verbose);

    //可视化rgp新的诱导点
    Eigen::MatrixXd inducing_points_rgp = rgp.getFlatInputs();
    std::vector<double> inducing_points_rgp_vec(inducing_points_rgp.data(),
                                                inducing_points_rgp.data() +
                                                    inducing_points_rgp.cols());
    std::cout << "Inducing points (RGP): " << inducing_points_rgp_vec.size()
              << std::endl;
    Eigen::VectorXd inducing_targets_rgp = rgp.getFlatTargets();
    std::vector<double> inducing_targets_rgp_vec(
        inducing_targets_rgp.data(),
        inducing_targets_rgp.data() + inducing_targets_rgp.rows());
    std::cout << "Inducing targets (RGP): " << inducing_targets_rgp_vec.size()
              << std::endl;
    plt::figure();
    plt::scatter(inducing_points_rgp_vec, inducing_targets_rgp_vec, 10,
                 {{"label", "Inducing Points"}});
    plt::xlim(-3.0, 6.0);
    plt::ylim(-5.0, 5.0);
    plt::title("Inducing Points (RGP)");
    plt::legend();
    plt::show();

    // if (epoch % 10 == 0 || epoch == max_epochs) {
    evaluator.visualize_epoch_results();
    evaluator.visualize_uncertainty_bands();
    // }
  }

  return EXIT_SUCCESS;
}

// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cg.h"
#include "ga.h"
#include "gp.h"
#include "gp_utils.h"
#include "matplotlibcpp.h"
#include "recursive_gp.h"
#include "rprop.h"
#include "sparse_gp.h"
#include <Eigen/Dense>

#include <chrono>
#include <iostream> // standard input/output
#include <vector>   // standard vector

#include "evaluate.h"

namespace plt = matplotlibcpp;

using namespace libgp;

int main(int argc, char const *argv[]) {
  bool verbose = true;
  int n = 400, m = 1000;
  // m will be determined by grid resolution
  double tss = 0, tss_ind = 0, error, error_ind, f, f_ind, y, var, var_ind;
  double css = 0;
  size_t input_dim = 2; // number of input dimensions

  // initialize Gaussian process for 2-D input using the squared exponential
  // covariance function with additive white noise.
  GaussianProcess gp(input_dim, "CovSum ( CovSEard, CovNoise)");

  SparseGaussianProcess sgp(input_dim, "CovSum ( CovSEard, CovNoise)");

  RecursiveGaussianProcess rgp(input_dim, "CovSum ( CovSEard, CovNoise)");

  // initialize hyper parameter vector
  Eigen::VectorXd params(gp.covf().get_param_dim());
  params << 0.0, 0.0, 0.0, 0.0;

  // set parameters of covariance function
  gp.covf().set_loghyper(params);
  sgp.covf().set_loghyper(params);
  rgp.covf().set_loghyper(params);

  // create training set
  std::shared_ptr<SampleSet> train_set(new SampleSet(input_dim));
  std::vector<double> x_train_0(n), x_train_1(n), y_train(n);
  // add training patterns
  for (int i = 0; i < n; ++i) {
    double x[] = {drand48() * 4 - 2, drand48() * 4 - 2};
    y = Utils::hill(x[0], x[1]) + Utils::randn() * 0.1;
    train_set->add(x, y);
    x_train_0.at(i) = x[0];
    x_train_1.at(i) = x[1];
    y_train.at(i) = y;
  }
  gp.set_sampleset(train_set);
  sgp.set_sampleset(train_set);
  rgp.set_sampleset(train_set);

  // create test set
  std::shared_ptr<SampleSet> test_set_ptr(new SampleSet(input_dim));
  for (int i = 0; i < m; ++i) {
    double x[] = {drand48() * 4 - 2, drand48() * 4 - 2};
    y = Utils::hill(x[0], x[1]);
    test_set_ptr->add(x, y);
  }

  std::vector<Eigen::VectorXd> inducing_points;
  // Generate uniformly spaced inducing points in [-2, 2] x [-2, 2]
  int grid_size = 6;
  for (int i = 0; i < grid_size; ++i) {
    for (int j = 0; j < grid_size; ++j) {
      Eigen::VectorXd point(input_dim);
      point(0) = -2.0 + 4.0 * i / (grid_size - 1);
      point(1) = -2.0 + 4.0 * j / (grid_size - 1);
      inducing_points.push_back(point);
    }
  }
  sgp.specify_inducingSet(inducing_points);
  rgp.specify_inducingSet(inducing_points);

  size_t epoch = 0;
  libgp::CG cg_sgp, cg_fullgp;
  size_t max_epochs = 1;
  std::vector<double> rmse(3), lml(3), mae(3), duration(3);

  std::vector<std::string> method_names = {"FullGP", "SVGP", "RSVGP"};
  GPEvaluation evaluator(method_names);
  evaluator.setTrainSet(train_set);
  evaluator.setTestSet(test_set_ptr);

  while (epoch < max_epochs) {
    epoch++;
    std::cout << "\033[34m==================== Epoch " << epoch
              << " ====================\033[0m" << std::endl;

    // FullGP
    auto epoch_t1 = std::chrono::high_resolution_clock::now();
    cg_fullgp.maximize(&gp, 100, 0);
    auto epoch_t2 = std::chrono::high_resolution_clock::now();
    duration[0] =
        std::chrono::duration<double, std::milli>(epoch_t2 - epoch_t1).count() /
        (epoch + 1);
    Eigen::VectorXd mean_gp, var_gp;
    gp.pred_diag(test_set_ptr, mean_gp, var_gp);
    evaluator.record_epoch_results("FullGP", mean_gp, var_gp,
                                   gp.log_likelihood(), duration[0], verbose);

    // SVGP
    epoch_t1 = std::chrono::high_resolution_clock::now();
    cg_sgp.maximize(&sgp, 100, 0);
    epoch_t2 = std::chrono::high_resolution_clock::now();
    duration[1] =
        std::chrono::duration<double, std::milli>(epoch_t2 - epoch_t1).count() /
        (epoch + 1);
    Eigen::VectorXd mean_sgp, var_sgp;
    sgp.pred_diag(test_set_ptr, mean_sgp, var_sgp);
    evaluator.record_epoch_results("SVGP", mean_sgp, var_sgp,
                                   sgp.log_likelihood(), duration[1], verbose);

    // RSVGP
    epoch_t1 = std::chrono::high_resolution_clock::now();
    rgp.epochUpdate(true);
    epoch_t2 = std::chrono::high_resolution_clock::now();
    duration[2] =
        std::chrono::duration<double, std::milli>(epoch_t2 - epoch_t1).count();
    Eigen::VectorXd mean_rgp, var_rgp;
    rgp.pred_diag(test_set_ptr, mean_rgp, var_rgp);
    evaluator.record_epoch_results("RSVGP", mean_rgp, var_rgp,
                                   rgp.log_likelihood(), duration[2], verbose);

    if (epoch % 10 == 0 || epoch == max_epochs) {
      evaluator.visualize_epoch_results();
      // evaluator.visualize_uncertainty_bands(); // Usually for 1D, might need
      // adaptation for 2D
    }
  }

  sgp.exportModelToYAML("sgp_model.yaml");
  gp.exportModelToYAML("gp_model.yaml");

  return EXIT_SUCCESS;
}
// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cg.h"
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

#include <casadi/casadi.hpp>

using namespace libgp;

int main(int argc, char const *argv[]) {
  bool verbose = true;
  int n = 200, m = 1000;
  double tss = 0, tss_ind = 0, error, error_ind, f, f_ind, y, var, var_ind;
  double css = 0;
  size_t input_dim = 1; // number of input dimensions

  // initialize Gaussian process for 2-D input using the squared exponential
  // covariance function with additive white noise.
  GaussianProcess gp(input_dim, "CovSum ( CovSEPeriodic, CovNoise)");

  SparseGaussianProcess sgp(input_dim, "CovSum ( CovSEPeriodic, CovNoise)");

  RecursiveGaussianProcess rgp(input_dim, "CovSum ( CovSEPeriodic, CovNoise)");

  // initialize hyper parameter vector
  Eigen::VectorXd params(gp.covf().get_param_dim());
  params << 1.0, 0.5, 2.5, 0.0, -2.0;

  // set parameters of covariance function
  gp.covf().set_loghyper(params);
  sgp.covf().set_loghyper(params);
  rgp.covf().set_loghyper(params);

  // create training set
  std::vector<double> x_train_0(n), y_train(n);
  std::shared_ptr<SampleSet> train_set(new SampleSet(input_dim));
  for (int i = 0; i < n; ++i) {
    double x[] = {10.0 * i / (n - 1) - 2.5};
    y = Utils::irregular_wave_function(x[0]) + Utils::randn() * 0.05;
    train_set->add(x, y);
    x_train_0.at(i) = x[0];
    y_train.at(i) = y;
  }
  gp.set_sampleset(train_set);
  sgp.set_sampleset(train_set);
  rgp.set_sampleset(train_set);

  // create test set
  std::shared_ptr<SampleSet> test_set_ptr(new SampleSet(input_dim));
  for (int i = 0; i < m; ++i) {
    double xi = 10.0 * i / (m - 1) + 7.5;
    double x[] = {xi};
    y = Utils::irregular_wave_function(xi);
    test_set_ptr->add(x, y);
  }

  std::vector<Eigen::VectorXd> inducing_points;
  // Generate uniformly spaced inducing points in [-2, 2]
  int grid_size = 20;
  for (int i = 0; i < grid_size; ++i) {
    Eigen::VectorXd point(input_dim);
    point(0) = -2.5 + 10.0 * i / (grid_size - 1);
    inducing_points.push_back(point);
  }
  sgp.specify_inducingSet(inducing_points);
  rgp.specify_inducingSet(inducing_points);

  size_t epoch = 0;
  libgp::CG cg_sgp, cg_fullgp;
  size_t max_epochs = 50;
  std::vector<double> rmse(3), lml(3), mae(3), duration(3);

  std::vector<std::string> method_names = {"FullGP", "SVGP", "RSVGP"};
  GPEvaluation evaluator(method_names);
  evaluator.setTrainSet(train_set);
  evaluator.setTestSet(test_set_ptr);

  while (epoch < max_epochs) {
    epoch++;
    std::cout << "\033[34m==================== Epoch " << epoch
              << " ====================\033[0m" << std::endl;
    auto epoch_t1 = std::chrono::high_resolution_clock::now();
    cg_fullgp.maximize(&gp, 50, 0);
    auto epoch_t2 = std::chrono::high_resolution_clock::now();
    duration[0] =
        std::chrono::duration<double, std::milli>(epoch_t2 - epoch_t1).count() /
        (epoch + 1);
    Eigen::VectorXd mean_gp, var_gp;
    gp.pred_diag(test_set_ptr, mean_gp, var_gp);
    evaluator.record_epoch_results("FullGP", mean_gp, var_gp,
                                   gp.log_likelihood(), duration[0], verbose);

    epoch_t1 = std::chrono::high_resolution_clock::now();
    cg_sgp.maximize(&sgp, 50, 0);
    epoch_t2 = std::chrono::high_resolution_clock::now();
    duration[1] =
        std::chrono::duration<double, std::milli>(epoch_t2 - epoch_t1).count() /
        (epoch + 1);
    std::cout << sgp.covf().get_loghyper().transpose() << std::endl;
    Eigen::VectorXd mean_sgp, var_sgp;
    sgp.pred_diag(test_set_ptr, mean_sgp, var_sgp);
    evaluator.record_epoch_results("SVGP", mean_sgp, var_sgp,
                                   sgp.log_likelihood(), duration[1], verbose);

    epoch_t1 = std::chrono::high_resolution_clock::now();
    // rgp.epochUpdate(false);
    epoch_t2 = std::chrono::high_resolution_clock::now();
    duration[2] =
        std::chrono::duration<double, std::milli>(epoch_t2 - epoch_t1).count();
    Eigen::VectorXd mean_rgp, var_rgp;
    rgp.pred_diag(test_set_ptr, mean_rgp, var_rgp);
    evaluator.record_epoch_results("RSVGP", mean_rgp, var_rgp,
                                   rgp.log_likelihood(), duration[2], verbose);

    evaluator.visualize_epoch_results();

    evaluator.visualize_uncertainty_bands();
  }

  return EXIT_SUCCESS;
}

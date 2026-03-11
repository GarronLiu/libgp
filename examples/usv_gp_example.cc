// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cg.h"
#include "configDir.h"
#include "gp.h"
#include "gp_utils.h"
#include "matplotlibcpp.h"
#include "rprop.h"
#include "sparse_gp.h"

#include <Eigen/Dense>

#include <cppad/cppad.hpp>               // the CppAD package
#include <cppad/example/cppad_eigen.hpp> // the CppAD/Eigen interface
#include <iostream>                      // standard input/output
#include <vector>                        // standard vector

namespace plt = matplotlibcpp;

using namespace libgp;

/// @brief 离散USV-GP模型：基于State_k,Action_k预测State_k+1
/// @param argc
/// @param argv
/// @return
int main(int argc, char const *argv[]) {

  double tss = 0, tss_u = 0, tss_v = 0, tss_r = 0, error, y, var;
  double sampling_interval = 0.1;

  // initialize Gaussian process
  size_t inducing_size = 100;
  SparseGaussianProcess gp_u((configDir + "/output_u_sim.txt").c_str(),
                             inducing_size);
  SparseGaussianProcess gp_v((configDir + "/output_v_sim.txt").c_str(),
                             inducing_size);
  SparseGaussianProcess gp_r((configDir + "/output_r_sim.txt").c_str(),
                             inducing_size);

  SampleSet *test_set;
  {
    GaussianProcess test((configDir + "/output_test.txt").c_str());
    test_set = new SampleSet(*test.sampleset);
  }

  int n = gp_u.sampleset->size();
  std::vector<double> u_train(n);
  std::vector<double> v_train(n);
  std::vector<double> r_train(n);
  std::vector<double> sample_id(n);
  // add training patterns
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd x_train = gp_u.sampleset->x(i);
    u_train.at(i) = x_train(0);
    v_train.at(i) = x_train(1);
    r_train.at(i) = x_train(2);
    sample_id.at(i) = i * 1;
  }
  std::cout << "Optimization begin!" << std::endl;
  libgp::CG cg;
  cg.maximize(&gp_u, 20, 0);

  cg.maximize(&gp_v, 20, 0);

  cg.maximize(&gp_r, 20, 0);

  std::cout << "Optimized hyper-parameters for u estimation"
            << gp_u.covf().get_loghyper().transpose() << std::endl;

  std::cout << "Optimized hyper-parameters for v estimation"
            << gp_v.covf().get_loghyper().transpose() << std::endl;

  std::cout << "Optimized hyper-parameters for r estimation"
            << gp_r.covf().get_loghyper().transpose() << std::endl;

  std::vector<double> u_pred(n);
  std::vector<double> v_pred(n);
  std::vector<double> r_pred(n);
  u_pred.at(0) = u_train.at(0);
  v_pred.at(0) = v_train.at(0);
  r_pred.at(0) = r_train.at(0);

  // total squared error
  double u_0 = u_pred.at(0);
  double v_0 = v_pred.at(0);
  double r_0 = r_pred.at(0);

  // on training set
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd x_eigen = gp_u.sampleset->x(i);
    double x[] = {x_eigen[0], x_eigen[1], x_eigen[2], x_eigen[3], x_eigen[4]};
    u_0 = gp_u.f(x);
    v_0 = gp_v.f(x);
    r_0 = gp_r.f(x);
    // var = gp_u.var(x);
    Eigen::Vector3d state_train(gp_u.sampleset->y(i), gp_v.sampleset->y(i),
                                gp_r.sampleset->y(i));
    Eigen::Vector3d state_pred(u_0, v_0, r_0);
    Eigen::Vector3d state_diff = state_train - state_pred;
    tss += state_diff.squaredNorm();
    tss_u += state_diff(0) * state_diff(0);
    tss_v += state_diff(1) * state_diff(1);
    tss_r += state_diff(2) * state_diff(2);
    u_pred.at(i) = u_0;
    v_pred.at(i) = v_0;
    r_pred.at(i) = r_0;
  }

  std::cout << "gpr mse on training set, component:" << tss / n
            << " u:" << tss_u / n << " v:" << tss_v / n << " r:" << tss_r / n
            << std::endl;
  plt::figure_size(1400, 1200);
  plt::suptitle(
      "gaussian process regression for USV maneuvering on training set");
  // visualize training data
  plt::subplot(3, 1, 1);

  std::map<std::string, std::string> keywords = {{"color", "k"},
                                                 {"label", "training"}};
  plt::scatter(sample_id, u_train, 1.0, keywords);

  keywords = {{"color", "b"}, {"label", "prediction"}};
  plt::plot(sample_id, u_pred, keywords);

  plt::ylabel("u(m/s)");

  plt::legend();

  plt::subplot(3, 1, 2);

  keywords = {{"color", "k"}, {"label", "training"}};
  plt::scatter(sample_id, v_train, 1.0, keywords);

  keywords = {{"color", "r"}, {"label", "prediction"}};
  plt::plot(sample_id, v_pred, keywords);

  plt::ylabel("v(m/s)");

  plt::legend();

  plt::subplot(3, 1, 3);

  keywords = {{"color", "k"}, {"label", "training"}};
  plt::scatter(sample_id, r_train, 1.0, keywords);

  keywords = {{"color", "y"}, {"label", "prediction"}};
  plt::plot(sample_id, r_pred, keywords);

  plt::ylabel("r(m/s)");

  plt::legend();

  plt::xlabel("sample id");

  plt::show();

  // on test set
  int m = test_set->size();
  std::vector<double> u_test(m);
  std::vector<double> v_test(m);
  std::vector<double> r_test(m);
  std::vector<double> t_test(m);
  for (int i = 0; i < m; ++i) {
    Eigen::VectorXd x_test = test_set->x(i);
    u_test.at(i) = x_test(0);
    v_test.at(i) = x_test(1);
    r_test.at(i) = x_test(2);
    t_test.at(i) = i * sampling_interval;
  }

  u_pred.resize(m);
  v_pred.resize(m);
  r_pred.resize(m);
  std::vector<double> t_pred(m);
  u_pred.at(0) = u_test.at(0);
  v_pred.at(0) = v_test.at(0);
  r_pred.at(0) = r_test.at(0);

  // initial state
  u_0 = u_pred.at(0);
  v_0 = v_pred.at(0);
  r_0 = r_pred.at(0);

  // on test set
  tss = 0.0;
  tss_u = 0.0;
  tss_v = 0.0;
  tss_r = 0.0;
  for (int i = 0; i < m - 1; ++i) {
    Eigen::VectorXd x_eigen = test_set->x(i);
    double x[] = {u_0, v_0, r_0, x_eigen[3], x_eigen[4]};
    u_0 = gp_u.f(x);
    v_0 = gp_v.f(x);
    r_0 = gp_r.f(x);
    var = gp_u.var(x);
    x_eigen = test_set->x(i + 1);
    Eigen::Vector3d state_test(x_eigen[0], x_eigen[1], x_eigen[2]);
    Eigen::Vector3d state_pred(u_0, v_0, r_0);
    Eigen::Vector3d state_diff = state_test - state_pred;
    tss += state_diff.squaredNorm();
    tss_u += state_diff(0) * state_diff(0);
    tss_v += state_diff(1) * state_diff(1);
    tss_r += state_diff(2) * state_diff(2);
    u_pred.at(i + 1) = u_0;
    v_pred.at(i + 1) = v_0;
    r_pred.at(i + 1) = r_0;
  }

  std::cout << "gpr mse on test set, component:" << tss / m
            << " u:" << tss_u / m << " v:" << tss_v / m << " r:" << tss_r / m
            << std::endl;
  plt::figure_size(1400, 1200);
  plt::suptitle("gaussian process regression for USV maneuvering on test set");
  // visualize training data
  plt::subplot(3, 1, 1);

  keywords = {{"color", "k"}, {"label", "test"}};
  plt::scatter(t_test, u_test, 1.0, keywords);

  keywords = {{"color", "b"}, {"label", "prediction"}};
  plt::plot(t_test, u_pred, keywords);

  plt::ylabel("u(m/s)");

  plt::legend();

  plt::subplot(3, 1, 2);

  keywords = {{"color", "k"}, {"label", "test"}};
  plt::scatter(t_test, v_test, 1.0, keywords);

  keywords = {{"color", "r"}, {"label", "prediction"}};
  plt::plot(t_test, v_pred, keywords);

  plt::ylabel("v(m/s)");

  plt::legend();

  plt::subplot(3, 1, 3);

  keywords = {{"color", "k"}, {"label", "training"}};
  plt::scatter(t_test, r_test, 1.0, keywords);

  keywords = {{"color", "y"}, {"label", "prediction"}};
  plt::plot(t_test, r_pred, keywords);

  plt::ylabel("r(m/s)");

  plt::xlabel("simulation time(s)");

  plt::legend();

  plt::show();

  return EXIT_SUCCESS;
}

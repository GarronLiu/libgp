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

#include <fstream>

namespace plt = matplotlibcpp;

using namespace libgp;

/// @brief 连续USV-GP加速度误差模型：基于State_{k},Action_{k}预测(State_{k+1} -
/// State_{Nominal}_{k+1})/SampleTime
/// @param argc
/// @param argv
/// @return
int main(int argc, char const *argv[]) {
  double tss = 0, tss_u = 0, tss_v = 0, tss_r = 0, error, y, var;
  double sampling_interval = 0.1;

  // initialize Gaussian process
  size_t inducing_size = 200;
  SparseGaussianProcess gp_u((configDir + "/udot_train.txt").c_str(),
                             inducing_size);
  SparseGaussianProcess gp_v((configDir + "/vdot_train.txt").c_str(),
                             inducing_size);
  SparseGaussianProcess gp_r((configDir + "/rdot_train.txt").c_str(),
                             inducing_size);

  SampleSet *udot_test_set, *vdot_test_set, *rdot_test_set;
  {
    GaussianProcess udot_test((configDir + "/udot_test.txt").c_str());
    GaussianProcess vdot_test((configDir + "/vdot_test.txt").c_str());
    GaussianProcess rdot_test((configDir + "/rdot_test.txt").c_str());
    udot_test_set = new SampleSet(*udot_test.sampleset);
    vdot_test_set = new SampleSet(*vdot_test.sampleset);
    rdot_test_set = new SampleSet(*rdot_test.sampleset);
  }

  int n = gp_u.sampleset->size();
  std::vector<double> acc_u_train(n);
  std::vector<double> acc_v_train(n);
  std::vector<double> acc_r_train(n);
  std::vector<double> sample_id(n);
  // add training patterns
  for (int i = 0; i < n; ++i) {
    acc_u_train.at(i) = gp_u.sampleset->y(i);
    acc_v_train.at(i) = gp_v.sampleset->y(i);
    acc_r_train.at(i) = gp_r.sampleset->y(i);
    sample_id.at(i) = i * 1;
  }
  std::cout << "Optimization begin!" << std::endl;
  libgp::CG cg;
  cg.maximize(&gp_u, 100, 0);

  cg.maximize(&gp_v, 50, 0);

  cg.maximize(&gp_r, 50, 0);

  std::cout << "Optimized hyper-parameters for acc_u estimation"
            << gp_u.covf().get_loghyper().transpose() << std::endl;

  std::cout << "Optimized hyper-parameters for acc_v estimation"
            << gp_v.covf().get_loghyper().transpose() << std::endl;

  std::cout << "Optimized hyper-parameters for acc_r estimation"
            << gp_r.covf().get_loghyper().transpose() << std::endl;

  std::vector<double> acc_u_pred(n);
  std::vector<double> acc_v_pred(n);
  std::vector<double> acc_r_pred(n);

  // total squared error
  double acc_u_0;
  double acc_v_0;
  double acc_r_0;

  // on training set
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd x_eigen = gp_u.sampleset->x(i);
    double x[] = {x_eigen[0], x_eigen[1], x_eigen[2], x_eigen[3], x_eigen[4]};
    acc_u_0 = gp_u.f(x);
    acc_v_0 = gp_v.f(x);
    acc_r_0 = gp_r.f(x);
    // var = gp_u.var(x);
    Eigen::Vector3d acc_train(gp_u.sampleset->y(i), gp_v.sampleset->y(i),
                              gp_r.sampleset->y(i));
    Eigen::Vector3d acc_pred(acc_u_0, acc_v_0, acc_r_0);
    Eigen::Vector3d acc_diff = acc_train - acc_pred;
    tss += acc_diff.squaredNorm();
    tss_u += acc_diff(0) * acc_diff(0);
    tss_v += acc_diff(1) * acc_diff(1);
    tss_r += acc_diff(2) * acc_diff(2);
    acc_u_pred.at(i) = acc_u_0;
    acc_v_pred.at(i) = acc_v_0;
    acc_r_pred.at(i) = acc_r_0;
  }

  std::cout << "gpr mse on training set, component:" << tss / n
            << " u:" << tss_u / n << " v:" << tss_v / n << " r:" << tss_r / n
            << std::endl;
  plt::figure_size(1400, 1200);
  plt::suptitle("gaussian process regression for USV dynamics (acceleration) "
                "on training set");
  // visualize training data
  plt::subplot(3, 1, 1);

  std::map<std::string, std::string> keywords = {{"color", "k"},
                                                 {"label", "training"}};
  plt::scatter(sample_id, acc_u_train, 1.0, keywords);

  keywords = {{"color", "b"}, {"label", "prediction"}};
  plt::plot(sample_id, acc_u_pred, keywords);

  plt::ylabel("u_dot(m/s^2)");

  plt::legend();

  plt::subplot(3, 1, 2);

  keywords = {{"color", "k"}, {"label", "training"}};
  plt::scatter(sample_id, acc_v_train, 1.0, keywords);

  keywords = {{"color", "r"}, {"label", "prediction"}};
  plt::plot(sample_id, acc_v_pred, keywords);

  plt::ylabel("v_dot(m/s^2)");

  plt::legend();

  plt::subplot(3, 1, 3);

  keywords = {{"color", "k"}, {"label", "training"}};
  plt::scatter(sample_id, acc_r_train, 1.0, keywords);

  keywords = {{"color", "y"}, {"label", "prediction"}};
  plt::plot(sample_id, acc_r_pred, keywords);

  plt::ylabel("r_dot(m/s^2)");

  plt::legend();

  plt::xlabel("sample id");

  plt::show();

  // on test set
  int m = udot_test_set->size();
  std::vector<double> udot_test(m);
  std::vector<double> vdot_test(m);
  std::vector<double> rdot_test(m);
  std::vector<double> t_test(m);

  for (int i = 0; i < m; ++i) {
    udot_test.at(i) = udot_test_set->y(i);
    vdot_test.at(i) = vdot_test_set->y(i);
    rdot_test.at(i) = rdot_test_set->y(i);
    t_test.at(i) = i * sampling_interval;
  }
  acc_u_pred.resize(m);
  acc_v_pred.resize(m);
  acc_r_pred.resize(m);
  std::vector<double> var_u_pred(m);
  std::vector<double> var_v_pred(m);
  std::vector<double> var_r_pred(m);
  std::vector<double> t_pred(m);

  // on test set
  tss = 0.0;
  tss_u = 0.0;
  tss_v = 0.0;
  tss_r = 0.0;
  for (int i = 0; i < m; ++i) {
    Eigen::VectorXd x_eigen = udot_test_set->x(i);
    double x[] = {x_eigen[0], x_eigen[1], x_eigen[2], x_eigen[3], x_eigen[4]};
    // acc_u_pred.at(i) = gp_u.f_sparse(x);
    // acc_v_pred.at(i) = gp_v.f_sparse(x);
    // acc_r_pred.at(i) = gp_r.f_sparse(x);
    var_u_pred.at(i) = gp_u.var(x);
    var_v_pred.at(i) = gp_v.var(x);
    var_r_pred.at(i) = gp_r.var(x);
    Eigen::Vector3d acc_test(udot_test_set->y(i), vdot_test_set->y(i),
                             rdot_test_set->y(i));
    Eigen::Vector3d acc_pred(acc_u_pred.at(i), acc_v_pred.at(i),
                             acc_r_pred.at(i));
    Eigen::Vector3d acc_diff = acc_test - acc_pred;
    tss += acc_diff.squaredNorm();
    tss_u += acc_diff(0) * acc_diff(0);
    tss_v += acc_diff(1) * acc_diff(1);
    tss_r += acc_diff(2) * acc_diff(2);
  }

  std::ofstream file("predictions.csv");
  file << "acc_u_pred,acc_v_pred,acc_r_pred,var_u,var_v,var_r\n";
  for (int i = 0; i < m; ++i) {
    file << acc_u_pred.at(i) << "," << acc_v_pred.at(i) << ","
         << acc_r_pred.at(i) << "," << var_u_pred.at(i) << ","
         << var_v_pred.at(i) << "," << var_r_pred.at(i) << "\n";
  }
  file.close();

  std::cout << "gpr mse on test set, component:" << tss / m
            << " u:" << tss_u / m << " v:" << tss_v / m << " r:" << tss_r / m
            << std::endl;
  plt::figure_size(1400, 1200);
  plt::suptitle("gaussian process regression for USV maneuvering on test set");
  // visualize training data
  plt::subplot(3, 1, 1);

  keywords = {{"color", "k"}, {"label", "test"}};
  plt::scatter(t_test, udot_test, 1.0, keywords);

  keywords = {{"color", "b"}, {"label", "prediction"}};
  plt::plot(t_test, acc_u_pred, keywords);

  plt::ylabel("u_dot(m/s^2)");

  plt::legend();

  plt::subplot(3, 1, 2);

  keywords = {{"color", "k"}, {"label", "test"}};
  plt::scatter(t_test, vdot_test, 1.0, keywords);

  keywords = {{"color", "r"}, {"label", "prediction"}};
  plt::plot(t_test, acc_v_pred, keywords);

  plt::ylabel("v_dot(m/s^2)");

  plt::legend();

  plt::subplot(3, 1, 3);

  keywords = {{"color", "k"}, {"label", "training"}};
  plt::scatter(t_test, rdot_test, 1.0, keywords);

  keywords = {{"color", "y"}, {"label", "prediction"}};
  plt::plot(t_test, acc_r_pred, keywords);

  plt::ylabel("r_dot(m/s^2)");

  plt::xlabel("time(s)");

  plt::legend();

  plt::show();

  // double x_[] = { 0, 0, 0, 0, 0 };
  // gp_u.f_sparse(x_);
  // gp_u.write_sparse(
  //     "/home/garronliu/7_Planning/usv_ipc_ws/src/usv_opt_tracker/include/usv_opt_tracker/output_u_sparse.txt");
  // gp_v.f_sparse(x_);
  // gp_v.write_sparse(
  //     "/home/garronliu/7_Planning/usv_ipc_ws/src/usv_opt_tracker/include/usv_opt_tracker/output_v_sparse.txt");
  // gp_r.f_sparse(x_);
  // gp_r.write_sparse(
  //     "/home/garronliu/7_Planning/usv_ipc_ws/src/usv_opt_tracker/include/usv_opt_tracker/output_r_sparse.txt");

  return EXIT_SUCCESS;
}

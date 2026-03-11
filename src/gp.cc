// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "cov_factory.h"

#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

namespace libgp {
const double log2pi = log(2 * M_PI);

const double initial_L_size = 1000;

void GaussianProcess::computeKernelMatrix(Eigen::MatrixXd &K_ab,
                                          SampleSet *set_a, SampleSet *set_b,
                                          CovarianceFunction *cf) {
  if (K_ab.rows() < set_a->size() || K_ab.cols() < set_b->size()) {
    K_ab.resize(set_a->size(), set_b->size());
  }

  for (size_t i = 0; i < set_a->size(); ++i) {
    for (size_t j = 0; j < set_b->size(); ++j) {
      K_ab(i, j) = cf->get(set_a->x(i), set_b->x(j));
    }
  }
}

void GaussianProcess::computeKernelMatrixLowerHalf(Eigen::MatrixXd &K_aa,
                                                   SampleSet *set_a,
                                                   CovarianceFunction *cf) {
  if (K_aa.rows() < set_a->size() || K_aa.cols() < set_a->size()) {
    K_aa.resize(set_a->size(), set_a->size());
  }
  for (size_t i = 0; i < set_a->size(); ++i) {
    for (size_t j = 0; j <= i; ++j) {
      K_aa(i, j) = cf->get(set_a->x(i), set_a->x(j));
    }
  }
}

GaussianProcess::GaussianProcess() {
  sampleset = NULL;
  cf = NULL;
}

GaussianProcess::GaussianProcess(size_t input_dim, std::string covf_def) {
  // set input dimensionality
  this->input_dim = input_dim;
  // create covariance function
  CovFactory factory;
  cf = factory.create(input_dim, covf_def);
  cf->loghyper_changed = 0;
  sampleset = new SampleSet(input_dim);
  L.resize(initial_L_size, initial_L_size);
}

GaussianProcess::GaussianProcess(const char *filename) {
  std::string fname(filename);
  if (fname.substr(fname.find_last_of(".") + 1) != "yaml") {
    /// TODO: optimize the efficiency of reading data from file
    sampleset = NULL;
    cf = NULL;
    int stage = 0;
    std::ifstream infile;
    double y;
    infile.open(filename);
    std::string s;
    double *x = NULL;
    L.resize(initial_L_size, initial_L_size);
    while (infile.good()) {
      getline(infile, s);
      // ignore empty lines and comments
      if (s.length() != 0 && s.at(0) != '#') {
        std::stringstream ss(s);
        if (stage > 2) {
          ss >> y;
          for (size_t j = 0; j < input_dim; ++j) {
            ss >> x[j];
          }

          add_pattern(x, y);
        } else if (stage == 0) {
          ss >> input_dim;
          sampleset = new SampleSet(input_dim);
          x = new double[input_dim];
        } else if (stage == 1) {
          CovFactory factory;
          cf = factory.create(input_dim, s);
          cf->loghyper_changed = 0;
        } else if (stage == 2) {
          Eigen::VectorXd params(cf->get_param_dim());
          for (size_t j = 0; j < cf->get_param_dim(); ++j) {
            ss >> params[j];
          }
          cf->set_loghyper(params);
        }
        stage++;
      }
    }
    infile.close();
    if (stage < 3) {
      std::cerr << "fatal error while reading " << filename << std::endl;
      exit(EXIT_FAILURE);
    }
    delete[] x;
  } else {
    YAML::Node config = YAML::LoadFile(filename);
    try {
      // 1. Sampling Points
      sampleset = new SampleSet(input_dim);
      if (config["sampling_points"]) {
        for (const auto &pt : config["sampling_points"]) {
          std::vector<double> vec = pt.as<std::vector<double>>();
          Eigen::VectorXd x =
              Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
          sampleset->add(x, 0.0);
        }
      } else {
        std::cout << "Warning: No sampling points provided in YAML file."
                  << std::endl;
        std::cout << " please add sampling points to use the GP model."
                  << std::endl;
      }
      // 2. Sampling Targets
      size_t n = sampleset->size();
      Eigen::VectorXd mean(n);
      if (config["sampling_targets"]) {
        std::vector<std::vector<double>> mean_vecs =
            config["sampling_targets"].as<std::vector<std::vector<double>>>();
        if (!mean_vecs.empty()) {
          std::vector<double> &v = mean_vecs[0];
          mean = Eigen::Map<Eigen::VectorXd>(v.data(), v.size());
          for (size_t i = 0; i < v.size(); ++i) {
            sampleset->set_y(i, v[i]);
          }
        }
      }
      L.resize(initial_L_size, initial_L_size);
      // 3.Kernel & Hyperparameters
      if (config["kernel"]) {
        YAML::Node kernel = config["kernel"];
        std::string kernel_name = kernel["type"].as<std::string>();

        // Determine input dimension from sampling points
        size_t dim = 0;
        if (config["sampling_points"] && config["sampling_points"].size() > 0) {
          dim = config["sampling_points"][0].size();
        }
        this->input_dim = dim;
        std::cout << " Input dimension set to: " << input_dim << std::endl;

        // Re-create covariance function
        CovFactory factory;
        cf = factory.create(input_dim, kernel_name);
        cf->loghyper_changed = 0;
        // Set params
        std::vector<double> params =
            kernel["hyperparameters"].as<std::vector<double>>();
        double noise = kernel["noise"].as<double>();
        params.push_back(noise);
        std::cout << " params size: " << params.size() << std::endl;
        Eigen::VectorXd log_params(params.size());
        for (size_t i = 0; i < params.size(); ++i)
          log_params(i) = std::log(params[i]);
        cf->set_loghyper(log_params);
      } else {
        std::cout << "Warning: No kernel provided in YAML file." << std::endl;
        std::cout << " please add kernel definition to use the GP model."
                  << std::endl;
      }

      std::cout << "Initialized GP from YAML model: " << filename << std::endl;

    } catch (...) {
      std::cerr << "Error loading GP model from YAML file." << std::endl;
    }
  }
}

GaussianProcess::GaussianProcess(const GaussianProcess &gp) {
  this->input_dim = gp.input_dim;
  sampleset = new SampleSet(*(gp.sampleset));
  alpha = gp.alpha;
  k_star = gp.k_star;
  alpha_needs_update = gp.alpha_needs_update;
  L = gp.L;

  // copy covariance function
  CovFactory factory;
  cf = factory.create(gp.input_dim, gp.cf->to_string());
  cf->loghyper_changed = gp.cf->loghyper_changed;
  cf->set_loghyper(gp.cf->get_loghyper());
}

GaussianProcess::~GaussianProcess() {
  // free memory
  if (sampleset != NULL)
    delete sampleset;
  if (cf != NULL)
    delete cf;
}

double GaussianProcess::f(const double x[]) {
  if (sampleset->empty())
    return 0;
  Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
  compute();
  update_alpha();
  update_k_star(x_star);
  return k_star.dot(alpha);
}

double GaussianProcess::var(const double x[]) {
  if (sampleset->empty())
    return 0;
  Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
  compute();
  update_alpha();
  update_k_star(x_star);
  int n = sampleset->size();
  Eigen::VectorXd v =
      L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(k_star);
  return cf->get(x_star, x_star) - v.dot(v);
}

void GaussianProcess::compute() {
  // can previously computed values be used?
  if (!cf->loghyper_changed)
    return;
  cf->loghyper_changed = false;
  int n = sampleset->size();
  // resize L if necessary
  if (n > L.rows())
    L.resize(n + initial_L_size, n + initial_L_size);
  // compute kernel matrix (lower triangle)
  computeKernelMatrixLowerHalf(L, sampleset, cf);
  // perform cholesky factorization
  // solver.compute(K.selfadjointView<Eigen::Lower>());
  L.topLeftCorner(n,n).diagonal().array() += 1e-6; // jitter for numerical stability
  L.topLeftCorner(n, n) =
      L.topLeftCorner(n, n).selfadjointView<Eigen::Lower>().llt().matrixL();
  alpha_needs_update = true;
}

void GaussianProcess::update_k_star(const Eigen::VectorXd &x_star) {
  k_star.resize(sampleset->size());
  for (size_t i = 0; i < sampleset->size(); ++i) {
    k_star(i) = cf->get(x_star, sampleset->x(i));
  }
}

void GaussianProcess::update_alpha() {
  // can previously computed values be used?
  if (!alpha_needs_update)
    return;
  alpha_needs_update = false;
  alpha.resize(sampleset->size());
  // Map target values to Matrix<double,Eigen::Dynamic,1>
  const std::vector<double> &targets = sampleset->y();
  Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
  int n = sampleset->size();
  alpha =
      L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(y); // L^(-1)*y
  L.topLeftCorner(n, n).triangularView<Eigen::Lower>().adjoint().solveInPlace(
      alpha); // L^(-double)*L^(-1)*y
}

void GaussianProcess::add_pattern(const double x[], double y) {
  // std::cout<< L.rows() << std::endl;
#if 0
    sampleset->add(x, y);
    cf->loghyper_changed = true;
    alpha_needs_update = true;
    cached_x_star = NULL;
    return;
#else
  int n = sampleset->size();
  sampleset->add(x, y);
  // create kernel matrix if sampleset is empty
  if (n == 0) {
    L(0, 0) = sqrt(cf->get(sampleset->x(0), sampleset->x(0)));
    cf->loghyper_changed = false;
    // recompute kernel matrix if necessary
  } else if (cf->loghyper_changed) {
    compute();
    // update kernel matrix
  } else {
    Eigen::VectorXd k(n);
    for (int i = 0; i < n; ++i) {
      k(i) = cf->get(sampleset->x(i), sampleset->x(n));
    }
    double kappa = cf->get(sampleset->x(n), sampleset->x(n));
    // resize L if necessary
    if (sampleset->size() > static_cast<std::size_t>(L.rows())) {
      L.conservativeResize(n + initial_L_size, n + initial_L_size);
    }
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(k);
    L.block(n, 0, 1, n) = k.transpose();
    L(n, n) = sqrt(kappa - k.dot(k));
  }
  alpha_needs_update = true;
#endif
}

void GaussianProcess::set_sampleset(const std::shared_ptr<SampleSet> &ss) {
  clear_sampleset();
  size_t N = ss->size();
  for (size_t i = 0; i < N; ++i) {
    add_pattern(ss->x(i).data(), ss->y(i));
  }
}

bool GaussianProcess::set_y(size_t i, double y) {
  if (sampleset->set_y(i, y)) {
    alpha_needs_update = true;
    return 1;
  }
  return false;
}

size_t GaussianProcess::get_sampleset_size() { return sampleset->size(); }

void GaussianProcess::clear_sampleset() { sampleset->clear(); }

void GaussianProcess::write(const char *filename) {
  // output
  std::ofstream outfile;
  outfile.open(filename);
  time_t curtime = time(0);
  tm now = *localtime(&curtime);
  char dest[BUFSIZ] = {0};
  strftime(dest, sizeof(dest) - 1, "%c", &now);
  outfile << "# " << dest << std::endl
          << std::endl
          << "# input dimensionality" << std::endl
          << input_dim << std::endl
          << std::endl
          << "# covariance function" << std::endl
          << cf->to_string() << std::endl
          << std::endl
          << "# log-hyperparameter" << std::endl;
  Eigen::VectorXd param = cf->get_loghyper();
  for (size_t i = 0; i < cf->get_param_dim(); i++) {
    outfile << std::setprecision(10) << param(i) << " ";
  }
  outfile << std::endl
          << std::endl
          << "# data (target value in first column)" << std::endl;
  for (size_t i = 0; i < sampleset->size(); ++i) {
    outfile << std::setprecision(10) << sampleset->y(i) << " ";
    for (size_t j = 0; j < input_dim; ++j) {
      outfile << std::setprecision(10) << sampleset->x(i)(j) << " ";
    }
    outfile << std::endl;
  }
  outfile.close();
}

CovarianceFunction &GaussianProcess::covf() { return *cf; }

size_t GaussianProcess::get_input_dim() { return input_dim; }

double GaussianProcess::log_likelihood() {
  compute();
  update_alpha();
  int n = sampleset->size();
  const std::vector<double> &targets = sampleset->y();
  Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
  double det = 2 * L.diagonal().head(n).array().log().sum();
  return -0.5 * y.dot(alpha) - 0.5 * det - 0.5 * n * log2pi;
}

Eigen::VectorXd GaussianProcess::log_likelihood_gradient() {
  compute();
  update_alpha();
  size_t n = sampleset->size();
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(cf->get_param_dim());
  Eigen::VectorXd g(grad.size());
  Eigen::MatrixXd W = Eigen::MatrixXd::Identity(n, n);

  // compute kernel matrix inverse
  L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(W);
  L.topLeftCorner(n, n).triangularView<Eigen::Lower>().transpose().solveInPlace(
      W);

  W = alpha * alpha.transpose() - W;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      cf->grad(sampleset->x(i), sampleset->x(j), g);
      if (i == j)
        grad += W(i, j) * g * 0.5;
      else
        grad += W(i, j) * g;
    }
  }
  return grad;
}

void GaussianProcess::update_hyperparameters(const Eigen::VectorXd &params) {
  cf->set_loghyper(params);
}

Eigen::VectorXd GaussianProcess::get_hyperparameters() {
  return cf->get_loghyper();
}

Eigen::VectorXd GaussianProcess::get_hyperparameter_lower_bound() {
  return cf->get_loghyper_lb();
}

Eigen::VectorXd GaussianProcess::get_hyperparameter_upper_bound() {
  return cf->get_loghyper_ub();
}

void GaussianProcess::pred_diag(const std::shared_ptr<SampleSet> testset,
                                Eigen::VectorXd &mean_pred,
                                Eigen::VectorXd &var_pred) {
  size_t N = testset->size();
  mean_pred.resize(N);
  var_pred.resize(N);
  for (size_t i = 0; i < N; ++i) {
    Eigen::VectorXd x_star = testset->x(i);
    mean_pred(i) = f(x_star.data());
    var_pred(i) = var(x_star.data());
  }
}

void GaussianProcess::pred_diag(const std::vector<Eigen::VectorXd> &testset,
                                Eigen::VectorXd &mean_pred,
                                Eigen::VectorXd &var_pred) {
  size_t N = testset.size();
  mean_pred.resize(N);
  var_pred.resize(N);
  for (size_t i = 0; i < N; ++i) {
    Eigen::VectorXd x_star = testset[i];
    mean_pred(i) = f(x_star.data());
    var_pred(i) = var(x_star.data());
  }
}

void GaussianProcess::pred_diag(Eigen::VectorXd &mean_pred,
                                Eigen::VectorXd &var_pred) {
  size_t N = sampleset->size();
  mean_pred.resize(N);
  var_pred.resize(N);
  for (size_t i = 0; i < N; ++i) {
    Eigen::VectorXd x_star = sampleset->x(i);
    mean_pred(i) = f(x_star.data());
    var_pred(i) = var(x_star.data());
  }
}

void GaussianProcess::pred_diag_derivative(Eigen::VectorXd &mean_deriv) {
  if (input_dim > 1) {
    std::cerr << "Error: pred_diag_derivative now only supports 1D input."
              << std::endl;
    return;
  }
  Eigen::VectorXd mean_deriv_temp;
  mean_deriv.resize(sampleset->size());
  mean_deriv.setZero();
  for (size_t i = 0; i < sampleset->size(); ++i) {
    for (size_t j = 0; j < sampleset->size(); ++j) {
      cf->grad_wrt_x1(sampleset->x(i), sampleset->x(j), mean_deriv_temp);
      mean_deriv(i) += mean_deriv_temp(0) * alpha(j);
    }
  }
}

void GaussianProcess::pred_diag_derivative(
    const std::vector<Eigen::VectorXd> &testset, Eigen::VectorXd &mean_deriv) {
  size_t N = testset.size();
  mean_deriv.resize(N);
  mean_deriv.setZero();
  Eigen::VectorXd mean_deriv_temp;
  for (size_t i = 0; i < testset.size(); ++i) {
    Eigen::VectorXd x_star = testset[i];
    for (size_t j = 0; j < sampleset->size(); ++j) {
      cf->grad_wrt_x1(x_star, sampleset->x(j), mean_deriv_temp);
      mean_deriv(i) += mean_deriv_temp(0) * alpha(j);
    }
  }
}

void GaussianProcess::validation(const std::shared_ptr<SampleSet> testset,
                                 double &mae, double &rmse, double &lml) {
  if (!testset) {
    std::cerr << "Test set not set!" << std::endl;
    return;
  }

  size_t N = testset->size();
  mae = 0;
  rmse = 0;
  for (size_t i = 0; i < N; ++i) {
    Eigen::VectorXd x_star = testset->x(i);
    double y_true = testset->y(i);
    double y_pred = f(x_star.data());
    double var_pred = var(x_star.data());
    double error = y_true - y_pred;
    rmse += error * error;
    mae += std::abs(error);
  }
  mae = mae / N;
  rmse = sqrt(rmse / N);
  lml = log_likelihood();
}

Eigen::MatrixXd GaussianProcess::getFlatInputs() {
  if (sampleset->size() < 1) {
    std::cout << "Sample set is empty." << std::endl;
    return Eigen::MatrixXd();
  }
  size_t n = sampleset->size();
  Eigen::MatrixXd flat_inducing_set(cf->get_input_dim(), n);
  for (size_t i = 0; i < n; ++i) {
    flat_inducing_set.col(i) = sampleset->x(i);
  }
  return flat_inducing_set;
}

Eigen::VectorXd GaussianProcess::getFlatTargets() {
  if (sampleset->size() < 1) {
    std::cout << "Sample set is empty." << std::endl;
    return Eigen::VectorXd();
  }
  const std::vector<double> &targets = sampleset->y();
  Eigen::Map<const Eigen::VectorXd> flat_targets(&targets[0], targets.size());
  return flat_targets;
}
Eigen::VectorXd GaussianProcess::getFlatAlpha() {
  compute();
  update_alpha();
  return alpha;
}

Eigen::VectorXd GaussianProcess::getFlatSamplingTargets() {
  if (sampleset->size() < 1) {
    std::cout << "Sample set is empty." << std::endl;
    return Eigen::VectorXd();
  }
  const std::vector<double> &targets = sampleset->y();
  Eigen::Map<const Eigen::VectorXd> flat_sampling_targets(&targets[0],
                                                          targets.size());
  return flat_sampling_targets;
}

Eigen::MatrixXd GaussianProcess::getFlatPosteriorCovMatrix() {
  compute();
  update_alpha();
  size_t n = sampleset->size();
  Eigen::MatrixXd K_post(n, n);

  K_post = L.topLeftCorner(n, n) * L.topLeftCorner(n, n).transpose();

  return K_post;
}

Eigen::VectorXd GaussianProcess::getFlatHyperparameters() {
  return cf->get_loghyper().array().exp();
}

void GaussianProcess::exportModelToYAML(const char *filename) {
  std::ofstream outfile(filename);
  if (!outfile.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing."
              << std::endl;
    return;
  }

  auto write_matrix_flat = [](std::ofstream &os, const Eigen::MatrixXd &mat) {
    for (size_t i = 0; i < mat.rows(); ++i) {
      os << " - [";
      for (size_t j = 0; j < mat.cols(); ++j) {
        os << mat(i, j);
        if (j < mat.cols() - 1)
          os << ", ";
      }
      os << "]" << std::endl;
    }
  };
  // Hyperparameters
  Eigen::VectorXd hypers =
      cf->get_loghyper()
          .array()
          .exp(); // Convert log hypers back to normal scale
  outfile << "kernel:" << std::endl;

  // Assuming the last hyperparameter is always noise variance for simple
  // kernels
  double noise = hypers(hypers.size() - 1);
  outfile << "  noise: " << noise << std::endl;

  outfile << "  hyperparameters: [";
  for (size_t i = 0; i < hypers.size() - 1; ++i) {
    outfile << hypers(i);
    if (i < hypers.size() - 2)
      outfile << ", ";
  }
  outfile << "]" << std::endl;

  outfile << "  type: \"" << cf->to_string() << "\"" << std::endl;

  if (sampleset->size() > 0) {
    // Sampling Points
    Eigen::MatrixXd sampling_points =
        Eigen::MatrixXd(sampleset->size(), input_dim);
    for (size_t i = 0; i < sampleset->size(); ++i) {
      sampling_points.row(i) = sampleset->x(i).transpose();
    }

    outfile << "sampling_points:" << std::endl;
    write_matrix_flat(outfile, sampling_points);

    // Sampling Targets
    Eigen::VectorXd sampling_targets = getFlatSamplingTargets();
    outfile << "sampling_targets: " << std::endl;
    write_matrix_flat(outfile, sampling_targets.transpose());
  }
  outfile.close();
  std::cout << "Model exported to " << filename << std::endl;
}

} // namespace libgp

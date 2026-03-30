// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_factory.h"
#include "sparse_gp.h"

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

SparseGaussianProcess::SparseGaussianProcess(size_t input_dim,
                                             std::string covf_def)
    : GaussianProcess(input_dim, covf_def) {
  double input_upper_vector[input_dim];
  double input_lower_vector[input_dim];
  double step_sizes[input_dim];
  int num_steps[input_dim];

  // 定义每个维度的上下界和步长
  for (int i = 0; i < input_dim; i++) {
    input_upper_vector[i] = 2.0;
    input_lower_vector[i] = -2.0;
    num_steps[i] = 6; //这里如果取太大，可能会导致Kuu不可逆，导致估算的方差发散
    step_sizes[i] =
        (input_upper_vector[i] - input_lower_vector[i]) / num_steps[i];
  }

  // 计算诱导集的大小
  int inducingSet_size = 1;
  for (int i = 0; i < input_dim; i++) {
    inducingSet_size *= num_steps[i];
  }

  //初始化诱导集
  inducingset = new SampleSet(input_dim);
  for (int i = 0; i < inducingSet_size; i++) {
    double x[input_dim];

    int index = i;
    for (int j = 0; j < input_dim; j++) {
      int step_index = index % num_steps[j];
      x[j] = input_lower_vector[j] + step_index * step_sizes[j];
      index /= num_steps[j];
    }
    // std::cout << "x: " << x[0] << " " << x[1] << std::endl;
    inducingset->add(x, 0);
  }
  L_K_RR.resize(inducingSet_size, inducingSet_size);
}

SparseGaussianProcess::SparseGaussianProcess(const char *filename) {
  std::string fname(filename);
  YAML::Node config = YAML::LoadFile(filename);
  try {
    // 1. Sampling Points
    sampleset = new SampleSet(input_dim);
    if (config["sampling_points"]) {
      for (const auto &pt : config["sampling_points"]) {
        std::vector<double> vec = pt.as<std::vector<double>>();
        Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
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
    // 4. Inducing Points
    inducingset = new SampleSet(input_dim);
    if (config["inducing_points"]) {
      for (const auto &pt : config["inducing_points"]) {
        std::vector<double> vec = pt.as<std::vector<double>>();
        Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
        inducingset->add(x, 0.0);
      }
    } else {
      std::cout << "Warning: No inducing points provided in YAML file."
                << std::endl;
      std::cout << " please add inducing points to use the SGP model."
                << std::endl;
    }
    size_t m = inducingset->size();
    // 5. inducing posterior mean
    Eigen::VectorXd mean_u(m);
    if (config["inducing_targets"]) {
      std::vector<std::vector<double>> mean_vecs =
          config["inducing_targets"].as<std::vector<std::vector<double>>>();
      if (!mean_vecs.empty()) {
        std::vector<double> &v = mean_vecs[0];
        mean_u = Eigen::Map<Eigen::VectorXd>(v.data(), v.size());
        for (size_t i = 0; i < v.size(); ++i) {
          inducingset->set_y(i, v[i]);
        }
      }
    } else {
      std::cout << "Warning: No inducing targets provided in YAML file."
                << std::endl;
      std::cout << " please add inducing targets to use the SGP model."
                << std::endl;
    }

    // 6. inducing posterior covariance
    cov_inducing.resize(m, m);
    if (config["inducing_cov"]) {
      std::vector<std::vector<double>> cov_rows =
          config["inducing_cov"].as<std::vector<std::vector<double>>>();
      for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
          cov_inducing(i, j) = cov_rows[i][j];
        }
      }
    }

    std::cout << "Initialized SparseGP from YAML model: " << filename
              << std::endl;

  } catch (...) {
    std::cerr << "Error loading SGP model from YAML file." << std::endl;
  }
}

SparseGaussianProcess::~SparseGaussianProcess() { // free memory
  if (inducingset != NULL)
    delete inducingset;
}

double SparseGaussianProcess::f(const double x[]) {

  if (inducingset->empty())
    return 0;

  Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
  compute();
  update_k_star(x_star);
  return k_star.dot(alpha_R);
}

double SparseGaussianProcess::var(const double x[]) {
  if (inducingset->empty())
    return 0;
  Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
  compute();
  update_k_star(x_star);
  double output = cf->get(x_star, x_star) - k_star.dot(Q_pred * k_star);
  if (output < 0.0) {
    // Numerical round-off can produce tiny negative variances; clamp to zero.
    if (output > -1e-12)
      output = 0.0;
    else {
      std::cerr << "Warning: predictive variance negative (" << output
                << "), clamping to zero." << std::endl;
      output = 0.0;
    }
  }
  return output;
}

void SparseGaussianProcess::pred_diag_derivative(Eigen::VectorXd &mean_deriv) {
  if (input_dim > 1) {
    std::cerr << "Error: pred_diag_derivative now only supports 1D input."
              << std::endl;
    return;
  }
  Eigen::VectorXd mean_deriv_temp;
  mean_deriv.resize(sampleset->size());
  mean_deriv.setZero();
  for (size_t i = 0; i < sampleset->size(); ++i) {
    for (size_t j = 0; j < inducingset->size(); ++j) {
      cf->grad_wrt_x1(sampleset->x(i), inducingset->x(j), mean_deriv_temp);
      mean_deriv(i) += mean_deriv_temp(0) * alpha_R(j);
    }
  }
}

void SparseGaussianProcess::pred_diag_derivative(
    const std::vector<Eigen::VectorXd> &testset, Eigen::VectorXd &mean_deriv) {
  size_t N = testset.size();
  mean_deriv.resize(N);
  mean_deriv.setZero();
  Eigen::VectorXd mean_deriv_temp;
  for (size_t i = 0; i < testset.size(); ++i) {
    Eigen::VectorXd x_star = testset[i];
    for (size_t j = 0; j < inducingset->size(); ++j) {
      cf->grad_wrt_x1(x_star, inducingset->x(j), mean_deriv_temp);
      mean_deriv(i) += mean_deriv_temp(0) * alpha_R(j);
    }
  }
}

void SparseGaussianProcess::compute() {
  if (!cf->loghyper_changed && !alpha_needs_update)
    return;
  cf->loghyper_changed = false;
  size_t m = inducingset->size();
  size_t n = sampleset->size();

  // calculate Kernel matrix K_RX from inducing set
  Eigen::MatrixXd K_RX, K_RR;
  K_RX.resize(m, n);
  K_RX.setZero();
  K_RR.resize(m, m);
  K_RR.setZero();

  computeKernelMatrix(K_RX, inducingset, sampleset, cf);
  computeKernelMatrixLowerHalf(K_RR, inducingset, cf);

  double noise_variance =
      std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2);

  K_RR.diagonal().array() -= noise_variance; // subtract noise variance
  K_RR.diagonal().array() += 1e-6;           // jitter for numerical stability
  L_K_RR = K_RR.selfadjointView<Eigen::Lower>().llt().matrixL();

  // Calculate covariance matrix of inducing set \Sigma_u = K_{uu} - v^double v
  U.resize(m, n);
  U = L_K_RR.triangularView<Eigen::Lower>().solve(K_RX);

  // Q_ff_diag = diag(U^T * U) = colSquaredNorm(U)
  Eigen::VectorXd Q_ff_diag = U.colwise().squaredNorm();

  Eigen::VectorXd K_XX_diag(n);
  for (size_t i = 0; i < n; ++i) {
    K_XX_diag(i) = cf->get(sampleset->x(i), sampleset->x(i)) - noise_variance;
  }
  K_XX_diag.array() += 1e-6; // jitter for numerical stability

  lambda.resize(n);
  lambda = param_alpha * (K_XX_diag - Q_ff_diag);
  lambda.array() += noise_variance;

  // 4. 计算 Woodbury 中间矩阵 B = I + U * Lambda^{-1} * U^T
  Eigen::VectorXd inv_lambda = lambda.cwiseInverse();

  // U_scaled = U * diag(inv_lambda)
  Eigen::MatrixXd U_scaled = U;
  for (size_t i = 0; i < n; ++i)
    U_scaled.col(i) *= inv_lambda(i);

  Eigen::MatrixXd B = Eigen::MatrixXd::Identity(m, m);
  // B = I + U_scaled * U^T -> O(M^2 * N)
  B.noalias() += U_scaled * U.transpose();

  L_B.resize(m, m);
  L_B = B.llt().matrixL();

  const std::vector<double> &targets = sampleset->y();
  Eigen::Map<const Eigen::VectorXd> y(&targets[0], n);
  Eigen::VectorXd y_tilde = y.cwiseProduct(inv_lambda); // Lambda^{-1} * y

  Eigen::VectorXd v = U * y_tilde; // U * Lambda^{-1} * y

  L_B.triangularView<Eigen::Lower>().solveInPlace(v); // L_B^{-1} * v
  L_B.triangularView<Eigen::Lower>().adjoint().solveInPlace(v);

  // alpha = Lambda^{-1} y - Lambda^{-1} U^T B^{-1} U Lambda^{-1} y
  //       = y_tilde - Lambda^{-1} U^T z
  alpha.resize(n);
  alpha =
      y_tilde - (U.transpose() * v)
                    .cwiseProduct(inv_lambda); // Lambda^{-1} * y - H^T * B^{-1}
                                               // * H * Lambda^{-1} * y

  Eigen::VectorXd temp = U * alpha;
  alpha_R.resize(m);
  alpha_R = L_K_RR.triangularView<Eigen::Lower>().adjoint().solve(temp);

  // compute the posterior mean of inducing points
  K_RR = K_RR.selfadjointView<Eigen::Lower>();
  Eigen::VectorXd mean_u = K_RR * alpha_R;
  for (size_t i = 0; i < m; ++i) {
    inducingset->set_y(i, mean_u(i));
  }

  // Q_pred = L_uu^{-T} * (I - B^{-1}) * L_uu^{-1}
  // var = k** - k*u * Q_pred * ku*
  Eigen::MatrixXd I_M = Eigen::MatrixXd::Identity(m, m);
  Eigen::MatrixXd B_inv = I_M;
  L_B.triangularView<Eigen::Lower>().solveInPlace(B_inv);
  L_B.triangularView<Eigen::Lower>().adjoint().solveInPlace(B_inv);

  Eigen::MatrixXd M_mid = I_M - B_inv; // M x M 对称矩阵

  // 计算 L^{-1}
  Eigen::MatrixXd L_inv = I_M;
  L_K_RR.triangularView<Eigen::Lower>().solveInPlace(L_inv);

  // 组合： Q = L^{-T} * M_mid * L^{-1}
  // Q = (L^{-1})^T * M_mid * L^{-1}
  Q_pred = L_inv.transpose() * M_mid * L_inv;

  alpha_needs_update = false;
}

void SparseGaussianProcess::update_k_star(const Eigen::VectorXd &x_star) {
  size_t m = inducingset->size();
  k_star.resize(m);
  for (size_t i = 0; i < m; ++i) {
    k_star(i) = cf->get(x_star, inducingset->x(i));
  }
}

void SparseGaussianProcess::add_pattern(const double x[], double y) {
  /* filling sample set */
  sampleset->add(x, y);
  size_t n = sampleset->size();
  if (n % 100 == 0) {
    std::cout << "added " << n << " samples to training set." << std::endl;
  }
}

void SparseGaussianProcess::specify_inducingSet(
    std::vector<Eigen::VectorXd> inducing_points, size_t m_random,
    size_t m_clusters) {
  if (!inducing_points.empty()) {
    inducingset->clear();
    for (const auto &point : inducing_points) {
      inducingset->add(point.data(), 0);
    }
    size_t m = inducingset->size();
    std::cout << "specify " << m << " inducing points from given set "
              << std::endl;
    // manual specification of inducing points
  } else if (m_random > 0) {
    size_t inducingSet_size =
        std::min(m_random, static_cast<size_t>(0.5 * sampleset->size()));
    std::srand(std::time(0));
    inducingset = new SampleSet(input_dim);
    //随机抽取一定数量的点作为诱导点
    std::default_random_engine generator(std::time(0));
    std::uniform_int_distribution<int> distribution(0, sampleset->size() - 1);
    for (int i = 0; i < inducingSet_size; i++) {
      int index = distribution(generator);
      inducingset->add(sampleset->x(index).data(), 0);
    }
    inducingSet_size = inducingset->size();
    std::cout << "initialize " << inducingSet_size
              << " inducing points randomly from sample set " << std::endl;
    // random selection of inducing points
  } else if (m_clusters > 0) {
    // Use K-means clustering to select inducing points
    size_t inducingSet_size =
        std::min(m_clusters, static_cast<size_t>(sampleset->size()));
    inducingset = new SampleSet(input_dim);

    // Initialize centroids randomly from the dataset
    std::vector<Eigen::VectorXd> centroids(inducingSet_size);
    std::vector<int> labels(sampleset->size());
    std::default_random_engine generator(std::time(0));
    std::uniform_int_distribution<int> distribution(0, sampleset->size() - 1);

    for (size_t i = 0; i < inducingSet_size; ++i) {
      int index = distribution(generator);
      centroids[i] = sampleset->x(index);
    }

    int max_iterations = 100;
    for (int iter = 0; iter < max_iterations; ++iter) {
      bool changed = false;
      std::vector<int> counts(inducingSet_size, 0);
      std::vector<Eigen::VectorXd> new_centroids(
          inducingSet_size, Eigen::VectorXd::Zero(input_dim));

      // Assignment step
      for (size_t i = 0; i < sampleset->size(); ++i) {
        double min_dist = 0.1;
        int best_cluster = 0;
        for (size_t j = 0; j < inducingSet_size; ++j) {
          double dist = (sampleset->x(i) - centroids[j]).squaredNorm();
          if (dist < min_dist) {
            min_dist = dist;
            best_cluster = j;
          }
        }
        if (labels[i] != best_cluster) {
          labels[i] = best_cluster;
          changed = true;
        }
        new_centroids[best_cluster] += sampleset->x(i);
        counts[best_cluster]++;
      }

      // Update step
      for (size_t j = 0; j < inducingSet_size; ++j) {
        if (counts[j] > 0) {
          centroids[j] = new_centroids[j] / counts[j];
        } else {
          // Re-initialize empty cluster randomly
          int index = distribution(generator);
          centroids[j] = sampleset->x(index);
        }
      }

      if (!changed)
        break;
    }

    // Add centroids to inducing set
    for (const auto &centroid : centroids) {
      inducingset->add(centroid.data(), 0);
    }
    std::cout << "initialize " << inducingSet_size
              << " inducing points using K-means clustering" << std::endl;
  }
  alpha_needs_update = true;
}

double SparseGaussianProcess::log_likelihood() {
  compute();
  size_t n = sampleset->size();
  const std::vector<double> &targets = sampleset->y();
  Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());

  // 1. Data Fit term: -0.5 * y^T * K^{-1} * y
  double data_fit = -0.5 * y.dot(alpha);

  // 2. Complexity Penalty: -0.5 * log|K|
  // Matrix Determinant Lemma: log|K| = log|Lambda| + log|B|
  double log_det_B = 2 * L_B.diagonal().array().log().sum();
  double log_det_lambda = lambda.array().log().sum();
  double log_det_K = log_det_lambda + log_det_B;

  // 3. Regularizer (Power EP term)
  double regularizer_term = 0.0;
  double noise_variance =
      std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2);

  if (std::abs(param_alpha - 1.0) > 1e-6) {
    Eigen::VectorXd ratio = lambda.array() / noise_variance;
    regularizer_term =
        0.5 * (1 - param_alpha) / param_alpha * ratio.array().log().sum();
  }

  return data_fit - 0.5 * log_det_K - 0.5 * n * log2pi - regularizer_term;
}

Eigen::VectorXd SparseGaussianProcess::log_likelihood_gradient() {
  // calculate gradient of collapsed lower bound to the sparse log marginal
  // likelihood
  compute();
  size_t n = sampleset->size();
  size_t m = inducingset->size();
  size_t grad_dim = cf->get_param_dim() + m * input_dim;
  Eigen::VectorXd grad(grad_dim);
  grad.setZero();

  // W = K_XX_bar^(-1) - K_XX_bar^(-1) * y * y^T * K_XX_bar^(-1)
  // K_XX_bar^(-1) = Lambda^{-1} - Lambda^{-1} * U^T * B^{-1} * U * Lambda^{-1}
  Eigen::VectorXd inv_lambda = lambda.cwiseInverse();
  Eigen::MatrixXd U_scaled = U;
  for (size_t i = 0; i < n; ++i) {
    U_scaled.col(i) *= inv_lambda(i);
  }

  Eigen::MatrixXd Y = L_B.triangularView<Eigen::Lower>().solve(U_scaled);
  Eigen::MatrixXd W = -Y.transpose() * Y;
  W.diagonal() += inv_lambda;
  W -= alpha * alpha.transpose();

  Eigen::MatrixXd Diag_W = W.diagonal().asDiagonal().toDenseMatrix();

  double noise_variance =
      std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2);
  Eigen::MatrixXd Diag_sigmaNoise_alphaDx_Inv =
      inv_lambda.asDiagonal().toDenseMatrix();

  Eigen::MatrixXd H_T =
      L_K_RR.triangularView<Eigen::Lower>().adjoint().solve(U);
  Eigen::MatrixXd H = H_T.transpose();

  Eigen::MatrixXd G_RX, G_RR;
  G_RX.resize(m, n);
  G_RR.resize(m, m);
  G_RX = H_T * (W - param_alpha * Diag_W -
                (1 - param_alpha) * Diag_sigmaNoise_alphaDx_Inv);
  G_RR = -G_RX * H;
  Eigen::VectorXd G_XX_diag =
      W.diagonal() +
      (1 - param_alpha) / param_alpha * Diag_sigmaNoise_alphaDx_Inv.diagonal();

  Eigen::VectorXd g_cov(cf->get_param_dim()); // temporary gradient storage
  Eigen::VectorXd g_ind(
      input_dim); // temporary gradient storage for inducing point location
  // gradient loop over K_RX
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      g_cov.setZero();
      g_ind.setZero();
      cf->grad(inducingset->x(i), sampleset->x(j), g_cov);
      cf->grad_wrt_x1(inducingset->x(i), sampleset->x(j), g_ind);
      grad.head(cf->get_param_dim()) -= G_RX(i, j) * g_cov;
      grad.segment(cf->get_param_dim() + i * input_dim, input_dim) -=
          G_RX(i, j) * g_ind;
    }
  }

  // gradient loop over K_RR
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      g_cov.setZero();
      g_ind.setZero();
      cf->grad(inducingset->x(i), inducingset->x(j), g_cov);
      if (i == j) {
        g_cov[cf->get_param_dim() - 1] = 0; // derivative wrt noise variance
        grad.head(cf->get_param_dim()) -= G_RR(i, j) * g_cov * 0.5;
      } else {
        grad.head(cf->get_param_dim()) -= G_RR(i, j) * g_cov;
        cf->grad_wrt_x1(inducingset->x(i), inducingset->x(j), g_ind);
        grad.segment(cf->get_param_dim() + i * input_dim, input_dim) -=
            G_RR(i, j) * g_ind;
        grad.segment(cf->get_param_dim() + j * input_dim, input_dim) +=
            G_RR(i, j) * g_ind;
      }
    }
  }

  // gradient loop over K_XX
  for (size_t i = 0; i < n; ++i) {
    g_cov.setZero();
    cf->grad(sampleset->x(i), sampleset->x(i), g_cov);
    double grad_noise = g_cov[cf->get_param_dim() - 1];
    g_cov *= param_alpha;
    g_cov[cf->get_param_dim() - 1] = grad_noise;
    grad.head(cf->get_param_dim()) -= G_XX_diag(i) * g_cov * 0.5;
  }

  grad[cf->get_param_dim() - 1] += (1 - param_alpha) / param_alpha * n;

  return grad;
}

void SparseGaussianProcess::update_hyperparameters(
    const Eigen::VectorXd &params) {
  cf->set_loghyper(params.head(cf->get_param_dim()));
  size_t m = inducingset->size();
  inducingset->clear();
  for (size_t i = 0; i < m; ++i) {
    Eigen::VectorXd x =
        params.segment(cf->get_param_dim() + i * input_dim, input_dim);
    inducingset->add(x, 0.0);
  }
  alpha_needs_update = true;
}

Eigen::VectorXd SparseGaussianProcess::get_hyperparameter_lower_bound() {
  Eigen::VectorXd lb;
  size_t m = inducingset->size();
  lb.resize(cf->get_param_dim() + m * input_dim);
  lb.head(cf->get_param_dim()) = cf->get_loghyper_lb();
  for (size_t i = 0; i < m; ++i) {
    lb.segment(cf->get_param_dim() + i * input_dim, input_dim) =
        inducingset->x(i) - 0.5 * Eigen::VectorXd::Ones(input_dim);
  }
  return lb;
}

Eigen::VectorXd SparseGaussianProcess::get_hyperparameter_upper_bound() {
  Eigen::VectorXd ub;
  size_t m = inducingset->size();
  ub.resize(cf->get_param_dim() + m * input_dim);
  ub.head(cf->get_param_dim()) = cf->get_loghyper_ub();
  for (size_t i = 0; i < m; ++i) {
    ub.segment(cf->get_param_dim() + i * input_dim, input_dim) =
        inducingset->x(i) + 0.5 * Eigen::VectorXd::Ones(input_dim);
  }
  return ub;
}

void SparseGaussianProcess::update_variational_parameters(
    const Eigen::VectorXd &params) {
  if (params.size() != inducingset->size() * input_dim) {
    std::cerr << "Error: Invalid size for variational parameters." << std::endl;
    return;
  }
  inducingset->clear();
  for (size_t i = 0; i < inducingset->size(); ++i) {
    Eigen::VectorXd x = params.segment(i * input_dim, input_dim);
    inducingset->add(x, 0.0);
  }
  alpha_needs_update = true;
}

Eigen::VectorXd SparseGaussianProcess::get_variational_parameters() {
  Eigen::VectorXd params(inducingset->size() * input_dim);
  for (size_t i = 0; i < inducingset->size(); ++i) {
    params.segment(i * input_dim, input_dim) = inducingset->x(i);
  }
  return params;
}

Eigen::VectorXd SparseGaussianProcess::get_hyperparameters() {
  Eigen::VectorXd params;
  Eigen::VectorXd cov_params = cf->get_loghyper();
  // return cov_params;
  size_t m = inducingset->size();
  params.resize(cov_params.size() + m * input_dim);
  params.head(cov_params.size()) = cov_params;
  for (size_t i = 0; i < m; ++i) {
    params.segment(cov_params.size() + i * input_dim, input_dim) =
        inducingset->x(i);
  }
  return params;
}

Eigen::MatrixXd SparseGaussianProcess::getFlatInputs() {
  if (inducingset->size() < 1) {
    std::cout << "Inducing set is empty." << std::endl;
    return Eigen::MatrixXd();
  }
  size_t m = inducingset->size();
  Eigen::MatrixXd flat_inducing_set(cf->get_input_dim(), m);
  for (size_t i = 0; i < m; ++i) {
    flat_inducing_set.col(i) = inducingset->x(i);
  }
  return flat_inducing_set;
}

Eigen::VectorXd SparseGaussianProcess::getFlatTargets() {
  if (inducingset->size() < 1) {
    std::cout << "Inducing set is empty." << std::endl;
    return Eigen::VectorXd();
  }
  const std::vector<double> &targets = inducingset->y();
  Eigen::Map<const Eigen::VectorXd> flat_inducing_targets(&targets[0],
                                                          targets.size());
  return flat_inducing_targets;
}

Eigen::MatrixXd SparseGaussianProcess::getFlatPosteriorCovMatrix() {
  if (inducingset->size() < 1) {
    std::cout << "Inducing set is empty." << std::endl;
    return Eigen::MatrixXd();
  }
  compute();
  Eigen::MatrixXd inducingPosteriorCov;
  // $$ \Sigma_u = L_{K_RR} B^{-1} L_{K_RR}^T $$
  Eigen::MatrixXd L_K_RR_T = L_K_RR.transpose();
  Eigen::MatrixXd X = L_B.triangularView<Eigen::Lower>().solve(L_K_RR_T);
  inducingPosteriorCov = X.transpose() * X;

  return inducingPosteriorCov;
}

Eigen::VectorXd SparseGaussianProcess::getFlatHyperparameters() {
  return cf->get_loghyper().array().exp();
}

Eigen::VectorXd SparseGaussianProcess::getFlatAlpha() {
  if (inducingset->size() < 1) {
    std::cout << "Inducing set is empty." << std::endl;
    return Eigen::VectorXd();
  }
  compute();
  return alpha_R;
}

void SparseGaussianProcess::exportModelToYAML(const char *filename) {
  compute(); // Ensure matrices are up to date

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

  // 1. Inducing Points
  Eigen::MatrixXd inducing_points = getFlatInputs();

  outfile << "inducing_points:" << std::endl;
  write_matrix_flat(outfile, inducing_points.transpose());

  // 2. Variational Mean (Posterior mean of inducing points)
  Eigen::VectorXd mean_u = getFlatTargets();

  outfile << "inducing_targets: " << std::endl;
  write_matrix_flat(outfile, mean_u.transpose());

  // 3. Variational Covariance (Posterior covariance of inducing points)
  Eigen::MatrixXd cov_u = getFlatPosteriorCovMatrix();
  outfile << "inducing_cov: " << std::endl;
  write_matrix_flat(outfile, cov_u);

  // 4. Hyperparameters
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
    // 5. Sampling Points
    Eigen::MatrixXd sampling_points =
        Eigen::MatrixXd(sampleset->size(), input_dim);
    for (size_t i = 0; i < sampleset->size(); ++i) {
      sampling_points.row(i) = sampleset->x(i).transpose();
    }

    outfile << "sampling_points:" << std::endl;
    write_matrix_flat(outfile, sampling_points);

    // 6. Sampling Targets
    Eigen::VectorXd sampling_targets = getFlatSamplingTargets();
    outfile << "sampling_targets: " << std::endl;
    write_matrix_flat(outfile, sampling_targets.transpose());
  }

  outfile.close();
  std::cout << "Model exported to " << filename << std::endl;
}

} // namespace libgp
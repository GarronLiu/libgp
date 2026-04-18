// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_factory.h"
#include "sparse_gp.h"

#include <chrono>
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

  // 初始化在线数据集
  batchset = new SampleSet(input_dim);
}

SparseGaussianProcess::SparseGaussianProcess(const char *filename) {
  std::string fname(filename);
  YAML::Node config = YAML::LoadFile(filename);
  try {
    // 1.Kernel & Hyperparameters
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
    // 2. Sampling Points & Sampling Targets
    std::cout << "Loading SGP model from YAML file: " << filename << std::endl;
    sampleset = new SampleSet(input_dim);
    if (config["sampling_points"] && config["sampling_points"].size() > 0) {
      for (const auto &pt : config["sampling_points"]) {
        std::vector<double> vec = pt.as<std::vector<double>>();
        Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
        sampleset->add(x, 0.0);
      }
      size_t n = sampleset->size();
      Eigen::VectorXd mean(n);
      std::vector<std::vector<double>> mean_vecs =
          config["sampling_targets"].as<std::vector<std::vector<double>>>();
      if (!mean_vecs.empty()) {
        std::vector<double> &v = mean_vecs[0];
        mean = Eigen::Map<Eigen::VectorXd>(v.data(), v.size());
        for (size_t i = 0; i < v.size(); ++i) {
          sampleset->set_y(i, v[i]);
        }
      }
    } else {
      std::cout << "Warning: No sampling points provided in YAML file."
                << std::endl;
      std::cout << " please add sampling points to use the GP model."
                << std::endl;
    }
    // 4. Inducing Points & inducing posterior mean & inducing posterior
    // covariance
    inducingset = new SampleSet(input_dim);
    if (config["inducing_points"] && config["inducing_targets"] &&
        config["inducing_cov"]) {
      for (const auto &pt : config["inducing_points"]) {
        std::vector<double> vec = pt.as<std::vector<double>>();
        Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
        inducingset->add(x, 0.0);
      }
      size_t m = inducingset->size();
      Eigen::VectorXd mean_u(m);
      std::vector<std::vector<double>> mean_vecs =
          config["inducing_targets"].as<std::vector<std::vector<double>>>();
      if (!mean_vecs.empty()) {
        std::vector<double> &v = mean_vecs[0];
        mean_u = Eigen::Map<Eigen::VectorXd>(v.data(), v.size());
        for (size_t i = 0; i < v.size(); ++i) {
          inducingset->set_y(i, v[i]);
        }
      }
      cov_inducing.resize(m, m);
      std::vector<std::vector<double>> cov_rows =
          config["inducing_cov"].as<std::vector<std::vector<double>>>();
      for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
          cov_inducing(i, j) = cov_rows[i][j];
        }
      }

      cf->loghyper_changed = false;
      alpha_needs_update = false;

      Eigen::MatrixXd K_RR;
      K_RR.resize(m, m);
      computeKernelMatrixLowerHalf(K_RR, inducingset, cf);
      double noise_variance =
          std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2);
      K_RR.diagonal().array() -= noise_variance;
      L_K_RR = chol_lower(K_RR);
      K_RR = K_RR.selfadjointView<Eigen::Lower>();

      // precompute alpha_R = L_K_RR^{-T} * mean_inducing for fast prediction
      alpha_R.resize(m);
      alpha_R = L_K_RR.triangularView<Eigen::Lower>().solve(mean_u);
      L_K_RR.triangularView<Eigen::Lower>().adjoint().solveInPlace(alpha_R);

      // precompute Q_pred = K_** - K_*R * K_RR^{-1} * K_R* for fast variance
      // prediction
      Q_pred.resize(m, m);
      Q_pred.setZero();
      Eigen::MatrixXd K_diff = K_RR - cov_inducing;
      Eigen::MatrixXd iK_RR = chol_inverse(K_RR);
      Q_pred.noalias() = iK_RR * K_diff * iK_RR;

    } else {
      std::cout << "Warning: No inducing points provided in YAML file."
                << std::endl;
      std::cout << " please add inducing points to use the SGP model."
                << std::endl;
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
  if (inducingset_pre != NULL)
    delete inducingset_pre;
  if (batchset != NULL)
    delete batchset;
}

double SparseGaussianProcess::f(const double x[]) {
  if (inducingset->empty())
    return 0;

  Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
  compute();
  update_alpha();
  update_k_star(x_star);
  return k_star.dot(alpha_R);
}

double SparseGaussianProcess::var(const double x[]) {
  if (inducingset->empty())
    return 0;
  Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
  compute();
  update_alpha();
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

  if (!cf->loghyper_changed)
    return;

  if (inducingset->empty())
    return;

  if (stream_update_mode) {
    if (batchset->empty())
      return;
    size_t param_dim = cf->get_param_dim();
    size_t m_a = inducingset_pre->size();
    size_t m = inducingset->size();
    size_t b = batchset->size();
    double noise_variance = std::exp(cf->get_loghyper()(param_dim - 1) * 2);

    Eigen::MatrixXd K_RX, K_RR, K_Ra;
    K_RX.resize(m, b);
    K_RX.setZero();
    K_RR.resize(m, m);
    K_RR.setZero();
    K_Ra.resize(m, m_a);
    K_Ra.setZero();

    computeKernelMatrix(K_RX, inducingset, batchset, cf);
    computeKernelMatrixLowerHalf(K_RR, inducingset, cf);
    computeKernelMatrix(K_Ra, inducingset, inducingset_pre, cf);

    K_RR.diagonal().array() -= noise_variance;
    K_RR = K_RR.selfadjointView<Eigen::Lower>();
    L_K_RR = chol_lower(K_RR);

    U.resize(m, b);
    U = L_K_RR.triangularView<Eigen::Lower>().solve(K_RX);
    U_a.resize(m, m_a);
    U_a = L_K_RR.triangularView<Eigen::Lower>().solve(K_Ra);

    Eigen::VectorXd Q_ff_diag = U.colwise().squaredNorm();

    Eigen::VectorXd K_XX_diag(b);
    for (size_t i = 0; i < b; i++) {
      K_XX_diag(i) = cf->get(batchset->x(i), batchset->x(i)) - noise_variance;
    }

    lambda.resize(b);
    lambda = param_alpha * (K_XX_diag - Q_ff_diag);
    lambda.array() += noise_variance;

    Eigen::VectorXd inv_lambda = lambda.cwiseInverse();

    lambda_a.resize(m_a, m_a);
    Eigen::MatrixXd K_aa(m_a, m_a);
    computeKernelMatrixLowerHalf(K_aa, inducingset_pre, cf);
    K_aa.diagonal().array() -= noise_variance;
    K_aa = K_aa.selfadjointView<Eigen::Lower>();
    Eigen::MatrixXd Q_aa = U_a.transpose() * U_a;
    lambda_a.noalias() = P_pre + param_alpha * (K_aa - Q_aa);
    // lambda_a.diagonal().array() += 0.01;  //
    // 添加遗忘因子，使得旧数据的影响逐渐减弱

    Eigen::MatrixXd U_scaled = U;
    for (size_t i = 0; i < b; ++i) {
      U_scaled.col(i) *= inv_lambda(i);
    }
    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(m, m);
    B.noalias() += U_scaled * U.transpose();

    L_B.resize(m, m);
    L_B = chol_lower(B);

    Eigen::MatrixXd inv_A(b, b);
    Eigen::MatrixXd temp = L_B.triangularView<Eigen::Lower>().solve(U_scaled);
    inv_A.noalias() = -temp.transpose() * temp;
    inv_A.diagonal() += inv_lambda;

    temp.resize(m_a, b);
    temp = U_a.transpose() * U;
    Eigen::MatrixXd C_inv_A = temp * inv_A;

    Eigen::MatrixXd M_shur(m_a, m_a);
    M_shur.noalias() = Q_aa + lambda_a - C_inv_A * temp.transpose();
    L_M_shur.resize(m_a, m_a);
    L_M_shur = chol_lower(M_shur);

    const std::vector<double> &targets = batchset->y();
    Eigen::Map<const Eigen::VectorXd> y(&targets[0], b);

    temp.resize(m_a, b);
    temp = L_M_shur.triangularView<Eigen::Lower>().solve(C_inv_A);
    Eigen::VectorXd y_tilde = temp * y;

    alpha.resize(b + m_a);
    alpha.head(b) = temp.transpose() * y_tilde;
    alpha.head(b) += inv_A * y;
    alpha.tail(m_a) =
        -L_M_shur.triangularView<Eigen::Lower>().adjoint().solve(y_tilde);

    Eigen::VectorXd y_a_tilde =
        L_M_shur.triangularView<Eigen::Lower>().solve(y_a);
    alpha.head(b) -= temp.transpose() * y_a_tilde;
    alpha.tail(m_a) +=
        L_M_shur.triangularView<Eigen::Lower>().adjoint().solve(y_a_tilde);

    Eigen::VectorXd mean_inducing;
    mean_inducing.resize(m);
    mean_inducing.setZero();
    mean_inducing.noalias() = K_RX * alpha.head(b) + K_Ra * alpha.tail(m_a);
    for (size_t i = 0; i < m; ++i) {
      inducingset->set_y(i, mean_inducing(i));
    }

  } else {
    if (sampleset->empty())
      return;
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

    // Calculate covariance matrix of inducing set \Sigma_u = K_{uu} - v^double
    // v
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
    alpha = y_tilde -
            (U.transpose() * v)
                .cwiseProduct(inv_lambda); // Lambda^{-1} * y - H^T * B^{-1}
                                           // * H * Lambda^{-1} * y
    Eigen::VectorXd mean_inducing = L_K_RR * U * alpha;
    for (size_t i = 0; i < m; ++i) {
      inducingset->set_y(i, mean_inducing(i));
    }
  }

  cf->loghyper_changed = false;
  llm_calculatable_flag = true;
  alpha_needs_update = true;
}

void SparseGaussianProcess::update_k_star(const Eigen::VectorXd &x_star) {
  size_t m = inducingset->size();
  k_star.resize(m);
  for (size_t i = 0; i < m; ++i) {
    k_star(i) = cf->get(x_star, inducingset->x(i));
  }
}

void SparseGaussianProcess::update_alpha() {
  if (!alpha_needs_update)
    return;
  alpha_needs_update = false;
  size_t m = inducingset->size();

  if (stream_update_mode) {
    size_t m_a = inducingset_pre->size();

    Eigen::MatrixXd L_K_RR_T = L_K_RR.transpose();
    Eigen::MatrixXd X = L_B.triangularView<Eigen::Lower>().solve(L_K_RR_T);
    cov_inducing = X.transpose() * X;

    Eigen::MatrixXd U_scaled = U;
    Eigen::VectorXd inv_lambda = lambda.cwiseInverse();
    for (size_t i = 0; i < batchset->size(); ++i) {
      U_scaled.col(i) *= inv_lambda(i);
    }

    Eigen::MatrixXd U_scaled_UT = U_scaled * U.transpose();
    Eigen::MatrixXd temp;
    temp.resize(m, m);
    temp = L_B.triangularView<Eigen::Lower>().solve(U_scaled_UT);
    Eigen::MatrixXd v(m_a, m);
    v.noalias() =
        U_a.transpose() * (U_scaled_UT - temp.transpose() * temp) * L_K_RR_T;
    L_M_shur.triangularView<Eigen::Lower>().solveInPlace(v);
    cov_inducing -= v.transpose() * v;

    Eigen::MatrixXd v_a(m_a, m);
    v_a.noalias() = U_a.transpose() * L_K_RR_T;

    L_M_shur.triangularView<Eigen::Lower>().solveInPlace(v_a);
    cov_inducing -= v_a.transpose() * v_a;

    cov_inducing += v.transpose() * v_a;
    cov_inducing += v_a.transpose() * v;

  } else {
    // $$ \Sigma_u = L_{K_RR} B^{-1} L_{K_RR}^T $$
    Eigen::MatrixXd L_K_RR_T = L_K_RR.transpose();
    Eigen::MatrixXd X = L_B.triangularView<Eigen::Lower>().solve(L_K_RR_T);
    cov_inducing = X.transpose() * X;
  }

  // precompute alpha_R = L_K_RR^{-T} * mean_inducing for fast prediction
  const std::vector<double> &targets = inducingset->y();
  Eigen::Map<const Eigen::VectorXd> mean_inducing(&targets[0], m);
  alpha_R.resize(m);
  alpha_R = L_K_RR.triangularView<Eigen::Lower>().solve(mean_inducing);
  L_K_RR.triangularView<Eigen::Lower>().adjoint().solveInPlace(alpha_R);

  // precompute Q_pred for fast variance prediction
  Q_pred.resize(m, m);
  Q_pred.setZero();
  Eigen::MatrixXd K_RR = L_K_RR * L_K_RR.transpose();
  Eigen::MatrixXd K_diff = K_RR - cov_inducing;
  Eigen::MatrixXd iK_RR = chol_inverse(K_RR);
  Q_pred.noalias() = iK_RR * K_diff * iK_RR;
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
  cf->loghyper_changed = true;
}

double SparseGaussianProcess::log_likelihood() {
  compute();

  if (!llm_calculatable_flag) {
    std::cerr << "Error: log-likelihood not calculatable. Please check if the "
                 "model is properly initialized and trained."
              << std::endl;
    return std::numeric_limits<double>::quiet_NaN();
  }

  if (stream_update_mode) {

    size_t b = batchset->size();
    size_t m_a = inducingset_pre->size();

    double noise_variance =
        std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2);
    Eigen::VectorXd y(b + m_a);
    y.head(b) = Eigen::Map<const Eigen::VectorXd>(&batchset->y()[0], b);
    y.tail(m_a) = y_a;
    double data_fit = y.dot(alpha);

    double log_det_B = 2 * L_B.diagonal().array().log().sum();
    double log_det_lambda = lambda.array().log().sum();
    double log_det_M_shur = 2 * L_M_shur.diagonal().array().log().sum();
    double log_det_K = log_det_lambda + log_det_B + log_det_M_shur;

    double regularizer_term = 0.0;
    regularizer_term -= static_cast<double>(b) * (1 - param_alpha) /
                        param_alpha * std::log(noise_variance);

    regularizer_term += (1 - param_alpha) / param_alpha * log_det_lambda;
    regularizer_term -= (1 - param_alpha) / param_alpha * logDet(lambda_a);

    return -0.5 * (static_cast<double>(b) * log2pi + log_det_K + data_fit +
                   regularizer_term + elbo_constant_init);

  } else {
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

  if (stream_update_mode) {
    /// TODO:
    size_t b = batchset->size();
    size_t m_a = inducingset_pre->size();
    Eigen::MatrixXd inv_A, H_T, H_a_T;
    H_T = L_K_RR.triangularView<Eigen::Lower>().adjoint().solve(U);
    H_a_T = L_K_RR.triangularView<Eigen::Lower>().adjoint().solve(U_a);
    Eigen::MatrixXd H = H_T.transpose();
    Eigen::MatrixXd H_a = H_a_T.transpose();

    Eigen::VectorXd inv_lambda = lambda.cwiseInverse();
    Eigen::MatrixXd U_scaled = U;
    for (size_t i = 0; i < b; ++i) {
      U_scaled.col(i) *= inv_lambda(i);
    }

    inv_A.resize(b, b);
    Eigen::MatrixXd temp = L_B.triangularView<Eigen::Lower>().solve(U_scaled);
    inv_A.noalias() = -temp.transpose() * temp;
    inv_A.diagonal() += inv_lambda;

    Eigen::MatrixXd W_11, W_12, W_21, W_22;
    W_11.resize(b, b);
    W_12.resize(b, m_a);
    W_21.resize(m_a, b);
    W_22.resize(m_a, m_a);

    W_21.noalias() = U_a.transpose() * U * inv_A;
    L_M_shur.triangularView<Eigen::Lower>().solveInPlace(W_21);
    W_11.noalias() = inv_A + W_21.transpose() * W_21;

    L_M_shur.triangularView<Eigen::Lower>().adjoint().solveInPlace(W_21);
    W_21.array() *= -1.0;

    W_22 = Eigen::MatrixXd::Identity(m_a, m_a);
    L_M_shur.triangularView<Eigen::Lower>().solveInPlace(W_22);
    L_M_shur.triangularView<Eigen::Lower>().adjoint().solveInPlace(W_22);

    W_11 -= alpha.head(b) * alpha.head(b).transpose();
    W_21 -= alpha.tail(m_a) * alpha.head(b).transpose();
    W_12 = W_21.transpose();
    W_22 -= alpha.tail(m_a) * alpha.tail(m_a).transpose();

    Eigen::MatrixXd G_RX, G_RR, G_Ra, G_aa;
    Eigen::VectorXd G_XX;

    Eigen::VectorXd diag_W_11 = W_11.diagonal();

    W_11.diagonal().array() *= (1.0 - param_alpha);

    G_RX = 2.0 * H_T * W_11;
    G_RX += 2.0 * H_a_T * W_21;
    Eigen::MatrixXd H_T_scaled = H_T;
    for (size_t i = 0; i < b; ++i) {
      H_T_scaled.col(i) *= inv_lambda(i);
    }
    G_RX -= 2.0 * (1.0 - param_alpha) * H_T_scaled;

    G_RR = -H_T * W_11 * H;
    G_RR -= H_T * W_12 * H_a;
    G_RR -= H_a_T * W_21 * H;
    G_RR -= (1.0 - param_alpha) * H_a_T * W_22 * H_a;
    G_RR += (1.0 - param_alpha) * H_T_scaled * H;
    Eigen::MatrixXd L_lambda_a = chol_lower(lambda_a);
    temp = L_lambda_a.triangularView<Eigen::Lower>().solve(H_a);
    G_RR -= (1.0 - param_alpha) * temp.transpose() * temp;

    G_Ra = 2.0 * H_T * W_12;
    G_Ra += 2.0 * (1.0 - param_alpha) * H_a_T * W_22;
    L_lambda_a.triangularView<Eigen::Lower>().adjoint().solveInPlace(temp);
    G_Ra += 2.0 * (1.0 - param_alpha) * temp.transpose();

    // G_aa^T = αW_22 - (1-α) Σ_a^{-1}
    G_aa = param_alpha * W_22;
    Eigen::MatrixXd inv_lambda_a = Eigen::MatrixXd::Identity(m_a, m_a);
    L_lambda_a.triangularView<Eigen::Lower>().solveInPlace(inv_lambda_a);
    L_lambda_a.triangularView<Eigen::Lower>().adjoint().solveInPlace(
        inv_lambda_a);
    G_aa -= (1.0 - param_alpha) * inv_lambda_a;

    // G_ff^T = αW_11 + (1-α) Σ_y^{-1}
    G_XX = param_alpha * diag_W_11;
    G_XX += (1.0 - param_alpha) * inv_lambda;

    Eigen::VectorXd g_cov(cf->get_param_dim());
    Eigen::VectorXd g_ind(input_dim);

    // Gradient loop over K_RX
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < b; ++j) {
        g_cov.setZero();
        g_ind.setZero();
        cf->grad(inducingset->x(i), batchset->x(j), g_cov);
        cf->grad_wrt_x1(inducingset->x(i), batchset->x(j), g_ind);
        grad.head(cf->get_param_dim()) += G_RX(i, j) * g_cov;
        grad.segment(cf->get_param_dim() + i * input_dim, input_dim) +=
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
          grad.head(cf->get_param_dim()) += G_RR(i, j) * g_cov;
        } else {
          g_cov[cf->get_param_dim() - 1] = 0; // derivative wrt noise variance
          grad.head(cf->get_param_dim()) += G_RR(i, j) * g_cov * 2.0;
          cf->grad_wrt_x1(inducingset->x(i), inducingset->x(j), g_ind);
          grad.segment(cf->get_param_dim() + i * input_dim, input_dim) +=
              G_RR(i, j) * g_ind * 2.0;
          grad.segment(cf->get_param_dim() + j * input_dim, input_dim) -=
              G_RR(j, i) * g_ind * 2.0;
        }
      }
    }

    // gradient loop over K_Ra
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < m_a; ++j) {
        g_cov.setZero();
        g_ind.setZero();
        cf->grad(inducingset->x(i), inducingset_pre->x(j), g_cov);
        cf->grad_wrt_x1(inducingset->x(i), inducingset_pre->x(j), g_ind);
        grad.head(cf->get_param_dim()) += G_Ra(i, j) * g_cov;
        grad.segment(cf->get_param_dim() + i * input_dim, input_dim) +=
            G_Ra(i, j) * g_ind;
      }
    }

    // gradient loop over K_aa
    for (size_t i = 0; i < m_a; ++i) {
      for (size_t j = 0; j <= i; ++j) {
        g_cov.setZero();
        cf->grad(inducingset_pre->x(i), inducingset_pre->x(j), g_cov);
        g_cov[cf->get_param_dim() - 1] = 0; // derivative wrt noise variance
        if (i == j) {
          grad.head(cf->get_param_dim()) += G_aa(i, j) * g_cov;
        } else {
          grad.head(cf->get_param_dim()) += G_aa(i, j) * g_cov * 2.0;
        }
      }
    }

    // gradient loop over K_XX
    for (size_t i = 0; i < b; ++i) {
      g_cov.setZero();
      cf->grad(batchset->x(i), batchset->x(i), g_cov);
      g_cov[cf->get_param_dim() - 1] = 0.0;
      grad.head(cf->get_param_dim()) += G_XX(i) * g_cov;
    }

    // gradient wrt noise variance (last hyperparameter)
    double noise_variance =
        std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2);
    grad[cf->get_param_dim() - 1] +=
        diag_W_11.array().sum() +
        (1 - param_alpha) / param_alpha * inv_lambda.sum();
    grad[cf->get_param_dim() - 1] *= noise_variance * 2.0;
    grad[cf->get_param_dim() - 1] -=
        (1 - param_alpha) / param_alpha * static_cast<double>(b) * 2.0;

    // 如果不更新原来的诱导点位置
    grad.segment(cf->get_param_dim(), m_a * input_dim).setZero();

    return -0.5 * grad;
  } else {
    // W = K_XX_bar^(-1) - K_XX_bar^(-1) * y * y^T * K_XX_bar^(-1)
    // K_XX_bar^(-1) = Lambda^{-1} - Lambda^{-1} * U^T * B^{-1} * U *
    // Lambda^{-1}
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
        W.diagonal() + (1 - param_alpha) / param_alpha *
                           Diag_sigmaNoise_alphaDx_Inv.diagonal();

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

    if(input_dim > 1){
      grad.segment(cf->get_param_dim(), m * input_dim).setZero(); // not updating inducing point locations
    }

    return grad;
  }
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
}

void SparseGaussianProcess::add_pattern_batch(const Eigen::VectorXd &x,
                                              double y) {
  size_t input_dim = x.size();
  assert(input_dim == this->input_dim);

  if (batchset == NULL) {
    std::cerr << "Error: batchset is not initialized. Call "
                 "storePosteriorPretrained() "
                 "before adding new patterns in online phase."
              << std::endl;
    return;
  }
  batchset->add(x.data(), y);

  addNewInducingPoints(x);

  cf->loghyper_changed = true; // trigger recomputation in compute()
}

// 在预训练阶段结束后，添加在线数据前调用，主要将预训练阶段的诱导点和相关矩阵保存下来，以便在线更新时使用
void SparseGaussianProcess::storePosteriorPretrained() {

  compute();
  update_alpha(); // in case that alpha and cov_inducing are not updated before
                  // this function is called

  // initialization for next online update phase
  if (inducingset_pre == NULL) {
    inducingset_pre = new SampleSet(static_cast<int>(get_input_dim()));
  } else {
    inducingset_pre->clear();
  }
  for (size_t i = 0; i < inducingset->size(); i++) {
    inducingset_pre->add(inducingset->x(i), inducingset->y(i));
  }
  std::cout << "Stored " << inducingset_pre->size()
            << " inducing points from pretraining phase." << std::endl;

  if (batchset == NULL) {
    batchset = new SampleSet(static_cast<int>(get_input_dim()));
  } else {
    for (size_t i = 0; i < batchset->size(); i++) {
      sampleset->add(batchset->x(i), batchset->y(i));
    }
    batchset->clear();
  }

  size_t m_a = inducingset_pre->size();

  // calculate K_aa_pre and its inverse for the pre-trained inducing points
  double noise_variance =
      std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2);
  Eigen::MatrixXd K_aa(m_a, m_a);
  computeKernelMatrixLowerHalf(K_aa, inducingset_pre, cf);
  K_aa.diagonal().array() -= noise_variance; // subtract noise variance
  K_aa_pre.resize(m_a, m_a);
  K_aa_pre = K_aa.selfadjointView<Eigen::Lower>(); // ensure K_aa_pre is lower
                                                   // triangular

  inv_K_aa_pre.resize(m_a, m_a);
  inv_K_aa_pre = chol_inverse(K_aa_pre);

  Sigma_u_pre.resize(m_a, m_a);
  Sigma_u_pre = cov_inducing;
  inv_Sigma_u_pre = chol_inverse(Sigma_u_pre);

  invP_pre = inv_Sigma_u_pre - inv_K_aa_pre; // inv(P_pre)
  P_pre = chol_inverse(invP_pre);

  const std::vector<double> &inducing_pre_targets = inducingset_pre->y();
  Eigen::Map<const Eigen::VectorXd> mean_a(&inducing_pre_targets[0], m_a);

  Eigen::VectorXd inv_Sigma_u_pre_mean_a = inv_Sigma_u_pre * mean_a;
  Eigen::VectorXd P_pre_inv_Sigma_u_pre_mean_a = P_pre * inv_Sigma_u_pre_mean_a;
  y_a = P_pre_inv_Sigma_u_pre_mean_a;

  elbo_constant_init =
      -logDet(K_aa_pre) + logDet(Sigma_u_pre); // constant w.r.t. new batch
  elbo_constant_init -=
      mean_a.dot((inv_Sigma_u_pre * P_pre_inv_Sigma_u_pre_mean_a -
                  inv_Sigma_u_pre_mean_a)); // constant w.r.t. new batch
  elbo_constant_init -=
      1 / param_alpha * logDet(P_pre); // constant w.r.t. new batch

  // reset novelty threshold for adding new inducing points in online phase
  Eigen::VectorXd gamma_values = inv_K_aa_pre.diagonal();
  gamma_values = gamma_values.array().inverse(); // gamma = 1 / diag(inv(K_RR))
  novelty_threshold = (gamma_values.maxCoeff() + gamma_values.minCoeff()) * 0.0005;
  // novelty_threshold = 0.05 * std::exp(cf->get_loghyper()(cf->get_param_dim() - 2) *
  //                     2.0); //使用signal_variance的1/5作为阈值
  // change flags
  invKRR_add_inducing_need_update = true;
  stream_update_mode = true;

  std::cout << "Stored posterior from pretraining phase. Ready for online updates."
            << std::endl;
}

void SparseGaussianProcess::addNewInducingPoints(Eigen::VectorXd x_t) {
  size_t m = inducingset->size();

  if (invKRR_add_inducing_need_update) {
    invK_RR_online = inv_K_aa_pre;
    invKRR_add_inducing_need_update = false;
  }

  double noise_variance =
      std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2.0);

  //计算novelty：gama = K_tt-K_tr * inv(K_rr) * K_rt
  Eigen::VectorXd k_rt(m);
  for (size_t j = 0; j < m; j++) {
    k_rt(j) = cf->get(x_t, inducingset->x(j));
  }

  Eigen::VectorXd v = invK_RR_online * k_rt;
  double k_tt = cf->get(x_t, x_t) - noise_variance;
  double gamma = k_tt - v.dot(k_rt.transpose());

  if (gamma > novelty_threshold) {
    inducingset->add(x_t, 0.0);
    //通过分块矩阵拓展的方式更新inv(K_RR)
    size_t new_m = inducingset->size();
    invK_RR_online.conservativeResize(new_m, new_m);
    double alpha = 1.0 / gamma;
    invK_RR_online.block(0, 0, m, m) += alpha * v * v.transpose();
    invK_RR_online.block(m, 0, 1, m) = -alpha * v.transpose();
    invK_RR_online.block(0, m, m, 1) = -alpha * v;
    invK_RR_online(m, m) = alpha;
    m = new_m;

  }

}

Eigen::VectorXd SparseGaussianProcess::get_hyperparameter_lower_bound() {
  Eigen::VectorXd lb;
  size_t m = inducingset->size();
  lb.resize(cf->get_param_dim() + m * input_dim);
  lb.head(cf->get_param_dim()) = cf->get_loghyper_lb();
  for (size_t i = 0; i < m; ++i) {
    lb.segment(cf->get_param_dim() + i * input_dim, input_dim) =
        inducingset->x(i) - 1.0 * Eigen::VectorXd::Ones(input_dim);
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
  cf->loghyper_changed = true;
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
  update_alpha();

  return cov_inducing;
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
  update_alpha();
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
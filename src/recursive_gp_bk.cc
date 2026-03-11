// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "recursive_gp.h"
#include "cov_factory.h"

#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <random>
#include <memory>

#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#define REGENERATE_FROM_SAMPLES
namespace libgp
{
const double log2pi = std::log(2 * M_PI);
const double initial_L_size = 1000;

RecursiveGaussianProcess::RecursiveGaussianProcess(size_t input_dim, std::string covf_def)
  : GaussianProcess(input_dim, covf_def)
{
  // 初始化诱导集
  inducingset = new SampleSet(input_dim);
}

RecursiveGaussianProcess::RecursiveGaussianProcess(const char* filename) : GaussianProcess(filename)
{
  if (sampleset->empty())
    return;
    
  size_t inducingSet_size = sampleset->size();
  
  // 使用现代随机数生成器
  std::mt19937 gen(std::time(0));
  
  inducingset = new SampleSet(input_dim);
  for (size_t i = 0; i < inducingSet_size; i++)
  {
    inducingset->add(sampleset->x(i).data(), sampleset->y(i));
  }
  
  std::cout << "initialize " << inducingset->size() << " inducing points from sample set " << std::endl;
}

RecursiveGaussianProcess::~RecursiveGaussianProcess()
{
  if (inducingset != nullptr) {
    delete inducingset;
    inducingset = nullptr;
  }
}

double RecursiveGaussianProcess::f(const double x[])
{
  if (inducingset->empty())
    return 0.0;
    
  Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
  compute();
  update_alpha();
  update_k_star(x_star);
  return k_star.dot(alpha_R);
}

double RecursiveGaussianProcess::var(const double x[])
{
  if (inducingset->empty())
    return 0.0;
    
  Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
  compute();
  update_alpha();
  update_k_star(x_star);
  
  Eigen::VectorXd v = L_R * k_star;
  
  return cf->get(x_star, x_star) - v.dot(v);
}

void RecursiveGaussianProcess::compute()
{
  if (!cf->loghyper_changed)
    return;
    
  if (inducingset->empty()) {
    std::cerr << "Error: inducing set is empty! Please Specify inducing points first!" << std::endl;
    return;
  }
  
  cf->loghyper_changed = false;
  size_t m = inducingset->size();
  size_t n = sampleset->size();

  Eigen::MatrixXd K_RR(m, m);
  K_RR.setZero();
  computeKernelMatrixLowerHalf(K_RR, inducingset, cf);
  
  double noise_variance = std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2.0);
  K_RR.diagonal().array() -= noise_variance;
  L_K_RR = K_RR.selfadjointView<Eigen::Lower>().llt().matrixL();

  if (!recursive_initialized) {
    L_R.resize(m, m);
    L_R.setZero();
    elbo_0 = -0.5 * static_cast<double>(sampleset->size()) * log2pi;
    elbo_dot_0.resize(cf->get_param_dim() + m * input_dim);
    elbo_dot_0.setZero();
  } else {
#ifdef REGENERATE_FROM_SAMPLES
    Eigen::MatrixXd K_RX(m, n);
    Eigen::MatrixXd K_XX(n, n);
    K_RX.setZero();
    K_XX.setZero();
    
    computeKernelMatrix(K_RX, inducingset, sampleset, cf);
    computeKernelMatrixLowerHalf(K_XX, sampleset, cf);

    K_XX.diagonal().array() -= noise_variance;

    L_K_RR = K_RR.selfadjointView<Eigen::Lower>().llt().matrixL();
    
    Eigen::MatrixXd H_T = L_K_RR.triangularView<Eigen::Lower>().solve(K_RX);
    L_K_RR.triangularView<Eigen::Lower>().adjoint().solveInPlace(H_T);

    Eigen::MatrixXd Q_ff = K_RX.transpose() * H_T;
    
    D_X = K_XX + K_XX.transpose();
    D_X.diagonal().array() *= 0.5;
    D_X -= Q_ff;

    auto K_XX_bar = Q_ff;
    K_XX_bar.diagonal() += param_alpha * D_X.diagonal();
    K_XX_bar.diagonal().array() += noise_variance;
    
    L = K_XX_bar.llt().matrixL();

    const std::vector<double>& targets = sampleset->y();
    Eigen::Map<const Eigen::VectorXd> y(targets.data(), n);
    
    alpha = L.triangularView<Eigen::Lower>().solve(y);
    L.triangularView<Eigen::Lower>().adjoint().solveInPlace(alpha);

    Eigen::VectorXd mean_u = K_RX * alpha;

    for (size_t i = 0; i < m; ++i) {
      inducingset->set_y(i, mean_u(i));
    }

    L_R = L.triangularView<Eigen::Lower>().solve(H_T.transpose());
#else
    Eigen::MatrixXd L_Lambda_0 = Lambda_0.llt().matrixL();
    Eigen::MatrixXd Sigma_u = L_Lambda_0.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(m, m));
    L_Lambda_0.triangularView<Eigen::Lower>().adjoint().solveInPlace(Sigma_u);
    K_RR = K_RR.selfadjointView<Eigen::Lower>();
    L_R = K_RR - Sigma_u;
    L_K_RR.triangularView<Eigen::Lower>().solveInPlace(L_R);
    L_K_RR.triangularView<Eigen::Lower>().adjoint().solveInPlace(L_R);
    L_R.transposeInPlace();
    L_K_RR.triangularView<Eigen::Lower>().solveInPlace(L_R);
    L_K_RR.triangularView<Eigen::Lower>().adjoint().solveInPlace(L_R);
    L_R = L_R.llt().matrixL().transpose();
#endif
  }
  alpha_needs_update = true;
}

void RecursiveGaussianProcess::epochUpdate(bool verbose)
{
  if (inducingset->empty()) {
    std::cerr << "Error: inducing set is empty! Please Specify inducing points first!" << std::endl;
    return;
  }
  if (sampleset->empty()) {
    std::cerr << "Error: sample set is empty! Please add training data first!" << std::endl;
    return;
  }
  
  size_t m = inducingset->size();
  size_t n = sampleset->size();
  size_t batch_size = 50;
  size_t param_dim = cf->get_param_dim() + m * input_dim;

  if (!adam_optimizer) {
    adam_optimizer.reset(new AdamOptimizer(param_dim, 0.005));
  }

  size_t max_batches = (n + batch_size - 1) / batch_size;
  auto random_indices = Utils::randperm(n);
  
  recursive_initialized = false;
  
  for (size_t k = 0; k < max_batches; ++k) {
    std::vector<Eigen::VectorXd> batch_inputs(batch_size);
    Eigen::VectorXd batch_targets(batch_size);
    
    size_t current_batch_size = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      size_t idx = k * batch_size + i;
      if (idx >= n) break;
      
      batch_inputs[i] = sampleset->x(random_indices[idx]);
      batch_targets(i) = sampleset->y(random_indices[idx]);
      current_batch_size++;
    }
    
    // 调整最后一次可能不满的 batch
    if (current_batch_size < batch_size) {
        batch_inputs.resize(current_batch_size);
        batch_targets.conservativeResize(current_batch_size);
    }
    
    batchUpdate(batch_inputs, batch_targets);

    Eigen::VectorXd params = get_hyperparameters();
    adam_optimizer->step(elbo_dot_0, params);
    update_hyperparameters(params);
  }
}

void RecursiveGaussianProcess::batchUpdate(const std::vector<Eigen::VectorXd>& batch_inputs, const Eigen::VectorXd& batch_targets)
{ 
  size_t m = inducingset->size();  
  size_t b = batch_inputs.size();
  
  if (b == 0) {
    std::cerr << "Error: no training data provided!" << std::endl;
    return;
  }
  if (batch_inputs[0].size() != input_dim) {
    std::cerr << "Error: input dimension mismatch!" << std::endl;
    return;
  }
  if (b != static_cast<size_t>(batch_targets.size())) {
    std::cerr << "Error: input and target size mismatch!" << std::endl;
    return;
  }
  
  // 使用 unique_ptr 管理临时 SampleSet
  auto batchSet = std::make_unique<SampleSet>(input_dim);
  for (size_t i = 0; i < b; i++) {
    batchSet->add(batch_inputs[i].data(), batch_targets(i));
  }

  Eigen::MatrixXd K_RX(m, b);
  Eigen::MatrixXd K_RR(m, m);
  Eigen::VectorXd diagK_XX(b);
  
  K_RX.setZero();
  K_RR.setZero();
  diagK_XX.setZero();

  computeKernelMatrixLowerHalf(K_RR, inducingset, cf);
  double noise_variance = std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2.0);
  K_RR.diagonal().array() -= noise_variance;
  
  L_K_RR = K_RR.selfadjointView<Eigen::Lower>().llt().matrixL();
  Eigen::MatrixXd iK_RR = L_K_RR.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(m, m));
  L_K_RR.triangularView<Eigen::Lower>().adjoint().solveInPlace(iK_RR);

  computeKernelMatrix(K_RX, inducingset, batchSet.get(), cf);

  for (size_t i = 0; i < b; i++) {
    diagK_XX(i) = cf->get(batchSet->x(i), batchSet->x(i)) - noise_variance;
  }
  
  if (!recursive_initialized) {
    eta_0.resize(m);
    eta_0.setZero();
    Lambda_0 = iK_RR;
    elbo_0 = -0.5 * static_cast<double>(sampleset->size()) * log2pi;
  }

  Eigen::MatrixXd H_T = iK_RR * K_RX;
  Eigen::MatrixXd H = H_T.transpose();
  Eigen::MatrixXd Q_XX = H * K_RX;

  Eigen::VectorXd diagV = diagK_XX - Q_XX.diagonal();
  diagV.array() *= param_alpha;
  diagV.array() += noise_variance;
  
  Eigen::MatrixXd iV(b, b);
  iV.setZero();
  iV.diagonal() = diagV.cwiseInverse();
  
  Eigen::MatrixXd L_Lambda_0 = Lambda_0.llt().matrixL();
  Eigen::MatrixXd Sigma_0 = L_Lambda_0.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(m, m));
  L_Lambda_0.triangularView<Eigen::Lower>().adjoint().solveInPlace(Sigma_0);

  Eigen::VectorXd Sigma_0_eta_0 = Sigma_0 * eta_0;
  Eigen::VectorXd r = batch_targets - H * Sigma_0_eta_0;

  Eigen::MatrixXd H_T_iV = H_T * iV;
  Eigen::VectorXd eta_1 = eta_0 + H_T_iV * batch_targets;
  
  Eigen::MatrixXd Lambda_1 = H_T_iV * H + Lambda_0;
  Eigen::MatrixXd L_Lambda_1 = Lambda_1.llt().matrixL();
  Eigen::MatrixXd Sigma_1 = L_Lambda_1.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(m, m));
  L_Lambda_1.triangularView<Eigen::Lower>().adjoint().solveInPlace(Sigma_1);
  
  Eigen::MatrixXd iS = iV - iV * H * Sigma_1 * H_T * iV;

  elbo_0 -= L_Lambda_1.diagonal().array().log().sum();
  elbo_0 += L_Lambda_0.diagonal().array().log().sum();
  elbo_0 -= 0.5 * diagV.array().log().sum();
  elbo_0 -= 0.5 * r.dot(iS * r);
  elbo_0 += (1.0 - param_alpha) / (2.0 * param_alpha) * std::log(noise_variance) * static_cast<double>(b);
  elbo_0 -= (1.0 - param_alpha) / (2.0 * param_alpha) * (diagV.array().log().sum());

  Eigen::VectorXd mean_u = Sigma_1 * eta_1;
  for (size_t i = 0; i < m; ++i) {
    inducingset->set_y(i, mean_u(i));
  }

  // Hyperparameter gradients calculation
  Eigen::MatrixXd Sigma_1_H_T_iV = Sigma_1 * H_T_iV;
  Eigen::VectorXd Sigma_1_H_T_iV_r = Sigma_1_H_T_iV * r;
  
  Eigen::MatrixXd G_Lambda_0 = Sigma_1 - Sigma_0;
  G_Lambda_0 += Sigma_1_H_T_iV_r * (2.0 * Sigma_0_eta_0 + Sigma_1_H_T_iV_r).transpose();
  
  Eigen::VectorXd iS_r = iS * r;
  Eigen::MatrixXd G_H = Sigma_1_H_T_iV.transpose() - iS_r * (Sigma_0_eta_0 + Sigma_1_H_T_iV_r).transpose();
  G_H *= 2.0;

  Eigen::MatrixXd G_V = iS - iS_r * iS_r.transpose();
  Eigen::VectorXd G_eta0 = -2.0 * Sigma_1_H_T_iV_r;

  size_t param_dim = cf->get_param_dim() + m * input_dim;
  std::vector<Eigen::MatrixXd> K_RX_dot(param_dim), K_RR_dot(param_dim);
  
  for (size_t i = 0; i < param_dim; i++) {
    K_RX_dot[i].resize(m, b);
    K_RX_dot[i].setZero();
    K_RR_dot[i].resize(m, m);
    K_RR_dot[i].setZero();
  }
  
  std::vector<Eigen::VectorXd> diagK_XX_dot(cf->get_param_dim() - 1);
  for (size_t i = 0; i < cf->get_param_dim() - 1; i++) {
    diagK_XX_dot[i].resize(b);
    diagK_XX_dot[i].setZero();
  }

  Eigen::VectorXd g_cov(cf->get_param_dim());
  Eigen::VectorXd g_ind(input_dim);
  
  // K_RX_dot
  for (size_t i = 0; i < m; i++) {
    size_t param_id = cf->get_param_dim() + i * input_dim;
    for (size_t j = 0; j < b; j++) {
      g_cov.setZero();
      cf->grad(inducingset->x(i), batchSet->x(j), g_cov);
      for (size_t p = 0; p < cf->get_param_dim(); p++) {
        K_RX_dot[p](i, j) = g_cov(p);
      }

      g_ind.setZero();
      cf->grad_wrt_x1(inducingset->x(i), batchSet->x(j), g_ind);
      for (size_t p = 0; p < input_dim; p++) {
        K_RX_dot[param_id + p](i, j) = g_ind(p);
      }
    }
  }
  
  // K_RR_dot
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j <= i; j++) {
      g_cov.setZero();
      cf->grad(inducingset->x(i), inducingset->x(j), g_cov);
      
      for (size_t p = 0; p < cf->get_param_dim() - 1; p++) {
        K_RR_dot[p](i, j) = g_cov(p);
        if (i != j) {
          K_RR_dot[p](j, i) = g_cov(p);
        }
      }
      
      if (i == j) continue;

      g_ind.setZero();
      cf->grad_wrt_x1(inducingset->x(i), inducingset->x(j), g_ind);
      
      size_t param_id_i = cf->get_param_dim() + i * input_dim;
      size_t param_id_j = cf->get_param_dim() + j * input_dim;
      
      for (size_t p = 0; p < input_dim; p++) {
        K_RR_dot[param_id_i + p](i, j) = g_ind(p);
        K_RR_dot[param_id_i + p](j, i) = g_ind(p);
        K_RR_dot[param_id_j + p](i, j) = -g_ind(p);
        K_RR_dot[param_id_j + p](j, i) = -g_ind(p);
      }
    }
  }

  // diagK_XX_dot
  for (size_t i = 0; i < b; i++) {
    g_cov.setZero();
    cf->grad(batchSet->x(i), batchSet->x(i), g_cov);
    for (size_t p = 0; p < cf->get_param_dim() - 1; p++) {
      diagK_XX_dot[p](i) = g_cov(p);
    }
  }
  
  if (!recursive_initialized) {
    eta_dot_0.resize(param_dim);
    Lambda_dot_0.resize(param_dim);
    for (size_t p = 0; p < param_dim; p++) {
      eta_dot_0[p].resize(m);
      eta_dot_0[p].setZero();
      Lambda_dot_0[p] = -iK_RR * K_RR_dot[p] * iK_RR;
    }
    elbo_dot_0.resize(param_dim);
    elbo_dot_0.setZero();
    recursive_initialized = true;
  }

  Eigen::VectorXd grad(param_dim);
  grad.setZero();
  
  for (size_t p = 0; p < param_dim; p++) {
      Eigen::MatrixXd H_dot = K_RX_dot[p].transpose() * iK_RR - K_RX.transpose() * iK_RR * K_RR_dot[p] * iK_RR;
      Eigen::VectorXd V_dot(b);
      V_dot.setZero();
      
      if (p == cf->get_param_dim() - 1) {
        V_dot.array() = 2.0 * noise_variance;
      } else {
        if (p < cf->get_param_dim() - 1) {
          V_dot = diagK_XX_dot[p];
        }
        Eigen::MatrixXd diagQ_XX_dot = (2.0 * H * K_RX_dot[p] - H * K_RR_dot[p] * H_T);
        V_dot -= diagQ_XX_dot.diagonal();
        V_dot.array() *= param_alpha;
      }

      grad(p) += G_Lambda_0.cwiseProduct(Lambda_dot_0[p]).sum();
      grad(p) += G_H.cwiseProduct(H_dot).sum();
      grad(p) += G_V.diagonal().cwiseProduct(V_dot).sum();
      grad(p) += G_eta0.dot(eta_dot_0[p]);

      if (p == cf->get_param_dim() - 1) {
        grad(p) -= 2.0 * (1.0 - param_alpha) / param_alpha * static_cast<double>(b);
      }

      grad(p) += (1.0 - param_alpha) / param_alpha * iV.diagonal().cwiseProduct(V_dot).sum();
      
      Eigen::MatrixXd iV_H_dot = iV * H_dot;
      Eigen::MatrixXd H_T_iV_V_dot_iV = H_T_iV * V_dot.asDiagonal() * iV;
      
      eta_dot_0[p] += (iV_H_dot.transpose() - H_T_iV_V_dot_iV) * batch_targets;
      Lambda_dot_0[p] += iV_H_dot.transpose() * H - H_T_iV_V_dot_iV * H + H_T * iV_H_dot;
  }
  
  elbo_dot_0 -= 0.5 * grad;
  eta_0 = eta_1;
  Lambda_0 = Lambda_1;
}

void RecursiveGaussianProcess::update_alpha()
{
  if (!alpha_needs_update)
    return;
    
  alpha_needs_update = false; 
  size_t m = inducingset->size();
  alpha_R.resize(m);
  
  const std::vector<double>& targets = inducingset->y();
  Eigen::Map<const Eigen::VectorXd> y_inducing(targets.data(), m);
  
  alpha_R = L_K_RR.triangularView<Eigen::Lower>().solve(y_inducing);
  L_K_RR.triangularView<Eigen::Lower>().adjoint().solveInPlace(alpha_R);
}

void RecursiveGaussianProcess::update_k_star(const Eigen::VectorXd& x_star)
{
  k_star.resize(inducingset->size());
  for (size_t i = 0; i < inducingset->size(); ++i) {
    k_star(i) = cf->get(x_star, inducingset->x(i));
  }
}

void RecursiveGaussianProcess::add_pattern(const double x[], double y)
{
  sampleset->add(x, y);
  size_t n = sampleset->size();
  if (n % 100 == 0) {
    std::cout << "added " << n << " samples to training set." << std::endl;
  }
}

void RecursiveGaussianProcess::specify_inducingSet(std::vector<Eigen::VectorXd> inducing_points, size_t m_random, size_t m_clusters)
{
  if (!inducing_points.empty()) {
    inducingset->clear();
    for (const auto& point : inducing_points) {
      inducingset->add(point.data(), 0.0);
    }
    std::cout << "specify " << inducingset->size() << " inducing points from given set " << std::endl;
  } else if (m_random > 0) {
    size_t inducingSet_size = std::min(m_random, static_cast<size_t>(0.5 * sampleset->size()));
    
    // 使用现代随机数生成器
    std::mt19937 gen(std::time(0));
    std::uniform_int_distribution<int> distribution(0, sampleset->size() - 1);
    
    inducingset->clear();
    for (size_t i = 0; i < inducingSet_size; i++) {
      int index = distribution(gen);
      inducingset->add(sampleset->x(index).data(), 0.0);
    }
    std::cout << "initialize " << inducingset->size() << " inducing points randomly from sample set " << std::endl;
  } else if (m_clusters > 0) {
    size_t inducingSet_size = std::min(m_clusters, static_cast<size_t>(0.5 * sampleset->size()));
    std::cout << "initialize " << inducingSet_size << " inducing points from sample set using K-means clustering" << std::endl;
    // TODO: Implement K-means clustering
  }
  alpha_needs_update = true;
}

void RecursiveGaussianProcess::write(const char* filename)
{
  if (sampleset->empty())
    return;
    
  compute();
  update_alpha();
  
  std::ofstream outfile(filename);
  if (!outfile.is_open()) {
    std::cerr << "Error: could not open file " << filename << " for writing." << std::endl;
    return;
  }

  time_t curtime = time(0);
  tm now = *localtime(&curtime);
  char dest[BUFSIZ] = { 0 };
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
  
  outfile << std::endl << std::endl << "# data (target value in first column)" << std::endl;
  for (size_t i = 0; i < inducingset->size(); ++i) {
    outfile << std::setprecision(10) << inducingset->y(i) << " ";
    for (size_t j = 0; j < input_dim; ++j) {
      outfile << std::setprecision(10) << inducingset->x(i)(j) << " ";
    }
    outfile << std::endl;
  }
  outfile.close();
}

double RecursiveGaussianProcess::log_likelihood()
{
  compute();
  update_alpha();
  return elbo_0;
}

Eigen::VectorXd RecursiveGaussianProcess::log_likelihood_gradient()
{
  compute();
  update_alpha();
  std::cout << "recursive log likelihood gradient: " << elbo_dot_0.transpose() << std::endl;
  return elbo_dot_0;
}

void RecursiveGaussianProcess::update_hyperparameters(const Eigen::VectorXd& params)
{
  cf->set_loghyper(params.head(cf->get_param_dim()));
  size_t m = inducingset->size();
  
  const std::vector<double>& targets = inducingset->y();
  Eigen::VectorXd y = Eigen::Map<const Eigen::VectorXd>(targets.data(), m);
  
  inducingset->clear();
  for (size_t i = 0; i < m; ++i) {
    Eigen::VectorXd x = params.segment(cf->get_param_dim() + i * input_dim, input_dim);
    inducingset->add(x, y(i));
  }
}

Eigen::VectorXd RecursiveGaussianProcess::get_hyperparameters()
{
  Eigen::VectorXd cov_params = cf->get_loghyper();
  size_t m = inducingset->size();
  
  Eigen::VectorXd params(cov_params.size() + m * input_dim);
  params.head(cov_params.size()) = cov_params;
  
  for (size_t i = 0; i < m; ++i) {
    params.segment(cov_params.size() + i * input_dim, input_dim) = inducingset->x(i);
  }
  return params;
}

Eigen::MatrixXd RecursiveGaussianProcess::getFlatInputs() {
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

Eigen::VectorXd RecursiveGaussianProcess::getFlatTargets() {
  if (inducingset->size() < 1) {
    std::cout << "Inducing set is empty." << std::endl;
    return Eigen::VectorXd();
  }
  const std::vector<double> &targets = inducingset->y();
  Eigen::Map<const Eigen::VectorXd> flat_inducing_targets(&targets[0],
                                                          targets.size());
  return flat_inducing_targets;
}

Eigen::MatrixXd RecursiveGaussianProcess::getFlatPosteriorCovMatrix() {
  if (inducingset->size() < 1) {
    std::cout << "Inducing set is empty." << std::endl;
    return Eigen::MatrixXd();
  }
  size_t m = inducingset->size();
  size_t n = sampleset->size();
  Eigen::MatrixXd K_RX, K_RR_half;
  K_RX.resize(m, n);
  K_RX.setZero();
  K_RR_half.resize(m, m);
  K_RR_half.setZero();

  computeKernelMatrix(K_RX, inducingset, sampleset, cf);
  computeKernelMatrixLowerHalf(K_RR_half, inducingset, cf);

  double noise_variance =
      std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2);
  K_RR_half.diagonal().array() -= noise_variance; // subtract noise variance
  Eigen::MatrixXd K_RR = K_RR_half + K_RR_half.transpose();
  K_RR.diagonal().array() *= 0.5;

  Eigen::MatrixXd inducingPosteriorCov;
  inducingPosteriorCov =
      L.triangularView<Eigen::Lower>().solve(K_RX.transpose());
  L.triangularView<Eigen::Lower>().adjoint().solveInPlace(inducingPosteriorCov);

  inducingPosteriorCov = K_RR - K_RX * inducingPosteriorCov;

  return inducingPosteriorCov;
}

Eigen::VectorXd RecursiveGaussianProcess::getFlatHyperparameters() {
  return cf->get_loghyper().array().exp();
}

Eigen::VectorXd RecursiveGaussianProcess::getFlatAlpha() {
  compute();
  update_alpha();
  return alpha_R;
}

void RecursiveGaussianProcess::setInducingTargetZeros()
{
  for (size_t i = 0; i < inducingset->size(); ++i) {
    inducingset->set_y(i, 0.0);
  }
}

}  // namespace
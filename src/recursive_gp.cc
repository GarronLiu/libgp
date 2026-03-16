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
#include <memory>
#include <random>
#include <sstream>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

namespace libgp {
const double log2pi = std::log(2 * M_PI);
const double initial_L_size = 1000;

RecursiveGaussianProcess::RecursiveGaussianProcess(size_t input_dim,
                                                   std::string covf_def)
    : GaussianProcess(input_dim, covf_def) {
  // 初始化诱导集
  inducingset = new SampleSet(input_dim);
}

RecursiveGaussianProcess::RecursiveGaussianProcess(const char *filename)
    : GaussianProcess(filename) {
  if (sampleset->empty())
    return;

  size_t inducingSet_size = sampleset->size();

  // 使用现代随机数生成器
  std::mt19937 gen(std::time(0));

  inducingset = new SampleSet(input_dim);
  for (size_t i = 0; i < inducingSet_size; i++) {
    inducingset->add(sampleset->x(i).data(), sampleset->y(i));
  }

  std::cout << "initialize " << inducingset->size()
            << " inducing points from sample set " << std::endl;
}

RecursiveGaussianProcess::~RecursiveGaussianProcess() {
  if (inducingset != nullptr) {
    delete inducingset;
    inducingset = nullptr;
  }
}

void RecursiveGaussianProcess::compute() {
  if (!cf->loghyper_changed)
    return;

  if (inducingset->empty()) {
    std::cerr << "Error: inducing set is empty! Please check!" << std::endl;
    return;
  }

  if (inducingset_pre->empty()) {
    // Use the pre-trained inducing set
    std::cerr << "Error: pre-trained inducing set is empty! Please check!"
              << std::endl;
    return;
  }

  cf->loghyper_changed = false;
  size_t m_a = inducingset_pre->size();
  size_t m = inducingset->size();
  double noise_variance =
      std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2);

  Eigen::MatrixXd K_aa(m_a, m_a);
  computeKernelMatrixLowerHalf(K_aa, inducingset_pre, cf);
  K_aa.diagonal().array() -= noise_variance + 1e-6; // subtract noise variance

  if (pass_pretrain_needed_flag) {
    K_aa_pre = K_aa;
    inv_K_aa_pre = chol_inverse(K_aa);
    inv_Sigma_u_pre = chol_inverse(Sigma_u_pre);

    invP_pre = inv_Sigma_u_pre - inv_K_aa_pre; // inv(P_pre)
    P_pre = chol_inverse(invP_pre);

    pass_pretrain_needed_flag = false;
  }

  Eigen::MatrixXd K_bb(m, m);
  computeKernelMatrixLowerHalf(K_bb, inducingset, cf);
  K_bb.diagonal().array() -= noise_variance + 1e-6; // subtract noise variance
  Lambda_0 = chol_inverse(K_bb);

  Eigen::MatrixXd K_ab;
  computeKernelMatrix(K_ab, inducingset_pre, inducingset, cf);

  Eigen::MatrixXd H = K_ab * Lambda_0; // 目前Lambda_0 = inv(K_bb)
  Eigen::MatrixXd H_K_Ra = H * K_ab.transpose();
  Eigen::MatrixXd Da = K_aa - H_K_Ra;
  Eigen::MatrixXd V = P_pre + param_alpha * (Da);
  Eigen::MatrixXd invV_H = chol_solve(V, H);

  Lambda_0 += H.transpose() * invV_H;

  const std::vector<double> &targets = inducingset_pre->y();
  Eigen::Map<const Eigen::VectorXd> mean_a(&targets[0], m_a);
  Eigen::VectorXd y_a = P_pre * inv_Sigma_u_pre * mean_a;
  eta_0 = invV_H.transpose() * y_a;

  Eigen::MatrixXd V_plus_H_K_Ra = V + H_K_Ra;
  Eigen::VectorXd alpha_y_a = chol_solve(V_plus_H_K_Ra, y_a);
  elbo_0 = -logDet(V_plus_H_K_Ra) - y_a.dot(alpha_y_a);

  Eigen::MatrixXd I_plus_alpha_invP_pre_Da =
      Eigen::MatrixXd::Identity(m_a, m_a);
  I_plus_alpha_invP_pre_Da += param_alpha * invP_pre * Da;
  elbo_0 += logDet(V) - (1 / param_alpha) * logDet(I_plus_alpha_invP_pre_Da);

  elbo_0 += logDet(K_aa_pre) - logDet(Sigma_u_pre);

  elbo_0 += mean_a.dot(
      (inv_Sigma_u_pre * P_pre * inv_Sigma_u_pre - inv_Sigma_u_pre) * mean_a);

  elbo_0 *= 0.5;

  // 初始化 eta_dot_0, Lambda_dot_0, elbo_dot_0

  int update_param_dim =
      cf->get_param_dim(); //在线学习时只更新超参数，诱导点位置不更新

  eta_dot_0.resize(update_param_dim);
  Lambda_dot_0.resize(update_param_dim);
  elbo_dot_0.resize(update_param_dim);

  //初始化基础的矩阵导数：
  std::vector<Eigen::MatrixXd> K_aa_dot(update_param_dim);
  std::vector<Eigen::MatrixXd> K_ab_dot(update_param_dim);
  std::vector<Eigen::MatrixXd> K_bb_dot(update_param_dim);
  for (size_t i = 0; i < update_param_dim; i++) {
    K_ab_dot[i] = Eigen::MatrixXd::Zero(m_a, m);
    K_bb_dot[i] = Eigen::MatrixXd::Zero(m, m);
    K_aa_dot[i] = Eigen::MatrixXd::Zero(m_a, m_a);
  }
  //填充导数矩阵
  Eigen::VectorXd g_hyper;
  size_t hyperParam_dim = cf->get_param_dim();
  // K_ab_dot
  for (size_t i = 0; i < m_a; i++) {
    for (size_t j = 0; j < m; j++) {
      g_hyper.setZero(hyperParam_dim);
      cf->grad(inducingset_pre->x(j), inducingset->x(i), g_hyper);
      for (size_t p = 0; p < update_param_dim; p++) {
        K_ab_dot[p](i, j) = g_hyper(p);
      }
    }
  }
  // K_bb_dot
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j <= i; j++) {
      g_hyper.setZero(hyperParam_dim);
      cf->grad(inducingset->x(j), inducingset->x(i), g_hyper);
      for (size_t p = 0; p < update_param_dim; p++) {
        K_bb_dot[p](i, j) = g_hyper(p);
        if (j != i)
          K_bb_dot[p](j, i) = g_hyper(p); // symmetry property
      }
    }
  }
  // K_aa_dot
  for (size_t i = 0; i < m_a; i++) {
    for (size_t j = 0; j <= i; j++) {
      g_hyper.setZero(hyperParam_dim);
      cf->grad(inducingset_pre->x(j), inducingset_pre->x(i), g_hyper);
      for (size_t p = 0; p < update_param_dim; p++) {
        K_aa_dot[p](i, j) = g_hyper(p);
        if (j != i)
          K_aa_dot[p](j, i) = g_hyper(p); // symmetry property
      }
    }
  }

  Eigen::MatrixXd inv_K_bb = chol_inverse(K_bb);
  Eigen::MatrixXd inv_Lambda_0 = K_bb;
  Eigen::MatrixXd S_0 = H * inv_Lambda_0 * H.transpose() + V;
  Eigen::MatrixXd inv_S_0 = chol_inverse(S_0);
  Eigen::MatrixXd Lambda_1 = Lambda_0;

  Eigen::MatrixXd H_dot_0;
  Eigen::MatrixXd V_dot_0;
  std::vector<Eigen::MatrixXd> Lambda_dot_1(update_param_dim);

  //定义矩阵逐元素相乘后所有元素求和的函数
  auto elementwise_sum = [](const Eigen::MatrixXd &mat,
                            const Eigen::MatrixXd &other) {
    return (mat.array() * other.array()).sum();
  };

  for (size_t p = 0; p < update_param_dim; p++) {
    Lambda_dot_0[p] = -inv_K_bb * K_bb_dot[p] * inv_K_bb;
    H_dot_0 = K_ab_dot[p] * inv_K_bb + K_ab * Lambda_dot_0[p];
    V_dot_0 = param_alpha * (K_aa_dot[p] - H_dot_0 * K_ab.transpose() -
                                H * K_ab_dot[p].transpose());
    Lambda_dot_1[p] = Lambda_dot_0[p] + H_dot_0.transpose() * invV_H -
                      invV_H.transpose() * V_dot_0 * invV_H +
                      invV_H.transpose() * H_dot_0.transpose();
    Eigen::MatrixXd invV = chol_inverse(V);
    eta_dot_0[p] = (H_dot_0.transpose() -invV_H.transpose() * V_dot_0 ) * invV * y_a;

    //计算dpsi_dLambda_0;
    elbo_dot_0[p] = 0;
    Eigen::MatrixXd elbo_deriv;
    {
      Eigen::VectorXd temp_vec = inv_Lambda_0 * H.transpose() * inv_S_0 * y_a;
      elbo_deriv = -inv_Lambda_0 + temp_vec * temp_vec.transpose();
    }
    elbo_dot_0[p] += elementwise_sum(elbo_deriv, Lambda_dot_0[p]);
    //计算dpsi_dH_0
    {
      Eigen::VectorXd temp_vec = inv_S_0 * y_a;
      elbo_deriv = -2.0 * temp_vec * temp_vec.transpose() * H * inv_Lambda_0;
    }
    elbo_dot_0[p] += elementwise_sum(elbo_deriv, H_dot_0);
    //计算dpsi_dV_0
    {
      Eigen::VectorXd temp_vec = inv_S_0 * y_a;
      
      elbo_deriv =
          -1.0 * temp_vec * temp_vec.transpose() + (1.0 / param_alpha) * invV;
    }
    elbo_dot_0[p] += elementwise_sum(elbo_deriv, V_dot_0);
    //计算dpsi_dLambda_1
    elbo_deriv = chol_inverse(Lambda_1);
    elbo_dot_0[p] += elementwise_sum(elbo_deriv, Lambda_dot_1[p]);
    elbo_dot_0[p] *= -0.5;
  }

  //最后用Lambda_dot_1更新Lambda_dot_0
  for (size_t p = 0; p < update_param_dim; p++) {
    Lambda_dot_0[p] = Lambda_dot_1[p];
  }
  //TODO:eta_dot_0
  
}

void RecursiveGaussianProcess::epochUpdate(bool verbose) {
  if (inducingset->empty()) {
    std::cerr
        << "Error: inducing set is empty! Please Specify inducing points first!"
        << std::endl;
    return;
  }
  if (sampleset->empty()) {
    std::cerr << "Error: sample set is empty! Please add training data first!"
              << std::endl;
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
      if (idx >= n)
        break;

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

void RecursiveGaussianProcess::batchUpdate(
    const std::vector<Eigen::VectorXd> &batch_inputs,
    const Eigen::VectorXd &batch_targets) {
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
  double noise_variance =
      std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2.0);
  K_RR.diagonal().array() -= noise_variance;

  L_K_RR = K_RR.selfadjointView<Eigen::Lower>().llt().matrixL();
  Eigen::MatrixXd iK_RR = L_K_RR.triangularView<Eigen::Lower>().solve(
      Eigen::MatrixXd::Identity(m, m));
  L_K_RR.triangularView<Eigen::Lower>().adjoint().solveInPlace(iK_RR);

  computeKernelMatrix(K_RX, inducingset, batchSet.get(), cf);

  // adding inducing point
  std::vector<double> novelty_scores(b);
  for (size_t i = 0; i < b; i++) {
    Eigen::VectorXd k_tr(m);
    for (size_t j = 0; j < m; j++) {
      k_tr(j) = cf->get(batchSet->x(i), inducingset->x(j));
    }
    Eigen::VectorXd v = L_K_RR.triangularView<Eigen::Lower>().solve(k_tr);
    double k_xx = cf->get(batchSet->x(i), batchSet->x(i));
    novelty_scores[i] = k_xx - v.dot(v);
  }
  // 若最大的 novelty score 超过阈值，则将该点加入诱导集
  double novelty_threshold =
      0.2 * std::exp(cf->get_loghyper()(cf->get_param_dim() - 2) *
                     2.0); //使用signal_variance的1/5作为阈值
  double last_novelty_score = 0.0;
  size_t last_novelty_index = 0;
  for (size_t i = 0; i < b; i++) {
    if (novelty_scores[i] > last_novelty_score) {
      last_novelty_score = novelty_scores[i];
      last_novelty_index = i;
    }
  }
  if (last_novelty_score > novelty_threshold) {
    inducingset->add(batchSet->x(last_novelty_index).data(), 0.0);
    size_t m_old = m;                  // 此时 inducingset 还没加点
    Eigen::MatrixXd iK_RR_old = iK_RR; // 备份旧的 K_uu^-1

    // 2. 添加点并更新 K_RR (K_uu)
    inducingset->add(batchSet->x(last_novelty_index).data(), 0.0);
    m = inducingset->size(); // m 现在是 m_old + 1

    K_RR.resize(m, m);
    K_RR.setZero();
    computeKernelMatrixLowerHalf(K_RR, inducingset, cf);
    double noise_variance =
        std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2.0);
    K_RR.diagonal().array() -= noise_variance; // Remove noise for K_uu

    L_K_RR = K_RR.selfadjointView<Eigen::Lower>().llt().matrixL();
    iK_RR = L_K_RR.triangularView<Eigen::Lower>().solve(
        Eigen::MatrixXd::Identity(m, m));
    L_K_RR.triangularView<Eigen::Lower>().adjoint().solveInPlace(iK_RR);
    // 现在 iK_RR 是 (m_old+1) x (m_old+1) 的新逆矩阵

    // 3. 更新后验精度矩阵 Lambda_0
    // 逻辑：Lambda_0_new = iK_RR_new + Pad(Lambda_0_old - iK_RR_old)

    // (A) 计算旧的数据信息增益 (Data Information Gain)
    Eigen::MatrixXd Info_old = Lambda_0 - iK_RR_old;

    // (B) 扩展信息矩阵 (新点没有历史数据信息，填 0)
    Eigen::MatrixXd Info_new = Eigen::MatrixXd::Zero(m, m);
    Info_new.topLeftCorner(m_old, m_old) = Info_old;

    // (C) 组合得到新的 Lambda_0
    Lambda_0 = iK_RR + Info_new;

    // 4. 更新 eta_0 (Natural Parameter m_u = Sigma * eta)
    // 简单地给新点补 0，假设新点的后验均值偏差仅由关联性决定
    Eigen::VectorXd eta_0_new = Eigen::VectorXd::Zero(m);
    eta_0_new.head(m_old) = eta_0;
    eta_0 = eta_0_new;
  }
  // adding inducing point

  // delete inducing point
  static size_t inducing_num_max = 70;
  if (m > inducing_num_max) {
    Eigen::VectorXd mean_u(m);
    for (size_t i = 0; i < m; ++i) {
      mean_u(i) = inducingset->y(i);
    }
    std::vector<double> scores(m);
    Eigen::MatrixXd Sigma_u = Eigen::MatrixXd::Identity(m, m);
    Eigen::MatrixXd L_Lambda = Lambda_0.llt().matrixL();
    L_Lambda.triangularView<Eigen::Lower>().solveInPlace(Sigma_u);
    L_Lambda.triangularView<Eigen::Lower>().adjoint().solveInPlace(Sigma_u);
    for (size_t i = 0; i < m; i++) {
      Eigen::VectorXd Q_du = iK_RR.row(i);
      double Q_dd_inv = K_RR(i, i);
      double v = Q_du.dot(mean_u);
      scores[i] = v * Q_dd_inv * v;
      scores[i] += Q_du.dot(Sigma_u * Q_du.transpose()) * Q_dd_inv;
      scores[i] += std::log(Lambda_0(i, i)) + std::log(Q_dd_inv);
    }

    size_t min_index = std::distance(
        scores.begin(), std::min_element(scores.begin(), scores.end()));

    // 更新 Lambda_0 和 eta_0 删除对应行列
    // 1. 删除 Lambda_0 的行/列
    Eigen::MatrixXd new_Lambda_0(m - 1, m - 1);

    // 复制四个区块
    // Top-Left
    if (min_index > 0) {
      new_Lambda_0.topLeftCorner(min_index, min_index) =
          Lambda_0.topLeftCorner(min_index, min_index);
    }
    // Top-Right
    if (min_index > 0 && min_index < m - 1) {
      new_Lambda_0.topRightCorner(min_index, m - 1 - min_index) =
          Lambda_0.topRightCorner(min_index, m - 1 - min_index);
    }
    // Bottom-Left
    if (min_index < m - 1 && min_index > 0) {
      new_Lambda_0.bottomLeftCorner(m - 1 - min_index, min_index) =
          Lambda_0.bottomLeftCorner(m - 1 - min_index, min_index);
    }
    // Bottom-Right
    if (min_index < m - 1) {
      new_Lambda_0.bottomRightCorner(m - 1 - min_index, m - 1 - min_index) =
          Lambda_0.bottomRightCorner(m - 1 - min_index, m - 1 - min_index);
    }
    Lambda_0 = new_Lambda_0;

    // 2. 删除 eta_0 的元素
    Eigen::VectorXd new_eta_0(m - 1);
    if (min_index > 0) {
      new_eta_0.head(min_index) = eta_0.head(min_index);
    }
    if (min_index < m - 1) {
      new_eta_0.tail(m - 1 - min_index) = eta_0.tail(m - 1 - min_index);
    }
    eta_0 = new_eta_0;

    // 3. 物理删除诱导点
    // libgp 的 SampleSet 似乎没有直接 remove 的 API？
    // 如果没有，需要重建 inducing set。
    // 假设这里我们需要手动移除。由于 SampleSet 是 vector 封装，访问受限。
    // 变通方案：创建一个新 SampleSet，拷贝除 min_index 外的所有点。
    SampleSet *new_inducing = new SampleSet(input_dim);
    for (size_t i = 0; i < m; ++i) {
      if (i == min_index)
        continue;
      new_inducing->add(inducingset->x(i).data(), inducingset->y(i));
    }
    delete inducingset;
    inducingset = new_inducing;

    m = m - 1; // 更新 size

    // 4. 更新相关的尺寸依赖变量
    // K_RR, iK_RR 需要更新，因为 batchUpdate 后续步骤依赖它们
    K_RR.resize(m, m);
    K_RR.setZero();
    computeKernelMatrixLowerHalf(K_RR, inducingset, cf);
    double noise_variance =
        std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2.0);
    K_RR.diagonal().array() -= noise_variance;

    L_K_RR = K_RR.selfadjointView<Eigen::Lower>().llt().matrixL();
    iK_RR = L_K_RR.triangularView<Eigen::Lower>().solve(
        Eigen::MatrixXd::Identity(m, m));
    L_K_RR.triangularView<Eigen::Lower>().adjoint().solveInPlace(iK_RR);

  } // delete inducing point

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
  Eigen::MatrixXd Sigma_0 = L_Lambda_0.triangularView<Eigen::Lower>().solve(
      Eigen::MatrixXd::Identity(m, m));
  L_Lambda_0.triangularView<Eigen::Lower>().adjoint().solveInPlace(Sigma_0);

  Eigen::VectorXd Sigma_0_eta_0 = Sigma_0 * eta_0;
  Eigen::VectorXd r = batch_targets - H * Sigma_0_eta_0;

  Eigen::MatrixXd H_T_iV = H_T * iV;
  Eigen::VectorXd eta_1 = eta_0 + H_T_iV * batch_targets;

  Eigen::MatrixXd Lambda_1 = H_T_iV * H + Lambda_0;
  Eigen::MatrixXd L_Lambda_1 = Lambda_1.llt().matrixL();
  Eigen::MatrixXd Sigma_1 = L_Lambda_1.triangularView<Eigen::Lower>().solve(
      Eigen::MatrixXd::Identity(m, m));
  L_Lambda_1.triangularView<Eigen::Lower>().adjoint().solveInPlace(Sigma_1);

  Eigen::MatrixXd iS = iV - iV * H * Sigma_1 * H_T * iV;

  elbo_0 -= L_Lambda_1.diagonal().array().log().sum();
  elbo_0 += L_Lambda_0.diagonal().array().log().sum();
  elbo_0 -= 0.5 * diagV.array().log().sum();
  elbo_0 -= 0.5 * r.dot(iS * r);
  elbo_0 += (1.0 - param_alpha) / (2.0 * param_alpha) *
            std::log(noise_variance) * static_cast<double>(b);
  elbo_0 -=
      (1.0 - param_alpha) / (2.0 * param_alpha) * (diagV.array().log().sum());

  Eigen::VectorXd mean_u = Sigma_1 * eta_1;
  for (size_t i = 0; i < m; ++i) {
    inducingset->set_y(i, mean_u(i));
  }

  // Hyperparameter gradients calculation
  Eigen::MatrixXd Sigma_1_H_T_iV = Sigma_1 * H_T_iV;
  Eigen::VectorXd Sigma_1_H_T_iV_r = Sigma_1_H_T_iV * r;

  Eigen::MatrixXd G_Lambda_0 = Sigma_1 - Sigma_0;
  G_Lambda_0 +=
      Sigma_1_H_T_iV_r * (2.0 * Sigma_0_eta_0 + Sigma_1_H_T_iV_r).transpose();

  Eigen::VectorXd iS_r = iS * r;
  Eigen::MatrixXd G_H = Sigma_1_H_T_iV.transpose() -
                        iS_r * (Sigma_0_eta_0 + Sigma_1_H_T_iV_r).transpose();
  G_H *= 2.0;

  Eigen::MatrixXd G_V = iS - iS_r * iS_r.transpose();
  Eigen::VectorXd G_eta0 = -2.0 * Sigma_1_H_T_iV_r;

  size_t param_dim = cf->get_param_dim();
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

  // K_RX_dot
  for (size_t i = 0; i < m; i++) {
    size_t param_id = cf->get_param_dim() + i * input_dim;
    for (size_t j = 0; j < b; j++) {
      g_cov.setZero();
      cf->grad(inducingset->x(i), batchSet->x(j), g_cov);
      for (size_t p = 0; p < cf->get_param_dim(); p++) {
        K_RX_dot[p](i, j) = g_cov(p);
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
    Eigen::MatrixXd H_dot = K_RX_dot[p].transpose() * iK_RR -
                            K_RX.transpose() * iK_RR * K_RR_dot[p] * iK_RR;
    Eigen::VectorXd V_dot(b);
    V_dot.setZero();

    if (p == cf->get_param_dim() - 1) {
      V_dot.array() = 2.0 * noise_variance;
    } else {
      if (p < cf->get_param_dim() - 1) {
        V_dot = diagK_XX_dot[p];
      }
      Eigen::MatrixXd diagQ_XX_dot =
          (2.0 * H * K_RX_dot[p] - H * K_RR_dot[p] * H_T);
      V_dot -= diagQ_XX_dot.diagonal();
      V_dot.array() *= param_alpha;
    }

    grad(p) += G_Lambda_0.cwiseProduct(Lambda_dot_0[p]).sum();
    grad(p) += G_H.cwiseProduct(H_dot).sum();
    grad(p) += G_V.diagonal().cwiseProduct(V_dot).sum();
    grad(p) += G_eta0.dot(eta_dot_0[p]);

    if (p == cf->get_param_dim() - 1) {
      grad(p) -=
          2.0 * (1.0 - param_alpha) / param_alpha * static_cast<double>(b);
    }

    grad(p) += (1.0 - param_alpha) / param_alpha *
               iV.diagonal().cwiseProduct(V_dot).sum();

    Eigen::MatrixXd iV_H_dot = iV * H_dot;
    Eigen::MatrixXd H_T_iV_V_dot_iV = H_T_iV * V_dot.asDiagonal() * iV;

    eta_dot_0[p] += (iV_H_dot.transpose() - H_T_iV_V_dot_iV) * batch_targets;
    Lambda_dot_0[p] +=
        iV_H_dot.transpose() * H - H_T_iV_V_dot_iV * H + H_T * iV_H_dot;
  }

  elbo_dot_0 -= 0.5 * grad;
  eta_0 = eta_1;
  Lambda_0 = Lambda_1;
}

void RecursiveGaussianProcess::specify_inducingSet(
    std::vector<Eigen::VectorXd> inducing_points, size_t m_random,
    size_t m_clusters) {
  if (!inducing_points.empty()) {
    inducingset->clear();
    for (const auto &point : inducing_points) {
      inducingset->add(point.data(), 0.0);
    }
    std::cout << "specify " << inducingset->size()
              << " inducing points from given set " << std::endl;
  } else if (m_random > 0) {
    size_t inducingSet_size =
        std::min(m_random, static_cast<size_t>(0.5 * sampleset->size()));

    // 使用现代随机数生成器
    std::mt19937 gen(std::time(0));
    std::uniform_int_distribution<int> distribution(0, sampleset->size() - 1);

    inducingset->clear();
    for (size_t i = 0; i < inducingSet_size; i++) {
      int index = distribution(gen);
      inducingset->add(sampleset->x(index).data(), 0.0);
    }
    std::cout << "initialize " << inducingset->size()
              << " inducing points randomly from sample set " << std::endl;
  } else if (m_clusters > 0) {
    size_t inducingSet_size =
        std::min(m_clusters, static_cast<size_t>(0.5 * sampleset->size()));
    std::cout << "initialize " << inducingSet_size
              << " inducing points from sample set using K-means clustering"
              << std::endl;
    // TODO: Implement K-means clustering
  }
  alpha_needs_update = true;
}

double RecursiveGaussianProcess::log_likelihood() {
  compute();
  update_alpha();
  return elbo_0;
}

Eigen::VectorXd RecursiveGaussianProcess::log_likelihood_gradient() {
  compute();
  update_alpha();
  std::cout << "recursive log likelihood gradient: " << elbo_dot_0.transpose()
            << std::endl;
  return elbo_dot_0;
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

void RecursiveGaussianProcess::setInducingTargetZeros() {
  for (size_t i = 0; i < inducingset->size(); ++i) {
    inducingset->set_y(i, 0.0);
  }
}

} // namespace libgp
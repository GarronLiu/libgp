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
    : SparseGaussianProcess(input_dim, covf_def) {}

RecursiveGaussianProcess::RecursiveGaussianProcess(const char *filename)
    : SparseGaussianProcess(filename) {
  std::cout << "Initialized RecursiveGaussianProcess from " << filename
            << std::endl;
}

RecursiveGaussianProcess::~RecursiveGaussianProcess() {
  if (inducingset_pre != nullptr) {
    delete inducingset_pre;
    inducingset_pre = nullptr;
  }
}

void RecursiveGaussianProcess::compute() {
  if (!cf->loghyper_changed)
    return;

  if (inducingset->empty()) {
    std::cerr << "Error: inducing set is empty! Please check!" << std::endl;
    return;
  }

  if (cov_inducing.size() == 0) {
    std::cerr << "Error: inducing covariance matrix is empty! Please check!"
              << std::endl;
    return;
  }
  size_t m = inducingset->size();
  Eigen::MatrixXd K_RR(inducingset->size(), inducingset->size());
  computeKernelMatrixLowerHalf(K_RR, inducingset, cf);
  auto noise_variance =
      std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2.0);
  K_RR.diagonal().array() -= noise_variance; // subtract noise variance
  L_K_RR = K_RR.selfadjointView<Eigen::Lower>()
               .llt()
               .matrixL();                     // compute Cholesky factor
  Eigen::MatrixXd iK_RR = chol_inverse(K_RR);  // compute inverse using Cholesky
  K_RR = K_RR.selfadjointView<Eigen::Lower>(); // ensure K_RR is lower
                                               // triangular for later use

  const std::vector<double> &inducing_targets = inducingset->y();
  Eigen::Map<const Eigen::VectorXd> mean_inducing(&inducing_targets[0], m);
  alpha_R = iK_RR * mean_inducing;

  Q_pred = iK_RR * (K_RR - cov_inducing) * iK_RR;
  cf->loghyper_changed = false;
}

void RecursiveGaussianProcess::addNewInducingPoints() {
  size_t m = inducingset->size();
  size_t n = sampleset->size();

  double novelty_threshold =
      0.05 * std::exp(cf->get_loghyper()(cf->get_param_dim() - 2) *
                      2.0); //使用signal_variance的1/5作为阈值
  Eigen::MatrixXd K_RR(m, m);
  computeKernelMatrixLowerHalf(K_RR, inducingset, cf);
  double noise_variance =
      std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2.0);
  auto invK_RR = chol_inverse(K_RR);
  for (size_t i = 0; i < n; i++) {
    //计算novelty：gama = K_tt-K_tr * inv(K_rr) * K_rt
    Eigen::VectorXd k_rt(m);
    Eigen::VectorXd x_t = sampleset->x(i);
    for (size_t j = 0; j < m; j++) {
      k_rt(j) = cf->get(x_t, inducingset->x(j));
    }
    Eigen::VectorXd v = invK_RR * k_rt;
    double k_tt = cf->get(x_t, x_t) - noise_variance;
    double gamma = k_tt - v.dot(k_rt.transpose());
    // std::cout << "Novelty (gamma) for sample " << i << ": " << gamma <<
    // std::endl;
    if (gamma > novelty_threshold) {
      inducingset->add(x_t, 0.0);
      //通过分块矩阵拓展的方式更新inv(K_RR)
      size_t new_m = inducingset->size();
      invK_RR.conservativeResize(new_m, new_m);
      double alpha = 1.0 / gamma;
      invK_RR.block(0, 0, m, m) += alpha * v * v.transpose();
      invK_RR.block(m, 0, 1, m) = -alpha * v.transpose();
      invK_RR.block(0, m, m, 1) = -alpha * v;
      invK_RR(m, m) = alpha;
      m = new_m;
    }
  }
}

void RecursiveGaussianProcess::deleteRedundantInducingPoints() {
  // // delete inducing point
  // static size_t inducing_num_max = 70;
  // if (m > inducing_num_max) {
  //   Eigen::VectorXd mean_u(m);
  //   for (size_t i = 0; i < m; ++i) {
  //     mean_u(i) = inducingset->y(i);
  //   }
  //   std::vector<double> scores(m);
  //   Eigen::MatrixXd Sigma_u = Eigen::MatrixXd::Identity(m, m);
  //   Eigen::MatrixXd L_Lambda = Lambda_0.llt().matrixL();
  //   L_Lambda.triangularView<Eigen::Lower>().solveInPlace(Sigma_u);
  //   L_Lambda.triangularView<Eigen::Lower>().adjoint().solveInPlace(Sigma_u);
  //   for (size_t i = 0; i < m; i++) {
  //     Eigen::VectorXd Q_du = iK_RR.row(i);
  //     double Q_dd_inv = K_RR(i, i);
  //     double v = Q_du.dot(mean_u);
  //     scores[i] = v * Q_dd_inv * v;
  //     scores[i] += Q_du.dot(Sigma_u * Q_du.transpose()) * Q_dd_inv;
  //     scores[i] += std::log(Lambda_0(i, i)) + std::log(Q_dd_inv);
  //   }

  //   size_t min_index = std::distance(
  //       scores.begin(), std::min_element(scores.begin(), scores.end()));

  //   // 更新 Lambda_0 和 eta_0 删除对应行列
  //   // 1. 删除 Lambda_0 的行/列
  //   Eigen::MatrixXd new_Lambda_0(m - 1, m - 1);

  //   // 复制四个区块
  //   // Top-Left
  //   if (min_index > 0) {
  //     new_Lambda_0.topLeftCorner(min_index, min_index) =
  //         Lambda_0.topLeftCorner(min_index, min_index);
  //   }
  //   // Top-Right
  //   if (min_index > 0 && min_index < m - 1) {
  //     new_Lambda_0.topRightCorner(min_index, m - 1 - min_index) =
  //         Lambda_0.topRightCorner(min_index, m - 1 - min_index);
  //   }
  //   // Bottom-Left
  //   if (min_index < m - 1 && min_index > 0) {
  //     new_Lambda_0.bottomLeftCorner(m - 1 - min_index, min_index) =
  //         Lambda_0.bottomLeftCorner(m - 1 - min_index, min_index);
  //   }
  //   // Bottom-Right
  //   if (min_index < m - 1) {
  //     new_Lambda_0.bottomRightCorner(m - 1 - min_index, m - 1 - min_index) =
  //         Lambda_0.bottomRightCorner(m - 1 - min_index, m - 1 - min_index);
  //   }
  //   Lambda_0 = new_Lambda_0;

  //   // 2. 删除 eta_0 的元素
  //   Eigen::VectorXd new_eta_0(m - 1);
  //   if (min_index > 0) {
  //     new_eta_0.head(min_index) = eta_0.head(min_index);
  //   }
  //   if (min_index < m - 1) {
  //     new_eta_0.tail(m - 1 - min_index) = eta_0.tail(m - 1 - min_index);
  //   }
  //   eta_0 = new_eta_0;

  //   // 3. 物理删除诱导点
  //   // libgp 的 SampleSet 似乎没有直接 remove 的 API？
  //   // 如果没有，需要重建 inducing set。
  //   // 假设这里我们需要手动移除。由于 SampleSet 是 vector 封装，访问受限。
  //   // 变通方案：创建一个新 SampleSet，拷贝除 min_index 外的所有点。
  //   SampleSet *new_inducing = new SampleSet(input_dim);
  //   for (size_t i = 0; i < m; ++i) {
  //     if (i == min_index)
  //       continue;
  //     new_inducing->add(inducingset->x(i).data(), inducingset->y(i));
  //   }
  //   delete inducingset;
  //   inducingset = new_inducing;

  //   m = m - 1; // 更新 size

  //   // 4. 更新相关的尺寸依赖变量
  //   // K_RR, iK_RR 需要更新，因为 batchUpdate 后续步骤依赖它们
  //   K_RR.resize(m, m);
  //   K_RR.setZero();
  //   computeKernelMatrixLowerHalf(K_RR, inducingset, cf);
  //   double noise_variance =
  //       std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2.0);
  //   K_RR.diagonal().array() -= noise_variance;

  //   L_K_RR = K_RR.selfadjointView<Eigen::Lower>().llt().matrixL();
  //   iK_RR = L_K_RR.triangularView<Eigen::Lower>().solve(
  //       Eigen::MatrixXd::Identity(m, m));
  //   L_K_RR.triangularView<Eigen::Lower>().adjoint().solveInPlace(iK_RR);

} // delete inducing point

/// TODO: 噪声参数的梯度需要特殊处理
void RecursiveGaussianProcess::updatePosteriorWithHistoryInfo(
    bool hyper_grad, bool inducing_grad) {
  size_t param_dim = cf->get_param_dim();
  size_t m_a = inducingset_pre->size();
  size_t m = inducingset->size();
  double noise_variance = std::exp(cf->get_loghyper()(param_dim - 1) * 2);
  Eigen::MatrixXd K_aa(m_a, m_a);
  computeKernelMatrixLowerHalf(K_aa, inducingset_pre, cf);
  K_aa.diagonal().array() -= noise_variance + 1e-6; // subtract noise variance
  K_aa = K_aa.selfadjointView<Eigen::Lower>();      // ensure K_aa is lower
                                                    // triangular for later use
  Eigen::MatrixXd K_bb(m, m);
  computeKernelMatrixLowerHalf(K_bb, inducingset, cf);
  K_bb.diagonal().array() -= noise_variance + 1e-6; // subtract noise variance
  Lambda_0 = chol_inverse(K_bb);
  elbo_0 = -logDet(Lambda_0);
  Eigen::MatrixXd K_ab;
  computeKernelMatrix(K_ab, inducingset_pre, inducingset, cf);
  Eigen::MatrixXd H = K_ab * Lambda_0; // 目前Lambda_0 = inv(K_bb)
  Eigen::MatrixXd H_K_ba = H * K_ab.transpose();
  Eigen::MatrixXd Da = K_aa - H_K_ba;
  Eigen::MatrixXd V = P_pre + param_alpha * (Da);
  Eigen::MatrixXd invV_H = chol_solve(V, H);
  Lambda_0 += H.transpose() * invV_H;
  elbo_0 += logDet(Lambda_0);

  const std::vector<double> &targets = inducingset_pre->y();
  Eigen::Map<const Eigen::VectorXd> mean_a(&targets[0], m_a);
  Eigen::VectorXd y_a = P_pre * inv_Sigma_u_pre * mean_a;
  eta_0 = invV_H.transpose() * y_a;

  Eigen::MatrixXd V_plus_H_K_ba = V + H_K_ba;
  Eigen::VectorXd alpha_y_a = chol_solve(V_plus_H_K_ba, y_a);
  elbo_0 += y_a.dot(alpha_y_a) + (1 / param_alpha) * logDet(V);

  Eigen::MatrixXd I_plus_alpha_invP_pre_Da =
      Eigen::MatrixXd::Identity(m_a, m_a) + param_alpha * invP_pre * Da;
  elbo_0 += (1 / param_alpha) * logDet(I_plus_alpha_invP_pre_Da);

  elbo_0 -= logDet(K_aa_pre) + logDet(Sigma_u_pre);

  elbo_0 -= mean_a.dot(
      (inv_Sigma_u_pre * P_pre * inv_Sigma_u_pre - inv_Sigma_u_pre) * mean_a);

  elbo_0 *= -0.5;

  if (!hyper_grad && !inducing_grad)
    return;
  // 初始化 eta_dot_0, Lambda_dot_0, elbo_dot_0

  size_t hyperparam_dim = cf->get_param_dim(); // 不更新噪声参数
  size_t new_inducing_start_idx = m_a;
  size_t new_inducing_points_num = m - m_a;
  size_t update_inducing_dim = (m - m_a) * input_dim;

  int variational_param_dim =
      hyperparam_dim + m * input_dim; //在线学习时只更新超参数，新的诱导点位置

  eta_dot_0.resize(variational_param_dim);
  Lambda_dot_0.resize(variational_param_dim);
  for (size_t i = 0; i < variational_param_dim; i++) {
    eta_dot_0[i].resize(m);
    eta_dot_0[i].setZero();
    Lambda_dot_0[i].resize(m, m);
    Lambda_dot_0[i].setZero();
  }
  elbo_dot_0.resize(variational_param_dim);
  elbo_dot_0.setZero();

  Eigen::MatrixXd inv_K_bb = chol_inverse(K_bb);
  Eigen::MatrixXd inv_Lambda_0 = K_bb;
  Eigen::MatrixXd S_0 = H * inv_Lambda_0 * H.transpose() + V;
  Eigen::MatrixXd inv_S_0 = chol_inverse(S_0);
  Eigen::MatrixXd Lambda_1 = Lambda_0;

  Eigen::MatrixXd H_dot_0;
  Eigen::MatrixXd V_dot_0;
  Eigen::MatrixXd Lambda_dot_1;

  //定义矩阵逐元素相乘后所有元素求和的函数
  auto elementwise_sum = [](const Eigen::MatrixXd &mat,
                            const Eigen::MatrixXd &other) {
    return (mat.array() * other.array()).sum();
  };

  //关于超参的梯度
  if (hyper_grad) {

    std::vector<Eigen::MatrixXd> K_aa_dot_hyper(hyperparam_dim);
    std::vector<Eigen::MatrixXd> K_ab_dot_hyper(hyperparam_dim);
    std::vector<Eigen::MatrixXd> K_bb_dot_hyper(hyperparam_dim);
    for (size_t i = 0; i < hyperparam_dim; i++) {
      K_ab_dot_hyper[i] = Eigen::MatrixXd::Zero(m_a, m);
      K_bb_dot_hyper[i] = Eigen::MatrixXd::Zero(m, m);
      K_aa_dot_hyper[i] = Eigen::MatrixXd::Zero(m_a, m_a);
    }
    //填充导数矩阵
    Eigen::VectorXd g_hyper;
    g_hyper.resize(hyperparam_dim);
    size_t update_hyperParam_dim =
        cf->get_param_dim() - 1; // 历史伪观测与噪声参数无关
    // K_ab_dot
    for (size_t i = 0; i < m_a; i++) {
      for (size_t j = 0; j < m; j++) {
        g_hyper.setZero();
        cf->grad(inducingset_pre->x(i), inducingset->x(j), g_hyper);
        for (size_t p = 0; p < update_hyperParam_dim; p++) {
          K_ab_dot_hyper[p](i, j) = g_hyper(p);
        }
      }
    }
    // K_bb_dot
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j <= i; j++) {
        g_hyper.setZero();
        cf->grad(inducingset->x(j), inducingset->x(i), g_hyper);
        for (size_t p = 0; p < update_hyperParam_dim; p++) {
          K_bb_dot_hyper[p](i, j) = g_hyper(p);
          if (j != i)
            K_bb_dot_hyper[p](j, i) = g_hyper(p); // symmetry property
        }
      }
    }
    // K_aa_dot
    for (size_t i = 0; i < m_a; i++) {
      for (size_t j = 0; j <= i; j++) {
        g_hyper.setZero();
        cf->grad(inducingset_pre->x(j), inducingset_pre->x(i), g_hyper);
        for (size_t p = 0; p < update_hyperParam_dim; p++) {
          K_aa_dot_hyper[p](i, j) = g_hyper(p);
          if (j != i)
            K_aa_dot_hyper[p](j, i) = g_hyper(p); // symmetry property
        }
      }
    }

    for (size_t p = 0; p < hyperparam_dim; p++) {
      if (p == hyperparam_dim - 1) {
        elbo_dot_0[p] = 0; //关于噪声的梯度为零
        Lambda_dot_0[p].resize(m, m);
        Lambda_dot_0[p].setZero();
        eta_dot_0[p].resize(m);
        eta_dot_0[p].setZero();

      } else {
        Lambda_dot_0[p] = -inv_K_bb * K_bb_dot_hyper[p] * inv_K_bb;
        H_dot_0 = K_ab_dot_hyper[p] * inv_K_bb + K_ab * Lambda_dot_0[p];
        V_dot_0 =
            param_alpha * (K_aa_dot_hyper[p] - H_dot_0 * K_ab.transpose() -
                           H * K_ab_dot_hyper[p].transpose());
        Lambda_dot_1 = Lambda_dot_0[p] + H_dot_0.transpose() * invV_H -
                       invV_H.transpose() * V_dot_0 * invV_H +
                       invV_H.transpose() * H_dot_0;
        Eigen::MatrixXd invV = chol_inverse(V);
        eta_dot_0[p] =
            (H_dot_0.transpose() - invV_H.transpose() * V_dot_0) * invV * y_a;
        //计算dpsi_dLambda_0;
        elbo_dot_0[p] = 0;
        Eigen::MatrixXd elbo_deriv;
        {
          Eigen::VectorXd temp_vec =
              inv_Lambda_0 * H.transpose() * inv_S_0 * y_a;
          elbo_deriv = -inv_Lambda_0 + temp_vec * temp_vec.transpose();
        }
        elbo_dot_0[p] += elementwise_sum(elbo_deriv, Lambda_dot_0[p]);
        //计算dpsi_dH_0
        {
          Eigen::VectorXd temp_vec = inv_S_0 * y_a;
          elbo_deriv =
              -2.0 * temp_vec * temp_vec.transpose() * H * inv_Lambda_0;
        }
        elbo_dot_0[p] += elementwise_sum(elbo_deriv, H_dot_0);
        //计算dpsi_dV_0
        {
          Eigen::VectorXd temp_vec = inv_S_0 * y_a;

          elbo_deriv = -1.0 * temp_vec * temp_vec.transpose() +
                       (1.0 / param_alpha) * invV;
        }
        elbo_dot_0[p] += elementwise_sum(elbo_deriv, V_dot_0);
        //计算dpsi_dLambda_1
        elbo_deriv = chol_inverse(Lambda_1);
        elbo_dot_0[p] += elementwise_sum(elbo_deriv, Lambda_dot_1);
        elbo_dot_0[p] *= -0.5;

        Lambda_dot_0[p] = Lambda_dot_1;
      }
    }
  }

  if (inducing_grad) {
    //关于诱导点的梯度
    std::vector<Eigen::MatrixXd> K_ba_dot_inducing(update_inducing_dim);
    std::vector<Eigen::MatrixXd> K_bb_dot_inducing(update_inducing_dim);
    for (size_t i = 0; i < update_inducing_dim; i++) {
      K_ba_dot_inducing[i] = Eigen::MatrixXd::Zero(m, m_a);
      K_bb_dot_inducing[i] = Eigen::MatrixXd::Zero(m, m);
    }
    //填充导数矩阵
    Eigen::VectorXd g_ind;
    g_ind.resize(input_dim);
    for (size_t i = 0; i < update_inducing_dim; i++) {
      K_bb_dot_inducing[i] = Eigen::MatrixXd::Zero(m, m);
      K_ba_dot_inducing[i] = Eigen::MatrixXd::Zero(m, m_a);
    }
    // K_ba_dot for inducing points
    for (size_t i = 0; i < new_inducing_points_num; i++) {
      size_t param_id = i * input_dim;
      size_t inducing_idx = new_inducing_start_idx + i;
      for (size_t j = 0; j < m_a; j++) {
        g_ind.setZero();
        cf->grad_wrt_x1(inducingset->x(inducing_idx), inducingset_pre->x(j),
                        g_ind);
        for (size_t p = 0; p < input_dim; p++) {
          K_ba_dot_inducing[param_id + p](inducing_idx, j) = g_ind(p);
        }
      }
    }
    // K_RR_dot for inducing points
    for (size_t i = 0; i < new_inducing_points_num; i++) {
      size_t inducing_idx_i = new_inducing_start_idx + i;
      for (size_t j = 0; j < i; j++) {
        size_t inducing_idx_j = new_inducing_start_idx + j;
        g_ind.setZero();
        cf->grad_wrt_x1(inducingset->x(inducing_idx_i),
                        inducingset->x(inducing_idx_j), g_ind);

        size_t param_id_i = i * input_dim;
        size_t param_id_j = j * input_dim;

        for (size_t p = 0; p < input_dim; p++) {
          K_bb_dot_inducing[param_id_i + p](inducing_idx_i, inducing_idx_j) =
              g_ind(p);
          K_bb_dot_inducing[param_id_i + p](inducing_idx_j, inducing_idx_i) =
              g_ind(p);
          K_bb_dot_inducing[param_id_j + p](inducing_idx_i, inducing_idx_j) =
              -g_ind(p);
          K_bb_dot_inducing[param_id_j + p](inducing_idx_j, inducing_idx_i) =
              -g_ind(p);
        }
      }
    }

    for (size_t p = 0; p < update_inducing_dim; p++) {
      size_t param_id = hyperparam_dim - 1 + m_a * input_dim + p;
      Lambda_dot_0[param_id] = -inv_K_bb * K_bb_dot_inducing[p] * inv_K_bb;
      H_dot_0 = K_ba_dot_inducing[p].transpose() * inv_K_bb +
                K_ab * Lambda_dot_0[param_id];
      V_dot_0 = param_alpha *
                (-H_dot_0 * K_ab.transpose() - H * K_ba_dot_inducing[p]);
      Lambda_dot_1 = Lambda_dot_0[param_id] + H_dot_0.transpose() * invV_H -
                     invV_H.transpose() * V_dot_0 * invV_H +
                     invV_H.transpose() * H_dot_0;
      Eigen::MatrixXd invV = chol_inverse(V);
      eta_dot_0[param_id] =
          (H_dot_0.transpose() - invV_H.transpose() * V_dot_0) * invV * y_a;
      //计算dpsi_dLambda_0;
      elbo_dot_0[param_id] = 0;
      Eigen::MatrixXd elbo_deriv;
      {
        Eigen::VectorXd temp_vec = inv_Lambda_0 * H.transpose() * inv_S_0 * y_a;
        elbo_deriv = -inv_Lambda_0 + temp_vec * temp_vec.transpose();
      }
      elbo_dot_0[param_id] +=
          elementwise_sum(elbo_deriv, Lambda_dot_0[param_id]);
      //计算dpsi_dH_0
      {
        Eigen::VectorXd temp_vec = inv_S_0 * y_a;
        elbo_deriv = -2.0 * temp_vec * temp_vec.transpose() * H * inv_Lambda_0;
      }
      elbo_dot_0[param_id] += elementwise_sum(elbo_deriv, H_dot_0);
      //计算dpsi_dV_0
      {
        Eigen::VectorXd temp_vec = inv_S_0 * y_a;

        elbo_deriv =
            -1.0 * temp_vec * temp_vec.transpose() + (1.0 / param_alpha) * invV;
      }
      elbo_dot_0[param_id] += elementwise_sum(elbo_deriv, V_dot_0);
      //计算dpsi_dLambda_1
      elbo_deriv = chol_inverse(Lambda_1);
      elbo_dot_0[param_id] += elementwise_sum(elbo_deriv, Lambda_dot_1);
      elbo_dot_0[param_id] *= -0.5;

      Lambda_dot_0[param_id] = Lambda_dot_1;
    }
  }
}

void RecursiveGaussianProcess::storeOldPosterior() {
  inducingset_pre = new SampleSet(static_cast<int>(get_input_dim()));
  for (size_t i = 0; i < inducingset->size(); i++) {
    inducingset_pre->add(inducingset->x(i), inducingset->y(i));
  }

  size_t m_a = inducingset_pre->size();

  double noise_variance =
      std::exp(cf->get_loghyper()(cf->get_param_dim() - 1) * 2);
  Eigen::MatrixXd K_aa(m_a, m_a);
  computeKernelMatrixLowerHalf(K_aa, inducingset_pre, cf);
  K_aa.diagonal().array() -= noise_variance + 1e-6; // subtract noise variance
  K_aa_pre = K_aa.selfadjointView<Eigen::Lower>();  // ensure K_aa_pre is lower
                                                    // triangular
  inv_K_aa_pre = chol_inverse(K_aa);
  inv_Sigma_u_pre = chol_inverse(cov_inducing);

  invP_pre = inv_Sigma_u_pre - inv_K_aa_pre; // inv(P_pre)
  P_pre = chol_inverse(invP_pre);
}

std::vector<double> RecursiveGaussianProcess::epochUpdate(bool verbose) {
  if (inducingset->empty()) {
    std::cerr
        << "Error: inducing set is empty! Please Specify inducing points first!"
        << std::endl;
    return {};
  }
  if (sampleset->empty()) {
    std::cerr << "Error: sample set is empty! Please add training data first!"
              << std::endl;
    return {};
  }

  storeOldPosterior();

  addNewInducingPoints();

  static size_t n_inducing_max = 75;
  size_t n_inducing_new = inducingset->size();
  // TODO:尝试对比逐点删除和批量删除的效果
  // 逐点删除 用while循环 while(n_inducing_new > n_inducing_max)()
  //批量删除
  if (n_inducing_new > n_inducing_max) {
    //基于历史伪观测更新新诱导点的后验
    updatePosteriorWithHistoryInfo(false, false);

    // TODO:
    // 完成一次新数据的后验更新，然后再执行后面删除诱导点的必要操作，不然可能刚新加入的诱导点就被删除了

    // TODO: 删除多余诱导点
    // deleteRedundantInducingPoints();
  }

  // return;

  size_t n = sampleset->size();
  size_t m = inducingset->size();
  size_t batch_size = 50;
  size_t param_dim =
      cf->get_param_dim() + m * input_dim; // 超参数维度 + 诱导点位置维度
  if (!adam_optimizer) {
    adam_optimizer.reset(new AdamOptimizer(param_dim, 0.002));
  }

  size_t max_batches = (n + batch_size - 1) / batch_size;
  std::vector<double> epoch_lml;
  for (size_t epoch = 0; epoch < 50; epoch++) {
    std::cout << "Epoch: " << epoch << std::endl;
    recursive_initialized = false;
    auto random_indices = Utils::randperm(n);
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

      batchUpdate(batch_inputs, batch_targets, true, true);

      Eigen::VectorXd params = get_hyperparameters();

      bool converged = adam_optimizer->step(elbo_dot_0, params);
      // update_hyperparameters(params);
      epoch_lml.push_back(elbo_0);
      if (converged) {
        std::cout << "Optimization converged at iteration "
                  << adam_optimizer->get_iteration() << std::endl;
        break; // 提前退出 Epoch 循环
      }
    }
    adam_optimizer->set_learning_rate_decay(0.9, 50);
  }

  Eigen::VectorXd mean_inducing = chol_solve(Lambda_0, eta_0);
  for (size_t i = 0; i < inducingset->size(); i++) {
    inducingset->set_y(i, mean_inducing(i));
  }
  cov_inducing = chol_inverse(Lambda_0);

  cf->loghyper_changed = true;

  return epoch_lml;
}

void RecursiveGaussianProcess::batchUpdate(
    const std::vector<Eigen::VectorXd> &batch_inputs,
    const Eigen::VectorXd &batch_targets, bool hyper_opt, bool inducing_opt) {
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

  if (!recursive_initialized) {
    updatePosteriorWithHistoryInfo(hyper_opt,
                                   inducing_opt); // 先计算一次 elbo_dot_0
    elbo_0 -= 0.5 * static_cast<double>(sampleset->size()) * log2pi;
    recursive_initialized = true;
  }

  // 使用 unique_ptr 管理临时 SampleSet
  auto batchSet = std::make_unique<SampleSet>(input_dim);
  for (size_t i = 0; i < b; i++) {
    batchSet->add(batch_inputs[i].data(), batch_targets(i));
  }
  size_t noise_idx = cf->get_param_dim() - 1;
  double noise_variance = std::exp(cf->get_loghyper()(noise_idx) * 2.0);

  /// STEP:计算基础矩阵
  Eigen::MatrixXd K_RR(m, m);
  computeKernelMatrixLowerHalf(K_RR, inducingset, cf);
  K_RR.diagonal().array() -= noise_variance;
  K_RR = K_RR.selfadjointView<Eigen::Lower>();
  Eigen::MatrixXd iK_RR = chol_inverse(K_RR);

  Eigen::MatrixXd K_RX(m, b);
  computeKernelMatrix(K_RX, inducingset, batchSet.get(), cf);

  /// STEP:计算中间矩阵
  Eigen::MatrixXd H_T = iK_RR * K_RX;
  Eigen::MatrixXd H = H_T.transpose();

  Eigen::VectorXd diag_V;
  Eigen::VectorXd diagK_XX(b);
  for (size_t i = 0; i < b; i++) {
    diagK_XX(i) = cf->get(batchSet->x(i), batchSet->x(i)) - noise_variance;
  }
  diag_V = param_alpha * (diagK_XX - (H * K_RX).diagonal());
  diag_V.array() += noise_variance;

  Eigen::VectorXd r;
  auto iLambda_0 = chol_inverse(Lambda_0);
  auto y = batch_targets;
  auto H_iLambda_0 = H * iLambda_0;
  r = y - H_iLambda_0 * eta_0;

  Eigen::MatrixXd S;
  S = H_iLambda_0 * H_T;
  S.diagonal() += diag_V;
  auto iS = chol_inverse(S);

  /// STEP: propagate natural mean and precision matrix
  auto inv_diag_V = diag_V.cwiseInverse();
  auto inv_V = inv_diag_V.asDiagonal();
  auto inv_V_H = inv_V * H;
  auto eta_1 = eta_0 + inv_V_H.transpose() * y;
  auto Lambda_1 = Lambda_0 + inv_V_H.transpose() * H;

  /// STEP: calculate approximate log likelihood
  double bound = r.dot(iS * r);
  bound += logDet(Lambda_1) - logDet(Lambda_0);
  bound += (1 / param_alpha) * diag_V.array().log().sum();
  bound -= (1 - param_alpha) / param_alpha * std::log(noise_variance) *
           static_cast<double>(b);
  elbo_0 -= 0.5 * bound;

  if (!hyper_opt && !inducing_opt) {
    /// STEP:propagate natural parameters
    eta_0 = eta_1;
    Lambda_0 = Lambda_1;
    return;
  }

  Eigen::MatrixXd H_dot;
  Eigen::VectorXd r_dot;
  Eigen::MatrixXd V_dot;
  Eigen::MatrixXd S_dot;
  Eigen::VectorXd eta_dot_1;
  Eigen::MatrixXd Lambda_dot_1;

  auto iLambda_0_eta_0 = iLambda_0 * eta_0;
  auto inv_V_y = inv_V * y;


  auto iLambda_1 = chol_inverse(Lambda_1);
  Eigen::VectorXd iS_r = iS * r;

  //定义矩阵逐元素相乘后所有元素求和的函数
  auto elementwise_sum = [](const Eigen::MatrixXd &mat,
                            const Eigen::MatrixXd &other) {
    return (mat.array() * other.array()).sum();
  };

  /// STEP:calculate derivatives of basic matrices
  int update_hyperParam_dim =
      cf->get_param_dim(); //在线学习时只更新超参数，诱导点位置不更新
  if (hyper_opt) {
    std::vector<Eigen::MatrixXd> K_RR_dot_hyper(noise_idx);
    std::vector<Eigen::MatrixXd> K_RX_dot_hyper(noise_idx);
    std::vector<Eigen::VectorXd> diagK_XX_dot_hyper(noise_idx);
    for (size_t i = 0; i < noise_idx; i++) {
      K_RR_dot_hyper[i] = Eigen::MatrixXd::Zero(m, m);
      K_RX_dot_hyper[i] = Eigen::MatrixXd::Zero(m, b);
      diagK_XX_dot_hyper[i] = Eigen::VectorXd::Zero(b);
    }
    Eigen::VectorXd g_hyper(update_hyperParam_dim);
    // K_RX_dot
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < b; j++) {
        g_hyper.setZero();
        cf->grad(batch_inputs[j], inducingset->x(i), g_hyper);
        for (size_t p = 0; p < noise_idx; p++) {
          K_RX_dot_hyper[p](i, j) = g_hyper(p);
        }
      }
    }
    // K_RR_dot
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j <= i; j++) {
        g_hyper.setZero();
        cf->grad(inducingset->x(j), inducingset->x(i), g_hyper);
        for (size_t p = 0; p < noise_idx; p++) {
          K_RR_dot_hyper[p](i, j) = g_hyper(p);
          if (j != i)
            K_RR_dot_hyper[p](j, i) = g_hyper(p); // symmetry property
        }
      }
    }
    // diagK_XX_dot
    for (size_t i = 0; i < b; i++) {
      g_hyper.setZero();
      cf->grad(batch_inputs[i], batch_inputs[i], g_hyper);
      for (size_t p = 0; p < noise_idx; p++) {
        diagK_XX_dot_hyper[p](i) = g_hyper(p);
      }
    }

    for (size_t p = 0; p < update_hyperParam_dim; p++) {
      if (p == noise_idx) {
        //专门处理噪声参数的梯度
        r_dot = H_iLambda_0 * Lambda_dot_0[p] * iLambda_0_eta_0 -
                H_iLambda_0 * eta_dot_0[p];
        V_dot = Eigen::MatrixXd::Zero(b, b);
        V_dot.diagonal().array() = 2 * noise_variance;
        S_dot = V_dot - H_iLambda_0 * Lambda_dot_0[p] * H_iLambda_0.transpose();

        eta_dot_1 = eta_dot_0[p] - inv_V_H.transpose() * V_dot * inv_V_y;

        Lambda_dot_1 = Lambda_dot_0[p] - inv_V_H.transpose() * V_dot * inv_V_H;

        double elbo_batch_dot = 0.0;
        /// STEP:calculate block derivatives
        //计算dpsi_dLambda_k
        elbo_batch_dot += elementwise_sum(iLambda_1, Lambda_dot_1);

        //计算dpsi_dLambda_k-1
        elbo_batch_dot -= elementwise_sum(iLambda_0, Lambda_dot_0[p]);

        //计算dpsi_drk
        elbo_batch_dot += r_dot.dot(2 * iS_r);

        //计算dpsi_dS_k
        elbo_batch_dot -= iS_r.dot(S_dot * iS_r);
        ;

        //计算dpsi_dV_k
        elbo_batch_dot += 1.0 / param_alpha * inv_diag_V.dot(V_dot.diagonal());

        //正则项中的噪声相关项
        elbo_batch_dot -=
            2 * (1.0 - param_alpha) / param_alpha * static_cast<double>(b);

        /// STEP:calculate final gradients
        elbo_dot_0[p] -= 0.5 * elbo_batch_dot;

        /// STEP:propagate gradients of natural parameters
        eta_dot_0[p] = eta_dot_1;
        Lambda_dot_0[p] = Lambda_dot_1;
      } else {
        /// STEP:calculate derivatives of intermediate matrices

        H_dot = K_RX_dot_hyper[p].transpose() * iK_RR -
                H * K_RR_dot_hyper[p] * iK_RR;

        r_dot = -H_dot * iLambda_0_eta_0 +
                H_iLambda_0 * Lambda_dot_0[p] * iLambda_0_eta_0 -
                H_iLambda_0 * eta_dot_0[p];

        auto d_dot = diagK_XX_dot_hyper[p] - (H_dot * K_RX).diagonal() -
                     (H * K_RX_dot_hyper[p]).diagonal();

        V_dot = param_alpha * d_dot.asDiagonal();

        S_dot = H_dot * H_iLambda_0.transpose() -
                H_iLambda_0 * Lambda_dot_0[p] * H_iLambda_0.transpose() +
                H_iLambda_0 * H_dot.transpose() + V_dot;

        eta_dot_1 = eta_dot_0[p] + H_dot.transpose() * inv_V_y -
                    inv_V_H.transpose() * V_dot * inv_V_y;

        Lambda_dot_1 = Lambda_dot_0[p] + H_dot.transpose() * inv_V_H -
                       inv_V_H.transpose() * V_dot * inv_V_H +
                       inv_V_H.transpose() * H_dot;

        double elbo_batch_dot = 0.0;
        /// STEP:calculate block derivatives
        //计算dpsi_dLambda_k
        elbo_batch_dot += elementwise_sum(iLambda_1, Lambda_dot_1);

        //计算dpsi_dLambda_k-1
        elbo_batch_dot -= elementwise_sum(iLambda_0, Lambda_dot_0[p]);

        //计算dpsi_drk
        elbo_batch_dot += r_dot.dot(2.0 * iS_r);

        //计算dpsi_dS_k
        elbo_batch_dot -= iS_r.dot(S_dot * iS_r);

        //计算dpsi_dV_k
        elbo_batch_dot += 1 / param_alpha * inv_diag_V.dot(V_dot.diagonal());

        /// STEP:calculate final gradients
        elbo_dot_0[p] -= 0.5 * elbo_batch_dot;

        /// STEP:propagate gradients of natural parameters
        eta_dot_0[p] = eta_dot_1;
        Lambda_dot_0[p] = Lambda_dot_1;
      }
    }
  }

  if (inducing_opt) {
    //处理诱导点相关的梯度(只更新新的诱导点位置的梯度，历史诱导点位置的梯度为零)
    size_t new_inducing_start_idx = inducingset_pre->size();
    size_t new_inducing_points_num = m - new_inducing_start_idx;
    size_t update_inducing_dim = (m - new_inducing_start_idx) * input_dim;
    std::vector<Eigen::MatrixXd> K_RR_dot_inducing(update_inducing_dim);
    std::vector<Eigen::MatrixXd> K_RX_dot_inducing(update_inducing_dim);
    for (size_t i = 0; i < update_inducing_dim; i++) {
      K_RR_dot_inducing[i] = Eigen::MatrixXd::Zero(m, m);
      K_RX_dot_inducing[i] = Eigen::MatrixXd::Zero(m, b);
    }
    Eigen::VectorXd g_ind(input_dim);
    // K_RX_dot for inducing points
    for (size_t i = 0; i < new_inducing_points_num; i++) {
      size_t param_id = i * input_dim;
      size_t inducing_idx = new_inducing_start_idx + i;
      for (size_t j = 0; j < b; j++) {
        g_ind.setZero();
        cf->grad_wrt_x1(inducingset->x(inducing_idx), batchSet->x(j), g_ind);
        for (size_t p = 0; p < input_dim; p++) {
          K_RX_dot_inducing[param_id + p](inducing_idx, j) = g_ind(p);
        }
      }
    }
    // K_RR_dot for inducing points
    for (size_t i = 0; i < new_inducing_points_num; i++) {
      size_t inducing_idx_i = new_inducing_start_idx + i;
      for (size_t j = 0; j < i; j++) {
        size_t inducing_idx_j = new_inducing_start_idx + j;
        g_ind.setZero();
        cf->grad_wrt_x1(inducingset->x(inducing_idx_i),
                        inducingset->x(inducing_idx_j), g_ind);

        size_t param_id_i = i * input_dim;
        size_t param_id_j = j * input_dim;

        for (size_t p = 0; p < input_dim; p++) {
          K_RR_dot_inducing[param_id_i + p](inducing_idx_i, inducing_idx_j) =
              g_ind(p);
          K_RR_dot_inducing[param_id_i + p](inducing_idx_j, inducing_idx_i) =
              g_ind(p);
          K_RR_dot_inducing[param_id_j + p](inducing_idx_i, inducing_idx_j) =
              -g_ind(p);
          K_RR_dot_inducing[param_id_j + p](inducing_idx_j, inducing_idx_i) =
              -g_ind(p);
        }
      }
    }

    //处理诱导点相关的梯度
    for (size_t p = 0; p < update_inducing_dim; p++) {
      size_t param_id =
          update_hyperParam_dim + new_inducing_start_idx * input_dim + p;
      /// STEP:calculate derivatives of intermediate matrices
      H_dot = K_RX_dot_inducing[p].transpose() * iK_RR -
              K_RX.transpose() * iK_RR * K_RR_dot_inducing[p] * iK_RR;

      r_dot = -H_dot * iLambda_0 * eta_0 - H_iLambda_0 * eta_dot_0[param_id] +
              H_iLambda_0 * Lambda_dot_0[param_id] * iLambda_0 * eta_0;

      auto d_dot =
          -(H_dot * K_RX).diagonal() - (H * K_RX_dot_inducing[p]).diagonal();

      V_dot = param_alpha * d_dot.asDiagonal();

      S_dot = H_dot * H_iLambda_0.transpose() -
              H_iLambda_0 * Lambda_dot_0[param_id] * H_iLambda_0.transpose() +
              H_iLambda_0 * H_dot.transpose() + V_dot;

      eta_dot_1 = eta_dot_0[param_id] + H_dot.transpose() * inv_V_y -
                  inv_V_H.transpose() * V_dot * inv_V_y;

      Lambda_dot_1 = Lambda_dot_0[param_id] + H_dot.transpose() * inv_V_H -
                     inv_V_H.transpose() * V_dot * inv_V_H +
                     inv_V_H.transpose() * H_dot;

      double elbo_batch_dot = 0.0;
      /// STEP:calculate block derivatives
      //计算dpsi_dLambda_k
      elbo_batch_dot += elementwise_sum(iLambda_1, Lambda_dot_1);

      //计算dpsi_dLambda_k-1
      elbo_batch_dot -= elementwise_sum(iLambda_0, Lambda_dot_0[param_id]);

      //计算dpsi_drk
      elbo_batch_dot += r_dot.dot(2 * iS_r);

      //计算dpsi_dS_k
      elbo_batch_dot -= iS_r.dot(S_dot * iS_r);

      //计算dpsi_dV_k
      elbo_batch_dot += 1 / param_alpha * inv_diag_V.dot(V_dot.diagonal());

      /// STEP:calculate final gradients
      elbo_dot_0[param_id] -= 0.5 * elbo_batch_dot;

      /// STEP:propagate gradients of natural parameters
      eta_dot_0[param_id] = eta_dot_1;
      Lambda_dot_0[param_id] = Lambda_dot_1;
    }
  }
  /// STEP:propagate natural parameters
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
  compute();

  return cov_inducing;
}

void RecursiveGaussianProcess::setInducingTargetZeros() {
  for (size_t i = 0; i < inducingset->size(); ++i) {
    inducingset->set_y(i, 0.0);
  }
}

} // namespace libgp
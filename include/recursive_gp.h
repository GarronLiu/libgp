// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2025, Garron Liu <ljr799910832@sjtu.edu.cn>
// All rights reserved.

/*!
 *
 *   \page licence Licensing
 *
 *     recursive gaussian process implementation
 *
 *      \verbinclude "../COPYING"
 */

#ifndef __RECURSIVE_GP_H__
#define __RECURSIVE_GP_H__
#include "KMeans.h"
#include "cov.h"
#include "gp.h"
#include "gp_utils.h"
#include "sampleset.h"
#include "sparse_gp.h"

#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include <cmath>
#include <memory>

namespace libgp {

class AdamOptimizer {
private:
  double learning_rate_;
  double beta1_;
  double beta2_;
  double epsilon_;
  double tolerance_; // 新增：收敛判断阈值

  int iteration_;
  Eigen::VectorXd m_; // 一阶矩估计
  Eigen::VectorXd v_; // 二阶矩估计

public:
  // 构造函数
  AdamOptimizer(int param_size, double learning_rate = 0.005,
                double beta1 = 0.9, double beta2 = 0.99, double epsilon = 1e-8,
                double tolerance = 1e-5, // 新增：默认阈值可以设为 1e-5
                int max_iterations = 1000)
      : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2),
        epsilon_(epsilon), tolerance_(tolerance), iteration_(0) {

    // 初始化矩估计向量
    m_ = Eigen::VectorXd::Zero(param_size);
    v_ = Eigen::VectorXd::Zero(param_size);
  }

  // 重置优化器状态
  void reset() {
    iteration_ = 0;
    m_.setZero();
    v_.setZero();
  }

  // 单步优化
  // gradient_func: 返回当前参数下的梯度向量
  // params: 当前参数值（会被原地更新）
  // 返回: 是否收敛
  bool step(const Eigen::VectorXd &grad, Eigen::VectorXd &params) {
    // check dimensions
    if (grad.size() != params.size()) {
      throw std::invalid_argument("Gradient and parameter size mismatch.");
    }

    iteration_++;

    // 更新一阶矩估计（动量）
    m_ = beta1_ * m_ + (1.0 - beta1_) * grad;

    // 更新二阶矩估计（RMSProp）
    v_ = beta2_ * v_ + (1.0 - beta2_) * grad.array().square().matrix();

    // 偏差修正
    double beta1_t = std::pow(beta1_, iteration_);
    double beta2_t = std::pow(beta2_, iteration_);

    Eigen::VectorXd m_hat = m_ / (1.0 - beta1_t);
    Eigen::VectorXd v_hat = v_ / (1.0 - beta2_t);

    // 更新参数
    params.array() +=
        learning_rate_ * m_hat.array() / (v_hat.array().sqrt() + epsilon_);

    if (grad.cwiseAbs().maxCoeff() < tolerance_) {
      return true;
    }

    return false;
  }

  // 获取当前迭代信息
  int get_iteration() const { return iteration_; }
  double get_learning_rate() const { return learning_rate_; }

  // 设置学习率衰减
  void set_learning_rate_decay(double decay_rate, int decay_step) {
    if (iteration_ % decay_step == 0 && iteration_ > 0) {
      learning_rate_ *= decay_rate;
    }
  }
};

class RecursiveGaussianProcess : public SparseGaussianProcess {
  /** Recursive Gaussian process regression.
   *  @author Garron Liu */
public:
  /** Create an instance of RecursiveGaussianProcess with given input
   * dimensionality and covariance function. */
  RecursiveGaussianProcess(size_t input_dim, std::string covf_def);

  /** Create an instance of RecursiveGaussianProcess from file. */
  RecursiveGaussianProcess(const char *filename);

  virtual ~RecursiveGaussianProcess();

  void specify_inducingSet(std::vector<Eigen::VectorXd> inducing_points,
                           size_t m_random = 0, size_t m_clusters = 0);

  double log_likelihood() override; // analytical collapsed evidence lower bound
                                    // for recursive sparse GP

  Eigen::VectorXd log_likelihood_gradient() override;

  Eigen::MatrixXd getFlatPosteriorCovMatrix() override;

  void batchUpdate(const std::vector<Eigen::VectorXd> &batch_inputs,
                   const Eigen::VectorXd &batch_targets, bool hyper_opt,
                   bool inducing_opt);

  std::vector<double> epochUpdate(bool verbose = true);

  void setInducingTargetZeros();

protected:
  using SparseGaussianProcess::alpha_R;
  using SparseGaussianProcess::cov_inducing;
  using SparseGaussianProcess::Q_pred;

  // PEP training condition
  double param_alpha = 0.5; // 0 < alpha <=1, alpha =1 corresponds to FITC,
                            // alpha->0 corresponds to VFE

  // variables related to pretrained posterior
  using SparseGaussianProcess::inv_K_aa_pre;
  using SparseGaussianProcess::inv_Sigma_u_pre;
  using SparseGaussianProcess::invP_pre;
  using SparseGaussianProcess::K_aa_pre;
  using SparseGaussianProcess::P_pre;
  using SparseGaussianProcess::Sigma_u_pre;

  SampleSet *inducingset_pre; // inducing points

  /// @brief auxiliary variable for recursive update

  Eigen::VectorXd eta_0;    // natural mean
  Eigen::MatrixXd Lambda_0; // natural precision matrix
  double elbo_0;
  Eigen::VectorXd elbo_dot_0;

  std::vector<Eigen::VectorXd>
      eta_dot_0; // auxiliary variable for recursive update
  std::vector<Eigen::MatrixXd>
      Lambda_dot_0; // auxiliary variable for recursive update

  size_t batch_count = 0;
  bool recursive_initialized = false;

  double novelty_threshold;

  // ADAM optimizer for hyperparameter optimization
  std::unique_ptr<AdamOptimizer> adam_optimizer;

  void compute() override;

  void addNewInducingPoints();

  void storeOldPosterior();

  void updatePosteriorWithHistoryInfo(bool hyper_grad, bool inducing_grad);

  void deleteRedundantInducingPoints();
};
} // namespace libgp

#endif /* __RECURSIVE_GP_H__ */

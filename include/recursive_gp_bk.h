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
#include "cov.h"
#include "sampleset.h"
#include "KMeans.h"
#include "gp.h"
#include "gp_utils.h"

#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include <cmath>
#include <memory>


namespace libgp
{

class AdamOptimizer {
private:
    double learning_rate_;
    double beta1_;
    double beta2_;
    double epsilon_;
    
    int iteration_;
    Eigen::VectorXd m_;  // 一阶矩估计
    Eigen::VectorXd v_;  // 二阶矩估计
    
public:
    // 构造函数
    AdamOptimizer(int param_size, 
                  double learning_rate = 0.005, 
                  double beta1 = 0.9, 
                  double beta2 = 0.999, 
                  double epsilon = 1e-3,
                  int max_iterations = 1000)
        : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), 
          epsilon_(epsilon), 
          iteration_(0) {
        
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
    void step(const Eigen::VectorXd& grad,
              Eigen::VectorXd& params) {
        //check dimensions
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
        params.array() += learning_rate_ * m_hat.array() / (v_hat.array().sqrt() + epsilon_);
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


class RecursiveGaussianProcess : public GaussianProcess
{
  /** Sparse Gaussian process regression.
 *  @author Garron Liu */
public:
  /** Create an instance of RecursiveGaussianProcess with given input
   * dimensionality and covariance function. */
  RecursiveGaussianProcess(size_t input_dim, std::string covf_def);

  /** Create an instance of RecursiveGaussianProcess from file. */
  RecursiveGaussianProcess(const char* filename);

  virtual ~RecursiveGaussianProcess();

  void write(const char* filename) override;

  /** Predict target value for given input on inducing set.
   *  @param x input vector
   *  @return predicted value */
  double f(const double x[]) override;

  /** Predict variance of prediction for given input on inducing set.
   *  @param x input vector
   *  @return predicted variance */
  // virtual double var(const double x[]);
  double var(const double x[]) override;

  /** Add input-output-pair to sample set.
   *  Add a copy of the given input-output-pair to sample set.
   *  @param x input array
   *  @param y output value
   */
  void add_pattern(const double x[], double y) override;

  void specify_inducingSet(std::vector<Eigen::VectorXd> inducing_points, size_t m_random = 0, size_t m_clusters = 0);

  double log_likelihood() override; //analytical collapsed evidence lower bound for recursive sparse GP
  
  Eigen::VectorXd log_likelihood_gradient() override;

  void update_hyperparameters(const Eigen::VectorXd& params) override;

  Eigen::VectorXd get_hyperparameters() override;

  SampleSet* get_inducingSet()
  {
    return inducingset;
  }

  Eigen::MatrixXd getFlatInputs() override;

  Eigen::VectorXd getFlatTargets() override;

  Eigen::MatrixXd getFlatPosteriorCovMatrix() override;

  Eigen::VectorXd getFlatHyperparameters() override;

  Eigen::VectorXd getFlatAlpha() override;

  void batchUpdate(const std::vector<Eigen::VectorXd>& batch_inputs, const Eigen::VectorXd& batch_targets);

  void epochUpdate(bool verbose = true);
  
  void setInducingTargetZeros();

protected:
  using GaussianProcess::alpha; //  alpha = (K_XX_bar + \sigma_n^2 I)^-1 * y
  using GaussianProcess::L; // Cholesky of (K_XX_bar + \sigma_n^2 I)

  using GaussianProcess::k_star; // f_star = k_star * alpha_R
  Eigen::VectorXd alpha_R; // K_RR^-1 * u
  Eigen::MatrixXd L_R;    // var_star = k_star_star - k_star^T * L_R^-T * L_R^-1 * k_star

  //PEP training condition
  double param_alpha = 0.5; //0 < alpha <=1, alpha =1 corresponds to FITC, alpha->0 corresponds to VFE

  Eigen::MatrixXd L_K_RR;       // cholesky(K_RR)

  Eigen::MatrixXd D_X; // K_XX - K_XR * K_RR^-1 * K_RX

  /// @brief auxiliary variable for recursive update
  Eigen::VectorXd eta_0;
  Eigen::MatrixXd Lambda_0; 

  std::vector<Eigen::VectorXd> eta_dot_0; // auxiliary variable for recursive update
  std::vector<Eigen::MatrixXd> Lambda_dot_0; // auxiliary variable for recursive update

  double elbo_0;
  Eigen::VectorXd elbo_dot_0;

  size_t batch_count = 0;
  bool recursive_initialized = false;

  //ADAM optimizer for hyperparameter optimization
  std::unique_ptr<AdamOptimizer> adam_optimizer;

  void update_k_star(const Eigen::VectorXd& x_star) override;

  void update_alpha() override;

  void compute() override;
  
  SampleSet* inducingset;  // inducing points
};
}  // namespace libgp

#endif /* __RECURSIVE_GP_H__ */

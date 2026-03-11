// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_se_ard.h"
#include <cmath>

namespace libgp
{

  CovSEard::CovSEard(): lb_ell(1e-3), ub_ell(1e2), lb_sf2(1e-3), ub_sf2(1e2) {}

  CovSEard::~CovSEard() = default;
  
  bool CovSEard::init(int n)
  {
    input_dim = n;
    param_dim = n + 1;
    ell.resize(input_dim);
    loghyper.resize(param_dim);
    loghyper.setZero(); // 显式初始化为0
    return true;
  }
  
  double CovSEard::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  { 
    // k(x,y) = sf2 * exp(-0.5 * sum((x_i - y_i)^2 / ell_i^2))
    const double z = (x1 - x2).cwiseQuotient(ell).squaredNorm();
    return sf2 * std::exp(-0.5 * z);
  }
  
  void CovSEard::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    // 确保输出向量大小正确
    if (grad.size() != param_dim) {
      grad.resize(param_dim);
    }

    // z_i = (x_i - y_i)^2 / ell_i^2
    const Eigen::VectorXd z = (x1 - x2).cwiseQuotient(ell).array().square();
    const double k = sf2 * std::exp(-0.5 * z.sum());
    
    // d(log k) / d(log ell_i) = z_i
    // d(k) / d(log ell_i) = k * z_i
    grad.head(input_dim) = z * k;
    
    // d(k) / d(log sf) = 2 * k
    grad(input_dim) = 2.0 * k;
  }

  void CovSEard::grad_wrt_x1(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad_input)
  {
    grad_input.resize(input_dim);
    
    if (&x1 == &x2) { 
      grad_input.setZero();
      return;
    }

    const double k = get(x1, x2);
    const Eigen::VectorXd diff = x2 - x1;
    
    // d(k) / d(x_i) = k * (x2_i - x1_i) / ell_i^2
    // 使用 array() 操作避免显式创建 ell_sq 临时向量，利用 Eigen 表达式模板优化
    grad_input = k * (diff.array() / ell.array().square());
  }

  void CovSEard::hessian_wrt_x1_x2(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::MatrixXd &hessian_input)
  {
    hessian_input.resize(input_dim, input_dim);
    
    if (&x1 == &x2) { 
      hessian_input.setZero();
      return;
    }

    const double k = get(x1, x2);
    const Eigen::VectorXd diff = x1 - x2;
    
    // 计算 Hessian 矩阵
    for (size_t i = 0; i < input_dim; ++i) {
      for (size_t j = 0; j < input_dim; ++j) {
        if (i == j) {
          hessian_input(i, j) = k * (1.0 / (ell(i) * ell(i)) - (diff(i) * diff(j)) / (ell(i) * ell(i) * ell(i) * ell(i)));
        } else {
          hessian_input(i, j) = -k * (diff(i) * diff(j)) / (ell(i) * ell(i) * ell(j) * ell(j));
        }
      }
    }
  }
  
  void CovSEard::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    
    // 使用向量化操作替代循环，更高效
    ell = p.head(input_dim).array().exp();
    sf2 = std::exp(2.0 * p(input_dim));
  }
  
  std::string CovSEard::to_string()
  {
    return "CovSEard";
  }

  Eigen::VectorXd CovSEard::get_loghyper_lb()
  {
    Eigen::VectorXd lb_(param_dim);
    lb_.head(input_dim).array() = lb_ell;
    lb_(input_dim) = lb_sf2;
    lb_.array() = lb_.array().log();
    return lb_;
  }

  Eigen::VectorXd CovSEard::get_loghyper_ub()
  {
    Eigen::VectorXd ub_(param_dim);
    ub_.head(input_dim).array() = ub_ell;
    ub_(input_dim) = ub_sf2;
    ub_.array() = ub_.array().log();
    return ub_;
  }
}
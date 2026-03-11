// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_se_periodic.h"
#include <cmath>

namespace libgp
{

  CovSEPeriodic::CovSEPeriodic() {}

  CovSEPeriodic::~CovSEPeriodic() {}

  bool CovSEPeriodic::init(int n)
  {
    input_dim = n;
    param_dim = 4;
    loghyper.resize(param_dim);
    loghyper.setZero();
#ifdef CPPAD
    /* Eigen::Matrix<bool, Dynamic, 1> loghyperSparsityVec(param_dim + 2*input_dim);
    loghyperSparsityVec.setZero();
    loghyperSparsityVec.head(param_dim).setOnes();
    loghyperSparsity.initPattern(loghyperSparsityVec); */
#endif

    return true;
  }
  
  double CovSEPeriodic::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  { 
    const auto x_diff = x1 - x2;
    const double ell_inv = 1.0 / ell;
    const double lp_inv = 1.0 / lp;
    const double T_inv = 1.0 / T;
    
    // 计算SE部分: exp(-0.5 * ||x_diff/ell||^2)
    const double squared_term = (x_diff * ell_inv).squaredNorm();
    
    // 计算周期部分: exp(-2 * sin^2(π*x_diff/T) / lp^2)
    const double phi = M_PI * x_diff(0) * T_inv;  // 假设1维输入
    const double sin_phi = std::sin(phi);
    const double sin_term = sin_phi * sin_phi * lp_inv * lp_inv;
    
    return sf2 * std::exp(-0.5 * squared_term - 2.0 * sin_term);
  }
  
  void CovSEPeriodic::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
#ifndef CPPAD
    const auto x_diff = x1 - x2;
    const double ell_inv = 1.0 / ell;
    const double lp_inv = 1.0 / lp;
    const double T_inv = 1.0 / T;
    
    // 计算基础项(复用)
    const double squared_term = (x_diff * ell_inv).squaredNorm();
    const double phi = M_PI * x_diff(0) * T_inv;
    const double sin_phi = std::sin(phi);
    const double sin_phi_sq = sin_phi * sin_phi;
    const double sin_term = sin_phi_sq * lp_inv * lp_inv;
    
    // 计算核函数值
    const double k = sf2 * std::exp(-0.5 * squared_term - 2.0 * sin_term);
    
    grad.resize(param_dim);
    
    // ∂k/∂log(ell) = k * ||x_diff||^2 / ell^2
    grad(0) = k * squared_term;
    
    // ∂k/∂log(lp) = k * 4 * sin^2(phi) / lp^2
    grad(1) = k * 4.0 * sin_term;
    
    // ∂k/∂log(T) = k * (4/lp^2) * sin(phi)*cos(phi) * phi
    // 利用 sin(2*phi) = 2*sin(phi)*cos(phi)
    const double sin_2phi = std::sin(2.0 * phi);
    grad(2) = k * 2.0 * lp_inv * lp_inv * sin_2phi * phi;
    
    // ∂k/∂log(sf) = 2k
    grad(3) = 2.0 * k;
#else
    // Using CppAD to compute the gradient
    Eigen::VectorXd combined_input(param_dim + 2 * input_dim);
    Eigen::VectorXd::Map(combined_input.data(), param_dim) = loghyper;
    Eigen::VectorXd::Map(combined_input.data() + param_dim, input_dim) = x1;
    Eigen::VectorXd::Map(combined_input.data() + param_dim + input_dim, input_dim) = x2;
    
    // Forward mode sparse Jacobian calculation
    if(combined_input.size() != f_.Domain())
    {
      // Garron Liu newly added //
      // 按照loghyper，x1，x2的顺序合并成一个向量
      ADVector c_input(param_dim + 2 * input_dim);
      c_input.head(param_dim) = eigen_to_ad(loghyper);
      c_input.segment(param_dim, input_dim) = eigen_to_ad(x1);
      c_input.tail(input_dim) = eigen_to_ad(x2);
      //start recording
      CppAD::Independent(c_input);
      
      auto loghyper_ad = c_input.head(param_dim);
      auto x1_ad = c_input.segment(param_dim, input_dim);
      auto x2_ad = c_input.tail(input_dim);

      ADVector ell_ad(input_dim);
      for(size_t i = 0; i < input_dim; ++i) ell_ad(i) = CppAD::exp(loghyper_ad(i));
      AD<double> sf2_ad = CppAD::exp(2.0 * loghyper_ad(input_dim));

      ADVector y(1);
      y(0) = sf2_ad * CppAD::exp(-0.5 * (x1_ad - x2_ad).cwiseQuotient(ell_ad).squaredNorm());
      // y(0) = sf2_ad * CppAD::exp( - ell_ad(0) * CppAD::acos(x1_ad.dot(x2_ad) / CppAD::sqrt(x1_ad.squaredNorm() * x2_ad.squaredNorm()+0.0001)) );

      CppAD::ADFun<double> f(c_input, y); 
      f.optimize();
      f_ = f;
      std::cout<<"Finished recording CppAD function for CovSEPeriodic."<<std::endl;

      // Garron Liu newly added //
    }
    Eigen::VectorXd jac(param_dim);
    f_.SparseJacobianReverse(combined_input, loghyperSparsity.sparsity(), loghyperSparsity.row(), loghyperSparsity.col(), jac, loghyperSparsity.workJacobian());

    grad = Eigen::VectorXd::Map(jac.data(), param_dim);
#endif
  }

  void CovSEPeriodic::grad_wrt_x1(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad_input)
  {
    grad_input.resize(input_dim);
    if (&x1 == &x2)
    { 
      grad_input.setZero();
    }
    const auto x_diff = x1 - x2;
    const double ell_inv = 1.0 / ell;
    const double ell_sq_inv = ell_inv * ell_inv;
    const double lp_inv = 1.0 / lp;
    const double lp_sq_inv = lp_inv * lp_inv;
    const double T_inv = 1.0 / T;
    
    // 计算基础项
    const double squared_term = (x_diff * ell_inv).squaredNorm();
    const double phi = M_PI * x_diff(0) * T_inv;
    const double sin_phi = std::sin(phi);
    const double cos_phi = std::cos(phi);
    const double sin_phi_sq = sin_phi * sin_phi;
    const double sin_term = sin_phi_sq * lp_sq_inv;
    
    // 计算核函数值
    const double k = sf2 * std::exp(-0.5 * squared_term - 2.0 * sin_term);
    
    // ∂k/∂x1 = k * [ -(x1-x2)/ell^2 - 4/lp^2 * sin(phi)*cos(phi) * π/T ]
    const double se_grad = -x_diff(0) * ell_sq_inv;
    const double periodic_grad = -4.0 * lp_sq_inv * sin_phi * cos_phi * M_PI * T_inv;
    
    grad_input(0) = k * (se_grad + periodic_grad);
  }
  
  void CovSEPeriodic::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    ell = exp(loghyper(0));
    lp = exp(loghyper(1));
    T = exp(loghyper(2));
    sf2 = exp(2*loghyper(3));
  }
  
  std::string CovSEPeriodic::to_string()
  {
    return "CovSEPeriodic";
  }
}


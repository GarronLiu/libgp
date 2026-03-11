// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov.h"
#include "gp_utils.h"

namespace libgp
{
  void CovarianceFunction::grad_wrt_x1(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad_input){
    grad_input.setZero();
  }
  
  void CovarianceFunction::hessian_wrt_x1_x2(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::MatrixXd &hessian_input){
    hessian_input.resize(input_dim, input_dim);
    hessian_input.setZero();
  }
  
  size_t CovarianceFunction::get_param_dim()
  {
    return param_dim;
  }
  
  size_t CovarianceFunction::get_input_dim()
  {
    return input_dim;
  }
  
  Eigen::VectorXd CovarianceFunction::get_loghyper()
  {
    return loghyper;
  }
  
  void CovarianceFunction::set_loghyper(const Eigen::VectorXd &p)
  {
    assert(p.size() == loghyper.size());
    loghyper = p;
    loghyper_changed = true;
  }
  
  void CovarianceFunction::set_loghyper(const double p[])
  {
    Eigen::Map<const Eigen::VectorXd> p_vec_map(p, param_dim);
    set_loghyper(p_vec_map);
  }

  
  Eigen::VectorXd CovarianceFunction::draw_random_sample(Eigen::MatrixXd &X)
  {
    assert (X.cols() == int(input_dim));  
    int n = X.rows();
    Eigen::MatrixXd K(n, n);
    Eigen::LLT<Eigen::MatrixXd> solver;
    Eigen::VectorXd y(n);
    // compute kernel matrix (lower triangle)
    for(int i = 0; i < n; ++i) {
      for(int j = i; j < n; ++j) {
        K(j, i) = get(X.row(j), X.row(i));
      }
      y(i) = Utils::randn();
    }
    // perform cholesky factorization
    solver = K.llt();  
    return solver.matrixL() * y;
  }

  Eigen::VectorXd CovarianceFunction::get_loghyper_lb()
  {
    Eigen::VectorXd lb_(param_dim);
    lb_.setConstant(-std::numeric_limits<double>::infinity());
    return lb_;
  }

  Eigen::VectorXd CovarianceFunction::get_loghyper_ub()
  {
    Eigen::VectorXd ub_(param_dim);
    ub_.setConstant(std::numeric_limits<double>::infinity());
    return ub_;
  }
}

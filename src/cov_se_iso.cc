// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_se_iso.h"
#include <cmath>

namespace libgp
{
  
  CovSEiso::CovSEiso() {}
  
  CovSEiso::~CovSEiso() {}
  
  bool CovSEiso::init(int n)
  {
    input_dim = n;
    param_dim = 2;
    loghyper.resize(param_dim);
    loghyper.setZero();
    return true;
  }
  
  double CovSEiso::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    double z = ((x1-x2)/ell).squaredNorm();
    return sf2*exp(-0.5*z);
  }
  
  void CovSEiso::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    double z = ((x1-x2)/ell).squaredNorm();
    double k = sf2*exp(-0.5*z);
    grad << k*z, 2*k;
  }

  void CovSEiso::grad_wrt_x1(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad_input)
  {
    grad_input.resize(input_dim);
    
    if (&x1 == &x2) { 
      grad_input.setZero();
      return;
    }

    double k = get(x1, x2);
    grad_input = k * (x2 - x1) / (ell * ell);
  }

  void CovSEiso::hessian_wrt_x1_x2(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::MatrixXd &hessian_input)
  {
    hessian_input.resize(input_dim, input_dim);

    if (&x1 == &x2) {
      hessian_input.setZero();
      return;
    }

    double k = get(x1, x2);
    Eigen::VectorXd diff = x1 - x2;

    for (size_t i = 0; i < input_dim; ++i) {
      for (size_t j = 0; j < input_dim; ++j) {
        if (i == j) {
          hessian_input(i, j) = k * (1.0 / (ell * ell) - (diff(i) * diff(j)) / (ell * ell * ell * ell));
        } else {
          hessian_input(i, j) = -k * (diff(i) * diff(j)) / (ell * ell * ell * ell);
        }
      }
    }
  }

  void CovSEiso::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    ell = exp(loghyper(0));
    sf2 = exp(2*loghyper(1));
  }
  
  std::string CovSEiso::to_string()
  {
    return "CovSEiso";
  }
  
}

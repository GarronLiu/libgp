// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_matern5_iso.h"
#include <cmath>

namespace libgp
{
  
  CovMatern5iso::CovMatern5iso() {}
  
  CovMatern5iso::~CovMatern5iso() {}
  
  bool CovMatern5iso::init(int n)
  {
    input_dim = n;
    param_dim = 2;
    loghyper.resize(param_dim);
    loghyper.setZero();
    sqrt5 = sqrt(5);
    return true;
  }
  
  double CovMatern5iso::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    double z = ((x1-x2)*sqrt5/ell).norm();
    return sf2*exp(-z)*(1+z+z*z/3);
  }
  
  void CovMatern5iso::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    grad.resize(param_dim);
    double z = ((x1-x2)*sqrt5/ell).norm();
    double k = sf2*exp(-z);
    double z_square = z*z;
    grad << k*(z_square + z_square*z)/3, 2*k*(1+z+z_square/3);
  }

  void CovMatern5iso::grad_wrt_x1(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad_input){
    double r = (x1-x2).norm();
    if (r < 1e-9) {
      grad_input.setZero();
      return;
    }
    double z = sqrt5 * r / ell;
    double dk_dr = -sf2 * exp(-z) * (5.0 * r / (3.0 * ell * ell)) * (1.0 + z);
    grad_input = dk_dr * (x1 - x2) / r;
  }
  void CovMatern5iso::hessian_wrt_x1_x2(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::MatrixXd &hessian_input){
    hessian_input.resize(input_dim, input_dim);
    double r = (x1-x2).norm();
    if (r < 1e-9) {
      hessian_input.setZero();
      double factor = 5.0 * sf2 / (3.0 * ell * ell);
      for(int i=0; i<input_dim; ++i) {
        hessian_input(i, i) = factor;
      }
      return;
    }
    double z = sqrt5 * r / ell;
    double factor1 = 5.0 * sf2 / (3.0 * ell * ell) * exp(-z);
    double term1 = (1.0 + z);
    double term2 = z * z / (1.0 + z); // This simplifies the derivation logic, but let's stick to standard derivatives.
    
    double coeff_I = factor1 * (1.0 + z);
    double coeff_xx = - factor1 * 5.0 / (ell * ell);
    
    hessian_input = coeff_I * Eigen::MatrixXd::Identity(input_dim, input_dim) + 
            coeff_xx * (x1 - x2) * (x1 - x2).transpose();
  }
  
  void CovMatern5iso::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    ell = exp(loghyper(0));
    sf2 = exp(2*loghyper(1));
  }
  
  std::string CovMatern5iso::to_string()
  {
    return "CovMatern5iso";
  }
  
}

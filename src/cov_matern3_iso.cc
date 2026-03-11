// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_matern3_iso.h"
#include <cmath>

namespace libgp
{
  
  CovMatern3iso::CovMatern3iso() {}
  
  CovMatern3iso::~CovMatern3iso() {}
  
  bool CovMatern3iso::init(int n)
  {
    input_dim = n;
    param_dim = 2;
    loghyper.resize(param_dim);
    loghyper.setZero();
    sqrt3 = sqrt(3);
    return true;
  }
  
  double CovMatern3iso::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    double z = ((x1-x2)*sqrt3/ell).norm();
    return sf2*exp(-z)*(1+z);
  }
  
  void CovMatern3iso::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    double z = ((x1-x2)*sqrt3/ell).norm();
    double k = sf2*exp(-z);
    grad << k*z*z, 2*k*(1+z);
  }

  void CovMatern3iso::grad_wrt_x1(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad_input){
    double r = (x1-x2).norm();
    if (r < 1e-9) {
      grad_input.setZero();
      return;
    }
    double z = sqrt3 * r / ell;
    // k(r) = sf2 * (1 + z) * exp(-z)
    // dk/dr = sf2 * [ (sqrt3/ell) * exp(-z) + (1+z) * (-sqrt3/ell) * exp(-z) ]
    //       = sf2 * (sqrt3/ell) * exp(-z) * [ 1 - (1+z) ]
    //       = sf2 * (sqrt3/ell) * exp(-z) * (-z)
    //       = -sf2 * (sqrt3/ell) * z * exp(-z)
    //       = -sf2 * (sqrt3/ell) * (sqrt3 * r / ell) * exp(-z)
    //       = -sf2 * (3 * r / (ell*ell)) * exp(-z)
    
    double dk_dr = -3.0 * sf2 * r / (ell * ell) * exp(-z);
    grad_input = dk_dr * (x1 - x2) / r;
  }
  void CovMatern3iso::hessian_wrt_x1_x2(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::MatrixXd &hessian_input){
    double r = (x1-x2).norm();
    if (r < 1e-9) {
      hessian_input.setZero();
      double val = 3.0 * sf2 / (ell * ell);
      for(int i=0; i<input_dim; ++i) {
      hessian_input(i, i) = val;
      }
      return;
    }
    double z = sqrt3 * r / ell;
    
    // k(r) = sf2 * (1 + z) * exp(-z)
    // First derivative wrt r: k'(r) = -3 * sf2 * r / ell^2 * exp(-z)
    // Second derivative wrt r: k''(r) = 3 * sf2 / ell^2 * exp(-z) * (3 * r^2 / (sqrt3 * ell * r) - 1) 
    //                                 = 3 * sf2 / ell^2 * exp(-z) * (sqrt3 * r / ell - 1)
    //                                 = 3 * sf2 / ell^2 * exp(-z) * (z - 1)

    // Hessian H_ij = - d^2 k / (dx1_i dx2_j)
    // Since d(x1-x2)/dx2 = -1, the cross derivative is positive wrt the kernel structure usually, 
    // but let's follow the chain rule: d/dx2 (dk/dx1)
    // dk/dx1 = k'(r) * (x1-x2)/r
    // d/dx2 (dk/dx1) = d/dx2 [ k'(r)/r * (x1-x2) ]
    //                = (x1-x2) * d/dx2(k'(r)/r) + k'(r)/r * d/dx2(x1-x2)
    //                = (x1-x2) * (d/dr(k'(r)/r) * dr/dx2) + k'(r)/r * (-I)
    // dr/dx2 = -(x1-x2)/r
    // d/dr(k'(r)/r) = (r*k''(r) - k'(r)) / r^2
    
    // Result: H = - [ (r*k''(r) - k'(r))/r^3 * (x1-x2)(x1-x2)^T * (-1) - k'(r)/r * I ]
    //           = (r*k''(r) - k'(r))/r^3 * (x1-x2)(x1-x2)^T + k'(r)/r * I
    
    // Let A = k'(r)/r = -3 * sf2 / ell^2 * exp(-z)
    // Let B = (r*k''(r) - k'(r))/r^3
    // r*k''(r) = r * 3 * sf2 / ell^2 * exp(-z) * (z - 1)
    // r*k''(r) - k'(r) = 3 * sf2 / ell^2 * exp(-z) * [ r(z-1) + r ]
    //                  = 3 * sf2 / ell^2 * exp(-z) * r * z
    //                  = 3 * sf2 / ell^2 * exp(-z) * r * (sqrt3 * r / ell)
    //                  = 3 * sqrt3 * sf2 / ell^3 * r^2 * exp(-z)
    // B = (3 * sqrt3 * sf2 / ell^3 * r^2 * exp(-z)) / r^3
    //   = 3 * sqrt3 * sf2 / (ell^3 * r) * exp(-z)
    
    double common_factor = 3.0 * sf2 / (ell * ell) * exp(-z);
    double A = -common_factor; // Coefficient for Identity
    double B = common_factor * sqrt3 / (ell * r); // Coefficient for outer product

    hessian_input = A * Eigen::MatrixXd::Identity(input_dim, input_dim) + 
            B * (x1 - x2) * (x1 - x2).transpose();
  }
  
  void CovMatern3iso::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    ell = exp(loghyper(0));
    sf2 = exp(2*loghyper(1));
  }
  
  std::string CovMatern3iso::to_string()
  {
    return "CovMatern3iso";
  }
  
}

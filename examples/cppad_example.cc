// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cg.h"
#include "configDir.h"
#include "gp.h"
#include "gp_utils.h"
#include "matplotlibcpp.h"
#include "rprop.h"
#include "sparse_gp.h"

#include <Eigen/Dense>

#include <cppad/cppad.hpp>               // the CppAD package
#include <cppad/example/cppad_eigen.hpp> // the CppAD/Eigen interface
#include <iostream>                      // standard input/output
#include <vector>                        // standard vector

namespace plt = matplotlibcpp;

using namespace libgp;

using CppAD::AD; // use AD as abbreviation for CppAD::AD
using CppAD::NearEqual;
using Eigen::Dynamic;
using Eigen::Matrix;
using std::vector; // use vector as abbreviation for std::vector

typedef Eigen::Matrix<AD<double>, Dynamic, 1> ADVector;
typedef Eigen::Matrix<AD<double>, Dynamic, Dynamic> ADMatrix;

namespace { // begin the empty namespace
// define the function Poly(a, x) = a[0] + a[1]*x[1] + ... + a[k-1]*x[k-1]
template <class Type>
Type spectralMetricKernel(const Eigen::Matrix<Type, Dynamic, 1> &hyperParams,
                          const Eigen::Matrix<Type, Dynamic, 1> &x1,
                          const Eigen::Matrix<Type, Dynamic, 1> &x2) {
  // hyperParams[0] = variance (sigma^2)
  // hyperParams[1] = lengthscale (l)
  Type sigma2 = hyperParams[0];
  Type l = hyperParams[1];

  // squared Euclidean norm of x
  Type x1_norm = Type(0);
  Type x2_norm = Type(0);

  x1_norm = x1.norm();
  x2_norm = x2.norm();

  // product of x1 and x2
  Type x1_x2 = Type(0);
  x1_x2 = x1.dot(x2);

  return sigma2 * CppAD::exp(-l * CppAD::acos(x1_x2 / (x1_norm * x2_norm)));
}

} // namespace

int main(int argc, char const *argv[]) {

  using CppAD::AD;   // use AD as abbreviation for CppAD::AD
  using std::vector; // use vector as abbreviation for std::vector

  // vector of hyper parameters
  size_t k = 2;             // number of hyper parameters
  ADVector ahyperParams(k); // vector of hyper parameters
  for (size_t i = 0; i < k; i++)
    ahyperParams[i] = 1.; // value of hyper parameters

  // domain space vector
  size_t n = 3;    // number of domain space variables
  ADVector ax1(n); // vector of domain space variables
  ax1[0] = 3.;     // value at which function is recorded
  ax1[1] = 3.;
  ax1[2] = 3.;

  ADVector ax2(n); // vector of domain space variables
  ax2[0] = 3.;
  ax2[1] = 2.;
  ax2[2] = 2.;

  // declare independent variables and start recording operation sequence
  CppAD::Independent(ahyperParams);

  // range space vector
  size_t m = 1;   // number of ranges space variables
  ADVector ay(m); // vector of ranges space variables
  ay[0] = spectralMetricKernel(ahyperParams, ax1,
                               ax2); // record operations that compute ay[0]

  // store operation sequence in f: X -> Y and stop recording
  CppAD::ADFun<double> f(ahyperParams, ay);

  // compute derivative using operation sequence stored in f
  vector<double> jac(m * k);     // Jacobian of f (m by n matrix)
  vector<double> hyperParams(k); // domain space vector
  hyperParams[0] = 1.;           // argument value for computing derivative
  hyperParams[1] = 1.;
  jac = f.Jacobian(hyperParams); // Jacobian for operation sequence

  // numerical finite-difference check
  Eigen::VectorXd hd(k);
  hd[0] = hyperParams[0];
  hd[1] = hyperParams[1];

  Eigen::VectorXd x1d(n), x2d(n);
  for (size_t i = 0; i < n; ++i) {
    x1d[i] = CppAD::Value(ax1[i]);
    x2d[i] = CppAD::Value(ax2[i]);
  }

  auto f_double = [&](const Eigen::VectorXd &hvec) {
    return spectralMetricKernel<double>(hvec, x1d, x2d);
  };

  double eps = 1e-6;
  std::vector<double> num_jac(k);
  for (size_t i = 0; i < k; ++i) {
    Eigen::VectorXd hplus = hd, hminus = hd;
    hplus[i] += eps;
    hminus[i] -= eps;
    double fplus = f_double(hplus);
    double fminus = f_double(hminus);
    num_jac[i] = (fplus - fminus) / (2.0 * eps);
    std::cout << "num_jac[" << i << "] = " << num_jac[i] << ", cppad jac[" << i
              << "] = " << jac[i] << std::endl;
  }

  // check if the derivative is correct
  int error_code;
  if (jac[0] == 142.)
    error_code = 0; // return code for correct case
  else
    error_code = 1; // return code for incorrect case

  return error_code;
}

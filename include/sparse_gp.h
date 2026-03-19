// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2025, Garron Liu <ljr799910832@sjtu.edu.cn>
// All rights reserved.

/*!
 *
 *   \page licence Licensing
 *
 *     sparse gaussian process implementation
 *
 *      \verbinclude "../COPYING"
 */

#ifndef __SPARSE_GP_H__
#define __SPARSE_GP_H__
#include "KMeans.h"
#include "cov.h"
#include "gp.h"
#include "gp_utils.h"
#include "sampleset.h"

#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include <cmath>

namespace libgp {
class SparseGaussianProcess : public GaussianProcess {
  /** Sparse Gaussian process regression.
   *  @author Garron Liu */
public:
  /** Create an instance of SparseGaussianProcess with given input
   * dimensionality and covariance function. */
  SparseGaussianProcess(size_t input_dim, std::string covf_def);

  /** Create an instance of SparseGaussianProcess from file with Random Sampling
   * of inducing points. */
  /** Create an instance of SparseGaussianProcess from file. */
  SparseGaussianProcess(const char *filename);

  virtual ~SparseGaussianProcess();

  double f(const double x[]) override;

  double var(const double x[]) override;

  /** Add input-output-pair to sample set.
   *  Add a copy of the given input-output-pair to sample set.
   *  @param x input array
   *  @param y output value
   */
  void add_pattern(const double x[], double y) override;

  void specify_inducingSet(std::vector<Eigen::VectorXd> inducing_points,
                           size_t m_random = 0, size_t m_clusters = 0);

  double log_likelihood()
      override; // analytical collapsed evidence lower bound for sparse GP

  Eigen::VectorXd log_likelihood_gradient() override;

  void update_hyperparameters(const Eigen::VectorXd &params) override;

  //update inducing points
  void update_variational_parameters(const Eigen::VectorXd &params);

  Eigen::VectorXd get_hyperparameters() override;

  Eigen::VectorXd get_hyperparameter_lower_bound() override;

  Eigen::VectorXd get_hyperparameter_upper_bound() override;

  //get inducing points
  Eigen::VectorXd get_variational_parameters();

  SampleSet *get_inducingSet() { return inducingset; }

  Eigen::MatrixXd getFlatInputs() override;

  Eigen::VectorXd getFlatTargets() override;

  Eigen::MatrixXd getFlatPosteriorCovMatrix() override;

  Eigen::VectorXd getFlatHyperparameters() override;

  Eigen::VectorXd getFlatAlpha() override;

  void exportModelToYAML(const char *filename);

protected:
  using GaussianProcess::alpha; //  alpha = (K_XX_bar + \sigma_n^2 I)^-1 * y
  using GaussianProcess::L;     // Cholesky of (K_XX_bar + \sigma_n^2 I)

  using GaussianProcess::k_star; // f_star = k_star * alpha_R
  Eigen::VectorXd alpha_R;       // K_RR^-1 * u
  Eigen::MatrixXd
      L_R; // var_star = k_star_star - k_star^T * L_R^-T * L_R^-1 * k_star

  // PEP training condition
  double param_alpha = 0.5; // 0 < alpha <=1, alpha =1 corresponds to FITC,
                            // alpha->0 corresponds to VFE

  /** Linear solver used to invert the covariance matrix. */
  //    Eigen::LLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>> solver;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> K_RR; // \Sigma_{u,u}
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
      L_K_RR; // cholesky(K_RR)

  // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  //     H_T; 
  // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  //     D_X; // K_XX - K_XR * K_RR^-1 * K_RX
  
  Eigen::VectorXd lambda;

  Eigen::MatrixXd L_B; //cholesky(woodburry identity的中间矩阵: B = I + H * Lambda^{-1} * H^T)

  Eigen::MatrixXd Q_pred;

  Eigen::MatrixXd cov_inducing;

  Eigen::MatrixXd U; // U = L_K_RR^{-1} * K_RX; U^T * U = K_XR * K_RR^(-1) * K_RX

  void update_k_star(const Eigen::VectorXd &x_star) override;

  void compute() override;

  SampleSet *inducingset; // full training set for sparse GP
};
} // namespace libgp

#endif /* __SPARSE_GP_H__ */

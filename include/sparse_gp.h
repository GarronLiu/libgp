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
// inline Cholesky helpers to simplify repeated patterns below
inline auto chol_lower(const Eigen::MatrixXd &A) -> Eigen::MatrixXd {
  return A.selfadjointView<Eigen::Lower>().llt().matrixL();
};

inline auto chol_inverse(const Eigen::MatrixXd &A) -> Eigen::MatrixXd {
  auto A_jitter = A + Eigen::MatrixXd::Identity(A.rows(), A.cols()) * 1e-8;
  Eigen::MatrixXd L = chol_lower(A_jitter);
  Eigen::MatrixXd inv = Eigen::MatrixXd::Identity(L.rows(), L.cols());
  L.triangularView<Eigen::Lower>().solveInPlace(inv);
  L.adjoint().triangularView<Eigen::Upper>().solveInPlace(inv);
  return inv;
};

// solve A * X = B using Cholesky of A (A must be SPD)
inline decltype(auto) chol_solve(const Eigen::MatrixXd &A, Eigen::MatrixXd B) {
  if (B.rows() != A.cols())
    throw std::invalid_argument("Incompatible matrix dimensions");
  auto X = B;
  auto A_jitter = A + Eigen::MatrixXd::Identity(A.rows(), A.cols()) * 1e-8;
  Eigen::MatrixXd L = chol_lower(A_jitter);
  L.triangularView<Eigen::Lower>().solveInPlace(X);
  L.adjoint().triangularView<Eigen::Upper>().solveInPlace(X);
  return X;
};

inline auto logDet(const Eigen::MatrixXd &A) -> double {
  auto A_jitter = A + Eigen::MatrixXd::Identity(A.rows(), A.cols()) * 1e-8;
  auto L = chol_lower(A_jitter);
  return 2.0 * L.diagonal().array().log().sum();
};

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

  void pred_diag_derivative(Eigen::VectorXd &mean_deriv) override;

  void pred_diag_derivative(const std::vector<Eigen::VectorXd> &testset,
                            Eigen::VectorXd &mean_deriv) override;

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

  // update inducing points
  void update_variational_parameters(const Eigen::VectorXd &params);

  Eigen::VectorXd get_hyperparameters() override;

  Eigen::VectorXd get_hyperparameter_lower_bound() override;

  Eigen::VectorXd get_hyperparameter_upper_bound() override;

  // get inducing points
  Eigen::VectorXd get_variational_parameters();

  SampleSet *get_inducingSet() { return inducingset; }

  Eigen::MatrixXd getFlatInputs() override;

  Eigen::VectorXd getFlatTargets() override;

  Eigen::MatrixXd getFlatPosteriorCovMatrix() override;

  Eigen::VectorXd getFlatHyperparameters() override;

  Eigen::VectorXd getFlatAlpha() override;

  void exportModelToYAML(const char *filename);

  void add_pattern_batch(const Eigen::VectorXd &x, double y);

  void check_gradient();

protected:
  Eigen::VectorXd alpha_R;

  using GaussianProcess::alpha;  //  alpha = (K_XX_bar + \sigma_n^2 I)^-1 * y
                                 // K_RR^-1 * u
  using GaussianProcess::k_star; // f_star = k_star * alpha_R

  // PEP training condition
  double param_alpha = 0.5; // 0 < alpha <=1, alpha =1 corresponds to FITC,
                            // alpha->0 corresponds to VFE

  /** Linear solver used to invert the covariance matrix. */
  //    Eigen::LLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>> solver;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
      L_K_RR; // cholesky(K_RR)

  Eigen::VectorXd lambda;
  Eigen::MatrixXd lambda_a; // a corresponding to history pseudo observations

  Eigen::MatrixXd L_B; // cholesky(woodburry identity的中间矩阵: B = I + H *
                       // Lambda^{-1} * H^T)
  Eigen::MatrixXd L_M_shur; // cholesky( M/A in shur complement)

  Eigen::MatrixXd Q_pred;

  Eigen::MatrixXd cov_inducing;

  // variables related to pretrained posterior
  bool stream_update_flag =
      false; // whether to use pretrained posterior for warm start
  Eigen::MatrixXd inv_K_aa_pre, K_aa_pre;
  Eigen::MatrixXd inv_Sigma_u_pre, Sigma_u_pre;
  Eigen::MatrixXd P_pre, invP_pre;

  Eigen::VectorXd y_a;       // history pseudo observations
  double elbo_constant_init; // constant term in ELBO that does not depend on
                             // new batch

  bool pretrained_stored = false;

  void storePosteriorPretrained();

  void addNewInducingPoints(Eigen::VectorXd x_t);

  double novelty_threshold;

  Eigen::MatrixXd
      U; // U = L_K_RR^{-1} * K_RX; U^T * U = K_XR * K_RR^(-1) * K_RX
  Eigen::MatrixXd
      U_a; // U_a = L_K_RR^{-1} * K_Ra; U_a^T * U_a = K_aR * K_RR^(-1) * K_Ra

  void update_k_star(const Eigen::VectorXd &x_star) override;

  void update_alpha() override;

  void compute() override;

  SampleSet *inducingset;     // full training set for sparse GP
  SampleSet *inducingset_pre; // inducing points of pretrained model
  SampleSet *batchset;        // new training points in the current batch

  size_t iterations = 0; // number of batch updates performed, used for learning rate scheduling
};
} // namespace libgp

#endif /* __SPARSE_GP_H__ */

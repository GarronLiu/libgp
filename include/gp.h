// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

/*!
 *
 *   \page licence Licensing
 *
 *     libgp - Gaussian process library for Machine Learning
 *
 *      \verbinclude "../COPYING"
 */

#ifndef __GP_H__
#define __GP_H__
#include "KMeans.h"
#include "cov.h"
#include "sampleset.h"

#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <yaml-cpp/yaml.h>

namespace libgp {
/** Gaussian process regression.
 *  @author Manuel Blum */
class GaussianProcess {
public:
  /** Empty initialization */
  GaussianProcess();

  /** Create and instance of GaussianProcess with given input dimensionality
   *  and covariance function. */
  GaussianProcess(size_t input_dim, std::string covf_def);

  /** Create and instance of GaussianProcess from file. */
  GaussianProcess(const char *filename);

  /** Copy constructor */
  GaussianProcess(const GaussianProcess &gp);

  virtual ~GaussianProcess();

  /** Write current gp model to file. */
  virtual void write(const char *filename);

  /** Predict target value for given input.
   *  @param x input vector
   *  @return predicted value */
  virtual double f(const double x[]);

  /** Predict variance of prediction for given input.
   *  @param x input vector
   *  @return predicted variance */
  virtual double var(const double x[]);

  /** Add input-output-pair to sample set.
   *  Add a copy of the given input-output-pair to sample set.
   *  @param x input array
   *  @param y output value
   */
  virtual void add_pattern(const double x[], double y);

  bool set_y(size_t i, double y);

  /** Get number of samples in the training set. */
  size_t get_sampleset_size();

  void set_sampleset(const std::shared_ptr<SampleSet> &ss);

  /** Clear sample set and free memory. */
  void clear_sampleset();

  /** Get reference on currently used covariance function. */
  CovarianceFunction &covf();

  /** Get input vector dimensionality. */
  size_t get_input_dim();

  virtual double log_likelihood();

  virtual Eigen::VectorXd log_likelihood_gradient();

  virtual void update_hyperparameters(const Eigen::VectorXd &params);

  virtual Eigen::VectorXd get_hyperparameters();

  virtual Eigen::VectorXd get_hyperparameter_lower_bound();

  virtual Eigen::VectorXd get_hyperparameter_upper_bound();

  void pred_diag(const std::shared_ptr<SampleSet> testset,
                 Eigen::VectorXd &mean, Eigen::VectorXd &var);
  
  void pred_diag(const std::vector<Eigen::VectorXd> &testset,
                 Eigen::VectorXd &mean, Eigen::VectorXd &var);

  void pred_diag(Eigen::VectorXd &mean_pred, Eigen::VectorXd &var_pred);

  virtual void pred_diag_derivative(Eigen::VectorXd &mean_deriv);

  virtual void pred_diag_derivative(const std::vector<Eigen::VectorXd> &testset,
                                     Eigen::VectorXd &mean_deriv);

  void validation(const std::shared_ptr<SampleSet> testset, double &mae,
                  double &rmse, double &lml);

  void exportModelToYAML(const char *filename);

  Eigen::VectorXd getFlatSamplingTargets();

  virtual Eigen::MatrixXd getFlatInputs();

  virtual Eigen::VectorXd getFlatTargets();

  virtual Eigen::MatrixXd getFlatPosteriorCovMatrix();

  virtual Eigen::VectorXd getFlatHyperparameters();

  virtual Eigen::VectorXd getFlatAlpha();

  virtual void check_gradient();

  SampleSet *sampleset;

protected:
  /** The covariance function of this Gaussian process. */
  CovarianceFunction *cf;

  /** The training sample set. */

  /** Alpha is cached for performance. */
  Eigen::VectorXd alpha;

  /** Last test kernel vector. */
  Eigen::VectorXd k_star;

  /** Linear solver used to invert the covariance matrix. */
  //    Eigen::LLT<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>> solver;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> L;

  /** Input vector dimensionality. */
  size_t input_dim;

  /** Update test input and cache kernel vector. */
  virtual void update_k_star(const Eigen::VectorXd &x_star);

  virtual void update_alpha();

  /** Compute covariance matrix and perform cholesky decomposition. */
  virtual void compute();

  bool alpha_needs_update;

  void computeKernelMatrix(
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &K_ab,
      SampleSet *set_a, SampleSet *set_b, CovarianceFunction *cf);

  void computeKernelMatrixLowerHalf(
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &K_aa,
      SampleSet *set_a, CovarianceFunction *cf);

private:
  /** No assignement */
  GaussianProcess &operator=(const GaussianProcess &);
};
} // namespace libgp

#endif /* __GP_H__ */

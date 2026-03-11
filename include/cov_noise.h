// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __COV_NOISE_H__
#define __COV_NOISE_H__

#include "cov.h"

namespace libgp
{
  
  /** Independent covariance function (white noise). 
   *  Parameters: signal noise, \f$\sigma^2\f$
   *  @author Manuel Blum
   *  @ingroup cov_group
   */
  class CovNoise : public CovarianceFunction
  {
  public:
    CovNoise ();
    virtual ~CovNoise ();
    bool init(int n);
    double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);
    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad);
    void set_loghyper(const Eigen::VectorXd &p);
    virtual std::string to_string();
    Eigen::VectorXd get_loghyper_lb();
    Eigen::VectorXd get_loghyper_ub();
  private:
    double s2;
    double lb; ///< lower bound for the noise variance>
    double ub; ///< upper bound for the noise variance>
  };
  
}

#endif /* __COV_NOISE_H__ */
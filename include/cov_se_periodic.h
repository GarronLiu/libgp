// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __COV_SE_PERIODIC_H__
#define __COV_SE_PERIODIC_H__

#include "cov.h"

namespace libgp
{
  
  /** Squared exponential covariance function multiplied with periodic covariance function.
   *  Computes the squared exponential covariance
   *  \f$k_{SE_PERIODIC}(x, y) :\alpha^2 \exp(-\frac{(x-y)^2}{2l^2})exp(-\frac{2\sin^2 (\frac{\pi}{\lambda}(x-y))}{l_p^2})\f$,
   *  with \f$\Lambda = diag(l^2, \dots, l^2)\f$ being the characteristic
   *  length scale and \f$\alpha\f$ describing the variability of the latent
   *  function (signal variance). The parameters \f$l^2, \alpha\f$ are expected
   *  in this order in the parameter array.
   *  @ingroup cov_group
   *  @author Manuel Blum
   */
  class CovSEPeriodic : public CovarianceFunction
  {
  public:
    CovSEPeriodic ();
    virtual ~CovSEPeriodic ();
    bool init(int n);
    double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);
    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad) override;
    void grad_wrt_x1(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad_input) override;
    void set_loghyper(const Eigen::VectorXd &p);
    virtual std::string to_string();
  private:
    double ell;
    double lp;
    double T;
    double sf2;
  };
  
}

#endif /* __COV_SE_PERIODIC_H__ */
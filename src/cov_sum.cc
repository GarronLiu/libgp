// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_sum.h"
#include "cmath"

namespace libgp
{
  
  CovSum::CovSum()
  {
  }
  
  CovSum::~CovSum()
  {
    delete first;
    delete second;
  }
  
  bool CovSum::init(int n, CovarianceFunction * first, CovarianceFunction * second)
  {
    this->input_dim = n;
    this->first = first;
    this->second = second;
    param_dim_first = first->get_param_dim();
    param_dim_second = second->get_param_dim();
    param_dim = param_dim_first + param_dim_second;
    loghyper.resize(param_dim);
    loghyper.setZero();
    return true;
  }
  
  double CovSum::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    return first->get(x1, x2) + second->get(x1, x2);
  }
  
  void CovSum::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    Eigen::VectorXd grad_first(param_dim_first);
    Eigen::VectorXd grad_second(param_dim_second);
    first->grad(x1, x2, grad_first);
    second->grad(x1, x2, grad_second);
    grad.head(param_dim_first) = grad_first;
    grad.tail(param_dim_second) = grad_second;
  }

  void CovSum::grad_wrt_x1(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad_input)
  {
    Eigen::VectorXd grad_input_first(input_dim);
    Eigen::VectorXd grad_input_second(input_dim);
    first->grad_wrt_x1(x1, x2, grad_input_first);
    second->grad_wrt_x1(x1, x2, grad_input_second);
    grad_input = grad_input_first + grad_input_second;
  }
  
  void CovSum::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    first->set_loghyper(p.head(param_dim_first));
    second->set_loghyper(p.tail(param_dim_second));
  }
  
  std::string CovSum::to_string()
  {
    return "CovSum("+first->to_string()+", "+second->to_string()+")";
  }

  Eigen::VectorXd CovSum::get_loghyper_lb()
  {
    Eigen::VectorXd lb_first = first->get_loghyper_lb();
    Eigen::VectorXd lb_second = second->get_loghyper_lb();
    Eigen::VectorXd lb_total(param_dim);
    lb_total.head(param_dim_first) = lb_first;
    lb_total.tail(param_dim_second) = lb_second;
    return lb_total;
  }

  Eigen::VectorXd CovSum::get_loghyper_ub()
  {
    Eigen::VectorXd ub_first = first->get_loghyper_ub();
    Eigen::VectorXd ub_second = second->get_loghyper_ub();
    Eigen::VectorXd ub_total(param_dim);
    ub_total.head(param_dim_first) = ub_first;
    ub_total.tail(param_dim_second) = ub_second;
    return ub_total;
  }
}

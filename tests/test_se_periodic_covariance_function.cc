// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "sparse_gp.h"
#include "recursive_gp.h"
#include "cg.h"
#include "gp_utils.h"
#include "matplotlibcpp.h"
#include "rprop.h"
#include <Eigen/Dense>

#include <iostream>                      // standard input/output
#include <vector>                        // standard vector
#include <chrono>

#include "evaluate.h"
#include <iomanip>

#include "cov.h"
#include "cov_factory.h"
#include "cov_se_periodic.h"  // Added include for CovSEPeriodic

namespace plt = matplotlibcpp;

using namespace libgp;

int main(int argc, char const *argv[]) {
  // 首先验证协方差函数梯度
  CovSEPeriodic test_cov;
  test_cov.init(1);
  Eigen::VectorXd test_params(4);
  test_params << 0.0, 0.0, 0.1, -1.0;  // log(ell), log(lp), log(T), log(sf)
  test_cov.set_loghyper(test_params);
  
      std::cout << "\n\033[32m==================== Gradient Verification ====================\033[0m" << std::endl;
    
    // 设置测试点
    Eigen::VectorXd x1(1), x2(1);
    x1 << 6.0;
    x2 << 5.5;
    
    // 有限差分步长
    double eps = 1e-6;
    
    // 获取当前超参数
    Eigen::VectorXd loghyper = test_cov.get_loghyper();
    int param_dim = loghyper.size();
    
    std::cout << "Test points: x1 = " << x1.transpose() << ", x2 = " << x2.transpose() << std::endl;
    std::cout << "Current loghyper: " << loghyper.transpose() << std::endl;
    std::cout << "\nGradient w.r.t. hyperparameters:" << std::endl;
    std::cout << std::setw(15) << "Parameter" 
              << std::setw(20) << "Analytical" 
              << std::setw(20) << "Numerical" 
              << std::setw(20) << "Rel. Error(%)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    // 计算解析梯度
    Eigen::VectorXd grad_analytical;
    test_cov.grad(x1, x2, grad_analytical);
    
    // 对每个超参数进行有限差分验证
    for(int i = 0; i < param_dim; ++i) {
        Eigen::VectorXd loghyper_plus = loghyper;
        Eigen::VectorXd loghyper_minus = loghyper;
        
        loghyper_plus(i) += eps;
        loghyper_minus(i) -= eps;
        
        // 计算 k(x1, x2) 在 loghyper+eps
        test_cov.set_loghyper(loghyper_plus);
        double k_plus = test_cov.get(x1, x2);
        
        // 计算 k(x1, x2) 在 loghyper-eps
        test_cov.set_loghyper(loghyper_minus);
        double k_minus = test_cov.get(x1, x2);
        
        // 数值梯度 (中心差分)
        double grad_numerical = (k_plus - k_minus) / (2.0 * eps);
        
        // 恢复原始超参数
        test_cov.set_loghyper(loghyper);
        
        // 计算相对误差
        double rel_error = std::abs(grad_analytical(i) - grad_numerical) / 
                          (std::abs(grad_analytical(i)) + 1e-10) * 100.0;
        
        std::string param_name;
        if(i == 0) param_name = "log(ell)";
        else if(i == 1) param_name = "log(lp)";
        else if(i == 2) param_name = "log(T)";
        else if(i == 3) param_name = "log(sf)";
        
        std::cout << std::setw(15) << param_name
                  << std::setw(20) << grad_analytical(i)
                  << std::setw(20) << grad_numerical
                  << std::setw(20) << rel_error << std::endl;
    }
    
    std::cout << "\nGradient w.r.t. input x1:" << std::endl;
    std::cout << std::setw(15) << "Dimension" 
              << std::setw(20) << "Analytical" 
              << std::setw(20) << "Numerical" 
              << std::setw(20) << "Rel. Error(%)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    // 计算输入梯度的解析值
    Eigen::VectorXd grad_input_analytical;
    test_cov.grad_wrt_input(x1, x2, grad_input_analytical);
    
    // 对每个输入维度进行有限差分验证
    for(int i = 0; i < x1.size(); ++i) {
        Eigen::VectorXd x1_plus = x1;
        Eigen::VectorXd x1_minus = x1;
        
        x1_plus(i) += eps;
        x1_minus(i) -= eps;
        
        double k_plus = test_cov.get(x1_plus, x2);
        double k_minus = test_cov.get(x1_minus, x2);
        
        double grad_input_numerical = (k_plus - k_minus) / (2.0 * eps);
        
        double rel_error = std::abs(grad_input_analytical(i) - grad_input_numerical) / 
                          (std::abs(grad_input_analytical(i)) + 1e-10) * 100.0;
        
        std::cout << std::setw(15) << ("x1[" + std::to_string(i) + "]")
                  << std::setw(20) << grad_input_analytical(i)
                  << std::setw(20) << grad_input_numerical
                  << std::setw(20) << rel_error << std::endl;
    }
    
    std::cout << "\033[32m================================================================\033[0m\n" << std::endl;


  return EXIT_SUCCESS;
}

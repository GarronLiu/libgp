/*
 * lbfgs.cc
 *
 *  Created on: [2025-12-09]
 *      Author: [Gemini3pro, Garron Liu]
 */

#include "lbfgs.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <limits>

using namespace std;

namespace libgp
{

LBFGS::LBFGS() : lml_(0), duration_(0), m_(10), tol_(1e-5)
{
}

LBFGS::~LBFGS()
{
}

void LBFGS::maximize(GaussianProcess* gp, size_t max_steps, bool verbose)
{
    auto start = std::chrono::high_resolution_clock::now();
    // 记录 LML 历史和时间消耗历史
    lml_history_.clear();
    lml_history_.reserve(max_steps);
    time_cost_history_.clear();
    time_cost_history_.reserve(max_steps);

    // 1. 初始化
    Eigen::VectorXd x = gp->get_hyperparameters();
    int n_dim = x.size();
    
    // 清空历史
    history_.clear();

    // 2. 计算初始值和梯度
    // 注意：LBFGS 通常用于最小化。我们要最大化 LML，所以目标函数 f = -LML
    double lml = gp->log_likelihood();
    Eigen::VectorXd grad_lml = gp->log_likelihood_gradient();
    
    double f = -lml;              // 最小化目标
    Eigen::VectorXd g = -grad_lml; // 目标梯度

    if (verbose) {
        cout << "LBFGS Start. Initial LML: " << lml << endl;
    }

    for (size_t k = 0; k < max_steps; ++k) {
        auto iter_start = std::chrono::high_resolution_clock::now();
        lml_history_.push_back(-f);
        // 3. 检查收敛性 (梯度范数)
        double g_norm = g.norm();
        if (g_norm < tol_) {
            if (verbose) cout << "LBFGS converged at step " << k << " with gradient norm " << g_norm << endl;
            break;
        }

        // 4. 计算搜索方向 d = -H * g
        Eigen::VectorXd d = compute_direction(g);

        // 5. 线搜索 (Line Search)
        Eigen::VectorXd x_new;
        double f_new;
        Eigen::VectorXd g_new_lml; // 临时存储 LML 梯度
        Eigen::VectorXd g_new;     // 目标函数梯度

        // 线搜索会更新 GP 内部参数以评估 f_new
        double alpha = line_search(gp, x, f, g, d, x_new, f_new, g_new);

        if (alpha == 0.0) {
            if (verbose) cout << "Line search failed at step " << k << endl;
            break;
        }

        // 6. 更新历史信息 (s, y)
        Eigen::VectorXd s = x_new - x;
        Eigen::VectorXd y = g_new - g;
        double ys = y.dot(s);

        if (ys > 1e-10) { // 保证正定性
            IterationData data;
            data.s = s;
            data.y = y;
            data.rho = 1.0 / ys;
            
            history_.push_back(data);
            if (history_.size() > m_) {
                history_.pop_front();
            }
        }

        // 7. 更新状态
        x = x_new;
        f = f_new;
        g = g_new;
        lml = -f;

        if (verbose && k % 10 == 0) {
            cout << "Step " << k << ": LML = " << lml << ", |g| = " << g.norm() << endl;
        }

        auto iter_end = std::chrono::high_resolution_clock::now();
        time_cost_history_.push_back(std::chrono::duration<double, std::milli>(iter_end - iter_start).count());
    }

    // 确保 GP 处于最佳参数状态
    gp->update_hyperparameters(x);
    lml_ = lml;

    auto end = std::chrono::high_resolution_clock::now();
    duration_ = std::chrono::duration<double, std::milli>(end - start).count();

    if (verbose) {
        cout << "LBFGS Optimization Finished. Final LML: " << lml_ << " Duration: " << duration_ << "ms" << endl;
        cout << "Optimized Hyperparameters: " << x.transpose() << endl;
    }
}

// 双循环递归 (Two-loop recursion) 计算 H_k * g_k
Eigen::VectorXd LBFGS::compute_direction(const Eigen::VectorXd& gradient) {
    Eigen::VectorXd q = gradient;
    int k = history_.size();
    std::vector<double> alpha_list(k);

    // Backward pass
    for (int i = k - 1; i >= 0; --i) {
        const auto& data = history_[i];
        alpha_list[i] = data.rho * data.s.dot(q);
        q -= alpha_list[i] * data.y;
    }

    // Initial Hessian approximation H_0
    // 使用 scaling trick: gamma = (s_last^T y_last) / (y_last^T y_last)
    Eigen::VectorXd r = q;
    if (k > 0) {
        const auto& last = history_.back();
        double gamma = last.s.dot(last.y) / last.y.dot(last.y);
        r *= gamma;
    } else {
        // 如果没有历史，退化为梯度下降，但给一个单位矩阵缩放
        // r = q; 
    }

    // Forward pass
    for (int i = 0; i < k; ++i) {
        const auto& data = history_[i];
        double beta = data.rho * data.y.dot(r);
        r += data.s * (alpha_list[i] - beta);
    }

    return -r; // 返回下降方向 (-H * g)
}

// 基于 Armijo 条件的回溯线搜索
double LBFGS::line_search(GaussianProcess* gp, 
                          const Eigen::VectorXd& x, 
                          double f, 
                          const Eigen::VectorXd& g, 
                          const Eigen::VectorXd& d,
                          Eigen::VectorXd& x_new,
                          double& f_new,
                          Eigen::VectorXd& g_new) {
    double alpha = 1.0;
    double c1 = 1e-4; // Armijo 参数
    double decay = 0.5;
    int max_ls_iter = 20;

    double g_dot_d = g.dot(d);
    
    // 如果方向不是下降方向 (由于数值误差)，重置为梯度方向
    if (g_dot_d > 0) {
        // std::cerr << "Warning: Ascent direction in LBFGS line search." << std::endl;
        return 0.0; 
    }

    for (int i = 0; i < max_ls_iter; ++i) {
        x_new = x + alpha * d;
        
        // 评估新位置
        gp->update_hyperparameters(x_new);
        double lml_new = gp->log_likelihood();
        f_new = -lml_new;

        // Armijo 条件: f(x + alpha*d) <= f(x) + c1 * alpha * g^T * d
        if (f_new <= f + c1 * alpha * g_dot_d) {
            // 找到合适的步长，计算新梯度供下一步使用
            Eigen::VectorXd grad_lml = gp->log_likelihood_gradient();
            g_new = -grad_lml;
            return alpha;
        }

        alpha *= decay;
    }

    return 0.0; // 线搜索失败
}

}
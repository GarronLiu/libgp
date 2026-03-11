/*
 * lbfgs.h
 *
 *  Created on: [2025-12-09]
 *      Author: [Gemini3pro, Garron Liu]
 */

#ifndef LBFGS_H_
#define LBFGS_H_

#include "gp.h"
#include <vector>
#include <deque>
#include <Eigen/Core>

namespace libgp
{

class LBFGS
{
public:
    LBFGS();
    virtual ~LBFGS();

    // 遵循与其他优化器相同的接口
    // max_steps: 最大迭代次数
    // tol: 梯度收敛阈值
    void maximize(GaussianProcess* gp, size_t max_steps=100, bool verbose=1);
    
    void getLMLandDuration(double& lml, double& duration) { lml = lml_; duration = duration_; }
    
    // 设置 LBFGS 历史窗口大小 (默认 10)
    void set_history_size(int m) { m_ = m; }
    // 设置收敛容差
    void set_tolerance(double tol) { tol_ = tol; }

    void get_lml_time_history(std::vector<double>& lml_history, std::vector<double>& time_cost_history) {
		lml_history = lml_history_;
		time_cost_history = time_cost_history_;
	}

private:
    // LBFGS 算法所需的历史数据结构
    struct IterationData {
        Eigen::VectorXd s; // 参数差值 (x_{k+1} - x_k)
        Eigen::VectorXd y; // 梯度差值 (g_{k+1} - g_k)
        double rho;        // 1 / (y^T * s)
    };

    // 回溯线搜索 (Backtracking Line Search)
    // 返回步长 alpha，如果失败返回 0.0
    double line_search(GaussianProcess* gp, 
                       const Eigen::VectorXd& x, 
                       double f, 
                       const Eigen::VectorXd& g, 
                       const Eigen::VectorXd& d,
                       Eigen::VectorXd& x_new,
                       double& f_new,
                       Eigen::VectorXd& g_new);

    // 双循环递归计算搜索方向
    Eigen::VectorXd compute_direction(const Eigen::VectorXd& gradient);

    // 历史存储
    std::deque<IterationData> history_;
    int m_; // 历史窗口大小
    double tol_; // 收敛容差

    // 结果记录
    double lml_;
    double duration_;

    std::vector<double> lml_history_;
    std::vector<double> time_cost_history_;
};

}

#endif /* LBFGS_H_ */
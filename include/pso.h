/*
 * pso.h
 *
 *  Created on: [Current Date]
 *      Author: [Your Name]
 */

#ifndef PSO_H_
#define PSO_H_

#include "gp.h"
#include <vector>
#include <Eigen/Core>

namespace libgp
{

class PSO
{
public:
    PSO();
    virtual ~PSO();

    // 遵循与其他优化器相同的接口
    // max_iterations: 最大迭代次数
    void maximize(GaussianProcess* gp, size_t max_iterations=100, bool verbose=1);
    
    void getLMLandDuration(double& lml, double& duration) { lml = lml_; duration = duration_; }

    // 设置 PSO 参数
    void set_population_size(size_t size) { population_size_ = size; }
    void set_inertia_weight(double w) { w_ = w; }
    void set_cognitive_coefficient(double c1) { c1_ = c1; }
    void set_social_coefficient(double c2) { c2_ = c2; }

    void get_lml_time_history(std::vector<double>& lml_history, std::vector<double>& time_cost_history) {
		lml_history = lml_history_;
		time_cost_history = time_cost_history_;
	}

private:
    struct Particle {
        Eigen::VectorXd position;      // 当前位置 (超参数)
        Eigen::VectorXd velocity;      // 当前速度
        Eigen::VectorXd best_position; // 个体历史最佳位置 (pBest)
        double best_fitness;           // 个体历史最佳适应度
        double current_fitness;        // 当前适应度
    };

    void initialize_swarm(size_t pop_size, int param_dim, const Eigen::VectorXd& initial_params,
                            const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);

    // 评估粒子适应度 (Log Likelihood)
    double evaluate_particle(GaussianProcess* gp, const Eigen::VectorXd& position);

    std::vector<Particle> swarm_;
    Eigen::VectorXd global_best_position_; // 全局最佳位置 (gBest)
    double global_best_fitness_;           // 全局最佳适应度

    double lml_;
    double duration_;

    // PSO 配置参数
    size_t population_size_;
    double w_;  // 惯性权重 (Inertia Weight)
    double c1_; // 个体认知系数 (Cognitive Coefficient)
    double c2_; // 社会学习系数 (Social Coefficient)

    std::vector<double> lml_history_;
    std::vector<double> time_cost_history_;
};

}

#endif /* PSO_H_ */

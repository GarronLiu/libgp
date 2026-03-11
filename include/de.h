/*
 * de.h
 *
 *  Created on: [Current Date]
 *      Author: [Your Name]
 */

#ifndef DE_H_
#define DE_H_

#include "gp.h"
#include <vector>
#include <Eigen/Core>

namespace libgp
{

class DE
{
public:
    DE();
    virtual ~DE();

    // 遵循与其他优化器相同的接口
    // max_generations: 最大迭代代数
    void maximize(GaussianProcess* gp, size_t max_generations=100, bool verbose=1);
    
    void getLMLandDuration(double& lml, double& duration) { lml = lml_; duration = duration_; }

    // 设置 DE 参数
    void set_population_size(size_t size) { population_size_ = size; }
    void set_crossover_rate(double cr) { crossover_rate_ = cr; }
    void set_differential_weight(double f) { differential_weight_ = f; }

    void get_lml_time_history(std::vector<double>& lml_history, std::vector<double>& time_cost_history) {
		lml_history = lml_history_;
		time_cost_history = time_cost_history_;
	}
private:
    struct Individual {
        Eigen::VectorXd genes; // 超参数
        double fitness;        // 适应度 (Log Likelihood)
    };

    void initialize_population(size_t pop_size, int param_dim, const Eigen::VectorXd& initial_params,
                               const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);
    void evaluate_individual(GaussianProcess* gp, Individual& ind);
    
    // 变异与交叉操作生成试验向量 (Trial Vector)
    Eigen::VectorXd create_trial_vector(int target_idx, int param_dim);

    // 边界约束处理 (Clamp)
    void enforce_bounds(Eigen::VectorXd& params);

    std::vector<Individual> population_;
    double lml_;
    double duration_;

    // DE 配置参数
    size_t population_size_;
    double crossover_rate_;      // CR: 交叉概率 [0, 1]
    double differential_weight_; // F:  缩放因子 [0, 2]
    std::vector<double> lml_history_;
    std::vector<double> time_cost_history_;
};

}

#endif /* DE_H_ */
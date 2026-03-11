/*
 * ga.h
 *
 *  Created on: [Current Date]
 *      Author: [Your Name]
 */

#ifndef GA_H_
#define GA_H_

#include "gp.h"
#include <vector>
#include <Eigen/Core>

namespace libgp
{

class GA
{
public:
    GA();
    virtual ~GA();
    
    // 遵循与 CG 类相似的接口
    void maximize(GaussianProcess* gp, size_t max_generations=100, bool verbose=1);
    void getLMLandDuration(double& lml, double& duration) { lml = lml_; duration = duration_; }

    void get_lml_time_history(std::vector<double>& lml_history, std::vector<double>& time_cost_history) {
		lml_history = lml_history_;
		time_cost_history = time_cost_history_;
	}
private:
    // 遗传算法特定的参数
    struct Individual {
        Eigen::VectorXd genes; // 超参数
        double fitness;        // 适应度 (Log Likelihood)
    };

    void initialize_population(size_t pop_size, int param_dim, const Eigen::VectorXd& initial_params,
                               const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);
    void evaluate_population(GaussianProcess* gp);
    const Individual& selection();
    void crossover(const Individual& p1, const Individual& p2, Individual& c1, Individual& c2);
    void mutation(Individual& ind, double mutation_rate);
    
    std::vector<Individual> population_;
    double lml_;
    double duration_;
    
    // GA 配置参数
    size_t population_size_;
    double mutation_rate_;
    double crossover_rate_;

    std::vector<double> lml_history_;
    std::vector<double> time_cost_history_;
};

}

#endif /* GA_H_ */
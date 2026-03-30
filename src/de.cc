/*
 * de.cc
 *
 *  Created on: [Current Date]
 *      Author: [Your Name]
 */

#include "de.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>

using namespace std;

namespace libgp
{

// 使用静态随机数引擎
static std::random_device rd;
static std::mt19937 rng(rd());

DE::DE() : lml_(0), duration_(0), 
           population_size_(50),    // DE 通常不需要像 GA 那么大的种群
           crossover_rate_(0.9),    // CR 通常较大
           differential_weight_(0.5) // F 通常在 [0.5, 1.0] 之间
{
}

DE::~DE()
{
}

void DE::maximize(GaussianProcess* gp, size_t max_generations, bool verbose)
{
    auto start = std::chrono::high_resolution_clock::now();

    // record lml history and time cost history
    lml_history_.clear();
    lml_history_.reserve(max_generations);
    time_cost_history_.clear();
    time_cost_history_.reserve(max_generations);

    Eigen::VectorXd initial_params = gp->get_hyperparameters();
    // 获取超参的上下界
    Eigen::VectorXd lb = gp->get_hyperparameter_lower_bound();
    Eigen::VectorXd ub = gp->get_hyperparameter_upper_bound();

    int param_dim = initial_params.size();

    // 1. 初始化种群
    // 确保种群大小至少为 4 (DE 需要至少 3 个其他个体来进行变异)
    if (population_size_ < 4) population_size_ = 4;
    initialize_population(population_size_, param_dim, initial_params, lb, ub);

    // 2. 评估初始种群
    for (auto& ind : population_) {
        evaluate_individual(gp, ind);
    }

    // 记录当前最佳
    auto best_it = std::max_element(population_.begin(), population_.end(), 
        [](const Individual& a, const Individual& b) {
            return a.fitness < b.fitness;
        });
    double best_fitness = best_it->fitness;

    if (verbose) {
        cout << "DE Start. Initial Best LML: " << best_fitness << endl;
    }

    // --- DE 主循环 ---
    for (size_t gen = 0; gen < max_generations; ++gen) {
        auto iter_start = std::chrono::high_resolution_clock::now();
        lml_history_.push_back(best_fitness);
        
        // 备份当前参数，避免在评估过程中破坏其他可能的状态
        Eigen::VectorXd gp_backup = gp->get_hyperparameters();

        for (size_t i = 0; i < population_size_; ++i) {
            // A. 变异与交叉 (Mutation & Crossover) -> 生成试验向量 u_i
            Eigen::VectorXd trial_genes = create_trial_vector(i, param_dim);
            
            // B. 边界处理 (使用 lb 和 ub 进行截断)
            for (int k = 0; k < param_dim; ++k) {
                trial_genes(k) = std::max(lb(k), std::min(trial_genes(k), ub(k)));
            }
            
            // 确保噪声具有合理的下界 (防止核矩阵奇异)，通常噪声是最后一个参数
            int noise_idx = param_dim - 1;
            double min_log_noise = -10.0; // 约等于 4.5e-5 在实数域
            if (trial_genes(noise_idx) < min_log_noise) {
                trial_genes(noise_idx) = min_log_noise;
            }

            // C. 选择 (Selection) -> 贪婪策略
            // 临时更新 GP 参数以评估试验个体
            gp->update_hyperparameters(trial_genes);
            double trial_fitness = gp->log_likelihood();
            
            if (std::isnan(trial_fitness) || std::isinf(trial_fitness)) {
                trial_fitness = -1e9;
            }

            // 评估后立即恢复，防止错误累积传递
            gp->update_hyperparameters(gp_backup);

            // 如果试验个体更好，则替换当前个体
            if (trial_fitness > population_[i].fitness) {
                population_[i].genes = trial_genes;
                population_[i].fitness = trial_fitness;
                
                if (trial_fitness > best_fitness) {
                    best_fitness = trial_fitness;
                }
            }
        }

        if (verbose && gen % 10 == 0) {
            cout << "Generation " << gen << ": Best LML = " << best_fitness << endl;
        }

        auto iter_end = std::chrono::high_resolution_clock::now();
        time_cost_history_.push_back(std::chrono::duration<double, std::milli>(iter_end - iter_start).count());
    }

    // 找到最终最佳参数并更新 GP
    best_it = std::max_element(population_.begin(), population_.end(), 
        [](const Individual& a, const Individual& b) {
            return a.fitness < b.fitness;
        });

    gp->update_hyperparameters(best_it->genes);
    lml_ = best_it->fitness;

    auto end = std::chrono::high_resolution_clock::now();
    duration_ = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (verbose) {
        cout << "DE Optimization Finished. Final LML: " << lml_ << " Duration: " << duration_ << "ms" << endl;
        cout << "Optimized Hyperparameters: " << best_it->genes.transpose() << endl;
    }
}

void DE::initialize_population(size_t pop_size, int param_dim, const Eigen::VectorXd& initial_params,
                               const Eigen::VectorXd& lb, const Eigen::VectorXd& ub) {
    population_.clear();
    population_.resize(pop_size);
    
    // 使用范围初始化，如果边界过大(无限)就限制在 [-10, 10] 之类的区间
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    // 第一个个体保留原始参数
    population_[0].genes = initial_params;
    for (int j = 0; j < param_dim; ++j) {
        population_[0].genes(j) = std::max(lb(j), std::min(population_[0].genes(j), ub(j)));
    }
    population_[0].fitness = -1e9;

    for (size_t i = 1; i < pop_size; ++i) {
        population_[i].genes = Eigen::VectorXd(param_dim);
        for (int j = 0; j < param_dim; ++j) {
            // 安全的边界决定
            double safe_lb = std::max(lb(j), -10.0);
            double safe_ub = std::min(ub(j), 10.0);
            
            // 采用基于边界的均匀采样
            population_[i].genes(j) = safe_lb + unif01(rng) * (safe_ub - safe_lb);
        }
        population_[i].fitness = -1e9;
    }
}

void DE::evaluate_individual(GaussianProcess* gp, Individual& ind) {
    gp->update_hyperparameters(ind.genes);
    ind.fitness = gp->log_likelihood();
    if (std::isnan(ind.fitness) || std::isinf(ind.fitness)) {
        ind.fitness = -1e9;
    }
}

// 生成试验向量 (DE/rand/1/bin 策略)
Eigen::VectorXd DE::create_trial_vector(int target_idx, int param_dim) {
    // 1. 选择三个互不相同且不等于 target_idx 的随机个体 r1, r2, r3
    int r1, r2, r3;
    std::uniform_int_distribution<int> dist(0, population_size_ - 1);
    
    do { r1 = dist(rng); } while (r1 == target_idx);
    do { r2 = dist(rng); } while (r2 == target_idx || r2 == r1);
    do { r3 = dist(rng); } while (r3 == target_idx || r3 == r1 || r3 == r2);

    const Eigen::VectorXd& x_r1 = population_[r1].genes;
    const Eigen::VectorXd& x_r2 = population_[r2].genes;
    const Eigen::VectorXd& x_r3 = population_[r3].genes;
    const Eigen::VectorXd& x_target = population_[target_idx].genes;

    Eigen::VectorXd v_trial = Eigen::VectorXd::Zero(param_dim);
    
    // 2. 变异 (Mutation): v = x_r1 + F * (x_r2 - x_r3)
    Eigen::VectorXd mutant = x_r1 + differential_weight_ * (x_r2 - x_r3);

    // 3. 交叉 (Crossover): Binomial Crossover
    std::uniform_real_distribution<double> rand_real(0.0, 1.0);
    std::uniform_int_distribution<int> rand_dim(0, param_dim - 1);
    
    // 确保至少有一个维度发生改变
    int j_rand = rand_dim(rng);

    for (int j = 0; j < param_dim; ++j) {
        if (rand_real(rng) < crossover_rate_ || j == j_rand) {
            v_trial(j) = mutant(j);
        } else {
            v_trial(j) = x_target(j);
        }
    }

    return v_trial;
}

void DE::enforce_bounds(Eigen::VectorXd& params) {
    // 简单的边界约束示例 (如果需要)
    // 例如限制 log 参数在 [-10, 10] 之间
    // for (int i = 0; i < params.size(); ++i) {
    //     if (params(i) > 10.0) params(i) = 10.0;
    //     if (params(i) < -10.0) params(i) = -10.0;
    // }
}

}
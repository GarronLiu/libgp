/*
 * ga.cc
 *
 *  Created on: [Current Date]
 *      Author: [Your Name]
 */

#include "ga.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>

using namespace std;

namespace libgp
{

// 使用静态随机数引擎，避免重复种子初始化
static std::random_device rd;
static std::mt19937 rng(rd());

GA::GA() : lml_(0), duration_(0), population_size_(100), mutation_rate_(0.1), crossover_rate_(0.9)
{
}

GA::~GA()
{
}

void GA::maximize(GaussianProcess* gp, size_t max_generations, bool verbose)
{
    auto start = std::chrono::high_resolution_clock::now();
    // 记录 LML 历史和时间消耗历史
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
    initialize_population(population_size_, param_dim, initial_params, lb, ub);

    // 2. 评估初始种群
    evaluate_population(gp);

    for (size_t gen = 0; gen < max_generations; ++gen) {
        auto iter_start = std::chrono::high_resolution_clock::now();

        // A. 排序：为了精英策略，先按适应度降序排列
        std::sort(population_.begin(), population_.end(), 
            [](const Individual& a, const Individual& b) {
                return a.fitness > b.fitness; // 降序：大的在前
            });
        lml_history_.push_back(population_[0].fitness);
        if (verbose && gen % 10 == 0) {
            cout << "Generation " << gen << ": Best LML = " << population_[0].fitness << endl;
        }

        // B. 精英策略：保留最好的几个个体直接进入下一代
        size_t elite_count = 2; // 保留前2名
        std::vector<Individual> new_population;
        new_population.reserve(population_size_);

        for (size_t i = 0; i < elite_count; ++i) {
            new_population.push_back(population_[i]);
        }

        // C. 生成剩余个体
        while (new_population.size() < population_size_) {
            // 1. 选择 (Selection)
            const Individual& p1 = selection();
            const Individual& p2 = selection();

            Individual c1 = p1;
            Individual c2 = p2;

            // 2. 交叉 (Crossover)
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            if (dist(rng) < crossover_rate_) {
                crossover(p1, p2, c1, c2);
            }

            // 3. 变异 (Mutation)
            mutation(c1, mutation_rate_);
            mutation(c2, mutation_rate_);

            // 4. 边界约束 (Clamping)
            // 交叉和变异可能会导致参数越界，此处强制拉回
            for (int k = 0; k < param_dim; ++k) {
                c1.genes(k) = std::max(lb(k), std::min(c1.genes(k), ub(k)));
                c2.genes(k) = std::max(lb(k), std::min(c2.genes(k), ub(k)));
            }

            new_population.push_back(c1);
            if (new_population.size() < population_size_) {
                new_population.push_back(c2);
            }
        }

        // D. 更新种群
        population_ = new_population;

        // E. 评估新一代
        evaluate_population(gp);

        auto iter_end = std::chrono::high_resolution_clock::now();
        time_cost_history_.push_back(std::chrono::duration<double, std::milli>(iter_end - iter_start).count());
    }

    // 找到最终最佳参数并更新 GP
    auto best_it = std::max_element(population_.begin(), population_.end(), 
        [](const Individual& a, const Individual& b) {
            return a.fitness < b.fitness;
        });

    gp->update_hyperparameters(best_it->genes);
    lml_ = best_it->fitness;

    auto end = std::chrono::high_resolution_clock::now();
    duration_ = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (verbose) {
        cout << "GA Optimization Finished. Final LML: " << lml_ << " Duration: " << duration_ << "ms" << endl;
        cout << "Optimized Hyperparameters: " << best_it->genes.transpose() << endl;
    }
}

void GA::initialize_population(size_t pop_size, int param_dim, const Eigen::VectorXd& initial_params,
                               const Eigen::VectorXd& lb, const Eigen::VectorXd& ub) {
    population_.clear();
    population_.resize(pop_size);
    
    std::normal_distribution<double> distribution(0.0, 0.5); // 较小的扰动

    // 第一个个体保留原始参数 (作为基准)
    population_[0].genes = initial_params;
    population_[0].fitness = -1e9; // 待评估

    for (size_t i = 1; i < pop_size; ++i) {
        population_[i].genes = initial_params;
        for (int j = 0; j < param_dim; ++j) {
             // 对初始参数进行随机扰动并截断
            double val = population_[i].genes(j) + distribution(rng);
            population_[i].genes(j) = std::max(lb(j), std::min(val, ub(j)));
        }
        population_[i].fitness = -1e9;
    }
}

void GA::evaluate_population(GaussianProcess* gp) {
    for (auto& ind : population_) {
        // 简单的缓存机制：如果适应度已经被计算过（非初始值），且我们假设GP数据没变，可以跳过
        // 但为了安全起见，这里每次都计算
        gp->update_hyperparameters(ind.genes);
        ind.fitness = gp->log_likelihood(); 
        
        // 处理数值不稳定的情况
        if (std::isnan(ind.fitness) || std::isinf(ind.fitness)) {
            ind.fitness = -1e9; // 给予极低的适应度
        }
    }
}

// 锦标赛选择 (Tournament Selection)
// 随机选择 k 个个体，返回其中最好的一个
const GA::Individual& GA::selection() {
    const int tournament_size = 3;
    std::uniform_int_distribution<int> dist(0, population_size_ - 1);
    
    int best_idx = dist(rng);
    double best_fit = population_[best_idx].fitness;

    for (int i = 1; i < tournament_size; ++i) {
        int idx = dist(rng);
        if (population_[idx].fitness > best_fit) {
            best_fit = population_[idx].fitness;
            best_idx = idx;
        }
    }
    return population_[best_idx];
}

// 算术交叉 (Arithmetic Crossover)
// 适用于连续数值优化
void GA::crossover(const Individual& p1, const Individual& p2, Individual& c1, Individual& c2) {
    // 生成混合系数 alpha
    std::uniform_real_distribution<double> dist(-0.25, 1.25); // 允许稍微外推，增加探索性
    double alpha = dist(rng);
    
    // c1 = alpha * p1 + (1 - alpha) * p2
    c1.genes = alpha * p1.genes + (1.0 - alpha) * p2.genes;
    c2.genes = (1.0 - alpha) * p1.genes + alpha * p2.genes;
}

// 高斯变异 (Gaussian Mutation)
void GA::mutation(Individual& ind, double mutation_rate) {
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::normal_distribution<double> noise_dist(0.0, 0.1); // 变异强度

    for (int i = 0; i < ind.genes.size(); ++i) {
        if (prob_dist(rng) < mutation_rate) {
            ind.genes(i) += noise_dist(rng);
        }
    }
}

// 注意：需要在 ga.h 中更新 selection 和 crossover 的函数签名以匹配此实现
// void selection() -> const Individual& selection();
// void crossover() -> void crossover(const Individual& p1, const Individual& p2, Individual& c1, Individual& c2);
// void mutation(double rate) -> void mutation(Individual& ind, double rate);

}
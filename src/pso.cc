/*
 * pso.cc
 *
 *  Created on: [Current Date]
 *      Author: [Your Name]
 */

#include "pso.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <limits>

using namespace std;

namespace libgp
{

// 使用静态随机数引擎
static std::random_device rd;
static std::mt19937 rng(rd());

PSO::PSO() : lml_(0), duration_(0), 
             population_size_(30), // 默认种群大小
             w_(0.729),            // 标准惯性权重
             c1_(1.49445),         // 标准认知系数
             c2_(1.49445)          // 标准社会系数
{
}

PSO::~PSO()
{
}

void PSO::maximize(GaussianProcess* gp, size_t max_iterations, bool verbose)
{
    auto start = std::chrono::high_resolution_clock::now();
    // 记录 LML 历史和时间消耗历史
    lml_history_.clear();
    lml_history_.reserve(max_iterations);
    time_cost_history_.clear();
    time_cost_history_.reserve(max_iterations);

    Eigen::VectorXd initial_params = gp->get_hyperparameters();
    // 获取超参的上下界
    Eigen::VectorXd lb = gp->get_hyperparameter_lower_bound();
    Eigen::VectorXd ub = gp->get_hyperparameter_upper_bound();

    int param_dim = initial_params.size();

    // 1. 初始化粒子群
    initialize_swarm(population_size_, param_dim, initial_params, lb, ub);

    // 2. 评估初始群体的适应度并找到初始 gBest
    global_best_fitness_ = -std::numeric_limits<double>::infinity();
    
    for (auto& particle : swarm_) {
        double fitness = evaluate_particle(gp, particle.position);
        
        particle.current_fitness = fitness;
        particle.best_fitness = fitness;
        particle.best_position = particle.position;

        if (fitness > global_best_fitness_) {
            global_best_fitness_ = fitness;
            global_best_position_ = particle.position;
        }
    }

    if (verbose) {
        cout << "PSO Start. Initial Best LML: " << global_best_fitness_ << endl;
    }

    // --- PSO 主循环 ---
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        auto iter_start = std::chrono::high_resolution_clock::now();
        lml_history_.push_back(global_best_fitness_);

        for (auto& particle : swarm_) {
            // A. 更新速度
            // v(t+1) = w*v(t) + c1*r1*(pBest - x(t)) + c2*r2*(gBest - x(t))
            
            for (int d = 0; d < param_dim; ++d) {
                double r1 = dist(rng);
                double r2 = dist(rng);
                
                particle.velocity(d) = w_ * particle.velocity(d) +
                                       c1_ * r1 * (particle.best_position(d) - particle.position(d)) +
                                       c2_ * r2 * (global_best_position_(d) - particle.position(d));
            }

            // B. 更新位置
            // x(t+1) = x(t) + v(t+1)
            particle.position += particle.velocity;

            // 边界约束及速度处理
            for(int d=0; d<param_dim; ++d) {
               if(particle.position(d) > ub(d)) {
                   particle.position(d) = ub(d);
                   particle.velocity(d) = 0.0; // 碰到边界速度归零
               } else if(particle.position(d) < lb(d)) {
                   particle.position(d) = lb(d);
                   particle.velocity(d) = 0.0;
               }
            }

            // C. 评估新位置
            double fitness = evaluate_particle(gp, particle.position);
            particle.current_fitness = fitness;

            // D. 更新个体最佳 (pBest)
            if (fitness > particle.best_fitness) {
                particle.best_fitness = fitness;
                particle.best_position = particle.position;
            }

            // E. 更新全局最佳 (gBest)
            if (fitness > global_best_fitness_) {
                global_best_fitness_ = fitness;
                global_best_position_ = particle.position;
            }
        }

        if (verbose && iter % 10 == 0) {
            cout << "Iteration " << iter << ": Best LML = " << global_best_fitness_ << endl;
        }

        auto iter_end = std::chrono::high_resolution_clock::now();
        time_cost_history_.push_back(std::chrono::duration<double, std::milli>(iter_end - iter_start).count());
    }

    // 将 GP 设置为找到的最佳参数
    gp->update_hyperparameters(global_best_position_);
    lml_ = global_best_fitness_;

    auto end = std::chrono::high_resolution_clock::now();
    duration_ = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (verbose) {
        cout << "PSO Optimization Finished. Final LML: " << lml_ << " Duration: " << duration_ << "ms" << endl;
        cout << "Optimized Hyperparameters: " << global_best_position_.transpose() << endl;
    }
}

void PSO::initialize_swarm(size_t pop_size, int param_dim, const Eigen::VectorXd& initial_params,
                            const Eigen::VectorXd& lb, const Eigen::VectorXd& ub) {
    swarm_.clear();
    swarm_.resize(pop_size);
    
    std::normal_distribution<double> pos_dist(0.0, 1.0); // 位置扰动
    std::uniform_real_distribution<double> vel_dist(-0.1, 0.1); // 初始微小速度

    // 第一个粒子保留原始参数 (精英策略)
    swarm_[0].position = initial_params;
    for (int j = 0; j < param_dim; ++j) {
        swarm_[0].position(j) = std::max(lb(j), std::min(swarm_[0].position(j), ub(j)));
    }
    swarm_[0].velocity = Eigen::VectorXd::Zero(param_dim);
    swarm_[0].best_position = swarm_[0].position;
    swarm_[0].best_fitness = -std::numeric_limits<double>::infinity();

    for (size_t i = 1; i < pop_size; ++i) {
        swarm_[i].position = initial_params;
        swarm_[i].velocity = Eigen::VectorXd(param_dim);
        
        for (int j = 0; j < param_dim; ++j) {
            // 在初始值附近随机撒点
            double val = initial_params(j) + pos_dist(rng);
            swarm_[i].position(j) = std::max(lb(j), std::min(val, ub(j)));
            swarm_[i].velocity(j) = vel_dist(rng);
        }
        
        swarm_[i].best_position = swarm_[i].position;
        swarm_[i].best_fitness = -std::numeric_limits<double>::infinity();
    }
}

double PSO::evaluate_particle(GaussianProcess* gp, const Eigen::VectorXd& position) {
    gp->update_hyperparameters(position);
    double lml = gp->log_likelihood();
    
    // 处理数值不稳定性
    if (std::isnan(lml) || std::isinf(lml)) {
        return -1e9; // 返回一个极小值
    }
    return lml;
}

} // namespace libgp
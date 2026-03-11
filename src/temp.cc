#include <iostream>
#include <vector>
#include <functional>
#include <memory>
#include <type_traits>
#include <Eigen/Dense>

// 前向声明
template<typename T>
class SystemIdentifier;

// 非线性状态空间系统类
template<typename StateFunc, typename OutputFunc>
class NonlinearStateSpaceSystem {
private:
    StateFunc state_func_;
    OutputFunc output_func_;
    int state_dim_;
    int input_dim_;
    int output_dim_;
    Eigen::VectorXd parameters_;

public:
    // 使用自动类型推导的构造函数
    NonlinearStateSpaceSystem(StateFunc state_func, OutputFunc output_func,
                            int state_dim, int input_dim, int output_dim)
        : state_func_(state_func), output_func_(output_func),
          state_dim_(state_dim), input_dim_(input_dim), output_dim_(output_dim),
          parameters_(Eigen::VectorXd::Zero(state_dim + output_dim)) {}
    
    // 移动语义支持
    NonlinearStateSpaceSystem(NonlinearStateSpaceSystem&&) = default;
    NonlinearStateSpaceSystem& operator=(NonlinearStateSpaceSystem&&) = default;
    
    // 禁止拷贝（因为有函数对象）
    NonlinearStateSpaceSystem(const NonlinearStateSpaceSystem&) = delete;
    NonlinearStateSpaceSystem& operator=(const NonlinearStateSpaceSystem&) = delete;
    
    // 使用auto返回类型推导
    auto simulate(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        return state_func_(x, u, parameters_);
    }
    
    auto getOutput(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        return output_func_(x, u, parameters_);
    }
    
    // 使用lambda的数值微分
    auto getJacobian(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        const double epsilon = 1e-6;
        Eigen::MatrixXd jacobian(state_dim_, state_dim_ + input_dim_);
        
        // 对状态变量的偏导
        for (int i = 0; i < state_dim_; ++i) {
            auto perturbed_x = [&, i](double delta) {
                Eigen::VectorXd x_perturbed = x;
                x_perturbed(i) += delta;
                return state_func_(x_perturbed, u, parameters_);
            };
            
            jacobian.col(i) = (perturbed_x(epsilon) - perturbed_x(-epsilon)) / (2 * epsilon);
        }
        
        // 对输入变量的偏导
        for (int i = 0; i < input_dim_; ++i) {
            auto perturbed_u = [&, i](double delta) {
                Eigen::VectorXd u_perturbed = u;
                u_perturbed(i) += delta;
                return state_func_(x, u_perturbed, parameters_);
            };
            
            jacobian.col(state_dim_ + i) = 
                (perturbed_u(epsilon) - perturbed_u(-epsilon)) / (2 * epsilon);
        }
        
        return jacobian;
    }
    
    // 设置参数（支持移动语义）
    void setParameters(Eigen::VectorXd&& params) {
        parameters_ = std::move(params);
    }
    
    void setParameters(const Eigen::VectorXd& params) {
        parameters_ = params;
    }
    
    // 获取维度信息
    constexpr int stateDim() const { return state_dim_; }
    constexpr int inputDim() const { return input_dim_; }
    constexpr int outputDim() const { return output_dim_; }
    const Eigen::VectorXd& parameters() const { return parameters_; }
};

// 使用C++14的自动返回类型推导的工厂函数
template<typename StateFunc, typename OutputFunc>
auto createSystem(StateFunc&& state_func, OutputFunc&& output_func,
                 int state_dim, int input_dim, int output_dim) {
    return NonlinearStateSpaceSystem<
        std::decay_t<StateFunc>, 
        std::decay_t<OutputFunc>>(
        std::forward<StateFunc>(state_func),
        std::forward<OutputFunc>(output_func),
        state_dim, input_dim, output_dim
    );
}

// 系统辨识器类
class SystemIdentifier {
private:
    std::function<void(const Eigen::VectorXd&)> parameter_update_callback_;
    
public:
    // 使用可变参数模板支持多种系统类型
    template<typename SystemType>
    auto identify(SystemType& system,
                 const std::vector<Eigen::VectorXd>& inputs,
                 const std::vector<Eigen::VectorXd>& outputs,
                 const std::vector<Eigen::VectorXd>& states) {
        
        static_assert(std::is_same<decltype(system.simulate(Eigen::VectorXd{}, Eigen::VectorXd{})), 
                                  Eigen::VectorXd>::value,
                     "System must have a simulate method returning Eigen::VectorXd");
        
        // 使用C++14的泛型lambda进行数据处理
        auto buildFeatureMatrix = [&system](const auto& states, const auto& inputs) {
            Eigen::MatrixXd features(states.size(), system.stateDim() * 2);
            
            for (size_t i = 0; i < states.size(); ++i) {
                // 构建特征向量：状态和输入的组合
                Eigen::VectorXd feature(system.stateDim() * 2);
                feature.head(system.stateDim()) = states[i];
                feature.tail(system.inputDim()) = inputs[i];
                features.row(i) = feature.transpose();
            }
            return features;
        };
        
        auto buildTargetMatrix = [](const auto& outputs) {
            Eigen::MatrixXd targets(outputs.size(), outputs[0].size());
            for (size_t i = 0; i < outputs.size(); ++i) {
                targets.row(i) = outputs[i].transpose();
            }
            return targets;
        };
        
        // 构建最小二乘问题
        Eigen::MatrixXd X = buildFeatureMatrix(states, inputs);
        Eigen::MatrixXd Y = buildTargetMatrix(outputs);
        
        // 使用正则化最小二乘
        double lambda = 1e-6; // 正则化参数
        Eigen::MatrixXd XTX = X.transpose() * X + 
                             lambda * Eigen::MatrixXd::Identity(X.cols(), X.cols());
        Eigen::MatrixXd theta = XTX.ldlt().solve(X.transpose() * Y);
        
        // 更新系统参数
        Eigen::VectorXd flat_theta(Eigen::Map<Eigen::VectorXd>(
            theta.data(), theta.rows() * theta.cols()));
        system.setParameters(std::move(flat_theta));
        
        // 回调通知
        if (parameter_update_callback_) {
            parameter_update_callback_(system.parameters());
        }
        
        return theta; // 返回参数矩阵
    }
    
    // 设置参数更新回调（C++14的泛型lambda）
    template<typename Callback>
    void setParameterUpdateCallback(Callback&& callback) {
        parameter_update_callback_ = std::forward<Callback>(callback);
    }
};

int main() {
    // 示例1：直接在main中定义非线性系统候选函数
    auto pendulum_state_func = [](const Eigen::VectorXd& x, 
                                 const Eigen::VectorXd& u,
                                 const Eigen::VectorXd& params) {
        // 单摆系统：x = [角度, 角速度], u = 扭矩
        Eigen::VectorXd x_next(2);
        double g = 9.8, L = 1.0, m = 1.0, b = 0.1;
        
        x_next(0) = x(0) + x(1);  // 角度更新
        x_next(1) = x(1) + (u(0) - m*g*L*sin(x(0)) - b*x(1)) / (m*L*L); // 角速度更新
        
        return x_next;
    };
    
    auto pendulum_output_func = [](const Eigen::VectorXd& x,
                                  const Eigen::VectorXd& u,
                                  const Eigen::VectorXd& params) {
        // 输出为角度
        return Eigen::VectorXd::Constant(1, x(0));
    };
    
    // 使用工厂函数创建系统（自动类型推导）
    auto pendulum_system = createSystem(
        pendulum_state_func, pendulum_output_func, 2, 1, 1
    );
    
    // 示例2：使用多项式非线性系统
    auto poly_state_func = [](const Eigen::VectorXd& x,
                             const Eigen::VectorXd& u,
                             const Eigen::VectorXd& params) {
        Eigen::VectorXd x_next = x;
        // 简单的多项式非线性：x_next = params[0]*x + params[1]*x² + params[2]*u
        if (params.size() >= 3) {
            x_next = params[0] * x + params[1] * x.array().square().matrix() + params[2] * u;
        }
        return x_next;
    };
    
    auto poly_system = createSystem(poly_state_func, pendulum_output_func, 2, 1, 1);
    
    // 创建辨识器
    SystemIdentifier identifier;
    
    // 设置参数更新回调（使用C++14的泛型lambda）
    identifier.setParameterUpdateCallback([](const Eigen::VectorXd& params) {
        std::cout << "参数已更新，新参数范数: " << params.norm() << std::endl;
    });
    
    // 生成测试数据
    std::vector<Eigen::VectorXd> inputs, outputs, states;
    int num_samples = 100;
    
    for (int i = 0; i < num_samples; ++i) {
        Eigen::VectorXd input(1), state(2), output(1);
        input << sin(i * 0.1);
        state << i * 0.01, i * 0.005;
        output << state(0) + 0.1 * noise(); // 添加噪声
        
        inputs.push_back(input);
        states.push_back(state);
        outputs.push_back(output);
    }
    
    // 进行系统辨识
    auto identified_params = identifier.identify(pendulum_system, inputs, outputs, states);
    
    std::cout << "辨识得到的参数矩阵大小: " 
              << identified_params.rows() << " x " << identified_params.cols() << std::endl;
    
    // 测试辨识后的系统
    Eigen::VectorXd test_x(2), test_u(1);
    test_x << 0.5, 0.1;
    test_u << 0.2;
    
    auto predicted_output = pendulum_system.getOutput(test_x, test_u);
    std::cout << "预测输出: " << predicted_output.transpose() << std::endl;
    
    return 0;
}

// 简单的噪声生成函数
double noise() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<> d(0, 0.1);
    return d(gen);
}
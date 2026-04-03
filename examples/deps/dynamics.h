#ifndef DYNAMICS_H
#define DYNAMICS_H

#include "utils.h"

using namespace casadi;
namespace CasadiUtils {
// ==========================================
// Nonlinear System Class
// ==========================================
class NonlinearSystem {
public:
  NonlinearSystem(const std::vector<std::vector<MX>> &candidate_basis,
                  const std::vector<std::vector<double>> &params,
                  const MX &state_vec)
      : NonlinearSystem(candidate_basis, params, state_vec, MX()) {}

  NonlinearSystem(const std::vector<std::vector<MX>> &candidate_basis,
                  const std::vector<std::vector<double>> &params,
                  const MX &state_vec, const MX &control_vec)
      : params_(params), candidate_basis_(candidate_basis),
        state_vec_(state_vec), control_vec_(control_vec) {
    state_dim_ = state_vec.size1();
    control_dim_ = control_vec.is_empty() ? 0 : control_vec.size1();

    total_param_dim_ = 0;
    for (const auto &p : params_)
      total_param_dim_ += p.size();
    sym_params_ = MX::sym("params", total_param_dim_);

    validateDimensions();
    initializeBasisFunctions();
    initializeContinuousDynamicsFunction();

    std::cout << "Nonlinear Dynamic System initialized. State dim: "
              << state_dim_ << ", Control dim: " << control_dim_ << std::endl;
  }

  ~NonlinearSystem() = default;

  void updateStructure(const std::vector<std::vector<MX>> &new_basis,
                       const std::vector<std::vector<double>> &new_params) {

    candidate_basis_ = new_basis;
    params_ = new_params;

    // 重新计算总参数维度
    total_param_dim_ = 0;
    for (const auto &p : params_)
      total_param_dim_ += p.size();

    // 重新创建符号变量，因为维度变了
    sym_params_ = MX::sym("params", total_param_dim_);

    validateDimensions();

    initializeBasisFunctions();

    initializeContinuousDynamicsFunction(); // 重新生成 f_continuous_
  }

  /// @brief 返回某状态维度的动力学加速度值
  /// @param state
  /// @param control
  /// @param state_idx 指定状态维度索引
  /// @return
  double getDynamicsInstance(const Eigen::VectorXd &state,
                             const Eigen::VectorXd &control,
                             size_t state_idx) const {
    using namespace casadi;
    if (state.size() != (int)state_dim_)
      throw std::runtime_error("State dimension mismatch.");
    if (control.size() != (int)control_dim_)
      throw std::runtime_error("Control dimension mismatch.");

    double state_derivative = 0.0;
    DM state_dm, control_dm, params_dm;
    CasadiUtils::eigen2casadi(state, state_dm);

    std::vector<DM> input_vec;
    input_vec.push_back(state_dm);

    if (control_dim_ > 0) {
      CasadiUtils::eigen2casadi(control, control_dm);
      input_vec.push_back(control_dm);
    }

    Eigen::VectorXd flat_params = getFlatParameters();
    CasadiUtils::eigen2casadi(flat_params, params_dm);
    input_vec.push_back(params_dm);

    std::vector<DM> res = f_continuous_(input_vec);
    state_derivative = double(res[0](state_idx));
    return state_derivative;
  }
  Function getContinuousDynamics() const { return f_continuous_; }
  MX getSymState() const { return state_vec_; }
  MX getSymControl() const { return control_vec_; }
  size_t getStateDim() const { return state_dim_; }
  size_t getControlDim() const { return control_dim_; }
  std::vector<std::vector<double>> getParameters() const { return params_; }
  MX getSymParams() const { return sym_params_; }

  Eigen::VectorXd getFlatParameters() const {
    Eigen::VectorXd flat(total_param_dim_);
    int idx = 0;
    for (const auto &row : params_) {
      for (double val : row)
        flat(idx++) = val;
    }
    return flat;
  }

  const std::vector<Function> &getBasisFunctions(size_t state_idx) const {
    return dfdt_basis_functions_.at(state_idx);
  }
  std::vector<std::vector<MX>> getCandidateBasis() const {
    return candidate_basis_;
  }

  void setParameters(const std::vector<std::vector<double>> &new_params) {
    if (new_params.size() != state_dim_)
      throw std::runtime_error("New parameter size mismatch.");
    params_ = new_params;
    initializeContinuousDynamicsFunction();
  }

private:
  size_t state_dim_;
  size_t control_dim_;
  MX state_vec_;
  MX control_vec_;
  std::vector<std::vector<MX>> candidate_basis_;
  std::vector<std::vector<Function>> dfdt_basis_functions_;
  std::vector<std::vector<double>> params_;
  Function f_continuous_;
  MX sym_params_;
  int total_param_dim_;

  void validateDimensions() {
    if (state_dim_ == 0)
      throw std::runtime_error("State dimension cannot be zero.");
    for (size_t i = 0; i < state_dim_; ++i) {
      std::cout << "State " << i << ": " << candidate_basis_[i].size()
                << " basis functions, " << params_[i].size() << " parameters."
                << std::endl;
      if (candidate_basis_[i].size() != params_[i].size()) {
        throw std::runtime_error("Basis/Param count mismatch for state " +
                                 std::to_string(i));
      }
    }
  }

  void initializeBasisFunctions() {
    dfdt_basis_functions_.resize(state_dim_);
    for (size_t i = 0; i < state_dim_; ++i) {
      dfdt_basis_functions_[i].resize(candidate_basis_[i].size());
      for (size_t j = 0; j < candidate_basis_[i].size(); ++j) {
        std::vector<MX> inputs = (control_dim_ > 0)
                                     ? std::vector<MX>{state_vec_, control_vec_}
                                     : std::vector<MX>{state_vec_};
        std::vector<std::string> input_names =
            (control_dim_ > 0) ? std::vector<std::string>{"state", "control"}
                               : std::vector<std::string>{"state"};
        dfdt_basis_functions_[i][j] =
            Function("f_" + std::to_string(i) + "_" + std::to_string(j), inputs,
                     {candidate_basis_[i][j]}, input_names, {"basis_value"});
      }
    }
  }

  void initializeContinuousDynamicsFunction() {
    MX state_dot = MX::zeros(state_dim_);
    int param_idx = 0;
    for (size_t i = 0; i < state_dim_; ++i) {
      for (size_t j = 0; j < candidate_basis_[i].size(); ++j) {
        state_dot(i) += sym_params_(param_idx++) * candidate_basis_[i][j];
      }
    }
    std::vector<MX> inputs = {state_vec_};
    std::vector<std::string> input_names = {"state"};
    if (control_dim_ > 0) {
      inputs.push_back(control_vec_);
      input_names.push_back("control");
    }
    inputs.push_back(sym_params_);
    input_names.push_back("params");
    f_continuous_ = Function("f_continuous", inputs, {state_dot}, input_names,
                             {"state_dot"});
  }
};

// ==========================================
// RK4 Simulator Class
// ==========================================
class RK4Simulator {
public:
  RK4Simulator() = default;
  ~RK4Simulator() = default;

  struct SGPModelCache {
    bool initialized = false;
    int input_dim;
    Eigen::VectorXd lengthscales;    // 长度尺度
    double signal_variance;          // sigma_f^2
    Eigen::MatrixXd inducing_inputs; // m x d
    Eigen::MatrixXd Q_matrix;        // = K_uu^-1 * (K_uu - Sigma_u) * K_uu^-1
  };

  void initSGPModels(size_t state_id, libgp::SparseGaussianProcess *sgp) {
    if (sgp) {
      switch (state_id) {
      case 3:
        cache_sgp_u_ = extractSGPModel(sgp);
        break;
      case 4:
        cache_sgp_v_ = extractSGPModel(sgp);
        break;
      case 5:
        cache_sgp_r_ = extractSGPModel(sgp);
        break;
      default:
        throw std::runtime_error("Invalid state ID");
      }
    }
  }

  void reset(const NonlinearSystem &nl_system) {
    Function f_continuous = nl_system.getContinuousDynamics();
    MX x = nl_system.getSymState();
    MX u = nl_system.getSymControl();
    MX p = nl_system.getSymParams();
    MX dt_sym = MX::sym("dt");
    flat_params_ = nl_system.getFlatParameters();
    bool has_control = (nl_system.getControlDim() > 0);

    auto call_f = [&](const MX &state_arg, const MX &control_arg) {
      std::vector<MX> args = {state_arg};
      if (has_control)
        args.push_back(control_arg);
      args.push_back(p);
      return f_continuous(args)[0];
    };

    MX K1 = call_f(x, u);
    MX K2 = call_f(x + dt_sym / 2 * K1, u);
    MX K3 = call_f(x + dt_sym / 2 * K2, u);
    MX K4 = call_f(x + dt_sym * K3, u);
    MX state_next = x + dt_sym / 6 * (K1 + 2 * K2 + 2 * K3 + K4);

    std::vector<MX> inputs = {x};
    std::vector<std::string> names = {"state"};
    if (has_control) {
      inputs.push_back(u);
      names.push_back("control");
    }
    inputs.push_back(p);
    names.push_back("params");
    inputs.push_back(dt_sym);
    names.push_back("dt");

    // Continuous dynamics function
    f_continuous_ = f_continuous;
    // Discrete dynamics function
    f_discrete_ =
        Function("f_discrete", inputs, {state_next}, names, {"state_next"});
    // Calculate Jacobian symbolically
    // Jacobian of state_next w.r.t x (state)
    MX J_x = MX::jacobian(state_next, x);

    // Discrete dynamics function derivative w.r.t. state
    // Note output name changed to "jac_state_next_state" for clarity
    df_discrete_dstate = Function("df_discrete_dstate", inputs, {J_x}, names,
                                  {"jac_state_next_state"});

    // Discrete dynamics function derivative w.r.t. control
    if (has_control) {
      // Jacobian of state_next w.r.t u (control)
      MX J_u = MX::jacobian(state_next, u);
      df_discrete_dcontrol = Function("df_discrete_dcontrol", inputs, {J_u},
                                      names, {"jac_state_next_control"});
    } else {
      // Handle no control case (return zero matrix or empty)
      df_discrete_dcontrol = Function();
    }
  }

  void reset(const Function &f_continuous, const MX &x, const MX &u) {
    MX dt_sym = MX::sym("dt");
    bool has_control = (u.size1() > 0);

    auto call_f = [&](const MX &state_arg, const MX &control_arg) {
      std::vector<MX> args = {state_arg};
      if (has_control)
        args.push_back(control_arg);
      return f_continuous(args)[0];
    };

    MX K1 = call_f(x, u);
    MX K2 = call_f(x + dt_sym / 2 * K1, u);
    MX K3 = call_f(x + dt_sym / 2 * K2, u);
    MX K4 = call_f(x + dt_sym * K3, u);
    MX state_next = x + dt_sym / 6 * (K1 + 2 * K2 + 2 * K3 + K4);

    std::vector<MX> inputs = {x};
    std::vector<std::string> names = {"state"};
    if (has_control) {
      inputs.push_back(u);
      names.push_back("control");
    }
    inputs.push_back(dt_sym);
    names.push_back("dt");

    f_discrete_ =
        Function("f_discrete", inputs, {state_next}, names, {"state_next"});
  }

  Eigen::VectorXd step(const Eigen::VectorXd &state,
                       const Eigen::VectorXd &control,
                       const Eigen::VectorXd &params, double dt) {
    if (f_discrete_.is_null())
      throw std::runtime_error("Simulator not initialized.");
    DM state_dm, control_dm, params_dm;
    CasadiUtils::eigen2casadi(state, state_dm);
    CasadiUtils::eigen2casadi(params, params_dm);
    std::vector<DM> input_vec = {state_dm};
    if (control.size() > 0) {
      CasadiUtils::eigen2casadi(control, control_dm);
      input_vec.push_back(control_dm);
    }
    input_vec.push_back(params_dm);
    input_vec.push_back(DM(dt));
    DM state_next_dm = f_discrete_(input_vec)[0];
    Eigen::VectorXd state_next_eigen;
    CasadiUtils::casadi2eigen(state_next_dm, state_next_eigen);
    return state_next_eigen;
  }

  Eigen::VectorXd step(const Eigen::VectorXd &state,
                       const Eigen::VectorXd &control, double dt) {
    if (f_discrete_.is_null())
      throw std::runtime_error("Simulator not initialized.");
    DM state_dm, control_dm;
    CasadiUtils::eigen2casadi(state, state_dm);
    CasadiUtils::eigen2casadi(control, control_dm);
    std::vector<DM> input_vec = {state_dm};
    if (control.size() > 0) {
      CasadiUtils::eigen2casadi(control, control_dm);
      input_vec.push_back(control_dm);
    }
    input_vec.push_back(DM(dt));
    DM state_next_dm = f_discrete_(input_vec)[0];
    Eigen::VectorXd state_next_eigen;
    CasadiUtils::casadi2eigen(state_next_dm, state_next_eigen);
    return state_next_eigen;
  }

  std::pair<Eigen::VectorXd, Eigen::MatrixXd>
  step_with_uncertainty(const Eigen::VectorXd &state,
                        const Eigen::VectorXd &control, double dt,
                        const Eigen::MatrixXd &cov_state) {
    return step_with_uncertainty(state, control, flat_params_, dt, cov_state);
  }

  // RK4 step with uncertainty propagation using PRECOMPUTED MATRICES
  // ----------------------------
  std::pair<Eigen::VectorXd, Eigen::MatrixXd>
  step_with_uncertainty(const Eigen::VectorXd &state,
                        const Eigen::VectorXd &control,
                        const Eigen::VectorXd &params, double dt,
                        const Eigen::MatrixXd &cov_state) {
    if (f_continuous_.is_null())
      throw std::runtime_error("Continuous dynamics not available.");

    // 1. 标准 RK4 均值预测 (保持不变)
    auto call_f_eigen = [&](const Eigen::VectorXd &x_arg,
                            const Eigen::VectorXd &u_arg,
                            const Eigen::VectorXd &p_arg) -> Eigen::VectorXd {
      DM x_dm, u_dm, p_dm;
      CasadiUtils::eigen2casadi(x_arg, x_dm);
      CasadiUtils::eigen2casadi(p_arg, p_dm);
      std::vector<DM> args = {x_dm};
      if (u_arg.size() > 0) {
        CasadiUtils::eigen2casadi(u_arg, u_dm);
        args.push_back(u_dm);
      }
      args.push_back(p_dm);
      DM res = f_continuous_(args)[0];
      Eigen::VectorXd out;
      CasadiUtils::casadi2eigen(res, out);
      return out;
    };

    // Jacobian of f_discrete w.r.t state
    auto call_df_dx_eigen =
        [&](const Eigen::VectorXd &x_arg, const Eigen::VectorXd &u_arg,
            const Eigen::VectorXd &p_arg) -> Eigen::MatrixXd {
      DM x_dm, u_dm, p_dm;
      CasadiUtils::eigen2casadi(x_arg, x_dm);
      CasadiUtils::eigen2casadi(p_arg, p_dm);
      std::vector<DM> args = {x_dm};
      if (u_arg.size() > 0) {
        CasadiUtils::eigen2casadi(u_arg, u_dm);
        args.push_back(u_dm);
      }
      args.push_back(p_dm);
      args.push_back(DM(dt));
      DM res = df_discrete_dstate(args)[0];
      Eigen::MatrixXd out;
      CasadiUtils::casadi2eigen(res, out);
      return out;
    };

    // Jacobian of f_discrete w.r.t control
    auto call_df_du_eigen =
        [&](const Eigen::VectorXd &x_arg, const Eigen::VectorXd &u_arg,
            const Eigen::VectorXd &p_arg) -> Eigen::MatrixXd {
      DM x_dm, u_dm, p_dm;
      CasadiUtils::eigen2casadi(x_arg, x_dm);
      CasadiUtils::eigen2casadi(p_arg, p_dm);
      std::vector<DM> args = {x_dm};
      if (u_arg.size() > 0) {
        CasadiUtils::eigen2casadi(u_arg, u_dm);
        args.push_back(u_dm);
      }
      args.push_back(p_dm);
      args.push_back(DM(dt));
      DM res = df_discrete_dcontrol(args)[0];
      Eigen::MatrixXd out;
      CasadiUtils::casadi2eigen(res, out);
      return out;
    };

    Eigen::MatrixXd Sigma_state_prior = cov_state;
    static Eigen::MatrixXd Sigma_input =
        Eigen::MatrixXd::Identity(control.size(), control.size()) * 0.01;

    Eigen::VectorXd K1 = call_f_eigen(state, control, params);
    Eigen::VectorXd x_k2 = state + (dt / 2.0) * K1;
    Eigen::VectorXd K2 = call_f_eigen(x_k2, control, params);
    Eigen::VectorXd x_k3 = state + (dt / 2.0) * K2;
    Eigen::VectorXd K3 = call_f_eigen(x_k3, control, params);
    Eigen::VectorXd x_k4 = state + dt * K3;
    Eigen::VectorXd K4 = call_f_eigen(x_k4, control, params);

    Eigen::VectorXd state_next =
        state + (dt / 6.0) * (K1 + 2.0 * K2 + 2.0 * K3 + K4);

    // 2. 构建 SGP 输入向量
    auto build_input = [&](const Eigen::VectorXd &x_st,
                           const Eigen::VectorXd &u_ct) -> Eigen::VectorXd {
      // 假设 SGP 输入为 [state_active, control]
      // 这里需要根据您实际训练时的 state mask 来调整。
      // 为简化，假设 active state 为 [u, v, r] (索引 3,4,5)
      // 如有不同，需传入 active_mask 参数。
      int active_dims = 3;
      int ctrl_dims = u_ct.size();
      Eigen::VectorXd z(active_dims + ctrl_dims);
      z << x_st(3), x_st(4), x_st(5), u_ct; // u,v,r + controls
      return z;
    };

    Eigen::VectorXd z1 = build_input(state, control);
    Eigen::VectorXd z2 = build_input(x_k2, control);
    Eigen::VectorXd z3 = build_input(x_k3, control);
    Eigen::VectorXd z4 = build_input(x_k4, control);

    // 3. 快速方差计算 (使用预计算矩阵) Sigma_Process
    auto fast_var = [&](const SGPModelCache &cache,
                        const Eigen::VectorXd &z) -> double {
      if (!cache.initialized) {
        std::cerr << "SGP Model Cache not initialized!" << std::endl;
        return 0.0;
      }

      return predictVarianceFast(cache, z);
    };

    double var_u =
        (dt * dt / 6.0) *
        (fast_var(cache_sgp_u_, z1) + 2 * fast_var(cache_sgp_u_, z2) +
         2 * fast_var(cache_sgp_u_, z3) + fast_var(cache_sgp_u_, z4));
    double var_v =
        (dt * dt / 6.0) *
        (fast_var(cache_sgp_v_, z1) + 2 * fast_var(cache_sgp_v_, z2) +
         2 * fast_var(cache_sgp_v_, z3) + fast_var(cache_sgp_v_, z4));
    double var_r =
        (dt * dt / 6.0) *
        (fast_var(cache_sgp_r_, z1) + 2 * fast_var(cache_sgp_r_, z2) +
         2 * fast_var(cache_sgp_r_, z3) + fast_var(cache_sgp_r_, z4));

    Eigen::MatrixXd Sigma_proc = Eigen::MatrixXd::Zero(6, 6);
    Sigma_proc(3, 3) = var_u;
    Sigma_proc(4, 4) = var_v;
    Sigma_proc(5, 5) = var_r;

    // First Order Approximation
    Eigen::MatrixXd Sigma_state_pred = Sigma_proc;

    auto df_dx = call_df_dx_eigen(state, control, params);
    auto df_du = call_df_du_eigen(state, control, params);

    auto df_dx_tran = df_dx.transpose();
    auto df_du_tran = df_du.transpose();

    Sigma_state_pred += df_dx * Sigma_state_prior * df_dx_tran;
    Sigma_state_pred += df_du * Sigma_input * df_du_tran;

    return {state_next, Sigma_state_pred};
  }

private:
  // ----------------------------------------------------
  // 提取 SGP 参数并计算 Q 矩阵
  // Q = K_uu^-1 * (K_uu - Sigma_u) * K_uu^-1
  // ----------------------------------------------------
  SGPModelCache extractSGPModel(libgp::SparseGaussianProcess *sgp) {
    SGPModelCache cache;
    if (!sgp)
      return cache;

    cache.input_dim = sgp->get_input_dim();
    Eigen::VectorXd hypers = sgp->covf().get_loghyper().array().exp();
    // 假设使用 CovSum(CovSEard, CovNoise) 或类似 ARD 核
    // hypers 结构: [l_1, ..., l_D, sigma_f, sigma_n]
    // 需要仔细确认 hyperparam 的排列顺序！
    // 若是 CovSEard: [l_1, ..., l_D, sigma_f]
    // 若是 CovSum(SE, Noise): [l_1, ..., l_D, sigma_f, sigma_n]

    // 提取 lengthscales
    cache.lengthscales = hypers.head(cache.input_dim);
    // 提取 signal variance (sigma_f^2)
    // 注意：libgp 的 hyper 是 log 形式，这里 exp 后是 sigma_f
    // 这里的索引取决于具体的 Covariance Function 组合。
    // 假设是 standard ARD + Noise 组合，sigma_f 在 input_dim 位置
    cache.signal_variance = hypers(cache.input_dim) * hypers(cache.input_dim);

    // 提取诱导点 (m x d)
    cache.inducing_inputs =
        sgp->getFlatInputs(); // libgp 可能是 d x m ? 需确认转置
    if (cache.inducing_inputs.rows() == cache.input_dim) {
      cache.inducing_inputs.transposeInPlace(); // 转为 m x d
    }

    size_t m = cache.inducing_inputs.rows();

    // 1. 获取先验核矩阵 K_uu
    Eigen::MatrixXd K_uu(m, m);
    // 手动计算 K_uu (避免重新调用 libgp 内部私有方法)
    // 对于 SE-ARD 核: k(x, x') = sigma_f^2 * exp(-0.5 * sum( (x_d - x'_d)^2 /
    // l_d^2 ))
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j <= i; ++j) {
        double dist = 0.0;
        for (int d = 0; d < cache.input_dim; ++d) {
          double diff =
              (cache.inducing_inputs(i, d) - cache.inducing_inputs(j, d)) /
              cache.lengthscales(d);
          dist += diff * diff;
        }
        double val = cache.signal_variance * std::exp(-0.5 * dist);
        K_uu(i, j) = val;
        K_uu(j, i) = val;
      }
      K_uu(i, i) += 1e-6; // jitter
    }

    // 2. 获取后验协方差矩阵 Sigma_u
    // libgp 需要提供接口: Eigen::MatrixXd getPosteriorCov();
    // 假设已添加到 SparseGaussianProcess 类中
    Eigen::MatrixXd Sigma_u =
        sgp->getFlatPosteriorCovMatrix(); // 需确保此方法存在

    // 3. 计算 Q = K_uu^-1 * (K_uu - Sigma_u) * K_uu^-1
    // 也可以写作 Q = K_uu^-1 - K_uu^-1 * Sigma_u * K_uu^-1
    Eigen::LLT<Eigen::MatrixXd> llt(K_uu);
    Eigen::MatrixXd K_uu_inv = Eigen::MatrixXd::Identity(m, m);
    llt.solveInPlace(K_uu_inv); // K_uu_inv 现在是逆矩阵

    Eigen::MatrixXd Diff = K_uu - Sigma_u;

    // Q = K_uu_inv * Diff * K_uu_inv
    cache.Q_matrix = K_uu_inv * Diff * K_uu_inv;

    cache.initialized = true;
    return cache;
  }

  // ----------------------------------------------------
  // sigma^2 = k** - k*u * Q * ku*
  // ----------------------------------------------------
  double predictVarianceFast(const SGPModelCache &cache,
                             const Eigen::VectorXd &z) {
    // 1. 计算 k_u* (诱导点与测试点的核向量 m x 1)
    int m = cache.inducing_inputs.rows();
    Eigen::VectorXd k_us(m);

    for (int i = 0; i < m; ++i) {
      double dist = 0.0;
      for (int d = 0; d < cache.input_dim; ++d) {
        double diff =
            (cache.inducing_inputs(i, d) - z(d)) / cache.lengthscales(d);
        dist += diff * diff;
      }
      k_us(i) = cache.signal_variance * std::exp(-0.5 * dist);
    }

    // 2. 计算方差
    // var = K(x,x) - k_us^T * Q * k_us
    double k_ss = cache.signal_variance; // + sigma_n^2 ?

    double reduction = k_us.dot(cache.Q_matrix * k_us);
    double var = k_ss - reduction;

    return std::max(0.0, var); // 保证非负
  }

  Function f_discrete_;
  Function f_continuous_;
  Function df_discrete_dstate;
  Function df_discrete_dcontrol;

  Eigen::VectorXd flat_params_;

  SGPModelCache cache_sgp_u_;
  SGPModelCache cache_sgp_v_;
  SGPModelCache cache_sgp_r_;
};

} // namespace CasadiUtils
#endif // DYNAMICS_H
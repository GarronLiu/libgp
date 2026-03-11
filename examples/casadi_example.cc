#include "cg.h"
#include "ga.h"
#include "gp.h"
#include "gp_utils.h"
#include "matplotlibcpp.h"
#include "recursive_gp.h"
#include "rprop.h"
#include "sparse_gp.h"
#include <Eigen/Dense>

#include <chrono>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "evaluate.h"

#include <casadi/casadi.hpp>

namespace plt = matplotlibcpp;
using namespace libgp;
using namespace casadi;

// ==========================================
// Utility Functions
// ==========================================
namespace CasadiUtils {
inline void eigen2casadi(const Eigen::VectorXd &eigen_vec, DM &casadi_vec) {
  if (eigen_vec.size() == 0) {
    casadi_vec = DM();
    return;
  }
  casadi_vec.resize(eigen_vec.size(), 1);
  for (size_t i = 0; i < (size_t)eigen_vec.size(); ++i) {
    casadi_vec(i) = eigen_vec(i);
  }
}

inline void casadi2eigen(const DM &casadi_vec, Eigen::VectorXd &eigen_vec) {
  if (casadi_vec.is_empty()) {
    eigen_vec.resize(0);
    return;
  }
  eigen_vec.resize(casadi_vec.size1());
  for (size_t i = 0; i < (size_t)casadi_vec.size1(); ++i) {
    eigen_vec(i) = double(casadi_vec(i));
  }
}

// Colorblind-friendly palette
std::vector<std::string> plot_colors_{"#0072B2", "#D55E00", "#009E73",
                                      "#CC79A7", "#F0E442", "#56B4E9",
                                      "#E69F00", "#000000"};
std::vector<std::string> plot_linestyles_{"-", "--",    "-.",
                                          ":", "solid", "dashed"};

void plot_format_init(int width_cm = 17.6, int height_cm = 4.5) {
  std::map<std::string, std::string> keywords = {{"figure.dpi", "300"}};
  plt::rcparams(keywords);
  keywords = {{"font.family", "Times New Roman"}};
  plt::rcparams(keywords);
  keywords = {{"font.size", "6.0"}};
  plt::rcparams(keywords);
  float cm_to_inch = 1.0 / 2.54;
  plt::figure_size(width_cm * cm_to_inch * 300, height_cm * cm_to_inch * 300);
  plt::tight_layout();
}

// Some dynamic system models

// 构造以诱导点为中心的 RBF kernel basis（返回每个 state 的 basis 列表）
std::vector<casadi::MX>
buildKernelBasis(const casadi::MX &state_sym, const casadi::MX &control_sym,
                 const std::vector<Eigen::VectorXd> &inducing_points,
                 const Eigen::VectorXd &lengthscales,
                 double process_covariance) {
  using namespace casadi;
  int state_dim = state_sym.size1();
  int control_dim = control_sym.is_empty() ? 0 : control_sym.size1();
  int z_dim = state_dim + control_dim;

  // 检查诱导点和z的维度一致性
  for (const auto &pt : inducing_points) {
    if (pt.size() != z_dim) {
      std::cerr << "pt.size(): " << pt.size() << ", z_dim: " << z_dim
                << std::endl;
      throw std::runtime_error("Inducing point dimension mismatch.");
    }
  }

  // 拼接输入符号 z = vertcat(state, control) 或仅 state
  MX z = state_sym;
  if (control_dim > 0) {
    z = vertcat(state_sym, control_sym);
  }

  // 预生成基函数（共享）
  std::vector<MX> kernel_basis;
  kernel_basis.reserve(inducing_points.size());

  // 将 lengthscales 转成 DM 常量（逐元素）
  std::vector<double> ls_vec(lengthscales.data(),
                             lengthscales.data() + lengthscales.size());
  DM ls_dm = DM::reshape(DM(ls_vec), ls_vec.size(), 1); // 列向量

  for (const auto &pt : inducing_points) {
    // 转换诱导点到 DM 列向量
    std::vector<double> pt_vec(pt.data(), pt.data() + pt.size());
    DM pt_dm = DM::reshape(DM(pt_vec), pt_vec.size(), 1);

    // 计算 squared Mahalanobis 距离: sum(((z - z_i) / ls)^2)
    MX z_ind = MX(pt_dm); // 常量诱导点
    MX diff = z - z_ind;
    // 对每个分量除以 lengthscale
    MX scaled = diff / MX(ls_dm);
    MX sq = dot(scaled, scaled);                // 标量
    MX k = process_covariance * exp(-0.5 * sq); // RBF kernel
    kernel_basis.push_back(k);
  }

  return kernel_basis;
}

} // namespace CasadiUtils

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

  void reset(const NonlinearSystem &nl_system) {
    Function f_continuous = nl_system.getContinuousDynamics();
    MX x = nl_system.getSymState();
    MX u = nl_system.getSymControl();
    MX p = nl_system.getSymParams();
    MX dt_sym = MX::sym("dt");
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

private:
  Function f_discrete_;
};

// ==========================================
// Base Class: ParamEstimator
// ==========================================
class ParamEstimator {
public:
  ParamEstimator(NonlinearSystem *nl_system) : nl_system_(nl_system) {
    state_dim_ = nl_system_->getStateDim();
    // Initialize RK4 for prediction/visualization
    rk4_simulator_.reset(*nl_system_);
  }
  virtual ~ParamEstimator() = default;

  void setData(const std::vector<Eigen::VectorXd> &state_data,
               const std::vector<Eigen::VectorXd> &control_data,
               const std::vector<double> &time_data) {
    state_data_ = state_data;
    control_data_ = control_data;
    time_data_ = time_data;
    validateData();
  }

  void setData(const std::vector<Eigen::VectorXd> &state_data,
               const std::vector<double> &time_data) {
    state_data_ = state_data;
    time_data_ = time_data;
    control_data_.clear();
    std::cout << "Data set. Size: " << state_data_.size() << std::endl;
  }

  void setTestData(const std::vector<Eigen::VectorXd> &state_test_data,
                   const std::vector<Eigen::VectorXd> &control_test_data,
                   const std::vector<double> &time_test_data) {
    state_test_data_ = state_test_data;
    control_test_data_ = control_test_data;
    time_test_data_ = time_test_data;
    validateData();
    std::cout << "Test data set. Size: " << state_test_data_.size()
              << std::endl;
  }

  void setTestData(const std::vector<Eigen::VectorXd> &state_test_data,
                   const std::vector<double> &time_test_data) {
    state_test_data_ = state_test_data;
    time_test_data_ = time_test_data;
    control_test_data_.clear();
    validateData();
    std::cout << "Test data set. Size: " << state_test_data_.size()
              << std::endl;
  }

  // Pure virtual function for estimation logic
  virtual void estimate(const std::vector<bool> &active_state_mask) = 0;

  // Visualization logic (shared)
  void visualizeFittingResults() {
    if (state_data_.empty()) {
      std::cerr << "No data to visualize." << std::endl;
      return;
    }
    if (state_pred_history_.empty()) {
      std::cerr << "No prediction history available. Run estimate() first."
                << std::endl;
      return;
    }

    size_t N = std::min(state_data_.size(), state_pred_history_.size());
    MX state_sym = nl_system_->getSymState();
    std::string state_name;

    // 1. Plot State Fitting
    std::cout << "Visualizing state fitting results on training set..."
              << std::endl;
    for (size_t i = 0; i < state_dim_; ++i) {
      std::vector<double> x_true_i(N), x_est_i(N), t_vec(N);
      for (size_t k = 0; k < N; ++k) {
        x_true_i[k] = state_data_[k](i);
        x_est_i[k] = state_pred_history_[k](i);
        t_vec[k] = time_data_[k];
      }

      CasadiUtils::plot_format_init(7.0, 4.5);
      plt::named_plot("Ground Truth", t_vec, x_true_i, "b.");
      plt::named_plot("Estimated", t_vec, x_est_i, "r-");

      if (i < (size_t)state_sym.size1()) {
        MX elem = state_sym(i);
        if (elem.is_symbolic())
          state_name = elem.name();
      }
      if (state_name.empty())
        state_name = "state_" + std::to_string(i);

      plt::title(state_name + " Tracking (Training Set)");
      plt::xlabel("Time [s]");
      plt::ylabel("State Value");
      plt::legend();
      plt::grid(true);
      plt::show();
    }

    if (!state_test_data_.empty()) {
      std::cout << "Visualizing state fitting results on test set..."
                << std::endl;
      // Generate prediction history on test set
      size_t N_test = state_test_data_.size();
      for (size_t i = 0; i < state_dim_; ++i) {
        std::vector<double> x_true_i(N_test), x_est_i(N_test), t_vec(N_test);
        for (size_t k = 0; k < N_test; ++k) {
          x_true_i[k] = state_test_data_[k](i);
          x_est_i[k] = state_pred_history_[k + N](i); // offset by training size
          t_vec[k] = time_test_data_[k];
        }

        CasadiUtils::plot_format_init(11.0, 8.0);
        plt::named_plot("Ground Truth", t_vec, x_true_i, "b.");
        plt::named_plot("Estimated", t_vec, x_est_i, "r-");

        if (i < (size_t)state_sym.size1()) {
          MX elem = state_sym(i);
          if (elem.is_symbolic())
            state_name = elem.name();
        }
        if (state_name.empty())
          state_name = "state_" + std::to_string(i);

        plt::title(state_name + " Tracking (Test Set)");
        plt::xlabel("Time [s]");
        plt::ylabel("State Value");
        plt::legend();
        plt::grid(true);
        plt::show();
      }
    }

    // 2. Plot Parameter Convergence (if available)
    if (!param_history_.empty()) {
      std::cout << "Visualizing parameter convergence..." << std::endl;
      size_t Tn = std::min(time_data_.size(), param_history_.size());
      std::vector<double> t_vec(Tn);
      for (size_t k = 0; k < Tn; ++k)
        t_vec[k] = time_data_[k];

      auto param_structure = nl_system_->getParameters();
      int total_flat = 0;
      for (const auto &v : param_structure)
        total_flat += v.size();

      // Map flat index to dimension
      std::vector<int> flat_to_dim(total_flat);
      int pos = 0;
      for (int d = 0; d < (int)state_dim_; ++d) {
        for (size_t k = 0; k < param_structure[d].size(); ++k)
          flat_to_dim[pos++] = d;
      }

      // Plot per dimension
      for (int d = 0; d < (int)state_dim_; ++d) {
        CasadiUtils::plot_format_init(11.0, 8.0);
        bool has_params = false;

        if (d < (int)state_sym.size1()) {
          MX elem = state_sym(d);
          if (elem.is_symbolic())
            state_name = elem.name();
        } else
          state_name = "state_" + std::to_string(d);

        int current_flat_idx = 0;
        // Find start index for this dimension
        for (int prev_d = 0; prev_d < d; ++prev_d)
          current_flat_idx += param_structure[prev_d].size();

        for (size_t p = 0; p < param_structure[d].size(); ++p) {
          int p_idx = current_flat_idx + p;
          std::vector<double> p_series(Tn);
          bool is_constant = true;
          for (size_t t = 0; t < Tn; ++t) {
            p_series[t] = param_history_[t](p_idx);
            if (t > 0 && std::abs(p_series[t] - p_series[t - 1]) > 1e-9)
              is_constant = false;
          }

          // Only plot if it changes or if we want to see everything
          has_params = true;
          std::string label =
              "$\\theta_{" + state_name + ',' + std::to_string(p) + "}$";
          std::string color =
              CasadiUtils::plot_colors_[p % CasadiUtils::plot_colors_.size()];
          std::string style =
              CasadiUtils::plot_linestyles_[p % CasadiUtils::plot_linestyles_
                                                    .size()];

          std::map<std::string, std::string> keywords;
          keywords["label"] = label;
          keywords["color"] = color;
          keywords["linestyle"] = style;
          plt::plot(t_vec, p_series, keywords);
        }

        if (has_params) {
          plt::title(state_name + " Parameters");
          plt::xlabel("Time [s]");
          plt::ylabel("Value");
          plt::legend();
          plt::grid(true);
          plt::show();
        }
        // Print final estimated parameters for this state
        std::cout << "Identified parameters for " << state_name << ":";
        for (size_t p = 0; p < param_structure[d].size(); ++p) {
          int p_idx = current_flat_idx + p;
          double final_val = param_history_.back()(p_idx);
          std::cout << ' ' << final_val;
        }
        std::cout << std::endl;
      }
    }
  }

  std::vector<std::vector<double>> getEstimatedParameters() const {
    return nl_system_->getParameters();
  }

protected:
  NonlinearSystem *nl_system_;
  RK4Simulator rk4_simulator_;
  size_t state_dim_;

  std::vector<Eigen::VectorXd> state_data_;
  std::vector<Eigen::VectorXd> control_data_;
  std::vector<double> time_data_;

  std::vector<Eigen::VectorXd> state_test_data_;
  std::vector<Eigen::VectorXd> control_test_data_;
  std::vector<double> time_test_data_;

  // Results storage
  std::vector<Eigen::VectorXd> state_pred_history_;
  std::vector<Eigen::VectorXd> param_history_;

  void validateData() {
    if (state_data_.size() != time_data_.size())
      throw std::runtime_error("State and time data size mismatch.");
    if (!control_data_.empty() && state_data_.size() != control_data_.size())
      throw std::runtime_error("State and control data size mismatch.");
    if (!state_test_data_.empty() &&
        state_test_data_.size() != time_test_data_.size())
      throw std::runtime_error("State test and time test data size mismatch.");
    if (!control_test_data_.empty() &&
        control_test_data_.size() != time_test_data_.size())
      throw std::runtime_error(
          "Control test and time test data size mismatch.");
  }

  void generatePredictionHistory() {
    state_pred_history_.clear();

    Eigen::VectorXd flat_params = nl_system_->getFlatParameters();
    Eigen::VectorXd x_curr = state_data_[0];
    state_pred_history_.push_back(x_curr);

    // on training data
    for (size_t k = 0; k < state_data_.size() - 1; ++k) {
      double dt = time_data_[k + 1] - time_data_[k];
      if (dt <= 0)
        dt = 1e-4;
      Eigen::VectorXd u =
          control_data_.empty() ? Eigen::VectorXd() : control_data_[k];

      x_curr = rk4_simulator_.step(x_curr, u, flat_params, dt);
      state_pred_history_.push_back(x_curr);
    }

    // Statistics on training data: RMSE, MAE and t-RMSE for each dimension
    Eigen::VectorXd mse(state_dim_);
    mse.setZero();
    Eigen::VectorXd mae(state_dim_);
    mae.setZero();
    for (size_t k = 0; k < state_data_.size(); ++k) {
      Eigen::VectorXd error = state_data_[k] - state_pred_history_[k];
      for (size_t d = 0; d < state_dim_; ++d) {
        mse[d] += error(d) * error(d);
        mae[d] += std::abs(error(d));
      }
      mse /= state_data_.size();
      mae /= state_data_.size();
    }

    Eigen::VectorXd rmse(state_dim_);
    rmse.setZero();
    for (size_t d = 0; d < state_dim_; ++d) {
      rmse[d] = std::sqrt(mse[d]);
    }
    std::cout << "Training Data Fitting Errors:" << std::endl;
    for (size_t d = 0; d < state_dim_; ++d) {
      std::cout << " State " << d << ": RMSE = " << rmse[d]
                << ", MAE = " << mae[d] << std::endl;
    }

    // on test data
    if (!state_test_data_.empty()) {
      x_curr = state_test_data_[0];
      state_pred_history_.push_back(x_curr);
      for (size_t k = 0; k < state_test_data_.size() - 1; ++k) {
        double dt = time_test_data_[k + 1] - time_test_data_[k];
        if (dt <= 0)
          dt = 1e-4;
        Eigen::VectorXd u = control_test_data_.empty() ? Eigen::VectorXd()
                                                       : control_test_data_[k];

        x_curr = rk4_simulator_.step(x_curr, u, flat_params, dt);
        state_pred_history_.push_back(x_curr);
      }
    }

    // Statistics on test data: RMSE, MAE and t-RMSE for each dimension
    if (!state_test_data_.empty()) {
      Eigen::VectorXd mse_test(state_dim_);
      mse_test.setZero();
      Eigen::VectorXd mae_test(state_dim_);
      mae_test.setZero();
      for (size_t k = 0; k < state_test_data_.size(); ++k) {
        Eigen::VectorXd error =
            state_test_data_[k] - state_pred_history_[k + state_data_.size()];
        for (size_t d = 0; d < state_dim_; ++d) {
          mse_test[d] += error(d) * error(d);
          mae_test[d] += std::abs(error(d));
        }
        mse_test /= state_test_data_.size();
        mae_test /= state_test_data_.size();
      }
      Eigen::VectorXd rmse_test(state_dim_);
      rmse_test.setZero();
      for (size_t d = 0; d < state_dim_; ++d) {
        rmse_test[d] = std::sqrt(mse_test[d]);
      }
      std::cout << "Test Data Fitting Errors:" << std::endl;
      for (size_t d = 0; d < state_dim_; ++d) {
        std::cout << " State " << d << ": RMSE = " << rmse_test[d]
                  << ", MAE = " << mae_test[d] << std::endl;
      }
    }
  }
};

// ==========================================
// Least Square Parameter Estimator
// ==========================================
class LSParameterEstimator : public ParamEstimator {
public:
  LSParameterEstimator(NonlinearSystem *nl_system)
      : ParamEstimator(nl_system) {}

  void estimate(const std::vector<bool> &active_state_mask) override {
    if (state_data_.size() < 3)
      throw std::runtime_error("Not enough data.");

    auto current_params = nl_system_->getParameters();
    size_t N_data = state_data_.size();

    // 1. Compute numerical derivatives
    std::vector<Eigen::VectorXd> x_dot_data =
        computeNumericalDerivatives(state_dim_, N_data);

    // 2. Least Squares Fitting
    for (size_t i = 0; i < state_dim_; ++i) {
      if (!active_state_mask[i])
        continue;

      size_t n_samples = N_data - 2;
      size_t n_params = current_params[i].size();

      Eigen::MatrixXd AT(n_params, n_samples);
      Eigen::VectorXd b(n_samples);

      for (size_t j = 0; j < n_samples; ++j) {
        size_t data_idx = j + 1;
        Eigen::VectorXd x_sample = state_data_[data_idx];
        Eigen::VectorXd u_sample = (control_data_.empty())
                                       ? Eigen::VectorXd()
                                       : control_data_[data_idx];
        AT.col(j) = getObservationVector(x_sample, u_sample, i);
        b(j) = x_dot_data[i](j);
      }

      Eigen::MatrixXd ATA = AT * AT.transpose();
      Eigen::VectorXd ATb = AT * b;
      ATA.diagonal().array() += 1e-1; // Ridge regularization

      Eigen::VectorXd param_estimated = ATA.ldlt().solve(ATb);

      current_params[i].resize(param_estimated.size());
      for (size_t k = 0; k < (size_t)param_estimated.size(); ++k) {
        current_params[i][k] = param_estimated(k);
      }
    }

    // Update system and generate prediction history for visualization
    nl_system_->setParameters(current_params);
    param_history_.clear();
    for (size_t k = 0; k < N_data; ++k)
      param_history_.push_back(nl_system_->getFlatParameters());
    generatePredictionHistory();
  }

private:
  Eigen::VectorXd getObservationVector(const Eigen::VectorXd &state_sample,
                                       const Eigen::VectorXd &control_sample,
                                       size_t id_state) {
    // 获取系统定义的基函数
    const auto &basis_funcs = nl_system_->getBasisFunctions(id_state);
    size_t param_count = basis_funcs.size();
    Eigen::VectorXd obs_vec(param_count);

    DM state_dm, control_dm;
    CasadiUtils::eigen2casadi(state_sample, state_dm);

    std::vector<DM> input_args;
    if (nl_system_->getControlDim() > 0) {
      CasadiUtils::eigen2casadi(control_sample, control_dm);
      input_args = {state_dm, control_dm};
    } else {
      input_args = {state_dm};
    }

    for (size_t i = 0; i < param_count; ++i) {
      DM result = basis_funcs[i](input_args)[0];
      obs_vec(i) = double(result);
    }
    return obs_vec;
  }

  std::vector<Eigen::VectorXd> computeNumericalDerivatives(size_t dim,
                                                           size_t N) {
    std::vector<Eigen::VectorXd> derivatives(dim);
    for (size_t i = 0; i < dim; ++i) {
      derivatives[i].resize(N - 2);
      for (size_t j = 1; j < N - 1; ++j) {
        double dt_fwd = time_data_[j + 1] - time_data_[j];
        double dt_bwd = time_data_[j] - time_data_[j - 1];
        double dx_fwd = state_data_[j + 1](i) - state_data_[j](i);
        double dx_bwd = state_data_[j](i) - state_data_[j - 1](i);
        derivatives[i](j - 1) = (dx_fwd / dt_fwd + dx_bwd / dt_bwd) / 2.0;
      }
    }
    return derivatives;
  }
};

// ==========================================
// UKF Parameter Estimator
// ==========================================
class UKFParameterEstimator : public ParamEstimator {
public:
  UKFParameterEstimator(NonlinearSystem *nl_system, double dt)
      : ParamEstimator(nl_system), dt_(dt) {}

  void initialize(const Eigen::VectorXd &x0,
                  const std::vector<std::vector<double>> &initial_params) {
    // Setup dimensions and indices based on active mask (set in estimate)
    // But for now we need to store initial guess
    current_full_params_.resize(nl_system_->getFlatParameters().size());
    int idx = 0;
    for (const auto &row : initial_params)
      for (double val : row)
        current_full_params_(idx++) = val;

    x_aug_init_ = x0;
  }

  void estimate(const std::vector<bool> &active_state_mask) override {
    setupUKF(active_state_mask);

    param_history_.clear();

    // Initial state
    x_aug_ = Eigen::VectorXd::Zero(aug_dim_);
    x_aug_.head(state_dim_) = x_aug_init_;
    for (int i = 0; i < active_param_dim_; ++i)
      x_aug_(state_dim_ + i) = current_full_params_(active_param_indices_[i]);

    // Loop through data
    for (size_t k = 0; k < state_data_.size(); ++k) {
      // Record current estimate
      param_history_.push_back(current_full_params_);

      if (k == state_data_.size() - 1)
        break;

      // Update step
      Eigen::VectorXd measurement = state_data_[k + 1];
      Eigen::VectorXd control =
          control_data_.empty() ? Eigen::VectorXd() : control_data_[k];

      updateStep(measurement, control);
    }

    updateSystemParameters();
    generatePredictionHistory();
    std::cout << "UKF Estimation Complete." << std::endl;
  }

private:
  double dt_;
  int aug_dim_, active_param_dim_;
  std::vector<int> active_param_indices_;
  Eigen::VectorXd current_full_params_;
  Eigen::VectorXd x_aug_init_;

  // UKF State
  Eigen::VectorXd x_aug_;
  Eigen::MatrixXd P_, Q_, R_;
  Eigen::VectorXd weights_m_, weights_c_;
  double lambda_;

  void setupUKF(const std::vector<bool> &mask) {
    auto params = nl_system_->getParameters();
    active_param_indices_.clear();
    int flat_idx = 0;
    for (size_t i = 0; i < state_dim_; ++i) {
      int n_p = params[i].size();
      if (mask[i]) {
        for (int k = 0; k < n_p; ++k)
          active_param_indices_.push_back(flat_idx + k);
      }
      flat_idx += n_p;
    }
    active_param_dim_ = active_param_indices_.size();
    aug_dim_ = state_dim_ + active_param_dim_;

    // UKF Parameters
    double alpha = 1e-3, beta = 2.0, kappa = 0.0;
    lambda_ = alpha * alpha * (aug_dim_ + kappa) - aug_dim_;

    weights_m_.resize(2 * aug_dim_ + 1);
    weights_c_.resize(2 * aug_dim_ + 1);
    weights_m_(0) = lambda_ / (aug_dim_ + lambda_);
    weights_c_(0) = weights_m_(0) + (1 - alpha * alpha + beta);
    for (int i = 1; i < 2 * aug_dim_ + 1; ++i) {
      weights_m_(i) = weights_c_(i) = 0.5 / (aug_dim_ + lambda_);
    }

    // Covariances
    P_ = Eigen::MatrixXd::Identity(aug_dim_, aug_dim_);
    P_.topLeftCorner(state_dim_, state_dim_) *= 1e-2;
    P_.bottomRightCorner(active_param_dim_, active_param_dim_) *= 1.0;

    Q_ = Eigen::MatrixXd::Identity(aug_dim_, aug_dim_);
    Q_.topLeftCorner(state_dim_, state_dim_) *= 1e-4;
    Q_.bottomRightCorner(active_param_dim_, active_param_dim_) *= 1e-6;

    R_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_) * 1e-2;
  }

  void updateStep(const Eigen::VectorXd &measurement,
                  const Eigen::VectorXd &control) {
    // 1. Sigma Points
    Eigen::MatrixXd sigma_points(aug_dim_, 2 * aug_dim_ + 1);
    Eigen::MatrixXd L = P_.llt().matrixL();
    double gamma = std::sqrt(aug_dim_ + lambda_);
    sigma_points.col(0) = x_aug_;
    for (int i = 0; i < aug_dim_; ++i) {
      sigma_points.col(i + 1) = x_aug_ + gamma * L.col(i);
      sigma_points.col(i + 1 + aug_dim_) = x_aug_ - gamma * L.col(i);
    }

    // 2. Prediction
    Eigen::MatrixXd sigma_pred = sigma_points;
    for (int i = 0; i < 2 * aug_dim_ + 1; ++i) {
      Eigen::VectorXd state_curr = sigma_points.col(i).head(state_dim_);
      Eigen::VectorXd active_p = sigma_points.col(i).tail(active_param_dim_);

      Eigen::VectorXd temp_full = current_full_params_;
      for (int k = 0; k < active_param_dim_; ++k)
        temp_full(active_param_indices_[k]) = active_p(k);

      Eigen::VectorXd state_next =
          rk4_simulator_.step(state_curr, control, temp_full, dt_);
      sigma_pred.col(i).head(state_dim_) = state_next;
      // Params stay same (random walk)
    }

    Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(aug_dim_);
    for (int i = 0; i < 2 * aug_dim_ + 1; ++i)
      x_pred += weights_m_(i) * sigma_pred.col(i);

    Eigen::MatrixXd P_pred = Q_;
    for (int i = 0; i < 2 * aug_dim_ + 1; ++i) {
      Eigen::VectorXd diff = sigma_pred.col(i) - x_pred;
      P_pred += weights_c_(i) * (diff * diff.transpose());
    }

    // 3. Update
    Eigen::MatrixXd Z_sigma = sigma_pred.topRows(state_dim_);
    Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(state_dim_);
    for (int i = 0; i < 2 * aug_dim_ + 1; ++i)
      z_pred += weights_m_(i) * Z_sigma.col(i);

    Eigen::MatrixXd S = R_;
    Eigen::MatrixXd Tc = Eigen::MatrixXd::Zero(aug_dim_, state_dim_);
    for (int i = 0; i < 2 * aug_dim_ + 1; ++i) {
      Eigen::VectorXd z_diff = Z_sigma.col(i) - z_pred;
      Eigen::VectorXd x_diff = sigma_pred.col(i) - x_pred;
      S += weights_c_(i) * (z_diff * z_diff.transpose());
      Tc += weights_c_(i) * (x_diff * z_diff.transpose());
    }

    Eigen::MatrixXd K = Tc * S.inverse();
    x_aug_ = x_pred + K * (measurement - z_pred);
    P_ = P_pred - K * S * K.transpose();

    // Update stored params
    Eigen::VectorXd est_active = x_aug_.tail(active_param_dim_);
    for (int k = 0; k < active_param_dim_; ++k)
      current_full_params_(active_param_indices_[k]) = est_active(k);
  }

  void updateSystemParameters() {
    std::vector<std::vector<double>> updated_params =
        nl_system_->getParameters();

    int flat_idx = 0;
    for (size_t i = 0; i < state_dim_; ++i) {
      int n_p = updated_params[i].size();
      for (int k = 0; k < n_p; ++k)
        updated_params[i][k] = current_full_params_(flat_idx++);
    }

    nl_system_->setParameters(updated_params);
  }
};

class SparseGPParameterEstimator : public ParamEstimator {
public:
  SparseGPParameterEstimator(NonlinearSystem *nl_system, double dt)
      : ParamEstimator(nl_system), dt_(dt) {
    gp_input_dim = nl_system->getStateDim() + nl_system->getControlDim();
  }
  void estimate(const std::vector<bool> &active_state_mask) override {
    // Placeholder for Sparse GP estimation logic
    auto updated_params = nl_system_->getParameters();
    auto state_mx = nl_system_->getSymState();
    auto control_mx = nl_system_->getSymControl();
    auto candidate_basis = nl_system_->getCandidateBasis();

    //
    std::vector<Eigen::VectorXd> gp_smoothed_derivatives(state_dim_);
    gp_smoothed_derivatives.resize(state_dim_);
    for(size_t i = 0; i < state_dim_; ++i) {
      gp_smoother_.reset(
          new libgp::GaussianProcess(1, "CovSum(CovSEard, CovNoise)"));
      // Initialize hyperparameters randomly
      Eigen::VectorXd params_gp(gp_smoother_->covf().get_param_dim());
      params_gp.setRandom();
      gp_smoother_->covf().set_loghyper(params_gp);

      for (size_t j = 0; j < state_data_.size(); ++j) {
        double state_val = state_data_[j](i);
        double time_val[] = {time_data_[j]};
        gp_smoother_->add_pattern(time_val, state_val);
      }

      // Optimize hyperparameters
      libgp::CG cg_optimizer;
      cg_optimizer.maximize(gp_smoother_.get(), 100, true);

      Eigen::VectorXd state_smoothed, state_variance;
      state_smoothed.resize(state_data_.size());
      state_variance.resize(state_data_.size());
      gp_smoother_->pred_diag(state_smoothed, state_variance);

      // plot smoothed vs original
      std::vector<double> t_vec(state_data_.size()), x_orig(state_data_.size()),
          x_smooth(state_data_.size());
      for (size_t j = 0; j < state_data_.size(); ++j) {
        t_vec[j] = time_data_[j];
        x_orig[j] = state_data_[j](i);
        x_smooth[j] = state_smoothed(j);
        state_data_[j](i) = state_smoothed(j); // update state data with smoothed value
      }
      CasadiUtils::plot_format_init(11.0, 8.0);
      plt::named_plot("Original", t_vec, x_orig, "b.");
      plt::named_plot("Smoothed", t_vec, x_smooth, "r-");
      plt::title("State " + std::to_string(i) + " Smoothing via Sparse GP");
      plt::xlabel("Time [s]");
      plt::ylabel("State Value");
      plt::legend();
      plt::grid(true);
      plt::show();

      // Numerical derivative computation (centered difference)
      Eigen::VectorXd derivatives;
      derivatives.resize(state_data_.size() - 2);
      for (size_t j = 1; j < state_data_.size() - 1; ++j) {
        double dt_fwd = time_data_[j + 1] - time_data_[j];
        double dt_bwd = time_data_[j] - time_data_[j - 1];
        double dx_fwd = state_data_[j + 1](i) - state_data_[j](i);
        double dx_bwd = state_data_[j](i) - state_data_[j - 1](i);
        derivatives(j - 1) = (dx_fwd / dt_fwd + dx_bwd / dt_bwd) / 2.0;
      }
      // GP smoothed derivative prediction
      gp_smoother_->pred_diag_derivative(gp_smoothed_derivatives[i]);

      // plot gp smoothed derivative vs numerical derivative
      std::vector<double> deriv_orig(state_data_.size() - 2),
          deriv_smooth(state_data_.size() - 2),
          t_deriv_vec(state_data_.size() - 2);
      for (size_t j = 1; j < state_data_.size() - 1; ++j) {
        t_deriv_vec[j - 1] = time_data_[j];
        deriv_orig[j - 1] = derivatives(j - 1);
        deriv_smooth[j - 1] = gp_smoothed_derivatives[i](j);
      }
      CasadiUtils::plot_format_init(11.0, 8.0);
      plt::named_plot("Numerical Derivative", t_deriv_vec, deriv_orig, "b.");
      plt::named_plot("GP Smoothed Derivative", t_deriv_vec, deriv_smooth,
                      "r-");
      plt::title("State " + std::to_string(i) + " Derivative Comparison");
      plt::xlabel("Time [s]");
      plt::ylabel("Derivative Value");
      plt::legend();
      plt::grid(true);
      plt::show();
    }



    for (size_t i = 0; i < state_dim_; ++i) {
      if (!active_state_mask[i])
        continue;

      // Implement Sparse GP fitting for state i
      sgp_state_space_model.reset(new SparseGaussianProcess(
          gp_input_dim, "CovSum(CovSEard, CovNoise)"));

      // Initialize hyperparameters randomly
      Eigen::VectorXd params(sgp_state_space_model->covf().get_param_dim());
      params.setRandom();
      sgp_state_space_model->covf().set_loghyper(params);

      Eigen::VectorXd derivatives;
      derivatives.resize(state_data_.size() - 2);
      for (size_t j = 1; j < state_data_.size() - 1; ++j) {
        double dt_fwd = time_data_[j + 1] - time_data_[j];
        double dt_bwd = time_data_[j] - time_data_[j - 1];
        double dx_fwd = state_data_[j + 1](i) - state_data_[j](i);
        double dx_bwd = state_data_[j](i) - state_data_[j - 1](i);
        derivatives(j - 1) = (dx_fwd / dt_fwd + dx_bwd / dt_bwd) / 2.0;
      }

      // Add training data to GP model
      for (size_t j = 1; j < state_data_.size()-1; ++j) {
        Eigen::VectorXd x_sample = state_data_[j];
        Eigen::VectorXd u_sample =
            (control_data_.empty()) ? Eigen::VectorXd() : control_data_[j];

        Eigen::VectorXd input_vec(gp_input_dim);
        input_vec.head(nl_system_->getStateDim()) = x_sample;
        if (nl_system_->getControlDim() > 0)
          input_vec.tail(nl_system_->getControlDim()) = u_sample;
        sgp_state_space_model->add_pattern(input_vec.data(),
                                           derivatives(j-1));// 注意索引对齐
      }

      // Specify inducing points
      // Calculate input bounds
      Eigen::VectorXd input_min = Eigen::VectorXd::Constant(gp_input_dim, 1e10);
      Eigen::VectorXd input_max =
          Eigen::VectorXd::Constant(gp_input_dim, -1e10);

      for (size_t j = 1; j < state_data_.size() - 1; ++j) {
        Eigen::VectorXd x_sample = state_data_[j];
        Eigen::VectorXd u_sample =
            (control_data_.empty()) ? Eigen::VectorXd() : control_data_[j];
        Eigen::VectorXd input_vec(gp_input_dim);
        input_vec.head(nl_system_->getStateDim()) = x_sample;
        if (nl_system_->getControlDim() > 0)
          input_vec.tail(nl_system_->getControlDim()) = u_sample;

        input_min = input_min.cwiseMin(input_vec);
        input_max = input_max.cwiseMax(input_vec);
      }

      // Generate inducing points (grid)
      int n_grid = 5; // Points per dimension
      static std::vector<Eigen::VectorXd> inducing_points;
      if (inducing_points.empty()) {
        std::function<void(int, Eigen::VectorXd)> generate_grid =
            [&](int d, Eigen::VectorXd pt) {
              if (d == (int)gp_input_dim) {
                inducing_points.push_back(pt);
                return;
              }
              double min_v = input_min(d);
              double max_v = input_max(d);
              double step = (n_grid > 1) ? (max_v - min_v) / (n_grid - 1) : 0.0;

              for (int k = 0; k < n_grid; ++k) {
                Eigen::VectorXd next_pt = pt;
                next_pt(d) = min_v + k * step;
                generate_grid(d + 1, next_pt);
              }
            };

        generate_grid(0, Eigen::VectorXd::Zero(gp_input_dim));
      }

      sgp_state_space_model->specify_inducingSet(inducing_points);

      libgp::GA ga_optimizer;

      ga_optimizer.maximize(sgp_state_space_model.get(), 100, true);

      // sgp_state_space_model->exportModelToYAML(("sparse_gp_model_state_" +
      // std::to_string(i) + ".yaml").c_str());
      // update gp candidate basis and parameters

      Eigen::VectorXd hyperparams =
          sgp_state_space_model->covf().get_loghyper().array().exp();
      Eigen::VectorXd lengthscales = hyperparams.head(gp_input_dim);
      std::cout << "hyperparams for state " << i << ": "
                << hyperparams.transpose() << std::endl;
      double process_covariance =
          hyperparams(gp_input_dim) * hyperparams(gp_input_dim);

      std::vector<Eigen::VectorXd> inducing_points_vec;
      auto inducing_points_mat = sgp_state_space_model->getFlatInputs();
      for (size_t idx = 0; idx < inducing_points_mat.cols(); ++idx) {
        inducing_points_vec.push_back(inducing_points_mat.col(idx));
      }
      candidate_basis[i] = CasadiUtils::buildKernelBasis(
          state_mx, control_mx, inducing_points_vec, lengthscales,
          process_covariance);

      updated_params[i].resize(sgp_state_space_model->getFlatAlpha().size());
      for (size_t k = 0;
           k < (size_t)sgp_state_space_model->getFlatAlpha().size(); ++k) {
        updated_params[i][k] = sgp_state_space_model->getFlatAlpha()(k);
      }
    }

    nl_system_->updateStructure(candidate_basis, updated_params);
    rk4_simulator_.reset(*nl_system_);
    std::cout << "Sparse GP Estimation Complete." << std::endl;
    generatePredictionHistory();
  };

private:
  double dt_;
  std::unique_ptr<SparseGaussianProcess> sgp_state_space_model;
  std::unique_ptr<GaussianProcess> gp_smoother_;
  std::vector<std::vector<double>> state_derivative_data_; // smoothed by GP
  size_t gp_input_dim;
};
// ==========================================
// Main Function
// ==========================================
int main(int argc, char const *argv[]) {
  try {
    plt::backend("TkAgg");
  } catch (...) {
  }

  try {
    std::cout << "\n\033[34m========== System Identification Framework "
                 "==========\033[0m\n"
              << std::endl;

    // 1. Define System (Lotka–Volterra)
    size_t state_dim = 2;
    std::vector<std::vector<double>> true_params = {{1.5, -1.0}, {1.0, -3.0}};

    MX x1 = MX::sym("x1");
    MX x2 = MX::sym("x2");
    MX state_sym = vertcat(x1, x2);
    std::vector<std::vector<MX>> basis = {{x1, x1 * x2}, {x1 * x2, x2}};

    NonlinearSystem system(basis, true_params, state_sym);

    // 2. Generate Data
    double dt = 0.05;
    RK4Simulator sim;
    sim.reset(system);
    Eigen::VectorXd x0(2);
    x0 << 2.0, 1.2;

    int N_sim = 800, N_train = 400, N_test = N_sim - N_train;
    std::vector<double> time_vec;
    std::vector<Eigen::VectorXd> state_hist;
    Eigen::VectorXd x_curr = x0;
    Eigen::VectorXd true_flat = system.getFlatParameters();

    for (int k = 0; k < N_sim; ++k) {
      time_vec.push_back(k * dt);
      state_hist.push_back(x_curr);
      x_curr = sim.step(x_curr, Eigen::VectorXd(), true_flat, dt);
    }

    // Split into training and test sets
    std::vector<Eigen::VectorXd> state_train(state_hist.begin(),
                                             state_hist.begin() + N_train);
    std::vector<Eigen::VectorXd> state_test(state_hist.begin() + N_train,
                                            state_hist.end());

    std::vector<double> time_train(time_vec.begin(),
                                   time_vec.begin() + N_train);
    std::vector<double> time_test(time_vec.begin() + N_train, time_vec.end());

    // Add Noise
    std::default_random_engine gen;
    std::normal_distribution<double> dist(0.0, 0.1);
    for (auto &s : state_train)
      for (int i = 0; i < s.size(); ++i)
        s(i) += dist(gen);

    // 3. Least Squares Estimation
    {
      std::cout << "\n--- Least Squares Estimation ---" << std::endl;
      LSParameterEstimator ls_est(&system);
      ls_est.setData(state_train, time_train);
      ls_est.setTestData(state_test, time_test);
      ls_est.estimate({true, true});
      ls_est.visualizeFittingResults();
    }

    // 4. UKF Estimation
    {
      std::cout << "\n--- UKF Estimation ---" << std::endl;
      // Reset system params to bad guess
      std::vector<std::vector<double>> guess = {{0.5, -0.5}, {0.5, -1.0}};
      system.setParameters(guess);

      UKFParameterEstimator ukf_est(&system, dt);
      ukf_est.setData(state_train, time_train);
      ukf_est.setTestData(state_test, time_test);
      ukf_est.initialize(x0, guess);
      ukf_est.estimate({true, true});
      ukf_est.visualizeFittingResults();
    }

    // 5. Sparse GP Estimation
    {
      std::cout << "\n--- Sparse GP Estimation ---" << std::endl;
      // Reset system params to bad guess
      std::vector<std::vector<double>> guess = {{0.5, -0.5}, {0.5, -1.0}};
      system.setParameters(guess);

      SparseGPParameterEstimator gp_est(&system, dt);
      gp_est.setData(state_train, time_train);
      gp_est.setTestData(state_test, time_test);
      gp_est.estimate({true, true});
      gp_est.visualizeFittingResults(); // Visualization not implemented
    }

    std::cout << "\n\033[34m========== Done ==========\033[0m\n" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
#include "gp.h"
#include "gp_utils.h"
#include "matplotlibcpp.h"
#include "recursive_gp.h"
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

#include "cg.h"
#include "ga.h"
#include "lbfgs.h"
#include "rprop.h"
#include "de.h"
#include "pso.h"


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

struct DataSet {
  std::vector<Eigen::VectorXd> state;
  std::vector<Eigen::VectorXd> control;
  std::vector<double> time;
};

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
  std::map<std::string, double> margins;
  margins["top"] = 0.9;
  margins["bottom"] = 0.15;
  margins["left"] = 0.15;
  margins["right"] = 0.95;
  plt::subplots_adjust(margins);
  // plt::tight_layout();
}

// Some dynamic system models

// 构造以诱导点为中心的 RBF kernel basis（返回每个 state 的 basis 列表）
std::vector<casadi::MX> buildKernelBasis(
    const casadi::MX &state_sym, const std::vector<bool> &active_state_mask,
    const casadi::MX &control_sym,
    const std::vector<Eigen::VectorXd> &inducing_points,
    const Eigen::VectorXd &lengthscales, double process_covariance) {
  using namespace casadi;
  MX state_sym_active;
  for (size_t d = 0; d < active_state_mask.size(); ++d) {
    if (active_state_mask[d] == false)
      continue;
    state_sym_active = vertcat(state_sym_active, state_sym(d));
  }
  int state_dim = state_sym_active.size1();
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
  MX z = state_sym_active;
  if (control_dim > 0) {
    z = vertcat(state_sym_active, control_sym);
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

// 保存时历到CSV文件
void save_to_csv(const std::string& filename_prefix, 
                 const std::vector<Eigen::VectorXd>& state,
                 const std::vector<Eigen::VectorXd>& control,
                 const std::vector<double>& time) {
    std::string filename = filename_prefix + ".csv";
    std::ofstream csv_file(filename);
    if (csv_file.is_open()) {
        // 自动生成表头
        csv_file << "time";
        if (!state.empty()) {
            for (int j = 0; j < state[0].size(); ++j) {
          csv_file << ",state_" << j;
            }
        }
        if (!control.empty()) {
            for (int j = 0; j < control[0].size(); ++j) {
          csv_file << ",control_" << j;
            }
        }
        csv_file << "\n";
        
        // 写入数据
        for (size_t i = 0; i < time.size(); ++i) {
            // time
            csv_file << time[i];
            
            // state (x, y, psi, u, v, r)
            if (i < state.size()) {
                for (int j = 0; j < state[i].size(); ++j) {
                    csv_file << "," << state[i](j);
                }
            }
            
            // control (input_1, input_2)
            if (i < control.size()) {
                for (int j = 0; j < control[i].size(); ++j) {
                    csv_file << "," << control[i](j);
                }
            } else {
                 csv_file << ",0,0"; // 防止越界
            }
            csv_file << "\n";
        }
        csv_file.close();
        std::cout << "Data saved to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

void save_lml_timecost_to_csv(const std::string& filename_prefix, 
                 const std::vector<double>& lmls,
                 const std::vector<double>& timecosts) {
    std::string filename = filename_prefix + "_LML_TimeCost.csv";
    std::ofstream csv_file(filename);
    if (csv_file.is_open()) {
        csv_file << "Generation_ID,LML,TimeCost\n";
        for (size_t i = 0; i < lmls.size(); ++i) {
            csv_file << i << "," << lmls[i] << "," << timecosts[i] << "\n";
        }
        csv_file.close();
        std::cout << "Data saved to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

bool verbose = true;

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

  void setData(CasadiUtils::DataSet &data_set) {
    state_data_ = data_set.state;
    control_data_ = data_set.control;
    time_data_ = data_set.time;
    validateData();
  }

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

  void setTestData(CasadiUtils::DataSet &data_set) {
    state_test_data_ = data_set.state;
    control_test_data_ = data_set.control;
    time_test_data_ = data_set.time;
    validateData();
    std::cout << "Test data set. Size: " << state_test_data_.size()
              << std::endl;
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
      std::vector<double> x_true_i(N), x_est_i(N), y_true_i(N), y_est_i(N);
      for (size_t k = 0; k < N; ++k) {
        x_true_i[k] = state_data_[k](0);
        x_est_i[k] = state_pred_history_[k](0);
        y_true_i[k] = state_data_[k](1);
        y_est_i[k] = state_pred_history_[k](1);
      }
    CasadiUtils::plot_format_init(17.0, 12.0);
      plt::named_plot("Ground Truth", x_true_i, y_true_i, "b.");
      plt::named_plot("Estimated", x_est_i, y_est_i, "r-");
      plt::title("Trajectory Fitting (Training Set)");
      plt::xlabel("x (ENU)");
      plt::ylabel("y (ENU)");
      plt::legend();
      plt::grid(true);
    plt::show();

    for (size_t i = 0; i < state_dim_; ++i) {
      std::vector<double> x_true_i(N), x_est_i(N), t_vec(N);
      for (size_t k = 0; k < N; ++k) {
        x_true_i[k] = state_data_[k](i);
        x_est_i[k] = state_pred_history_[k](i);
        t_vec[k] = time_data_[k];
      }

      CasadiUtils::plot_format_init(17.0, 12.0);
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

        CasadiUtils::plot_format_init(17.0, 12.0);
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
        CasadiUtils::plot_format_init(17.0, 12.0);
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
    }
    mse /= state_data_.size();
    mae /= state_data_.size();

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
      }
      mse_test /= state_test_data_.size();
      mae_test /= state_test_data_.size();
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
  SparseGPParameterEstimator(NonlinearSystem *nl_system, double dt, size_t clusters,
                            size_t max_iters, size_t downsample_rate, std::string optimizer_option, size_t run_id=0)
      : ParamEstimator(nl_system), dt_(dt), clusters_(clusters),
        max_iters_(max_iters), downsample_rate_(downsample_rate), optimizer_option_(optimizer_option), run_id_(run_id) {
    std::cout << "Sparse GP Parameter Estimator initialized. " << std::endl;
  }
  void estimate(const std::vector<bool> &active_state_mask) override {
    // Placeholder for Sparse GP estimation logic
    auto updated_params = nl_system_->getParameters();
    auto state_mx = nl_system_->getSymState();
    auto control_mx = nl_system_->getSymControl();
    auto candidate_basis = nl_system_->getCandidateBasis();
    size_t active_state_dim = 0;
    for(size_t i = 0; i<state_dim_; ++i) {
      if (active_state_mask[i]) {
        active_state_dim += 1;
      }
    }

/*     std::vector<Eigen::VectorXd> state_smoothed(state_data_.size());
    for (size_t j = 0; j < state_data_.size(); ++j) {
      state_smoothed[j] = state_data_[j];
    }
    std::vector<Eigen::VectorXd> gp_smoothed_derivatives(state_dim_);
    gp_smoothed_derivatives.resize(state_dim_);
    for (size_t i = 0; i < state_dim_; ++i) {

      if (!active_state_mask[i])
        continue;
      active_state_dim += 1;

      if (i == 3) {
        gp_smoother_.reset(
            new libgp::GaussianProcess(1, "CovSum(CovSEiso, CovNoise)"));
      } else {
        gp_smoother_.reset(
            new libgp::GaussianProcess(1, "CovSum(CovMatern5iso, CovNoise)"));
      }

      // Initialize hyperparameters randomly
      Eigen::VectorXd params_gp(gp_smoother_->covf().get_param_dim());
      params_gp.setZero();
      gp_smoother_->covf().set_loghyper(params_gp);

      for (size_t j = 0; j < state_data_.size(); ++j) {
        double state_val = state_data_[j](i);
        double time_val[] = {time_data_[j]};
        gp_smoother_->add_pattern(time_val, state_val);
      }

      // Optimize hyperparameters
      libgp::LBFGS cg_optimizer;
      cg_optimizer.set_tolerance(1e-3);
      cg_optimizer.maximize(gp_smoother_.get(), 20, true);


      Eigen::VectorXd state_smoothed_per_dim, state_variance;
      state_smoothed_per_dim.resize(state_data_.size());
      state_variance.resize(state_data_.size());
      gp_smoother_->pred_diag(state_smoothed_per_dim, state_variance);
      gp_smoother_->pred_diag_derivative(gp_smoothed_derivatives[i]);

      // plot smoothed vs original
      std::vector<double> t_vec(state_data_.size()), x_orig(state_data_.size()),
          x_smooth(state_data_.size());
      for (size_t j = 0; j < state_data_.size(); ++j) {
        t_vec[j] = time_data_[j];
        x_orig[j] = state_data_[j](i);
        x_smooth[j] = state_smoothed_per_dim(j);
        state_smoothed[j](i) =
            x_smooth[j]; // update state data with smoothed value
      }

      if (CasadiUtils::verbose) {
        CasadiUtils::plot_format_init(17.0, 12.0);
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
          double dx_fwd = x_orig[j + 1] - x_orig[j];
          double dx_bwd = x_orig[j] - x_orig[j - 1];
          derivatives(j - 1) = (dx_fwd / dt_fwd + dx_bwd / dt_bwd) / 2.0;
        }
        // GP smoothed derivative prediction

        // plot gp smoothed derivative vs numerical derivative

        std::vector<double> deriv_orig(state_data_.size() - 2),
            deriv_smooth(state_data_.size() - 2),
            t_deriv_vec(state_data_.size() - 2);
        for (size_t j = 1; j < state_data_.size() - 1; ++j) {
          t_deriv_vec[j - 1] = time_data_[j];
          deriv_orig[j - 1] = derivatives(j - 1);
          deriv_smooth[j - 1] = gp_smoothed_derivatives[i](j);
        }
        CasadiUtils::plot_format_init(17.0, 12.0);
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
    } */

    gp_input_dim = active_state_dim + nl_system_->getControlDim();
    std::vector<size_t> active_state_idx;
    for (size_t state_idx = 0; state_idx < state_dim_; ++state_idx) {
      if (active_state_mask[state_idx])
        active_state_idx.push_back(state_idx);
    }
    auto state_active_ =
        [&](const Eigen::VectorXd &full_state) -> Eigen::VectorXd {
      Eigen::VectorXd state_active_vec(active_state_idx.size());
      for (size_t i = 0; i < active_state_idx.size(); ++i) {
        state_active_vec(i) = full_state(active_state_idx[i]);
      }
      return state_active_vec;
    };

    for (size_t i = 0; i < state_dim_; ++i) {
      if (!active_state_mask[i])
        continue;

      // Implement Sparse GP fitting for state i
      sgp_state_space_model.reset(new SparseGaussianProcess(
          gp_input_dim, "CovSum(CovSEard, CovNoise)"));

      // Initialize hyperparameters randomly
      Eigen::VectorXd params(sgp_state_space_model->covf().get_param_dim());
      params.setZero();
      sgp_state_space_model->covf().set_loghyper(params);
      
      // Add training data to GP model
      size_t N_samples =
          static_cast<size_t>(state_smoothed_data_.size() / downsample_rate_);
      for (size_t j = 0; j < N_samples; ++j) {
        size_t data_idx = static_cast<size_t>(j * downsample_rate_);
        Eigen::VectorXd x_sample = state_active_(state_smoothed_data_[data_idx]);
        Eigen::VectorXd u_sample = (control_data_.empty())
                                       ? Eigen::VectorXd()
                                       : control_data_[data_idx];
        Eigen::VectorXd input_vec(gp_input_dim);
        input_vec.resize(gp_input_dim);
        input_vec.head(active_state_dim) = x_sample;
        if (nl_system_->getControlDim() > 0)
          input_vec.tail(nl_system_->getControlDim()) = u_sample;

        //计算残差动力学
        double residual_derivative =
            state_derivative_data_[i](data_idx) -
            nl_system_->getDynamicsInstance(state_smoothed_data_[data_idx], u_sample,
                                            i); // 注意索引对齐
        sgp_state_space_model->add_pattern(input_vec.data(),
                                           residual_derivative); // 注意索引对齐
      }

      // Specify inducing points
      static std::vector<Eigen::VectorXd> inducing_points;
      sgp_state_space_model->specify_inducingSet(inducing_points, 0, clusters_);

      std::vector<double> lml_history;
      std::vector<double> time_cost_history;

      // optimize hyperparameters with different options
      if (optimizer_option_ == "GA") {
        libgp::GA ga_optimizer;
        ga_optimizer.maximize(sgp_state_space_model.get(), max_iters_, false);
        ga_optimizer.get_lml_time_history(lml_history, time_cost_history);
      } else if (optimizer_option_ == "DE") {
        libgp::DE de_optimizer;
        de_optimizer.maximize(sgp_state_space_model.get(), max_iters_, false);
        de_optimizer.get_lml_time_history(lml_history, time_cost_history);
      } else if (optimizer_option_ == "PSO") {
        libgp::PSO pso_optimizer;
        pso_optimizer.maximize(sgp_state_space_model.get(), max_iters_, false);
        pso_optimizer.get_lml_time_history(lml_history, time_cost_history);
      } else if (optimizer_option_ == "CG") {
        libgp::CG cg_optimizer;
        cg_optimizer.maximize(sgp_state_space_model.get(), max_iters_, false);
        cg_optimizer.get_lml_time_history(lml_history, time_cost_history);
      } else {
        std::cerr << "Unknown optimizer option: " << optimizer_option_ << std::endl;
      }

      if (!lml_history.empty()) {
        CasadiUtils::plot_format_init(17.0, 12.0);
        // Plot LML History
        std::vector<double> iterations(lml_history.size());
        std::iota(iterations.begin(), iterations.end(), 0);

        plt::subplot(2, 1, 1);
        plt::named_plot("LML", iterations, lml_history, "b-");
        plt::title("Log Marginal Likelihood History (State " +
           std::to_string(i) + ")");
        plt::xlabel("Iteration");
        plt::ylabel("LML");
        plt::grid(true);

        // Plot Time Cost History
        if (!time_cost_history.empty()) {
          plt::subplot(2, 1, 2);
          plt::named_plot("Time Cost", iterations, time_cost_history, "r-");
          plt::title("Computation Time per Iteration");
          plt::xlabel("Iteration");
          plt::ylabel("Time [s]");
          plt::grid(true);
        }

        plt::tight_layout();
        plt::show();
      }

      // export lml and time cost history
      CasadiUtils::save_lml_timecost_to_csv(
          optimizer_option_ + "_RunID=" + std::to_string(run_id_) + "_State_" +
          std::to_string(i) + "_LML_TimeCost",
          lml_history, time_cost_history);

      sgp_state_space_model->exportModelToYAML((optimizer_option_ + "_sparse_gp_model_state_" +
      std::to_string(i) + "_runID="+ std::to_string(run_id_) + ".yaml").c_str());

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
      std::vector<casadi::MX> gp_basis = CasadiUtils::buildKernelBasis(
          state_mx, active_state_mask, control_mx, inducing_points_vec,
          lengthscales, process_covariance);
      candidate_basis[i].insert(candidate_basis[i].end(), gp_basis.begin(),
                                gp_basis.end());

      for (size_t k = 0;
           k < (size_t)sgp_state_space_model->getFlatAlpha().size(); ++k) {
        updated_params[i].insert(updated_params[i].end(),
                                 sgp_state_space_model->getFlatAlpha()(k));
      }
    }

    nl_system_->updateStructure(candidate_basis, updated_params);
    rk4_simulator_.reset(*nl_system_);
    std::cout << "Sparse GP Estimation Complete." << std::endl;

    // Generate prediction history
    generatePredictionHistory();
    std::vector<Eigen::VectorXd> state_pred_training_history(state_pred_history_.begin(), state_pred_history_.begin() + state_data_.size());
    std::vector<Eigen::VectorXd> state_pred_test_history(state_pred_history_.begin() + state_data_.size(), state_pred_history_.end());
    CasadiUtils::save_to_csv((optimizer_option_ + "_RunID=" +std::to_string(run_id_)+"_TrainingSet_prediction").c_str(),
         state_pred_training_history,control_data_,time_data_);
    CasadiUtils::save_to_csv((optimizer_option_ + "_RunID=" +std::to_string(run_id_)+"_TestSet_prediction").c_str(),
         state_pred_test_history,control_test_data_,time_test_data_);
    // Generate summary

  };

  void setTrainingTarget(const std::vector<Eigen::VectorXd> &state_input, const std::vector<Eigen::VectorXd> &target) {
    state_derivative_data_.resize(target.size());
    state_smoothed_data_.resize(state_input.size());
    for (size_t i = 0; i < target.size(); ++i) {
      state_derivative_data_[i] = target[i];
    }
    for (size_t i = 0; i < state_input.size(); ++i) {
      state_smoothed_data_[i] = state_input[i];
    }
  }

private:
  double dt_;
  std::unique_ptr<SparseGaussianProcess> sgp_state_space_model;
  // std::unique_ptr<GaussianProcess> gp_smoother_;
  std::vector<Eigen::VectorXd> state_derivative_data_; // smoothed by GP
  std::vector<Eigen::VectorXd> state_smoothed_data_;
  size_t gp_input_dim;
  size_t clusters_ = 50;
  size_t max_iters_ = 100;
  size_t downsample_rate_ = 5;
  std::string optimizer_option_ = "DE";
  size_t run_id_ = 0;
};

// 1. 构建 KVLCC2 动力学模型 (Ground Truth)
Function build_kvlcc2_dynamics(double &out_L_pp) {
  // 物理参数定义
  double U0 = 0.76;
  double L_pp = 7.0;
  double d = 0.455;
  // double displacement = 3.272; // Unused
  double rho = 1025.0;
  double x_G = 0.244 / L_pp;
  double x_P = -0.5;

  out_L_pp = L_pp; // 输出 L_pp 供后续绘图使用

  double m = 3.27 * 1000 / 1025 / 0.5 / L_pp / L_pp / d;
  double Iz = 0.017897;
  double X_u_dot = -0.022;
  double Y_v_dot = -0.223;
  double N_r_dot = -0.011;

  // 船体系数
  double X_vv = -0.040, X_vr = 0.002, X_rr = 0.011, X_vvvv = 0.771, X_0 = 0.022;
  double Y_v = -0.315, Y_r = 0.083, Yvvv = -1.607, Y_vvr = 0.379,
         Y_vrr = -0.391, Y_rrr = 0.008;
  double N_v = -0.137, N_r = -0.049, Nvvv = -0.030, N_vvr = -0.294,
         N_vrr = 0.055, N_rrr = -0.013;

  // 推进器参数
  double wp0 = 0.40, Dp = 0.216, HR = 0.345, eta = Dp / HR, tp = 0.220,
         C1 = 2.0;

  // 舵参数
  double epsilon = 1.09, ka = 0.5, lR = -0.710, AR = 0.0539, f_alpha = 2.747;
  double tR = 0.387, aH = 0.312, xR = -0.5 * L_pp, xH = -0.464 * L_pp;

  // 符号变量
  MX x = MX::sym("x"), y = MX::sym("y"), psi = MX::sym("psi");
  MX u = MX::sym("u"), v = MX::sym("v"), r = MX::sym("r");
  MX U = sqrt(u * u + v * v);
  MX u_ = u / U, v_ = v / U, r_ = r * L_pp / U;
  MX state_sym = vertcat(x, y, psi, u, v, r);

  MX np = MX::sym("np"), delta = MX::sym("delta");
  MX control_sym = vertcat(np, delta);

  // 力学计算
  MX force_scale = 0.5 * rho * L_pp * d * U * U;
  MX moment_scale = force_scale * L_pp;

  MX X_hull =
      force_scale * (-X_0 + X_vv * v_ * v_ + (X_vr + m - Y_v_dot) * v_ * r_ +
                     (X_rr + x_G * m) * r_ * r_ + X_vvvv * v_ * v_ * v_ * v_);
  MX Y_hull = force_scale * (Y_v * v_ + (Y_r - (m - X_u_dot) * u_) * r_ +
                             Yvvv * v_ * v_ * v_ + Y_vvr * v_ * v_ * r_ +
                             Y_vrr * v_ * r_ * r_ + Y_rrr * r_ * r_ * r_);
  MX N_hull = moment_scale * (N_v * v_ + (N_r - x_G * m * u_) * r_ +
                              Nvvv * v_ * v_ * v_ + N_vvr * v_ * v_ * r_ +
                              N_vrr * v_ * r_ * r_ + N_rrr * r_ * r_ * r_);

  // 粘性阻力 (注意：此处保留原逻辑，尽管可能存在物理上的重复计算)
  double viscosity = 1.188 * 10e-6;
  double wet_surface = 13.144;
  MX Re = U * L_pp / viscosity;
  MX Cf = 0.075 / pow(log10(Re) - 2, 2);
  MX X_viscous = -0.5 * rho * U * U * wet_surface * Cf;

  // 推进器
  MX beta = atan2(-v, u); // Note: NED definition kept as per original
  MX beta_p = beta - x_P * r_;
  MX C2 = 1.35 + sign(beta_p) * 0.25;
  MX wp = 1 - (1 - wp0) * (1.0 + (1.0 - exp(-C1 * abs(beta_p))) * (C2 - 1.0));
  MX Jp = u * (1 - wp) / (np * Dp + 1e-6);
  MX KT = (0.2931 - 0.2753 * Jp - 0.1385 * Jp * Jp);
  MX Xp = (1 - tp) * rho * pow(np, 2) * pow(Dp, 4) * KT;

  // 舵
  MX u_R =
      epsilon * u * (1 - wp) *
      sqrt(eta * pow(1 + ka * (sqrt(1 + 8 * KT / M_PI / Jp / Jp) - 1.0), 2) +
           1 - eta);
  MX beta_R = beta - lR * r_;
  MX gamma_R = 0.1225 * sign(beta_R) + 0.5175;
  MX v_R = U * gamma_R * beta_R;
  MX U_R = sqrt(u_R * u_R + v_R * v_R);
  MX alpha_R = delta - v_R / u_R;
  MX FN = 0.5 * rho * AR * U_R * U_R * f_alpha * sin(alpha_R);

  MX XR = -(1 - tR) * FN * sin(delta);
  MX YR = -(1 + aH) * FN * cos(delta);
  MX NR = -(xR + aH * xH) * FN * cos(delta);

  // 质量矩阵
  Eigen::Matrix3d M_mat;
  M_mat << (m - X_u_dot) * (0.5 * rho * L_pp * L_pp * d), 0.0, 0.0,
      0.0, (m - Y_v_dot) * (0.5 * rho * L_pp * L_pp * d), x_G * m * (0.5 * rho * L_pp * L_pp * d) * L_pp,
      0.0, x_G * m * (0.5 * rho * L_pp * L_pp * d) * L_pp, (Iz - N_r_dot + x_G * x_G * m) * (0.5 * rho * L_pp * L_pp * d) * L_pp * L_pp;
  Eigen::Matrix3d Minv = M_mat.inverse();

  MX state_dot = vertcat(
      u * cos(psi) - v * sin(psi), u * sin(psi) + v * cos(psi), r,
      Minv(0, 0) *
          (X_hull + XR + Xp +
           X_viscous), // Added X_viscous back to match original logic flow
      Minv(1, 1) * (Y_hull + YR) + Minv(1, 2) * (N_hull + NR),
      Minv(2, 1) * (Y_hull + YR) + Minv(2, 2) * (N_hull + NR));

  return Function("state_dot_func", {state_sym, control_sym}, {state_dot});
}

// 2. 运行 Zig-Zag 仿真
CasadiUtils::DataSet run_sequential_zigzag_simulation(Function &state_dot_func, double dt,
                                                      const std::vector<double>& desired_heading_deg_vec,
                                                      const std::vector<double>& max_rudder_deg_vec,
                                                      int repeats_per_case) {
    if (desired_heading_deg_vec.size() != max_rudder_deg_vec.size()) {
        throw std::runtime_error("Size mismatch between heading and rudder vectors.");
    }

    CasadiUtils::DataSet data;

    // 提取符号变量维度用于初始化 Simulator
    MX state_sym = MX::sym("state", 6);
    MX control_sym = MX::sym("control", 2);

    RK4Simulator sim;
    sim.reset(state_dot_func, state_sym, control_sym);

    Eigen::VectorXd x_curr(6);
    x_curr << 0.0, 0.0, 0.0, 1.179, 0.0, 0.0;
    Eigen::VectorXd u_curr(2);
    u_curr << 17.95, 0.0; // Initial RPM and Rudder

    double time = 0.0;
    
    // 遍历每一个测试工况 (Case)
    for (size_t case_idx = 0; case_idx < desired_heading_deg_vec.size(); ++case_idx) {
        
        double target_heading_deg = desired_heading_deg_vec[case_idx];
        double max_rudder_deg = max_rudder_deg_vec[case_idx];
        
        // 转换为弧度
        double target_heading_rad = target_heading_deg * M_PI / 180.0;
        double max_rudder_rad = max_rudder_deg * M_PI / 180.0;
        
        // 物理参数
        double heading_tolerance = 0.02; // 航向切换容差 (rad)
        double delta_rate = 7.6 / 180 * M_PI; // 舵转速 (rad/s) -- KVLCC2 standard? 这里保持原有的固定值
        double np_input = 17.95; // 固定推力 RPM

        int completed_half_cycles = 0; // 记录完成了多少次半周期切换
        // repeats_per_case * 2 是因为一正一反算一次完整重复，这有两个切换动作
        // Zigzag逻辑通常是：左转 -> 达标回舵 -> 右转 -> 达标左舵
        // 传统的 20/20 Zigzag 定义通常是一次完整操作。
        // 这里定义：1次 repeat = 左舵达标 + 右舵达标 (两个半周期)
        
        int target_half_cycles = repeats_per_case * 2; 

        bool turning_left = true; // 初始向左转舵
        bool reached_switch_point = false; // 是否触发过切换逻辑
        
        // 当前阶段的目标舵角
        // 初始动作为：向左打到最大舵角
        double current_rudder_command = max_rudder_rad; 

        // 动态检测 current_heading_switch_target
        // Zigzag逻辑：检测的是 Heading 是否超过 Switch Value。
        // 标准 Zigzag (e.g. 10/10): 当 Heading > 10 deg -> Rudder = -10 deg
        double current_heading_trigger = target_heading_rad; // 初始触发点是正向 (左) Heading

        // 循环直到完成指定次数的往返
        while (completed_half_cycles < target_half_cycles) {
            
            // 1. 记录数据
            data.time.push_back(time);
            data.state.push_back(x_curr);
            data.control.push_back(u_curr);

            // 2. 状态更新
            double psi = x_curr(2);
            // 归一化 psi 到 [-pi, pi]
            while (psi > M_PI) psi -= 2 * M_PI;
            while (psi < -M_PI) psi += 2 * M_PI;

            // 3. 切换逻辑检查
            bool switch_triggered = false;
            
            if (turning_left) {
                // 当前正在向左转（建立正向航向），Command是正舵角
                // 检查航向是否超过触发阈值 (e.g. > +10 deg)
                if (psi > current_heading_trigger) {
                    switch_triggered = true;
                }
            } else {
                // 当前正在向右转（建立负向航向），Command是负舵角
                // 检查航向是否低于触发阈值 (e.g. < -10 deg)
                if (psi < current_heading_trigger) {
                    switch_triggered = true;
                }
            }

            // 执行切换
            if (switch_triggered && !reached_switch_point) {
                reached_switch_point = true;
                completed_half_cycles++;
                
                // 反转方向
                turning_left = !turning_left;
                
                if (turning_left) {
                    // 下一个目标：往左打舵，触发点变为正向 Heading
                    current_rudder_command = max_rudder_rad;
                    current_heading_trigger = target_heading_rad;
                } else {
                    // 下一个目标：往右打舵，触发点变为负向 Heading
                    current_rudder_command = -max_rudder_rad;
                    current_heading_trigger = -target_heading_rad;
                }
                
                // 如果已经完成了所有循环，立即终止当前Case
                if (completed_half_cycles >= target_half_cycles) break;
            }

            // 重置单次触发锁 (简单的滞后处理防止抖动，实际上Zigzag是大惯性，很难抖动回来)
            if (std::abs(psi - current_heading_trigger) > heading_tolerance * 5) {
                 reached_switch_point = false; 
            }

            // 4. 执行舵机控制 (一阶响应或线性变化)
            // 舵角向 Command 靠近，速率受限
            if (u_curr(1) < current_rudder_command) {
                u_curr(1) = std::min(u_curr(1) + delta_rate * dt, current_rudder_command);
            } else {
                u_curr(1) = std::max(u_curr(1) - delta_rate * dt, current_rudder_command);
            }
            
            u_curr(0) = np_input; // RPM 保持

            // 5. 物理步进
            x_curr = sim.step(x_curr, u_curr, dt);
            time += dt;
        }

        // Case 之间的过渡：回正舵并在中位保持一段时间，以便让船稳定下来？
        // 或者直接接下一个 Case。为了数据连贯性，这里让舵回中一段时间。
        double reset_duration = 10.0; // 10秒回中缓冲
        double t_reset_start = time;
        while (time - t_reset_start < reset_duration) {
             data.time.push_back(time);
             data.state.push_back(x_curr);
             data.control.push_back(u_curr);
             
             // 舵角回零
             if (u_curr(1) > 0) u_curr(1) = std::max(u_curr(1) - delta_rate * dt, 0.0);
             else u_curr(1) = std::min(u_curr(1) + delta_rate * dt, 0.0);
             
             x_curr = sim.step(x_curr, u_curr, dt);
             time += dt;
        }
    }
    
    return data;
}

CasadiUtils::DataSet run_sequential_turning_circle_simulation(Function &state_dot_func, double dt,
                                                              const std::vector<double>& max_rudder_deg_vec) {
    CasadiUtils::DataSet data;

    // 提取符号变量维度用于初始化 Simulator
    MX state_sym = MX::sym("state", 6);
    MX control_sym = MX::sym("control", 2);

    RK4Simulator sim;
    sim.reset(state_dot_func, state_sym, control_sym);

    Eigen::VectorXd x_curr(6);
    // 初始状态: [x, y, psi, u, v, r]
    x_curr << 0.0, 0.0, 0.0, 0.76, 0.0, 0.0;
    Eigen::VectorXd u_curr(2);
    u_curr << 17.95, 0.0; // [rpm, delta]

    double time = 0.0;
    double delta_rate = 7.6 / 180 * M_PI; // 舵角变化率 (rad/s)
    double np_input = 17.95;

    // 遍历每个设定的最大舵角
    for (double max_rudder_deg : max_rudder_deg_vec) {
        // 只执行左转
        // 设定目标舵角 (假设 NED 坐标，Turn Left -> Negative Delta)
        // 从逻辑一致性考虑，之前 Turning Left = true, if turn_left target = -abs(rad)
        double target_rudder_rad = -std::abs(max_rudder_deg * M_PI / 180.0);

        double start_turn_time = time + 20.0; // 20秒直航稳定
        
        bool maneuver_complete = false;
        
        // 安全机制：单次机动最大时长
        double maneuver_start_time = time;
        double accumulated_psi_change = 0.0;
        double prev_psi = x_curr(2);
        
        while (!maneuver_complete) {
            // 1. 记录
            data.time.push_back(time);
            data.state.push_back(x_curr);
            data.control.push_back(u_curr);

            // 2. 舵角控制
            if (time >= start_turn_time) {
                double d = u_curr(1);
                if (d < target_rudder_rad) {
                    u_curr(1) = std::min(d + delta_rate * dt, target_rudder_rad);
                } else {
                    u_curr(1) = std::max(d - delta_rate * dt, target_rudder_rad);
                }
            } else {
                // 直航阶段保持舵角归零（或者保持上一次的状态逐渐归零）
                if (u_curr(1) > 0) u_curr(1) = std::max(u_curr(1) - delta_rate * dt, 0.0);
                else u_curr(1) = std::min(u_curr(1) + delta_rate * dt, 0.0);
            }
            u_curr(0) = np_input;

            // 3. 状态步进
            x_curr = sim.step(x_curr, u_curr, dt);
            
            // 4. 结束判定：航向改变量 > 360度 (2*PI)
            if (time >= start_turn_time) {
                double current_psi = x_curr(2);
                double d_psi = current_psi - prev_psi;
                while (d_psi > M_PI) d_psi -= 2 * M_PI;
                while (d_psi < -M_PI) d_psi += 2 * M_PI;
                accumulated_psi_change += std::abs(d_psi);
                prev_psi = current_psi;

                if (accumulated_psi_change > 2.0 * M_PI) { // 360 degrees
                    maneuver_complete = true;
                }
            } else {
                 prev_psi = x_curr(2); //直航阶段重置基准
                 accumulated_psi_change = 0.0;
            }
            
            // 超时保护
            if (time - maneuver_start_time > 2000.0) maneuver_complete = true;

            time += dt;
        }

        // 完成一个回转后，让舵归零并稳定一段时间，以便衔接下一个
        double stable_duration = 20.0;
        double t_stable_end = time + stable_duration;
        while (time < t_stable_end) {
            data.time.push_back(time);
            data.state.push_back(x_curr);
            data.control.push_back(u_curr);

            // 舵归零
            if (u_curr(1) > 0) u_curr(1) = std::max(u_curr(1) - delta_rate * dt, 0.0);
            else u_curr(1) = std::min(u_curr(1) + delta_rate * dt, 0.0);
            
            x_curr = sim.step(x_curr, u_curr, dt);
            time += dt;
        }
    }

    return data;
}

// 3. 可视化仿真结果
void visualize_simulation(const CasadiUtils::DataSet &data, double L_pp) {
  size_t N = data.time.size();
  std::vector<double> x_vec(N), y_vec(N), psi_vec(N);
  std::vector<double> u_vec(N), v_vec(N), r_vec(N);
  std::vector<double> np_vec(N), delta_vec(N);

  for (size_t k = 0; k < N; ++k) {
    x_vec[k] = data.state[k](0) / L_pp;
    y_vec[k] = data.state[k](1) / L_pp;
    psi_vec[k] = data.state[k](2) * 57.3;
    u_vec[k] = data.state[k](3);
    v_vec[k] = data.state[k](4);
    r_vec[k] = data.state[k](5) * 57.3;
    np_vec[k] = data.control[k](0);
    delta_vec[k] = data.control[k](1) * 57.3;
  }
  if (CasadiUtils::verbose) {
    CasadiUtils::plot_format_init(17.0, 12.0);
    plt::plot(x_vec, y_vec);
    plt::xlabel("x/Lpp [-]");
    plt::ylabel("y/Lpp [-]");
    plt::grid(true);
    plt::axis("equal");
    plt::title("KVLCC2 L7 model Simulation Trajectory (Zig-Zag Maneuver)");
    plt::show();

    CasadiUtils::plot_format_init(17.0, 12.0);
    plt::plot(data.time, delta_vec);
    plt::plot(data.time, psi_vec, "--");
    plt::title("Rudder Angle and Heading vs Time");
    plt::xlabel("Time [s]");
    plt::ylabel("Rudder Angle [deg]");
    plt::grid(true);
    plt::show();

    plt::subplot(3, 1, 1);
    plt::plot(data.time, u_vec);
    plt::ylabel("u [m/s]");
    plt::grid(true);
    plt::subplot(3, 1, 2);
    plt::plot(data.time, v_vec);
    plt::ylabel("v [m/s]");
    plt::grid(true);
    plt::subplot(3, 1, 3);
    plt::plot(data.time, r_vec);
    plt::ylabel("r [deg/s]");
    plt::xlabel("Time [s]");
    plt::grid(true);

    plt::suptitle("Simulation Data History");
    plt::tight_layout();
    plt::show();
  }
}

// 4. 数据集准备
 CasadiUtils::DataSet add_noise(const CasadiUtils::DataSet &raw_data) {
  CasadiUtils::DataSet noisy_data;
  noisy_data.state = raw_data.state;
  noisy_data.control = raw_data.control;
  noisy_data.time = raw_data.time;

  // 添加噪声到训练集
  std::default_random_engine gen;
  std::normal_distribution<double> dist_pos(0.0, 0.03);
  std::normal_distribution<double> dist_vel(0.0, 0.03);
  std::normal_distribution<double> dist_deg(0.0, M_PI / 180.0 *
                                                     1.0); // 1 deg in radians
  std::normal_distribution<double> dist_rate(
      0.0, M_PI / 180.0 * 0.015); // 1 deg in radians
  bool skip_first = true;
  for (auto &s : noisy_data.state) {
    if (skip_first) {
      skip_first = false;
      continue; // skip noise for the first state
    }
    // position noise
    for (int i = 0; i < 2; ++i)
      s(i) += dist_pos(gen);
    s(2) += dist_deg(gen); // heading noise
    // velocity noise
    for (int i = 3; i < 5; ++i)
      s(i) += dist_vel(gen);
    s(5) += dist_rate(gen); // yaw rate noise
  }

  return noisy_data;
}

std::pair<CasadiUtils::DataSet, CasadiUtils::DataSet>
prepare_trainingSet(const char *filename, int downSample = 1) {
  CasadiUtils::DataSet train_set, test_set;
  CasadiUtils::DataSet raw_data;
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + std::string(filename));
  }

  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    std::vector<double> values;
    while (std::getline(ss, cell, ',')) {
      try {
        values.push_back(std::stod(cell));
      } catch (...) {
        // Ignore parsing errors (e.g. headers)
      }
    }

    if (values.size() >= 9) {
      raw_data.time.push_back(values[0]);

      // Map CSV columns to state: [x, y, psi, u, v, r]
      // CSV: time,x, y, psi, u, v, r, th_l, th_r
      Eigen::VectorXd st(6);
      st << values[1], values[2], values[3], values[4], values[5], values[6];
      raw_data.state.push_back(st);

      // Map CSV columns to control: [throttle_left, throttle_right]
      Eigen::VectorXd ctrl(2);
      ctrl << values[7], values[8];
      raw_data.control.push_back(ctrl);
    }
  }

  visualize_simulation(raw_data, 1.0);

  // 迭代器辅助
  auto s_begin = raw_data.state.begin();
  auto c_begin = raw_data.control.begin();
  auto t_begin = raw_data.time.begin();
  size_t N_total = raw_data.time.size();
  // 训练集切片
  for (int i = 0; i < raw_data.time.size(); i+= downSample) {
    train_set.state.push_back(raw_data.state[i]);
    train_set.control.push_back(raw_data.control[i]);
    train_set.time.push_back(raw_data.time[i]);
  }

  // 测试集切片
  test_set.state.assign(s_begin, raw_data.state.end());
  test_set.control.assign(c_begin, raw_data.control.end());
  test_set.time.assign(t_begin, raw_data.time.end());

  return {train_set, test_set};
}

// ==========================================
// Main Function
// ==========================================
int main(int argc, char const *argv[]) {
  // if (argc != 1) {
  //   std::cout
  //       << " Usage: ./maneuverring_example <verbose>"
  //       << std::endl;
  //   return -1;
  // }
  CasadiUtils::verbose = std::stod(argv[1]);

  try {
    plt::backend("TkAgg");
  } catch (...) {
  }

  try {
    std::cout << "\n\033[34m========== System Identification Framework "
                 "==========\033[0m\n"
              << std::endl;

    // 1. 初始化与模型构建
    double L_pp = 0; // 将在 build 函数中被赋值
    Function state_dot_func = build_kvlcc2_dynamics(L_pp);
    std::cout << "KVLCC2 Dynamics Model Built. L_pp: " << L_pp << std::endl;

    // 2. 运行仿真 (Ground Truth Generation)
    double dt = 0.2;
    int N_sim = 4000;
    CasadiUtils::DataSet sim_data_1 =
        run_sequential_turning_circle_simulation(state_dot_func, dt, {10.0,20.0,30.0});
    CasadiUtils::DataSet sim_data_2 =
        run_sequential_turning_circle_simulation(state_dot_func, dt, {25.0});

    // 3. 可视化仿真数据
    visualize_simulation(sim_data_1, L_pp);

    size_t clusters = 50;
    size_t max_iters = 50;
    size_t downsample_rate = 5;

    // 4. 数据集准备 (Train/Test Split & Noise)
    int N_skip = 1000;
    int N_train = 2000;
    auto train_data =
        add_noise(sim_data_1);
    auto test_data = sim_data_2;
    // --- 保存 TrainingSet.csv ---
    CasadiUtils::save_to_csv( "TrainingSet",train_data.state, train_data.control, train_data.time);
    CasadiUtils::save_to_csv( "TestSet",test_data.state, test_data.control, test_data.time);

    // 1. Define ship maneuvering system
    size_t state_dim = 3;
    std::vector<std::vector<double>> init_params = {
        {1.0, -1.0},
        {1.0, 1.0},
        {1.0},
        {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
        {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
        {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}};

    MX x = MX::sym("x");
    MX y = MX::sym("y");
    MX psi = MX::sym("psi");
    MX u = MX::sym("u");
    MX v = MX::sym("v");
    MX r = MX::sym("r");
    MX state_sym = vertcat(x, y, psi, u, v, r);
    MX np = MX::sym("np");
    MX delta = MX::sym("delta");
    MX control_sym = vertcat(np, delta);
    std::vector<std::vector<MX>> basis = {
        {u * cos(psi), v * sin(psi)},
        {u * sin(psi), v * cos(psi)},
        {r},
        {u * u, v * v, r * r, v * r, delta * delta, v * v * r, v * r * r},
        {v, abs(v) * v, r, v * r, abs(r) * r, delta, v * delta, v * v * delta,
         r * delta, v * v * v, r * r * r},
        {v, abs(v) * v, r, v * r, abs(r) * r, delta, v * delta, v * v * delta,
         r * delta, v * v * v, r * r * r},
    };

    // Resize init_params to match the basis size
    init_params[3].assign(basis[3].size(), 0.0);
    init_params[4].assign(basis[4].size(), 0.0);
    init_params[5].assign(basis[5].size(), 0.0);

    NonlinearSystem system(basis, init_params, state_sym, control_sym);
    std::vector<std::vector<double>> optimized_params;
    // 3. Least Squares Estimation
    // {
    //   std::cout << "\n--- Least Squares Estimation ---" << std::endl;
    //   LSParameterEstimator ls_est(&system);
    //   ls_est.setData(train_data.state, train_data.control, train_data.time);
    //   ls_est.setTestData(test_data.state, test_data.control, test_data.time);
    //   ls_est.estimate({false, false, false, true, true, true});
    //   ls_est.visualizeFittingResults();

    // }

    // 4. UKF Estimation
    {
      std::cout << "\n--- UKF Estimation ---" << std::endl;
      std::vector<std::vector<double>> guess = {
          {1.0, -1.0},
          {1.0, 1.0},
          {1.0},
          {0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0},
          {0.0, -0.0, -0.0, -0.0, -0.0, -0.0},
          {0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0}};
      guess[3].resize(basis[3].size(), 0.0);
      guess[4].resize(basis[4].size(), 0.0);
      guess[5].resize(basis[5].size(), 0.0);

      system.setParameters(guess);
      // Reset system params to bad guess
      UKFParameterEstimator ukf_est(&system, dt * 1);
      ukf_est.setData(train_data.state, train_data.control, train_data.time);
      ukf_est.setTestData(test_data.state, test_data.control, test_data.time);
      Eigen::VectorXd x0(state_dim);
      x0 = train_data.state[0];
      ukf_est.initialize(x0, guess);
      ukf_est.estimate({false, false, false, true, true, true});
      if (CasadiUtils::verbose)
        ukf_est.visualizeFittingResults();
      optimized_params = ukf_est.getEstimatedParameters();
    }

    // 5.1 Preprocessing of data before SGP regression：smooth state time series data with one-dimensional GP
    size_t state_dim_ = train_data.state[0].size();
    std::vector<Eigen::VectorXd> state_smoothed(train_data.state.size());
    std::vector<Eigen::VectorXd> gp_smoothed_derivatives(state_dim_);
    {
    for (size_t j = 0; j < train_data.state.size(); ++j) {
      state_smoothed[j] = train_data.state[j];
    }
    std::unique_ptr<libgp::GaussianProcess> gp_smoother_;
    for (size_t i = 3; i < state_dim_; ++i) {
      
      if (i == 3) {
        gp_smoother_.reset(
            new libgp::GaussianProcess(1, "CovSum(CovSEiso, CovNoise)"));
      } else {
        gp_smoother_.reset(
            new libgp::GaussianProcess(1, "CovSum(CovMatern5iso, CovNoise)"));
      }

      // Initialize hyperparameters randomly
      Eigen::VectorXd params_gp(gp_smoother_->covf().get_param_dim());
      params_gp.setZero();
      gp_smoother_->covf().set_loghyper(params_gp);

      for (size_t j = 0; j < train_data.state.size(); ++j) {
        double state_val = train_data.state[j](i);
        double time_val[] = {train_data.time[j]};
        gp_smoother_->add_pattern(time_val, state_val);
      }

      // Optimize hyperparameters
      libgp::LBFGS cg_optimizer;
      cg_optimizer.set_tolerance(1e-3);
      cg_optimizer.maximize(gp_smoother_.get(), 20, true);


      Eigen::VectorXd state_smoothed_per_dim, state_variance;
      state_smoothed_per_dim.resize(train_data.state.size());
      state_variance.resize(train_data.state.size());
      gp_smoother_->pred_diag(state_smoothed_per_dim, state_variance);
      gp_smoother_->pred_diag_derivative(gp_smoothed_derivatives[i]);

      // plot smoothed vs original
      std::vector<double> t_vec(train_data.state.size()), x_orig(train_data.state.size()),
          x_smooth(train_data.state.size());
      for (size_t j = 0; j < train_data.state.size(); ++j) {
        t_vec[j] = train_data.time[j];
        x_orig[j] = train_data.state[j](i);
        x_smooth[j] = state_smoothed_per_dim(j);
        state_smoothed[j](i) =
            x_smooth[j]; // update state data with smoothed value
      }

      if (CasadiUtils::verbose) {
        CasadiUtils::plot_format_init(17.0, 12.0);
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
        derivatives.resize(train_data.state.size() - 2);
        for (size_t j = 1; j < train_data.state.size() - 1; ++j) {
          double dt_fwd = train_data.time[j + 1] - train_data.time[j];
          double dt_bwd = train_data.time[j] - train_data.time[j - 1];
          double dx_fwd = x_orig[j + 1] - x_orig[j];
          double dx_bwd = x_orig[j] - x_orig[j - 1];
          derivatives(j - 1) = (dx_fwd / dt_fwd + dx_bwd / dt_bwd) / 2.0;
        }
        // GP smoothed derivative prediction

        // plot gp smoothed derivative vs numerical derivative
        std::vector<double> deriv_orig(train_data.state.size() - 2),
            deriv_smooth(train_data.state.size() - 2),
            t_deriv_vec(train_data.state.size() - 2);
        for (size_t j = 1; j < train_data.state.size() - 1; ++j) {
          t_deriv_vec[j - 1] = train_data.time[j];
          deriv_orig[j - 1] = derivatives(j - 1);
          deriv_smooth[j - 1] = gp_smoothed_derivatives[i](j);
        }
        CasadiUtils::plot_format_init(17.0, 12.0);
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
    }
    }

    // 5.2 Sparse GP Full Dynamics Estimation
    size_t max_runs = 1;
    std::vector<std::string> algorithms = { "PSO", "DE"};
    for(size_t run_id = 0; run_id < max_runs; run_id++)
    {
      for(size_t algorithm_id = 0; algorithm_id < algorithms.size(); algorithm_id++)
      {
        // Reset system params to bad guess
        optimized_params[3].resize(basis[3].size(), 0.0);
        optimized_params[4].resize(basis[4].size(), 0.0);
        optimized_params[5].resize(basis[5].size(), 0.0);
        std::cout << "\n--- Sparse GP Estimation ---" << std::endl;
        NonlinearSystem gp_system(basis, optimized_params, state_sym,
                                  control_sym);

        SparseGPParameterEstimator gp_est(&gp_system, dt, clusters, max_iters,
                                        downsample_rate, algorithms[algorithm_id], run_id);
        gp_est.setData(train_data.state, train_data.control, train_data.time);
        gp_est.setTrainingTarget(state_smoothed, gp_smoothed_derivatives);
        gp_est.setTestData(test_data.state, test_data.control, test_data.time);
        gp_est.estimate({false, false, false, true, true, true});
        if (CasadiUtils::verbose)
          gp_est.visualizeFittingResults(); // Visualization not implemented
      }
    }

    
    
    std::cout << "\n\033[34m========== Done ==========\033[0m\n" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
#ifndef PARAM_ESTIMATOR_H
#define PARAM_ESTIMATOR_H

#include "dynamics.h"
#include "utils.h"

// ==========================================
// Base Class: ParamEstimator
// ==========================================
namespace CasadiUtils {
class ParamEstimator {
public:
  ParamEstimator(NonlinearSystem *nl_system) : nl_system_(nl_system) {
    state_dim_ = nl_system_->getStateDim();
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

  virtual void estimate(const std::vector<bool> &active_state_mask) = 0;

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
      std::vector<double> x_true_i_v(N), x_est_i_v(N), t_vec(N);
      for (size_t k = 0; k < N; ++k) {
        x_true_i_v[k] = state_data_[k](i);
        x_est_i_v[k] = state_pred_history_[k](i);
        t_vec[k] = time_data_[k];
      }

      CasadiUtils::plot_format_init(17.0, 12.0);
      plt::named_plot("Ground Truth", t_vec, x_true_i_v, "b.");
      plt::named_plot("Estimated", t_vec, x_est_i_v, "r-");

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
      size_t N_test = state_test_data_.size();
      for (size_t i = 0; i < state_dim_; ++i) {
        std::vector<double> x_true_i_v(N_test), x_est_i_v(N_test),
            t_vec(N_test);
        for (size_t k = 0; k < N_test; ++k) {
          x_true_i_v[k] = state_test_data_[k](i);
          x_est_i_v[k] =
              state_pred_history_[k + N](i); // offset by training size
          t_vec[k] = time_test_data_[k];
        }

        CasadiUtils::plot_format_init(17.0, 12.0);
        plt::named_plot("Ground Truth", t_vec, x_true_i_v, "b.");
        plt::named_plot("Estimated", t_vec, x_est_i_v, "r-");

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

      std::vector<int> flat_to_dim(total_flat);
      int pos = 0;
      for (int d = 0; d < (int)state_dim_; ++d) {
        for (size_t k = 0; k < param_structure[d].size(); ++k)
          flat_to_dim[pos++] = d;
      }

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

    for (size_t k = 0; k < state_data_.size() - 1; ++k) {
      double dt = time_data_[k + 1] - time_data_[k];
      if (dt <= 0)
        dt = 1e-4;
      Eigen::VectorXd u =
          control_data_.empty() ? Eigen::VectorXd() : control_data_[k];
      x_curr = rk4_simulator_.step(x_curr, u, flat_params, dt);
      state_pred_history_.push_back(x_curr);
    }

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
// LSParameterEstimator
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

    std::vector<Eigen::VectorXd> x_dot_data =
        computeNumericalDerivatives(state_dim_, N_data);

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
      ATA.diagonal().array() += 1e-1; // Ridge

      Eigen::VectorXd param_estimated = ATA.ldlt().solve(ATb);
      current_params[i].resize(param_estimated.size());
      for (size_t k = 0; k < (size_t)param_estimated.size(); ++k) {
        current_params[i][k] = param_estimated(k);
      }
    }

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

    x_aug_ = Eigen::VectorXd::Zero(aug_dim_);
    x_aug_.head(state_dim_) = x_aug_init_;
    for (int i = 0; i < active_param_dim_; ++i)
      x_aug_(state_dim_ + i) = current_full_params_(active_param_indices_[i]);

    for (size_t k = 0; k < state_data_.size(); ++k) {
      param_history_.push_back(current_full_params_);
      if (k == state_data_.size() - 1)
        break;

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

    double alpha = 0.1, beta = 2.0, kappa = 1.0;
    lambda_ = alpha * alpha * (aug_dim_ + kappa) - aug_dim_;

    weights_m_.resize(2 * aug_dim_ + 1);
    weights_c_.resize(2 * aug_dim_ + 1);
    weights_m_(0) = lambda_ / (aug_dim_ + lambda_);
    weights_c_(0) = weights_m_(0) + (1 - alpha * alpha + beta);
    for (int i = 1; i < 2 * aug_dim_ + 1; ++i) {
      weights_m_(i) = weights_c_(i) = 0.5 / (aug_dim_ + lambda_);
    }

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
    Eigen::MatrixXd sigma_points(aug_dim_, 2 * aug_dim_ + 1);
    Eigen::MatrixXd L = P_.llt().matrixL();
    double gamma = std::sqrt(aug_dim_ + lambda_);
    sigma_points.col(0) = x_aug_;
    for (int i = 0; i < aug_dim_; ++i) {
      sigma_points.col(i + 1) = x_aug_ + gamma * L.col(i);
      sigma_points.col(i + 1 + aug_dim_) = x_aug_ - gamma * L.col(i);
    }

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
    }

    Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(aug_dim_);
    for (int i = 0; i < 2 * aug_dim_ + 1; ++i)
      x_pred += weights_m_(i) * sigma_pred.col(i);

    Eigen::MatrixXd P_pred = Q_;
    for (int i = 0; i < 2 * aug_dim_ + 1; ++i) {
      Eigen::VectorXd diff = sigma_pred.col(i) - x_pred;
      P_pred += weights_c_(i) * (diff * diff.transpose());
    }

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

// ==========================================
// SparseGPParameterEstimator
// ==========================================
class SparseGPParameterEstimator : public ParamEstimator {
public:
  SparseGPParameterEstimator(NonlinearSystem *nl_system, double dt,
                             size_t clusters, size_t max_iters,
                             size_t downsample_rate,
                             std::string optimizer_option,
                             std::string file_prefix)
      : ParamEstimator(nl_system), dt_(dt), clusters_(clusters),
        max_iters_(max_iters), downsample_rate_(downsample_rate),
        optimizer_option_(optimizer_option), file_prefix_(file_prefix) {
    std::cout << "Sparse GP Parameter Estimator initialized. " << std::endl;
  }

  void estimate(const std::vector<bool> &active_state_mask) override {
    auto updated_params = nl_system_->getParameters();
    auto state_mx = nl_system_->getSymState();
    auto control_mx = nl_system_->getSymControl();
    auto candidate_basis = nl_system_->getCandidateBasis();
    size_t active_state_dim = 0;
    for (size_t i = 0; i < state_dim_; ++i) {
      if (active_state_mask[i]) {
        active_state_dim += 1;
      }
    }

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

      sgp_state_space_model.reset(new libgp::SparseGaussianProcess(
          gp_input_dim, "CovSum(CovSEard, CovNoise)"));

      Eigen::VectorXd params(sgp_state_space_model->covf().get_param_dim());
      params.setZero();
      sgp_state_space_model->covf().set_loghyper(params);

      size_t N_samples =
          static_cast<size_t>(state_smoothed_data_.size() / downsample_rate_);
      for (size_t j = 0; j < N_samples; ++j) {
        size_t data_idx = static_cast<size_t>(j * downsample_rate_);
        Eigen::VectorXd x_sample =
            state_active_(state_smoothed_data_[data_idx]);
        Eigen::VectorXd u_sample = (control_data_.empty())
                                       ? Eigen::VectorXd()
                                       : control_data_[data_idx];
        Eigen::VectorXd input_vec(gp_input_dim);
        input_vec.head(active_state_dim) = x_sample;
        if (nl_system_->getControlDim() > 0)
          input_vec.tail(nl_system_->getControlDim()) = u_sample;

        double residual_derivative =
            state_derivative_data_[i](data_idx) -
            nl_system_->getDynamicsInstance(state_smoothed_data_[data_idx],
                                            u_sample, i);
        sgp_state_space_model->add_pattern(input_vec.data(),
                                           residual_derivative);
      }

      static std::vector<Eigen::VectorXd> inducing_points;
      sgp_state_space_model->specify_inducingSet(inducing_points, 0, clusters_);

      std::vector<double> lml_history, time_cost_history;

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
        std::cerr << "Unknown optimizer option: " << optimizer_option_
                  << std::endl;
      }

      // if (!lml_history.empty()) {
      //   CasadiUtils::plot_format_init(17.0, 12.0);
      //   std::vector<double> iterations(lml_history.size());
      //   std::iota(iterations.begin(), iterations.end(), 0);

      //   plt::subplot(2, 1, 1);
      //   plt::named_plot("LML", iterations, lml_history, "b-");
      //   plt::title("Log Marginal Likelihood History (State " +
      //   std::to_string(i) + ")"); plt::xlabel("Iteration");
      //   plt::ylabel("LML");
      //   plt::grid(true);

      //   if (!time_cost_history.empty()) {
      //     plt::subplot(2, 1, 2);
      //     plt::named_plot("Time Cost", iterations, time_cost_history, "r-");
      //     plt::title("Computation Time per Iteration");
      //     plt::xlabel("Iteration");
      //     plt::ylabel("Time [ms]");
      //     plt::grid(true);
      //   }

      //   plt::tight_layout();
      //   plt::show();
      // }
      if (!lml_history.empty() && !time_cost_history.empty()) {
        CasadiUtils::save_lml_timecost_to_csv(file_prefix_ + "_State_" +
                                                  std::to_string(i),
                                              lml_history, time_cost_history);
      }

      sgp_state_space_model->exportModelToYAML((file_prefix_ +
                                                "_sparse_gp_model_state_" +
                                                std::to_string(i) + ".yaml")
                                                   .c_str());

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

    generatePredictionHistory();
    std::vector<Eigen::VectorXd> state_pred_training_history(
        state_pred_history_.begin(),
        state_pred_history_.begin() + state_data_.size());
    std::vector<Eigen::VectorXd> state_pred_test_history(
        state_pred_history_.begin() + state_data_.size(),
        state_pred_history_.end());
    CasadiUtils::save_to_csv((file_prefix_ + "_TrainingSet_prediction").c_str(),
                             state_pred_training_history, control_data_,
                             time_data_);
    CasadiUtils::save_to_csv((file_prefix_ + "_TestSet_prediction").c_str(),
                             state_pred_test_history, control_test_data_,
                             time_test_data_);
  };

  void setTrainingTarget(const std::vector<Eigen::VectorXd> &state_input,
                         const std::vector<Eigen::VectorXd> &target) {
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
  std::unique_ptr<libgp::SparseGaussianProcess> sgp_state_space_model;
  std::vector<Eigen::VectorXd> state_derivative_data_;
  std::vector<Eigen::VectorXd> state_smoothed_data_;
  size_t gp_input_dim;
  size_t clusters_ = 50;
  size_t max_iters_ = 100;
  size_t downsample_rate_ = 5;
  std::string optimizer_option_ = "DE";
  std::string file_prefix_ = "";
};
} // namespace CasadiUtils
#endif // PARAM_ESTIMATOR_H
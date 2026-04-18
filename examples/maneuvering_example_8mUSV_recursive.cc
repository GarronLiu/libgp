#include "deps/core.h"
using namespace CasadiUtils;

MX state_sym;
MX control_sym;
// std::vector<std::unique_ptr<RecursiveGaussianProcess>> rgp_vec;
std::vector<std::unique_ptr<SparseGaussianProcess>> rgp_vec;
RK4Simulator rk4_simulator_;
auto active_state_mask = {false, false, false, true, true, true};

std::vector<std::vector<MX>> init_candidate_basis;
std::vector<std::vector<double>> init_updated_params;

void rk4_update() {
  size_t gp_input_dim = rgp_vec[0]->get_input_dim();

  auto candidate_basis = init_candidate_basis;
  auto updated_params = init_updated_params;
  for (size_t i = 0; i < rgp_vec.size(); ++i) {
    std::cout << "Loaded Recursive GP model for state " << i + 3 << std::endl;

    Eigen::VectorXd hyperparams =
        rgp_vec[i]->covf().get_loghyper().array().exp();
    Eigen::VectorXd lengthscales = hyperparams.head(gp_input_dim);
    std::cout << "hyperparams for state " << i << ": "
              << hyperparams.transpose() << std::endl;

    double process_covariance =
        hyperparams(gp_input_dim) * hyperparams(gp_input_dim);
    std::vector<Eigen::VectorXd> inducing_points_vec;

    auto inducing_points_mat = rgp_vec[i]->getFlatInputs();
    for (size_t idx = 0; idx < inducing_points_mat.cols(); ++idx) {
      inducing_points_vec.push_back(inducing_points_mat.col(idx));
    }

    std::vector<casadi::MX> state_gp_basis = CasadiUtils::buildKernelBasis(
        state_sym, active_state_mask, control_sym, inducing_points_vec,
        lengthscales, process_covariance);
    candidate_basis[i + 3].insert(candidate_basis[i + 3].end(),
                                  state_gp_basis.begin(), state_gp_basis.end());
    auto alpha_vec = rgp_vec[i]->getFlatAlpha();
    for (size_t k = 0; k < alpha_vec.size(); ++k) {
      updated_params[i + 3].insert(updated_params[i + 3].end(), alpha_vec(k));
    }
  }

  NonlinearSystem system(candidate_basis, updated_params, state_sym,
                         control_sym);

  rk4_simulator_.reset(system);

  for (size_t i = 0; i < rgp_vec.size(); ++i) {
    rk4_simulator_.initSGPModels(i + 3, rgp_vec[i].get());
  }
  std::cout << "RK4 simulator initialized with loaded sparse GP models."
            << std::endl;
}

void visualize_full_state_with_cov(
    const std::vector<Eigen::VectorXd> &state_vec,
    const std::vector<Eigen::MatrixXd> &cov_vec,
    const std::vector<Eigen::VectorXd> &true_state_vec,
    const std::vector<double> &time_vec) {
  if (true_state_vec.empty()) {
    std::cerr << "No data to visualize." << std::endl;
    return;
  }
  if (state_vec.empty()) {
    std::cerr << "No prediction history available. Run estimate() first."
              << std::endl;
    return;
  }
  if (state_vec.size() != cov_vec.size() ||
      state_vec.size() != true_state_vec.size() ||
      state_vec.size() != time_vec.size()) {
    std::cerr << "Size mismatch among state_vec, cov_vec, true_state_vec, and "
                 "time_vec."
              << std::endl;
    return;
  }
  auto state_dim = state_vec[0].size();
  // Visualize state trajectory with uncertainty On Training Data
  CasadiUtils::plot_format_init(34.0, 24.0);
  for (size_t d = 0; d < state_dim; ++d) {

    std::vector<double> t_vec(time_vec.begin(), time_vec.end());
    std::vector<double> state_mean, state_std;
    std::vector<double> state_true;

    for (size_t k = 0; k < state_vec.size(); ++k) {
      state_mean.push_back(state_vec[k](d));
      state_std.push_back(std::sqrt(cov_vec[k](d, d)));
      state_true.push_back(true_state_vec[k](d));
    }

    // Create 2x3 subplot for six states
    int nrows = 2, ncols = 3;

    int row = d / ncols;
    int col = d % ncols;
    int idx = d + 1; // subplot index starts from 1
    plt::subplot(nrows, ncols, idx);

    plt::plot(t_vec, state_mean, {{"label", "Mean"}, {"color", "blue"}});
    std::vector<double> state_mean_minus_state_std(state_mean.size());
    for (size_t i = 0; i < state_mean.size(); ++i) {
      state_mean_minus_state_std[i] = state_mean[i] - state_std[i];
    }
    std::vector<double> state_mean_plus_state_std(state_mean.size());
    for (size_t i = 0; i < state_mean.size(); ++i) {
      state_mean_plus_state_std[i] = state_mean[i] + state_std[i];
    }
    plt::fill_between(
        t_vec, state_mean_minus_state_std, state_mean_plus_state_std,
        {{"alpha", "0.2"}, {"label", "Uncertainty"}, {"color", "blue"}});

    plt::plot(t_vec, state_true, {{"label", "True"}, {"color", "green"}});
    plt::title("State " + std::to_string(d));
    plt::xlabel("Time [s]");

    plt::grid(true);

    // Show the figure after all subplots are drawn

    if (d == 0) {

      plt::suptitle("State Trajectory with Uncertainty");
    }
    if (d == state_dim - 1) {
      plt::legend();
      plt::tight_layout();
      plt::show();
    }
  }
  return;
}

void save_to_csv(const std::string &filename_prefix,
                 const std::string &experiment_name,
                 const std::vector<Eigen::VectorXd> &state_vec,
                 const std::vector<Eigen::MatrixXd> &cov_vec,
                 const std::vector<double> &time_vec) {
  std::string filename =
      filename_prefix + "/8mUSV_prediction_" + experiment_name + ".csv";
  std::ofstream csv_file(filename);
  if (csv_file.is_open()) {
    // 自动生成表头
    csv_file << "time";
    if (!state_vec.empty()) {
      for (int j = 0; j < state_vec[0].size(); ++j) {
        csv_file << ",state_" << j;
      }
    }

    if (!cov_vec.empty()) {
      for (int j = 0; j < cov_vec[0].size(); ++j) {
        csv_file << ",cov_" << j;
      }
    }
    csv_file << "\n";

    // 写入数据
    for (size_t i = 0; i < time_vec.size(); ++i) {
      // time
      csv_file << time_vec[i];

      // state (x, y, psi, u, v, r)
      if (i < state_vec.size()) {
        for (int j = 0; j < state_vec[i].size(); ++j) {
          csv_file << "," << state_vec[i](j);
        }
      }

      // covariance
      if (i < cov_vec.size()) {
        for (int j = 0; j < cov_vec[i].size(); ++j) {
          csv_file << "," << cov_vec[i](j);
        }
      } else {
        csv_file << ",0,0,0,0,0,0"; // 防止越界
      }
      csv_file << "\n";
    }
    csv_file.close();
    std::cout << "Data saved to " << filename << std::endl;
  } else {
    std::cerr << "Unable to open file: " << filename << std::endl;
  }
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

  std::string experiment_name = "8mUSV_recursive";
  std::string ws_path =
      (argc > 2) ? argv[2]
                 : "/home/garronliu/2_Tracking_control/GP-MPC/libgp-master/result/8mUSV_Recursive/ZigZag10deg";
  std::string model_path = (argc > 3)
                               ? argv[3]
                               : "/home/garronliu/2_Tracking_control/"
                                 "GP-MPC/libgp-master/result/8mUSV_Recursive";
  try {
    plt::backend("TkAgg");
  } catch (...) {
  }
  try {
    std::cout << "\n\033[34m========== System Identification Framework "
                 "==========\033[0m\n"
              << std::endl;

    // 1. 从文件中读取数据集
    std::string train_set_path = ws_path + "/TrainSet.csv";
    auto train_data = prepareDataSet(train_set_path.c_str(), 1);
    double dt = train_data.time[1] - train_data.time[0];
    dt = std::max(1e-4, dt);
    if (CasadiUtils::verbose) {
      visualize_simulation(train_data);
    }

    // 2. Define ship maneuvering system
    size_t state_dim = 3;
    init_updated_params = {
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
    state_sym = vertcat(x, y, psi, u, v, r);
    MX tl = MX::sym("tl");
    MX tr = MX::sym("tr");
    control_sym = vertcat(tl, tr);
    init_candidate_basis = {
        {u * cos(psi), v * sin(psi)},
        {u * sin(psi), v * cos(psi)},
        {r},
        {u * u, v * v, r * r, v * r, tr * tr, v * v * r, v * r * r},
        {v, abs(v) * v, r, v * r, abs(r) * r, tr, v * tr, v * v * tr, r * tr,
         v * v * v, r * r * r},
        {v, abs(v) * v, r, v * r, abs(r) * r, tr, v * tr, v * v * tr, r * tr,
         v * v * v, r * r * r},
    };

    // Resize init_params to match the basis size
    init_updated_params[3].assign(init_candidate_basis[3].size(), 0.0);
    init_updated_params[4].assign(init_candidate_basis[4].size(), 0.0);
    init_updated_params[5].assign(init_candidate_basis[5].size(), 0.0);

    // 3. 加载预训练SGP模型
    rgp_vec.clear();
    rgp_vec.resize(3);
    std::vector<std::string> model_files = {
        model_path + "/8mUSV_GA_RandomID=0_sparse_gp_model_state_3.yaml",
        model_path + "/8mUSV_GA_RandomID=0_sparse_gp_model_state_4.yaml",
        model_path + "/8mUSV_GA_RandomID=0_sparse_gp_model_state_5.yaml"};
    for (size_t i = 0; i < model_files.size(); ++i) {
      // rgp_vec[i] =
      //     std::make_unique<RecursiveGaussianProcess>(model_files[i].c_str());
      // rgp_vec[i]->sampleset->clear(); // Clear the sample set to prepare for
      //                                 // online updates
      rgp_vec[i] =
          std::make_unique<SparseGaussianProcess>(model_files[i].c_str());
    }

    // 4. 构建RK4模型进行预测
    rk4_update();

    Eigen::MatrixXd state_meas_cov = []() {
      Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(6, 6);
      cov(0, 0) = std::pow(0.1, 2);                  // x
      cov(1, 1) = std::pow(0.1, 2);                  // y
      cov(2, 2) = std::pow(M_PI / 180.0, 2);         // psi
      cov(3, 3) = std::pow(0.03, 2);                 // u
      cov(4, 4) = std::pow(0.03, 2);                 // v
      cov(5, 5) = std::pow(M_PI / 180.0 * 0.015, 2); // r
      return cov;
    }(); // 定义测量状态的协方差 （考虑测量的不确定度）

    // 4.1 记录多批次数据预测结果和误差
    std::vector<double> time_vec;
    std::vector<Eigen::VectorXd> state_true_vec;
    size_t epoch_size = 500; // 每批次包含100个时间步

    // 4.1.1 无递归更新
    std::vector<Eigen::VectorXd> state_pred_vec_wo_recursive;
    std::vector<Eigen::MatrixXd> state_cov_vec_wo_recursive;
    Eigen::VectorXd state_curr = train_data.state[0];
    Eigen::MatrixXd cov_curr = state_meas_cov;
    state_pred_vec_wo_recursive.push_back(state_curr);
    state_cov_vec_wo_recursive.push_back(cov_curr);
    for (size_t k = 0; k < train_data.state.size() - 1; ++k) {
      if (k % epoch_size == 0) {
        // reset to true state at the beginning of each batch
        state_curr = train_data.state[k + 1];
        cov_curr = state_meas_cov;
      } else {
        // propagate with RK4 Simulator
        double dt = train_data.time[k + 1] - train_data.time[k];
        if (dt <= 0)
          dt = 1e-4;
        Eigen::VectorXd u = train_data.control.empty() ? Eigen::VectorXd()
                                                       : train_data.control[k];
        std::tie(state_curr, cov_curr) =
            rk4_simulator_.step_with_uncertainty(state_curr, u, dt, cov_curr);
      }
      state_pred_vec_wo_recursive.push_back(state_curr);
      state_cov_vec_wo_recursive.push_back(cov_curr);
    }
    std::cout << "Completed prediction without recursive update." << std::endl;
    visualize_full_state_with_cov(state_pred_vec_wo_recursive,
                                  state_cov_vec_wo_recursive, train_data.state,
                                  train_data.time);
    save_to_csv(ws_path, "wo_rec", state_pred_vec_wo_recursive,
                state_cov_vec_wo_recursive, train_data.time);

    // 4.1.2 有递归更新
    // 4.1.2.1 有超参更新
    {
      std::vector<Eigen::VectorXd> state_pred_vec_recursive_hyperUpdate;
      std::vector<Eigen::MatrixXd> state_cov_vec_recursive_hyperUpdate;
      state_curr = train_data.state[0];
      cov_curr = state_meas_cov;
      state_pred_vec_recursive_hyperUpdate.push_back(state_curr);
      state_cov_vec_recursive_hyperUpdate.push_back(cov_curr);
      bool pretrained_need_store =
          true; // 只在第一次进入循环时存储预训练模型的后验分布
      for (size_t k = 0; k < train_data.state.size() - 1; ++k) {
        if (k % epoch_size == 0 && k > 0) {
          // reset to true state at the beginning of each batch
          state_curr = train_data.state[k + 1];
          cov_curr = state_meas_cov;

          libgp::CG optimizer;
          optimizer.maximize(rgp_vec[0].get(), 15, 0);
          optimizer.maximize(rgp_vec[1].get(), 5, 0);
          optimizer.maximize(rgp_vec[2].get(), 10, 0);

          pretrained_need_store = true;
          rk4_update();
        } else {
          // propagate with RK4 Simulator
          double dt = train_data.time[k + 1] - train_data.time[k];
          if (dt <= 0)
            dt = 1e-4;
          Eigen::VectorXd u = train_data.control.empty()
                                  ? Eigen::VectorXd()
                                  : train_data.control[k];
          std::tie(state_curr, cov_curr) =
              rk4_simulator_.step_with_uncertainty(state_curr, u, dt, cov_curr);
        }
        state_pred_vec_recursive_hyperUpdate.push_back(state_curr);
        state_cov_vec_recursive_hyperUpdate.push_back(cov_curr);

        // 计算uvr的中心微分
        if (k > 0 && k < train_data.state.size() - 2) {
          double dt_fwd = train_data.time[k + 1] - train_data.time[k];
          double dt_bwd = train_data.time[k] - train_data.time[k - 1];
          auto dx_fwd = train_data.state[k + 1].segment(3, 3) -
                        train_data.state[k].segment(3, 3);
          auto dx_bwd = train_data.state[k].segment(3, 3) -
                        train_data.state[k - 1].segment(3, 3);
          auto uvr_derivatives = (dx_fwd / dt_fwd + dx_bwd / dt_bwd) / 2.0;
          Eigen::VectorXd gp_input(5);
          gp_input << train_data.state[k](3), train_data.state[k](4),
              train_data.state[k](5), train_data.control[k](0),
              train_data.control[k](1);

          if (pretrained_need_store) {
            for (int dim = 0; dim < 3; ++dim) {
              rgp_vec[dim]->storePosteriorPretrained();
            }
            pretrained_need_store = false;
          }
          for (int dim = 0; dim < 3; ++dim) {
            rgp_vec[dim]->add_pattern_batch(gp_input, uvr_derivatives(dim));
            // rgp_vec[dim]->add_pattern(gp_input.data(), uvr_derivatives(dim));
          }
        }
      }
      std::cout << "Completed prediction with recursive update(hyperparameter "
                   "update)."
                << std::endl;
      visualize_full_state_with_cov(state_pred_vec_recursive_hyperUpdate,
                                    state_cov_vec_recursive_hyperUpdate,
                                    train_data.state, train_data.time);
      save_to_csv(ws_path, "rec", state_pred_vec_recursive_hyperUpdate,
                  state_cov_vec_recursive_hyperUpdate, train_data.time);

      // 重新传播一次，看是否能对整段轨迹进行更好的拟合
      std::vector<Eigen::VectorXd> state_pred_vec_recursive_hyperUpdate_reprop;
      std::vector<Eigen::MatrixXd> state_cov_vec_recursive_hyperUpdate_reprop;
      state_curr = train_data.state[0];
      cov_curr = state_meas_cov;
      state_pred_vec_recursive_hyperUpdate_reprop.push_back(state_curr);
      state_cov_vec_recursive_hyperUpdate_reprop.push_back(cov_curr);

      for (size_t k = 0; k < train_data.state.size() - 1; ++k) {
        if (k % epoch_size == 0 && k > 0) {
          // reset to true state at the beginning of each batch
          state_curr = train_data.state[k + 1];
          cov_curr = state_meas_cov;
        } else {
          // propagate with RK4 Simulator
          double dt = train_data.time[k + 1] - train_data.time[k];
          if (dt <= 0)
            dt = 1e-4;
          Eigen::VectorXd u = train_data.control.empty()
                                  ? Eigen::VectorXd()
                                  : train_data.control[k];
          std::tie(state_curr, cov_curr) =
              rk4_simulator_.step_with_uncertainty(state_curr, u, dt, cov_curr);
        }
        state_pred_vec_recursive_hyperUpdate_reprop.push_back(state_curr);
        state_cov_vec_recursive_hyperUpdate_reprop.push_back(cov_curr);
      }
      std::cout << "Repropagate prediction after hyperparameter update."
                << std::endl;
      visualize_full_state_with_cov(state_pred_vec_recursive_hyperUpdate_reprop,
                                    state_cov_vec_recursive_hyperUpdate_reprop,
                                    train_data.state, train_data.time);
      save_to_csv(ws_path, "rec_reprop",
                  state_pred_vec_recursive_hyperUpdate_reprop,
                  state_cov_vec_recursive_hyperUpdate_reprop, train_data.time);
    }

    // 4.1.2.2 无超参更新
    {
      //重新加载预训练模型，确保在无超参更新的情况下进行预测
      rgp_vec.clear();
      rgp_vec.resize(3);
      for (size_t i = 0; i < model_files.size(); ++i) {
        rgp_vec[i] =
            std::make_unique<SparseGaussianProcess>(model_files[i].c_str());
      }
      rk4_update();

      std::vector<Eigen::VectorXd> state_pred_vec_recursive_wo_hyperUpdate;
      std::vector<Eigen::MatrixXd> state_cov_vec_recursive_wo_hyperUpdate;
      state_curr = train_data.state[0];
      cov_curr = state_meas_cov;
      state_pred_vec_recursive_wo_hyperUpdate.push_back(state_curr);
      state_cov_vec_recursive_wo_hyperUpdate.push_back(cov_curr);
      bool pretrained_need_store = true;
      for (size_t k = 0; k < train_data.state.size() - 1; ++k) {
        if (k % epoch_size == 0 && k > 0) {
          // reset to true state at the beginning of each batch
          state_curr = train_data.state[k + 1];
          cov_curr = state_meas_cov;

          pretrained_need_store = true;
          rk4_update();

        } else {
          // propagate with RK4 Simulator
          double dt = train_data.time[k + 1] - train_data.time[k];
          if (dt <= 0)
            dt = 1e-4;
          Eigen::VectorXd u = train_data.control.empty()
                                  ? Eigen::VectorXd()
                                  : train_data.control[k];
          std::tie(state_curr, cov_curr) =
              rk4_simulator_.step_with_uncertainty(state_curr, u, dt, cov_curr);
        }
        state_pred_vec_recursive_wo_hyperUpdate.push_back(state_curr);
        state_cov_vec_recursive_wo_hyperUpdate.push_back(cov_curr);

        // 计算uvr的中心微分
        if (k > 0 && k < train_data.state.size() - 2) {
          double dt_fwd = train_data.time[k + 1] - train_data.time[k];
          double dt_bwd = train_data.time[k] - train_data.time[k - 1];
          auto dx_fwd = train_data.state[k + 1].segment(3, 3) -
                        train_data.state[k].segment(3, 3);
          auto dx_bwd = train_data.state[k].segment(3, 3) -
                        train_data.state[k - 1].segment(3, 3);
          auto uvr_derivatives = (dx_fwd / dt_fwd + dx_bwd / dt_bwd) / 2.0;
          Eigen::VectorXd gp_input(5);
          gp_input << train_data.state[k](3), train_data.state[k](4),
              train_data.state[k](5), train_data.control[k](0),
              train_data.control[k](1);

          if (pretrained_need_store) {
            for (int dim = 0; dim < 3; ++dim) {
              rgp_vec[dim]->storePosteriorPretrained();
            }
            pretrained_need_store = false;
          }
          for (int dim = 0; dim < 3; ++dim) {
            rgp_vec[dim]->add_pattern_batch(gp_input, uvr_derivatives(dim));
          }
        }
      }
      std::cout << "Completed prediction with recursive update(without "
                   "hyperparameter update)."
                << std::endl;
      visualize_full_state_with_cov(state_pred_vec_recursive_wo_hyperUpdate,
                                    state_cov_vec_recursive_wo_hyperUpdate,
                                    train_data.state, train_data.time);
      save_to_csv(ws_path, "rec_wo_hyperUpdate",
                  state_pred_vec_recursive_wo_hyperUpdate,
                  state_cov_vec_recursive_wo_hyperUpdate, train_data.time);

      // 重新传播一次，看是否能对整段轨迹进行更好的拟合
      std::vector<Eigen::VectorXd>
          state_pred_vec_recursive_wo_hyperUpdate_reprop;
      std::vector<Eigen::MatrixXd>
          state_cov_vec_recursive_wo_hyperUpdate_reprop;
      state_curr = train_data.state[0];
      cov_curr = state_meas_cov;
      state_pred_vec_recursive_wo_hyperUpdate_reprop.push_back(state_curr);
      state_cov_vec_recursive_wo_hyperUpdate_reprop.push_back(cov_curr);

      for (size_t k = 0; k < train_data.state.size() - 1; ++k) {
        if (k % epoch_size == 0 && k > 0) {
          // reset to true state at the beginning of each batch
          state_curr = train_data.state[k + 1];
          cov_curr = state_meas_cov;
        } else {
          // propagate with RK4 Simulator
          double dt = train_data.time[k + 1] - train_data.time[k];
          if (dt <= 0)
            dt = 1e-4;
          Eigen::VectorXd u = train_data.control.empty()
                                  ? Eigen::VectorXd()
                                  : train_data.control[k];
          std::tie(state_curr, cov_curr) =
              rk4_simulator_.step_with_uncertainty(state_curr, u, dt, cov_curr);
        }
        state_pred_vec_recursive_wo_hyperUpdate_reprop.push_back(state_curr);
        state_cov_vec_recursive_wo_hyperUpdate_reprop.push_back(cov_curr);
      }
      std::cout << "Repropagate prediction after hyperparameter update."
                << std::endl;
      visualize_full_state_with_cov(
          state_pred_vec_recursive_wo_hyperUpdate_reprop,
          state_cov_vec_recursive_wo_hyperUpdate_reprop, train_data.state,
          train_data.time);
      save_to_csv(ws_path, "rec_wo_hyperUpdate_reprop",
                  state_pred_vec_recursive_wo_hyperUpdate_reprop,
                  state_cov_vec_recursive_wo_hyperUpdate_reprop, train_data.time);
    }
    std::cout << "\n\033[34m========== Done ==========\033[0m\n" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
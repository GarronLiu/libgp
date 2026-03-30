#include "deps/core.h"
using namespace CasadiUtils;

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
    // 0. 命名实验类别
    std::string experiment_name = "KVLCC2_Circle";
    // 1. 初始化与模型构建
    double L_pp = 0; // 将在 build 函数中被赋值
    Function state_dot_func = build_kvlcc2_dynamics(L_pp);
    std::cout << "KVLCC2 Dynamics Model Built. L_pp: " << L_pp << std::endl;

    // 2. 运行仿真 (Ground Truth Generation)
    double dt = 0.1;
    CasadiUtils::DataSet sim_data_1 =
        run_sequential_turning_circle_simulation(state_dot_func, dt, {15.0,20.0,30.0});
    CasadiUtils::DataSet sim_data_2 =
        run_sequential_turning_circle_simulation(state_dot_func, dt, {25.0});

    //保存未加噪的训练集
    CasadiUtils::save_to_csv("TrainingSet_wo_noise", sim_data_1.state,
                             sim_data_1.control, sim_data_1.time);

    // 3. 可视化仿真数据
    if (CasadiUtils::verbose) {
      CasadiUtils::visualize_simulation(sim_data_1, L_pp, true);
    }

    // 4. 数据集准备 (add Noise)
    auto train_data = add_noise(sim_data_1);
    auto test_data = sim_data_2;
    // --- 保存 TrainingSet.csv ---
    CasadiUtils::save_to_csv("TrainingSet_noise", train_data.state,
                             train_data.control, train_data.time);
    CasadiUtils::save_to_csv("TestSet", sim_data_2.state, sim_data_2.control,
                             sim_data_2.time);

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

    // / 5.1 Preprocessing of data before SGP regression：smooth state time
    // / series data with one-dimensional GP
    std::cout << "\n\033[34m========== State Smoothing via Sparse GP "
                 "==========\033[0m\n"
              << std::endl;
    size_t state_dim_ = train_data.state[0].size();
    std::vector<Eigen::VectorXd> state_smoothed(train_data.state.size());
    std::vector<Eigen::VectorXd> gp_smoothed_derivatives(state_dim_);
    {
      for (size_t j = 0; j < train_data.state.size(); ++j) {
        state_smoothed[j] = train_data.state[j];
      }
      std::unique_ptr<libgp::SparseGaussianProcess> gp_smoother_;
      for (size_t i = 3; i < state_dim_; ++i) {

        if (i == 3) {
          gp_smoother_.reset(new libgp::SparseGaussianProcess(
              1, "CovSum(CovSEiso, CovNoise)"));
        } else {
          gp_smoother_.reset(new libgp::SparseGaussianProcess(
              1, "CovSum(CovMatern5iso, CovNoise)"));
        }
        // 根据时间范围设置诱导点
        double time_interval = 1.0;
        double duration = train_data.time.back() - train_data.time.front();
        size_t inducing_points_num =
            static_cast<size_t>(ceil((duration) / time_interval));
        std::vector<Eigen::VectorXd> t_inducing_points;
        t_inducing_points.reserve(inducing_points_num);
        for (size_t j = 0; j < inducing_points_num; ++j) {
          double time_val = static_cast<double>(j) * time_interval + train_data.time.front();
          Eigen::VectorXd pt(1);
          pt << time_val;
          t_inducing_points.push_back(pt);
        }

        gp_smoother_->specify_inducingSet(t_inducing_points);

        // Initialize hyperparameters to zeros
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
        cg_optimizer.maximize(gp_smoother_.get(), 50, true);

        Eigen::VectorXd state_smoothed_per_dim, state_variance;
        state_smoothed_per_dim.resize(train_data.state.size());
        state_variance.resize(train_data.state.size());
        gp_smoother_->pred_diag(state_smoothed_per_dim, state_variance);
        gp_smoother_->pred_diag_derivative(gp_smoothed_derivatives[i]);

        // plot smoothed vs original
        std::vector<double> t_vec(train_data.state.size()),
            x_orig(train_data.state.size()), x_smooth(train_data.state.size());
        for (size_t j = 0; j < train_data.state.size(); ++j) {
          t_vec[j] = train_data.time[j];
          x_orig[j] = train_data.state[j](i);
          x_smooth[j] = state_smoothed_per_dim(j);
          state_smoothed[j](i) =
              x_smooth[j]; // update state data with smoothed value
        }

        //可视化平滑结果
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
          plt::named_plot("Numerical Derivative", t_deriv_vec, deriv_orig,
                          "b.");
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

    /// 保存平滑后的训练集
    CasadiUtils::save_to_csv("TrainingSet_Smoothed", state_smoothed,
                             train_data.control, train_data.time);

    /// 5.2 Sparse GP Full Dynamics Estimation
    size_t max_runs = 10;
    size_t clusters = 50;
    size_t max_iters = 50;
    size_t downsample_rate = 2;
    std::vector<std::string> algorithms = {"DE"};
    for (size_t run_id = 0; run_id < max_runs; run_id++) {
      for (size_t algorithm_id = 0; algorithm_id < algorithms.size();
           algorithm_id++) {
        // Reset system params to bad guess
        init_params[3].resize(basis[3].size(), 0.0);
        init_params[4].resize(basis[4].size(), 0.0);
        init_params[5].resize(basis[5].size(), 0.0);
        std::cout << "\n--- Sparse GP Estimation ---" << std::endl;
        NonlinearSystem gp_system(basis, init_params, state_sym, control_sym);

        std::string file_prefix = experiment_name + "_" +
                                  algorithms[algorithm_id] +
                                  "_RandomID=" + std::to_string(run_id);
        SparseGPParameterEstimator gp_est(
            &gp_system, dt, clusters, max_iters, downsample_rate,
            algorithms[algorithm_id], file_prefix);
        gp_est.setData(train_data.state, train_data.control, train_data.time);
        gp_est.setTrainingTarget(state_smoothed, gp_smoothed_derivatives);
        gp_est.setTestData(test_data.state, test_data.control, test_data.time);
        gp_est.estimate({false, false, false, true, true, true});
        if (CasadiUtils::verbose) {
          gp_est.visualizeFittingResults(); // Visualization not implemented
        }
      }
    }

    std::cout << "\n\033[34m========== Done ==========\033[0m\n" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
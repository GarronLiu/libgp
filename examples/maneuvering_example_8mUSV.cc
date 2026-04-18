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

  std::string experiment_name = "8mUSV";
  std::string dataset_path_1 =
      (argc > 2) ? argv[2]
                 : "/home/garronliu/2_Tracking_control/GP-MPC/libgp-master/"
                   "dataset/rosbag/csv/lemniscate.csv";
  std::string dataset_path_2 =
      (argc > 3) ? argv[3]
                 : "/home/garronliu/2_Tracking_control/GP-MPC/libgp-master/"
                   "dataset/rosbag/csv/zig_zag_20_20.csv";
  try {
    plt::backend("TkAgg");
  } catch (...) {
  }
  try {
    std::cout << "\n\033[34m========== System Identification Framework "
                 "==========\033[0m\n"
              << std::endl;

    // 1. 从文件中读取数据集
    auto train_data = prepareDataSet(dataset_path_1.c_str(), 5);
    double dt = train_data.time[1] - train_data.time[0];
    auto test_data = prepareDataSet(dataset_path_2.c_str(), 5);
    dt = std::max(1e-4, dt);
    if (CasadiUtils::verbose) {
      visualize_simulation(train_data);
    }

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

    // MX tl = MX::sym("tl");
    // MX tr = MX::sym("tr");
    // MX control_sym = vertcat(tl, tr);
    // std::vector<std::vector<MX>> basis = {
    //   {u * cos(psi), v * sin(psi)},
    //   {u * sin(psi), v * cos(psi)},
    //   {r},
    //   // Surge dynamics: Linear damping, Quadratic damping, Coriolis, Thrust
    //   sum {u, v * r, r * r, tl, tr},
    //   // Sway dynamics: Linear damping, Quadratic damping, Coriolis
    //   {v, r, v * abs(v), r * abs(r), u * v, u * r, abs(v) * r, abs(r) * v,
    //   tl, tr},
    //   // Yaw dynamics: Linear damping, Quadratic damping, Coriolis, Thrust
    //   difference {v, r, v * abs(v), r * abs(r), u * v, u * r, abs(v) * r,
    //   abs(r) * v, tl, tr},
    // };

    // Resize init_params to match the basis size
    init_params[3].assign(basis[3].size(), 0.0);
    init_params[4].assign(basis[4].size(), 0.0);
    init_params[5].assign(basis[5].size(), 0.0);

    NonlinearSystem system(basis, init_params, state_sym, control_sym);

    size_t state_dim_ = train_data.state[0].size();
    std::vector<Eigen::VectorXd> state_smoothed(train_data.state.size());
    std::vector<Eigen::VectorXd> gp_smoothed_derivatives(state_dim_);

    for (size_t i = 0; i < state_dim_; ++i) {
      gp_smoothed_derivatives[i] = Eigen::VectorXd::Zero(train_data.state.size());
    }

    for (size_t j = 0; j < train_data.state.size(); ++j) {
      state_smoothed[j] = train_data.state[j];
    }

    for (size_t i = 3; i < state_dim_; ++i) {
      for (size_t j = 1; j < train_data.state.size() - 1; ++j) {
      double dt_fwd = train_data.time[j + 1] - train_data.time[j];
      double dt_bwd = train_data.time[j] - train_data.time[j - 1];
      double dx_fwd = train_data.state[j + 1](i) - train_data.state[j](i);
      double dx_bwd = train_data.state[j](i) - train_data.state[j - 1](i);
      gp_smoothed_derivatives[i](j) = (dx_fwd / dt_fwd + dx_bwd / dt_bwd) / 2.0;
      }
      gp_smoothed_derivatives[i](0) = gp_smoothed_derivatives[i](1);
      gp_smoothed_derivatives[i](train_data.state.size() - 1) = gp_smoothed_derivatives[i](train_data.state.size() - 2);
    }

    size_t max_runs = 10;
    size_t clusters = 50;
    size_t max_iters = 200;
    size_t downsample_rate = 1;
    std::vector<std::string> algorithms = {"CG"};
    // 6. Sparse GP Full Dynamics Estimation
    for (size_t run_id = 0; run_id < max_runs; run_id++) {
      for (size_t algorithm_id = 0; algorithm_id < algorithms.size();
          algorithm_id++) {
        // Reset system params to bad guess
        init_params[3].resize(basis[3].size(), 0.0);
        init_params[4].resize(basis[4].size(), 0.0);
        init_params[5].resize(basis[5].size(), 0.0);
        std::cout << "\n--- Sparse GP Estimation ---" << std::endl;
        NonlinearSystem gp_system(basis, init_params, state_sym, control_sym);

        std::string file_prefix =
            experiment_name + "_" + algorithms[algorithm_id] +"_RandomID=" + std::to_string(run_id);
        SparseGPParameterEstimator gp_est(&gp_system, dt, clusters, max_iters,
                                          downsample_rate,
                                          algorithms[algorithm_id], file_prefix);
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
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

  std::string experiment_name = "8mUSV_recursive";
  std::string dataset_path_1 =
      (argc > 2) ? argv[2]
                 : "/home/garronliu/2_Tracking_control/GP-MPC/libgp-master/"
                   "dataset/rosbag/csv/zig_zag_20_20.csv";
  std::string model_path =
      (argc > 3) ? argv[3]
                 : "/home/garronliu/2_Tracking_control/GP-MPC/libgp-master/"
                   "dataset/rosbag/csv/";
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
    dt = std::max(1e-4, dt);
    if (CasadiUtils::verbose) {
      visualize_simulation(train_data);
    }

    // 2. Define ship maneuvering system
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

    // 3. 加载预训练SGP模型
    RecursiveGaussianProcess rgp_u("sgp_model.yaml");
    RecursiveGaussianProcess rgp_v("sgp_model.yaml");
    RecursiveGaussianProcess rgp_r("sgp_model.yaml");
    // 4. 构建RK4模型进行预测

    // 4.1 记录多批次数据预测结果和误差
    // 4.1.1 无递归更新

    // 4.1.2 有递归更新

    // 4.1.2.1 有超参更新

    // 4.1.2.2 无超参更新


   
    std::cout << "\n\033[34m========== Done ==========\033[0m\n" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
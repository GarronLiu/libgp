#ifndef UTILS_H
#define UTILS_H

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <casadi/casadi.hpp>

#include "evaluate.h"
#include "gp.h"
#include "gp_utils.h"
#include "matplotlibcpp.h"
#include "recursive_gp.h"
#include "sparse_gp.h"

#include "cg.h"
#include "de.h"
#include "ga.h"
#include "lbfgs.h"
#include "pso.h"
#include "rprop.h"

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

namespace CasadiUtils {
using namespace casadi;
bool verbose = true;
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

inline void casadi2eigen(const DM &casadi_vec, Eigen::MatrixXd &eigen_mat) {
  if (casadi_vec.is_empty()) {
    eigen_mat.resize(0, 0);
    return;
  }
  eigen_mat.resize(casadi_vec.size1(), casadi_vec.size2());
  for (size_t i = 0; i < (size_t)casadi_vec.size1(); ++i) {
    for (size_t j = 0; j < (size_t)casadi_vec.size2(); ++j) {
      eigen_mat(i, j) = double(casadi_vec(i, j));
    }
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
void save_to_csv(const std::string &filename_prefix,
                 const std::vector<Eigen::VectorXd> &state,
                 const std::vector<Eigen::VectorXd> &control,
                 const std::vector<double> &time) {
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

void save_lml_timecost_to_csv(const std::string &filename_prefix,
                              const std::vector<double> &lmls,
                              const std::vector<double> &timecosts) {
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
    for (int i = 3; i < 5; ++i) {
      s(i) += dist_vel(gen);
    }
    s(5) += dist_rate(gen); // yaw rate noise
  }

  return noisy_data;
}

std::pair<CasadiUtils::DataSet, CasadiUtils::DataSet>
prepare_training_test_Set(const char *filename, int downSample = 1) {
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

  // 迭代器辅助
  auto s_begin = raw_data.state.begin();
  auto c_begin = raw_data.control.begin();
  auto t_begin = raw_data.time.begin();
  size_t N_total = raw_data.time.size();
  // 训练集切片
  for (int i = 0; i < raw_data.time.size(); i += downSample) {
    train_set.state.push_back(raw_data.state[i]);
    train_set.control.push_back(raw_data.control[i]);
    train_set.time.push_back(raw_data.time[i]);
  }

  // 测试集切片
  test_set.state.assign(s_begin, raw_data.state.end());
  test_set.control.assign(c_begin, raw_data.control.end());
  test_set.time.assign(t_begin, raw_data.time.end());

  std::cout << "Total samples: " << N_total
            << ", Train samples: " << train_set.time.size()
            << ", Test samples: " << test_set.time.size() << std::endl;

  return {train_set, test_set};
}

CasadiUtils::DataSet prepareDataSet(const char *filename, int downSample = 1){
  CasadiUtils::DataSet data_set;
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

  // 迭代器辅助
  auto s_begin = raw_data.state.begin();
  auto c_begin = raw_data.control.begin();
  auto t_begin = raw_data.time.begin();
  size_t N_total = raw_data.time.size();
  // 训练集切片
  for (int i = 0; i < raw_data.time.size(); i += downSample) {
    data_set.state.push_back(raw_data.state[i]);
    data_set.control.push_back(raw_data.control[i]);
    data_set.time.push_back(raw_data.time[i]);
  }

  std::cout << "Total samples: " << N_total;

  return data_set;
}

// 3. 可视化轨迹状态
void visualize_simulation(const CasadiUtils::DataSet &data, double L_pp = 1.0,
                          bool kvlcc2 = false) {
  size_t N = data.time.size();
  std::vector<double> x_vec(N), y_vec(N), psi_vec(N);
  std::vector<double> u_vec(N), v_vec(N), r_vec(N);
  std::vector<double> control_1(N), control_2(N);

  for (size_t k = 0; k < N; ++k) {
    x_vec[k] = data.state[k](0) / L_pp;
    y_vec[k] = data.state[k](1) / L_pp;
    psi_vec[k] = data.state[k](2) * 57.3;
    u_vec[k] = data.state[k](3);
    v_vec[k] = data.state[k](4);
    r_vec[k] = data.state[k](5) * 57.32;
    control_1[k] = data.control[k](0);
    control_2[k] = data.control[k](1) * ((kvlcc2) ? 57.325 : 1.0);
  }

  if (CasadiUtils::verbose) {
    CasadiUtils::plot_format_init(17.0, 12.0);
    plt::plot(x_vec, y_vec);

    plt::xlabel((L_pp != 1.0) ? "x/Lpp [-]" : "x [m]");
    plt::ylabel((L_pp != 1.0) ? "y/Lpp [-]" : "y [m]");

    plt::grid(true);
    plt::axis("equal");
    plt::title("Ship Planar Motion Trajectory");
    plt::show();

    if (kvlcc2) {
      CasadiUtils::plot_format_init(17.0, 12.0);
      plt::plot(data.time, control_2);
      plt::plot(data.time, psi_vec, "--");
      plt::title("Rudder Angle and Heading vs Time");
      plt::xlabel("Time [s]");
      plt::ylabel("Rudder Angle [deg]");
      plt::grid(true);
      plt::show();
    }

    plt::subplot(5, 1, 1);
    plt::plot(data.time, u_vec);
    plt::ylabel("u [m/s]");
    plt::grid(true);
    plt::subplot(5, 1, 2);
    plt::plot(data.time, v_vec);
    plt::ylabel("v [m/s]");
    plt::grid(true);
    plt::subplot(5, 1, 3);
    plt::plot(data.time, r_vec);
    plt::ylabel("r [deg/s]");
    plt::xlabel("Time [s]");
    plt::grid(true);
    plt::subplot(5, 1, 4);
    plt::plot(data.time, control_1);
    plt::ylabel((kvlcc2) ? "Thruster [rps]" : "Throttle 1 Signal [-]");
    plt::grid(true);
    plt::subplot(5, 1, 5);
    plt::plot(data.time, control_2);
    plt::ylabel((kvlcc2) ? "Rudder Angle [deg]" : "Throttle 2 Signal [-]");
    plt::grid(true);

    plt::tight_layout();
    plt::show();
  }
}


// wrap single angle to [-pi, pi)
inline double wrapAngle(double a) {
  const double PI = 3.14159265358979323846;
  a = std::fmod(a + PI, 2.0 * PI);
  if (a < 0) a += 2.0 * PI;
  return a - PI;
}

// wrap angles in Eigen vector at given index (in-place)
inline void wrapAngles(Eigen::VectorXd &x, const std::vector<int> &angle_indices) {
  for (int idx : angle_indices) {
    if (idx >= 0 && idx < x.size())
      x(idx) = wrapAngle(x(idx));
  }
}

// ensure symmetric and positive definite (eigenvalue floor)
inline void regularizeCovariance(Eigen::MatrixXd &P, double eps = 1e-8) {
  // enforce symmetry
  P = 0.5 * (P + P.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(P);
  if (es.info() != Eigen::Success) {
    // fallback: add jitter
    P += eps * Eigen::MatrixXd::Identity(P.rows(), P.cols());
    return;
  }
  Eigen::VectorXd vals = es.eigenvalues();
  Eigen::MatrixXd vecs = es.eigenvectors();
  double min_eig = eps;
  for (int i = 0; i < vals.size(); ++i)
    if (vals(i) < min_eig) vals(i) = min_eig;
  P = vecs * vals.asDiagonal() * vecs.transpose();
}

} // namespace CasadiUtils

#endif // UTILS_H
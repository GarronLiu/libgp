#include "dynamics.h"
#include "utils.h"
// 1. 构建 KVLCC2 动力学模型 (Ground Truth)
namespace CasadiUtils {

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
  M_mat << (m - X_u_dot) * (0.5 * rho * L_pp * L_pp * d), 0.0, 0.0, 0.0,
      (m - Y_v_dot) * (0.5 * rho * L_pp * L_pp * d),
      x_G * m * (0.5 * rho * L_pp * L_pp * d) * L_pp, 0.0,
      x_G * m * (0.5 * rho * L_pp * L_pp * d) * L_pp,
      (Iz - N_r_dot + x_G * x_G * m) * (0.5 * rho * L_pp * L_pp * d) * L_pp *
          L_pp;
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
CasadiUtils::DataSet run_zigzag_simulation(Function &state_dot_func, double dt,
                                           int N_sim,
                                           double desired_heading_deg) {
  CasadiUtils::DataSet data;

  // 提取符号变量维度用于初始化 Simulator
  MX state_sym = MX::sym("state", 6);
  MX control_sym = MX::sym("control", 2);

  RK4Simulator sim;
  sim.reset(state_dot_func, state_sym, control_sym);

  Eigen::VectorXd x_curr(6);
  x_curr << 0.0, 0.0, 0.0, 0.76, 0.0, 0.0;
  Eigen::VectorXd u_curr(2);
  u_curr << 17.95, 0.0;

  // Zig-zag 参数
  double desired_heading = M_PI * desired_heading_deg / 180.0;
  double heading_tolerance = 0.02;
  double np_input = 17.95;
  double np_rate = 0.25;
  double delta_rate = 7.6 / 180 * M_PI;
  bool turning_left = true;
  bool reached_target = false;

  for (int k = 0; k < N_sim; ++k) {
    // 航向误差处理
    double heading_error = x_curr(2) - desired_heading;
    while (heading_error > M_PI)
      heading_error -= 2 * M_PI;
    while (heading_error < -M_PI)
      heading_error += 2 * M_PI;

    // 切换逻辑
    if (std::abs(heading_error) < heading_tolerance && !reached_target) {
      reached_target = true;
      desired_heading = -desired_heading;
      turning_left = !turning_left;
    }

    // 控制输入更新
    if (turning_left) {
      u_curr(0) = np_input;
      u_curr(1) = std::max(u_curr(1) - delta_rate * dt,
                           desired_heading); // Note: Logic kept as original
    } else {
      u_curr(0) = np_input;
      u_curr(1) = std::min(u_curr(1) + delta_rate * dt,
                           desired_heading); // Note: Logic kept as original
    }

    if (reached_target && std::abs(heading_error) > heading_tolerance * 2) {
      reached_target = false;
    }

    // 存储
    data.time.push_back(k * dt);
    data.state.push_back(x_curr);
    data.control.push_back(u_curr);

    // 步进
    x_curr = sim.step(x_curr, u_curr, dt);
  }
  return data;
}

CasadiUtils::DataSet run_sequential_zigzag_simulation(
    Function &state_dot_func, double dt,
    const std::vector<double> &desired_heading_deg_vec,
    const std::vector<double> &max_rudder_deg_vec, int repeats_per_case) {
  if (desired_heading_deg_vec.size() != max_rudder_deg_vec.size()) {
    throw std::runtime_error(
        "Size mismatch between heading and rudder vectors.");
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
  for (size_t case_idx = 0; case_idx < desired_heading_deg_vec.size();
       ++case_idx) {

    double target_heading_deg = desired_heading_deg_vec[case_idx];
    double max_rudder_deg = max_rudder_deg_vec[case_idx];

    // 转换为弧度
    double target_heading_rad = target_heading_deg * M_PI / 180.0;
    double max_rudder_rad = max_rudder_deg * M_PI / 180.0;

    // 物理参数
    double heading_tolerance = 0.02; // 航向切换容差 (rad)
    double delta_rate =
        7.6 / 180 *
        M_PI; // 舵转速 (rad/s) -- KVLCC2 standard? 这里保持原有的固定值
    double np_input = 17.95; // 固定推力 RPM

    int completed_half_cycles = 0; // 记录完成了多少次半周期切换
    // repeats_per_case * 2 是因为一正一反算一次完整重复，这有两个切换动作
    // Zigzag逻辑通常是：左转 -> 达标回舵 -> 右转 -> 达标左舵
    // 传统的 20/20 Zigzag 定义通常是一次完整操作。
    // 这里定义：1次 repeat = 左舵达标 + 右舵达标 (两个半周期)

    int target_half_cycles = repeats_per_case * 2;

    bool turning_left = true;          // 初始向左转舵
    bool reached_switch_point = false; // 是否触发过切换逻辑

    // 当前阶段的目标舵角
    // 初始动作为：向左打到最大舵角
    double current_rudder_command = max_rudder_rad;

    // 动态检测 current_heading_switch_target
    // Zigzag逻辑：检测的是 Heading 是否超过 Switch Value。
    // 标准 Zigzag (e.g. 10/10): 当 Heading > 10 deg -> Rudder = -10 deg
    double current_heading_trigger =
        target_heading_rad; // 初始触发点是正向 (左) Heading

    // 循环直到完成指定次数的往返
    while (completed_half_cycles < target_half_cycles) {

      // 1. 记录数据
      data.time.push_back(time);
      data.state.push_back(x_curr);
      data.control.push_back(u_curr);

      // 2. 状态更新
      double psi = x_curr(2);
      // 归一化 psi 到 [-pi, pi]
      while (psi > M_PI)
        psi -= 2 * M_PI;
      while (psi < -M_PI)
        psi += 2 * M_PI;

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
        if (completed_half_cycles >= target_half_cycles)
          break;
      }

      // 重置单次触发锁
      // (简单的滞后处理防止抖动，实际上Zigzag是大惯性，很难抖动回来)
      if (std::abs(psi - current_heading_trigger) > heading_tolerance * 5) {
        reached_switch_point = false;
      }

      // 4. 执行舵机控制 (一阶响应或线性变化)
      // 舵角向 Command 靠近，速率受限
      if (u_curr(1) < current_rudder_command) {
        u_curr(1) =
            std::min(u_curr(1) + delta_rate * dt, current_rudder_command);
      } else {
        u_curr(1) =
            std::max(u_curr(1) - delta_rate * dt, current_rudder_command);
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
      if (u_curr(1) > 0)
        u_curr(1) = std::max(u_curr(1) - delta_rate * dt, 0.0);
      else
        u_curr(1) = std::min(u_curr(1) + delta_rate * dt, 0.0);

      x_curr = sim.step(x_curr, u_curr, dt);
      time += dt;
    }
  }

  return data;
}

CasadiUtils::DataSet run_sequential_turning_circle_simulation(
    Function &state_dot_func, double dt,
    const std::vector<double> &max_rudder_deg_vec) {
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
    // 从逻辑一致性考虑，之前 Turning Left = true, if turn_left target =
    // -abs(rad)
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
        if (u_curr(1) > 0)
          u_curr(1) = std::max(u_curr(1) - delta_rate * dt, 0.0);
        else
          u_curr(1) = std::min(u_curr(1) + delta_rate * dt, 0.0);
      }
      u_curr(0) = np_input;

      // 3. 状态步进
      x_curr = sim.step(x_curr, u_curr, dt);

      // 4. 结束判定：航向改变量 > 360度 (2*PI)
      if (time >= start_turn_time) {
        double current_psi = x_curr(2);
        double d_psi = current_psi - prev_psi;
        while (d_psi > M_PI)
          d_psi -= 2 * M_PI;
        while (d_psi < -M_PI)
          d_psi += 2 * M_PI;
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
      if (time - maneuver_start_time > 2000.0)
        maneuver_complete = true;

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
      if (u_curr(1) > 0)
        u_curr(1) = std::max(u_curr(1) - delta_rate * dt, 0.0);
      else
        u_curr(1) = std::min(u_curr(1) + delta_rate * dt, 0.0);

      x_curr = sim.step(x_curr, u_curr, dt);
      time += dt;
    }
  }

  return data;
}

CasadiUtils::DataSet run_turning_circle_simulation(Function &state_dot_func,
                                                   double dt,
                                                   double max_rudder_deg,
                                                   bool turn_left = true) {
  CasadiUtils::DataSet data;

  // 提取符号变量维度用于初始化 Simulator
  MX state_sym = MX::sym("state", 6);
  MX control_sym = MX::sym("control", 2);

  RK4Simulator sim;
  sim.reset(state_dot_func, state_sym, control_sym);

  Eigen::VectorXd x_curr(6);
  // 初始状态: [x, y, psi, u, v, r]
  x_curr << 0.0, 0.0, 0.0, 1.179, 0.0, 0.0;
  Eigen::VectorXd u_curr(2);
  u_curr << 17.95, 0.0; // [rpm, delta]

  double target_rudder_rad = max_rudder_deg * M_PI / 180.0;
  // 如果是右转，舵角通常定义为正(STARBOARD)还是负(PORT)?
  // 文中坐标系通常 NED: z向下. 右舵产生正 yaw rate (r>0).
  // 若 KVLCC2 模型遵循常规: 右舵(Positive Delta) -> Positive r -> Positive Psi
  // change (Turn Right) 请根据您的坐标系核实。此处假设 Positive Delta = Turn
  // Right. 若 turn_left = true (Turn Left) -> Negative Delta.
  if (turn_left)
    target_rudder_rad = -std::abs(target_rudder_rad);
  else
    target_rudder_rad = std::abs(target_rudder_rad);

  double delta_rate = 7.6 / 180 * M_PI; // 舵角变化率 (rad/s)

  // 仿真参数
  double time = 0.0;
  double start_turn_time = 50.0; // 直航稳态建立时间

  // 判定结束条件：航向角改变 540度 (1.5圈)
  double prev_psi = x_curr(2);
  double accumulated_psi_change = 0.0;

  double max_sim_time = 2000.0;

  while (time < max_sim_time) {
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
    }

    // 3. 状态步进
    x_curr = sim.step(x_curr, u_curr, dt);

    // 4. 结束判定
    double current_psi = x_curr(2);
    double d_psi = current_psi - prev_psi;
    while (d_psi > M_PI)
      d_psi -= 2 * M_PI;
    while (d_psi < -M_PI)
      d_psi += 2 * M_PI;
    accumulated_psi_change += std::abs(d_psi);
    prev_psi = current_psi;

    if (accumulated_psi_change > 3.0 * M_PI && time > start_turn_time + 100) {
      break;
    }

    time += dt;
  }

  return data;
}
} // namespace CasadiUtils
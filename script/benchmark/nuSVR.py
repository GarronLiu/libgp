import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
from sklearn.multioutput import MultiOutputRegressor

import argparse
import os

def full_dynamics(state, u_ctrl, model, scaler):
    """联合运动学和SVM动力学的全状态导数"""
    # state = [x, y, psi, u, v, r]
    psi_angle = state[2]
    u = state[3]
    v = state[4]
    r = state[5]

    # 运动学 (Kinematics)
    x_dot = u * np.cos(psi_angle) - v * np.sin(psi_angle)
    y_dot = u * np.sin(psi_angle) + v * np.cos(psi_angle)
    psi_dot = r

    # 动力学 (SVM Dynamics)
    xu = np.hstack(([u, v, r], u_ctrl)).reshape(1, -1)

    xu_scaled = scaler.transform(xu)
    dyn_dot = model.predict(xu_scaled).flatten()

    return np.array([x_dot, y_dot, psi_dot, dyn_dot[0], dyn_dot[1], dyn_dot[2]])

def rk4_step_full(state, u_ctrl, dt, model, scaler):
    """全状态RK4积分"""
    k1 = full_dynamics(state, u_ctrl, model, scaler)
    k2 = full_dynamics(state + 0.5 * dt * k1, u_ctrl, model, scaler)
    k3 = full_dynamics(state + 0.5 * dt * k2, u_ctrl, model, scaler)
    k4 = full_dynamics(state + dt * k3, u_ctrl, model, scaler)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name of the model to train",
                        default="KVLCC2")
    parser.add_argument("--train_file", type=str, help="Path to the training file",
                        default=None)
    parser.add_argument("--test_file", type=str, help="Path to the testing file",
                        default=None)
    parser.add_argument("--export_dir", type=str, help="Directory to export results",
                        default=".")
    parser.add_argument("--verbose", type=bool, help="Use lemniscate dataset",
                        default=True)
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment",
                        default=None)
    if (parser.parse_args().train_file):
        df = pd.read_csv(parser.parse_args().train_file)
    else:
        print("No training file provided.")
        exit()

    # 提取运动学状态 (x, y, psi) 和 动力学状态 (u, v, r)
    X_kin = df[['state_0', 'state_1', 'state_2']].values
    X_dyn = df[['state_3', 'state_4', 'state_5']].values
    X_full = np.hstack((X_kin, X_dyn))

    U = df[['control_0', 'control_1']].values
    time = df['time'].values
    
    # Clean duplicates in time
    dt_arr_check = np.diff(time)
    unique_mask = np.append([True], dt_arr_check > 1e-6)
    time = time[unique_mask]
    X_kin = X_kin[unique_mask]
    X_dyn = X_dyn[unique_mask]
    X_full = X_full[unique_mask]
    U = U[unique_mask]
    
    dt = time[1] - time[0]

    # 剔除异常离群点 (使用基于移动中位数的绝对偏差方法)
    def remove_outliers(data, window=5):
        df_temp = pd.DataFrame(data)
        rolling_median = df_temp.rolling(window=window, center=True, min_periods=1).median()
        mad = (df_temp - rolling_median).abs().rolling(window=window, center=True, min_periods=1).median()
        threshold = 3 * mad
        outliers = (df_temp - rolling_median).abs() > threshold
        return df_temp.mask(outliers).interpolate(method='linear').bfill().ffill().values

    # X_kin = remove_outliers(X_kin)
    # X_dyn = remove_outliers(X_dyn)
    # U = remove_outliers(U)
    # 平滑时历曲线 (使用移动平均滤波)
    window_size = 5
    X_kin = pd.DataFrame(X_kin).rolling(window=window_size, min_periods=1).mean().values
    X_dyn = pd.DataFrame(X_dyn).rolling(window=window_size, min_periods=1).mean().values
    X_full = np.hstack((X_kin, X_dyn))
    U = pd.DataFrame(U).rolling(window=window_size, min_periods=1).mean().values
    # 检查时历数据
    if parser.parse_args().verbose:
        plt.figure(figsize=(12, 10))
        plt.subplot(3, 1, 1)
        plt.plot(time, X_kin)
        plt.title('Kinematic States (x, y, psi)')
        plt.legend(['x', 'y', 'psi'])
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(time, X_dyn)
        plt.title('Dynamic States (u, v, r)')
        plt.legend(['u', 'v', 'r'])
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(time, U)
        plt.title('Control Inputs')
        plt.legend(['control_0', 'control_1'])
        plt.xlabel('Time')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # 计算动力学状态导数用于训练Koopman
    X_dyn_dot = np.zeros_like(X_dyn)
    for i in range(X_dyn.shape[1]):
        X_dyn_dot[:, i] = np.gradient(X_dyn[:, i], time)
    
    # Check for NaNs or Infs that could cause SVD to fail
    mask = ~np.isnan(X_dyn_dot).any(axis=1) & ~np.isinf(X_dyn_dot).any(axis=1)
    X_dyn = X_dyn[mask]
    U = U[mask]
    X_full = X_full[mask]
    time = time[mask]
    X_dyn_dot = X_dyn_dot[mask]

    if parser.parse_args().verbose:
        plt.figure(figsize=(12, 4))
        plt.plot(time, X_dyn_dot)
        plt.title('Dynamic States Derivative (u_dot, v_dot, r_dot)')
        plt.legend(['u_dot', 'v_dot', 'r_dot'])
        plt.xlabel('Time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Check for NaNs or Infs one more time before scaler
    mask2 = ~np.isnan(X_dyn).any(axis=1) & ~np.isinf(X_dyn).any(axis=1) & ~np.isnan(X_dyn_dot).any(axis=1) & ~np.isinf(X_dyn_dot).any(axis=1)
    X_dyn = X_dyn[mask2]
    U = U[mask2]
    X_full = X_full[mask2]
    time = time[mask2]
    X_dyn_dot = X_dyn_dot[mask2]
    
    # 保证 time 微分不为 0 (解决np.gradient 除以0警告)
    dt_arr = np.diff(time)
    good_time_mask = dt_arr > 1e-6
    good_time_mask = np.append([True], good_time_mask)  # shift for size match
    
    X_dyn = X_dyn[good_time_mask]
    U = U[good_time_mask]
    X_full = X_full[good_time_mask]
    time = time[good_time_mask]
    X_dyn_dot = X_dyn_dot[good_time_mask]
    
    dt = time[1] - time[0] if len(time) > 1 else 0.1

    X_U_combined = np.hstack((X_dyn, U))
    scaler = StandardScaler().fit(X_U_combined)
    scaler.scale_[scaler.scale_ == 0.0] = 1.0
    X_U_scaled = scaler.transform(X_U_combined)

    # 采用 NuSVR: nu 参数即可预先决定支持向量的比例 (0 < nu <= 1)。这里设置 nu=0.5 意味着大约有 50% 的训练样本会成为支持向量。
    print("Training Support Vector Machine (NuSVR)...")
    svr = NuSVR(nu=0.5, C=1.0, kernel='rbf') 
    model = MultiOutputRegressor(svr)
    model.fit(X_U_scaled, X_dyn_dot)
    print("Training finished.")

    # 预测 (包含运动学积分)
    X_full_pred = np.zeros_like(X_full)
    X_full_pred[0] = X_full[0]

    for i in range(1, len(time)):
        u_current = U[i-1]
        next_state = rk4_step_full(X_full_pred[i-1], u_current, dt, model, scaler)
        # 裁剪状态范围以防止 RK4 积分爆炸
        next_state = np.clip(next_state, -1e6, 1e6)
        X_full_pred[i] = next_state

    # 绘制轨迹对比
    if parser.parse_args().verbose:
        plt.figure(figsize=(15, 10))
        
        # 1. Path Plot
        plt.subplot(2, 2, 1)
        plt.plot(X_full[:, 0], X_full[:, 1], label='True Path', linewidth=2)
        plt.plot(X_full_pred[:, 0], X_full_pred[:, 1], label='SVM + Kinematics Path', linestyle='--')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Path Integration Compare (Train Set)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # 2. u Velocity Plot
        plt.subplot(2, 2, 2)
        plt.plot(time, X_full[:, 3], label='True u', linewidth=2)
        plt.plot(time, X_full_pred[:, 3], label='Pred u', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('u (m/s)')
        plt.title('Surge Velocity (u) Compare')
        plt.legend()
        plt.grid(True)
        
        # 3. v Velocity Plot
        plt.subplot(2, 2, 3)
        plt.plot(time, X_full[:, 4], label='True v', linewidth=2)
        plt.plot(time, X_full_pred[:, 4], label='Pred v', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('v (m/s)')
        plt.title('Sway Velocity (v) Compare')
        plt.legend()
        plt.grid(True)

        # 4. r Velocity Plot
        plt.subplot(2, 2, 4)
        plt.plot(time, X_full[:, 5], label='True r', linewidth=2)
        plt.plot(time, X_full_pred[:, 5], label='Pred r', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('r (rad/s)')
        plt.title('Yaw Rate (r) Compare')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    
    # 保存koopman在训练集上的预测结果
    df_pred = pd.DataFrame(
        np.column_stack((time, X_full_pred, U)),
        columns=['time', 'state_0', 'state_1', 'state_2', 'state_3', 'state_4', 'state_5', 'control_0', 'control_1']
    )
    export_path = os.path.join(parser.parse_args().export_dir, f"{parser.parse_args().model_name}_{parser.parse_args().experiment_name}_KPM_TrainingSet_prediction.csv")
    df_pred.to_csv(export_path, index=False)
    print(f"Koopman predictions saved to {export_path}")

    # 在测试集验证
    if parser.parse_args().test_file:
        print(f"\nValidating on {parser.parse_args().test_file}...")
        df_test = pd.read_csv(parser.parse_args().test_file)
        X_kin_test = df_test[['state_0', 'state_1', 'state_2']].values
        X_dyn_test = df_test[['state_3', 'state_4', 'state_5']].values
        X_test_full = np.hstack((X_kin_test, X_dyn_test))

        U_test = df_test[['control_0', 'control_1']].values
        time_test = df_test['time'].values
        dt_test = time_test[1] - time_test[0]

        X_test_full_pred = np.zeros_like(X_test_full)
        X_test_full_pred[0] = X_test_full[0]

        for i in range(1, len(time_test)):
            u_current = U_test[i-1]
            next_state = rk4_step_full(X_test_full_pred[i-1], u_current, dt_test, model, scaler)
            next_state = np.clip(next_state, -1e6, 1e6)
            X_test_full_pred[i] = next_state
    
        # 保存SVR在测试集上的预测结果
        df_pred = pd.DataFrame(
            np.column_stack((time_test, X_test_full_pred, U_test)),
            columns=['time', 'state_0', 'state_1', 'state_2', 'state_3', 'state_4', 'state_5', 'control_0', 'control_1']
        )
        export_path = os.path.join(parser.parse_args().export_dir, f"{parser.parse_args().model_name}_{parser.parse_args().experiment_name}_KPM_TestSet_prediction.csv")
        df_pred.to_csv(export_path, index=False)
        print(f"Koopman predictions saved to {export_path}")
    
        rmse = np.sqrt(np.mean((X_test_full[:, :2] - X_test_full_pred[:, :2])**2))
        print(f"Test RMSE: {rmse:.4f}")
        state_names = ['x', 'y', 'psi', 'u', 'v', 'r']
        for j, name in enumerate(state_names):
            state_rmse = np.sqrt(np.mean((X_test_full[:, j] - X_test_full_pred[:, j])**2))
            print(f"Test RMSE for {name}: {state_rmse:.4f}")
        if parser.parse_args().verbose:
        # Overriding the title function temporarily to inject RMSE
            original_title = plt.title
            plt.title = lambda *args, **kwargs: original_title(f'Path Integration Compare (Test Set) | RMSE: {rmse:.4f}', **kwargs)
            # 绘制测试集轨迹对比
            plt.figure(figsize=(15, 10))
            
            # 1. Path Plot
            plt.subplot(2, 2, 1)
            plt.plot(X_test_full[:, 0], X_test_full[:, 1], label='Test True Path', linewidth=2)
            plt.plot(X_test_full_pred[:, 0], X_test_full_pred[:, 1], label='SVM + Kinematics Path', linestyle='--')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title('Path Integration Compare (Test Set)')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            
            # Restore title function for other plots
            plt.title = original_title

            # 2. u Velocity Plot
            plt.subplot(2, 2, 2)
            plt.plot(time_test, X_test_full[:, 3], label='True u', linewidth=2)
            plt.plot(time_test, X_test_full_pred[:, 3], label='Pred u', linestyle='--')
            plt.xlabel('Time (s)')
            plt.ylabel('u (m/s)')
            plt.title('Test Surge Velocity (u) Compare')
            plt.legend()
            plt.grid(True)
            
            # 3. v Velocity Plot
            plt.subplot(2, 2, 3)
            plt.plot(time_test, X_test_full[:, 4], label='True v', linewidth=2)
            plt.plot(time_test, X_test_full_pred[:, 4], label='Pred v', linestyle='--')
            plt.xlabel('Time (s)')
            plt.ylabel('v (m/s)')
            plt.title('Test Sway Velocity (v) Compare')
            plt.legend()
            plt.grid(True)

            # 4. r Velocity Plot
            plt.subplot(2, 2, 4)
            plt.plot(time_test, X_test_full[:, 5], label='True r', linewidth=2)
            plt.plot(time_test, X_test_full_pred[:, 5], label='Pred r', linestyle='--')
            plt.xlabel('Time (s)')
            plt.ylabel('r (rad/s)')
            plt.title('Test Yaw Rate (r) Compare')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

    # 注意：SVM 是非参数模型，无法像 Koopman 那样打印解析表达式，这里已经去除打印解析多项式代码。

if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# 配置参数
base_dir = "result/8mUSV/"
benchmarks = ["Koopman", "nuSVR($\\nu=0.2$)", "nuSVR($\\nu=0.03$)", "SGP-DE(ind_num=50)"]
prediction_files = [
    "8mUSV_None_KPM_TrainingSet_prediction.csv",
    "8mUSV_None_nuSVR_nu=2e-1_TrainingSet_prediction.csv",
    "8mUSV_None_nuSVR_nu=3e-2_TrainingSet_prediction.csv",
    "8mUSV_DE_RandomID=8_TrainingSet_prediction.csv"
]
states = [3, 4, 5]
cmap = plt.get_cmap('tab10')
colors = {benchmark: cmap(i % 10) for i, benchmark in enumerate(benchmarks)}

# 配置图保存参数
save_fig = True
fig_dir = "result/convergence_analysis_figures/8mUSV"
save_format = "svg"
dpi = 300

if save_fig and not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

def plot_accuracy_comparison(title):

    # 读取真实数据集
    true_train_filepath = os.path.join(base_dir, "8mUSV_TrainingSet_Smoothed.csv")
    
    if not os.path.exists(true_train_filepath):
        print("True dataset file not found!")
        return
        
    true_train_df = pd.read_csv(true_train_filepath)
    
    # 读取四个benchmark的预测数据
    predictions = {}
    for benchmark, pred_file in zip(benchmarks, prediction_files):
        filepath = os.path.join(base_dir, pred_file)
        if os.path.exists(filepath):
            predictions[benchmark] = pd.read_csv(filepath)
        else:
            print(f"Warning: File not found {filepath}")
            predictions[benchmark] = None
    
    all_states = [0, 1, 2, 3, 4, 5]
    rad_to_deg = 180.0 / np.pi
    
    # 绘制六个状态的时历曲线
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(title, fontsize=16)
    
    label_list = ["x(m)", "y(m)", "$\\psi$(deg)", "u(m/s)", "v(m/s)", "r(deg/s)"]
    
    for state_idx, label in enumerate(label_list):
        ax = axes[state_idx // 3, state_idx % 3]
        
        # 绘制真实数据
        true_data = true_train_df[f'state_{state_idx}'].values
        if state_idx in [2, 5]:  # 角度和角速度转换为度
            true_data = true_data * rad_to_deg
        ax.plot(true_data, 'k-', label='Ground Truth', linewidth=2)
        
        # 绘制每个benchmark的预测数据
        for benchmark in benchmarks:
            if predictions[benchmark] is not None:
                pred_data = predictions[benchmark][f'state_{state_idx}'].values
                if state_idx in [2, 5]:
                    pred_data = pred_data * rad_to_deg
                
                # 计算RMSE
                min_len = min(len(pred_data), len(true_data))
                rmse = np.sqrt(np.mean((pred_data[:min_len] - true_data[:min_len])**2))
                
                ax.plot(pred_data, color=colors[benchmark], label=f'{benchmark} (RMSE: {rmse:.4f})', linewidth=2)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(label)
        ax.grid(True, linestyle='--', alpha=0.7)
        if state_idx == 0:
            ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()
    if save_fig:
        save_path = os.path.join(fig_dir, f"accuracy_comparison_timeseries.{save_format}")
        fig.savefig(save_path, format=save_format, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # 绘制xy平面轨迹 - 分开四个子图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{title} - XY Trajectory Comparison", fontsize=16)
    
    true_x = true_train_df['state_0'].values
    true_y = true_train_df['state_1'].values
    
    # 其他三个子图: 每个benchmark的预测轨迹
    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx]
        # 绘制Ground Truth作为参考
        ax.plot(true_x, true_y, 'k--', label='Ground Truth', linewidth=2, alpha=0.5)
        
        if predictions[benchmark] is not None:
            pred_x = predictions[benchmark]['state_0'].values
            pred_y = predictions[benchmark]['state_1'].values
            
            # 计算xy联合RMSE
            min_len = min(len(pred_x), len(true_x))
            rmse_xy = np.sqrt(np.mean((pred_x[:min_len] - true_x[:min_len])**2 + (pred_y[:min_len] - true_y[:min_len])**2))
            
            ax.plot(pred_x, pred_y, color=colors[benchmark], label=f'{benchmark}\n(RMSE: {rmse_xy:.4f})', linewidth=2)
        
        ax.set_title(benchmark, fontsize=12)
        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('y (m)', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    if save_fig:
        save_path = os.path.join(fig_dir, f"accuracy_comparison_trajectory.{save_format}")
        fig.savefig(save_path, format=save_format, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    plot_accuracy_comparison("RMSE and MAE Comparison")
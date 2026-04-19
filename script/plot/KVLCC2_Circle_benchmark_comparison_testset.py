import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import utils

# 配置参数
base_dir = "result/Circle_benchmark/"
benchmarks = ["Koopman(order=3)", "nuSVR($\\nu=0.05$)", "SGP-DE(ind_num=50)", "SGP-CG(ind_num=50)"]
prediction_files = [
    "KVLCC2_Circle_KPM_TestSet_prediction.csv",
    "KVLCC2_Circle_nuSVR_TestSet_prediction.csv",
    "KVLCC2_Circle_DE_RandomID=0_TestSet_prediction.csv",
    "KVLCC2_Circle_CG_RandomID=9_TestSet_prediction.csv"
]
states = [3, 4, 5]
cmap = plt.get_cmap('tab10')
colors = {benchmark: cmap(i % 10) for i, benchmark in enumerate(benchmarks)}

# 配置图保存参数
save_fig = True
fig_dir = "result/convergence_analysis_figures/Circle_benchmark"
save_format = "svg"
dpi = 300

if save_fig and not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

def plot_accuracy_comparison(title):

    # 读取真实数据集
    true_train_filepath = os.path.join(base_dir, "TestSet.csv")
    
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
    utils.setup_matplotlib_style(single_column=False, figure_height=6)
    fig, axes = plt.subplots(2, 3)
    
    label_list = ["x(m)", "y(m)", "$\\psi$(deg)", "u(m/s)", "v(m/s)", "r(deg/s)"]
    
    for state_idx, label in enumerate(label_list):
        ax = axes[state_idx // 3, state_idx % 3]
        
        # 绘制真实数据
        true_data = true_train_df[f'state_{state_idx}'].values
        time_data = true_train_df['time'].values
        if state_idx in [2, 5]:  # 角度和角速度转换为度
            true_data = true_data * rad_to_deg
        ax.plot(time_data, true_data, 'k--', label='GT')
        
        # 绘制每个benchmark的预测数据
        for benchmark in benchmarks:
            if predictions[benchmark] is not None:
                pred_data = predictions[benchmark][f'state_{state_idx}'].values
                time_data = predictions[benchmark]['time'].values
                if state_idx in [2, 5]:
                    pred_data = pred_data * rad_to_deg
                
                # 计算RMSE
                min_len = min(len(pred_data), len(true_data))
                rmse = np.sqrt(np.mean((pred_data[:min_len] - true_data[:min_len])**2))
                
                ax.plot(time_data, pred_data, color=colors[benchmark], label=f'{benchmark} (RMSE: {rmse:.4f})')
        if state_idx // 3 == 1:
            ax.set_xlabel('time(s)')
        # 设置标题并贴近坐标轴上边缘
        ax.set_title(label)
        
        if state_idx == 2:
            ax.legend()
    
    plt.tight_layout()
    # plt.subplots_adjust(top=0.93)
    plt.show()
    if save_fig:
        save_path = os.path.join(fig_dir, f"accuracy_comparison_timeseries_testset.{save_format}")
        fig.savefig(save_path, format=save_format, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # 绘制xy平面轨迹 - 分开四个子图
    utils.setup_matplotlib_style(single_column=True, figure_height=4.5)
    fig, axes = plt.subplots()
    
    true_x = true_train_df['state_0'].values
    true_y = true_train_df['state_1'].values
    
    # 其他三个子图: 每个benchmark的预测轨迹
    for idx, benchmark in enumerate(benchmarks):
        ax = axes
        # 绘制Ground Truth作为参考
        if idx == 0:  # 只在第一个子图绘制GT
            ax.plot(true_x, true_y, 'k--', label='GT', alpha=0.5)
        
        if predictions[benchmark] is not None:
            pred_x = predictions[benchmark]['state_0'].values
            pred_y = predictions[benchmark]['state_1'].values
            
            # 计算xy联合RMSE
            min_len = min(len(pred_x), len(true_x))
            rmse_xy = np.sqrt(np.mean((pred_x[:min_len] - true_x[:min_len])**2 + (pred_y[:min_len] - true_y[:min_len])**2))
            
            ax.plot(pred_x, pred_y, color=colors[benchmark], label=f'{benchmark}\n(RMSE: {rmse_xy:.4f})')
        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    if save_fig:
        save_path = os.path.join(fig_dir, f"accuracy_comparison_trajectory_testset.{save_format}")
        fig.savefig(save_path, format=save_format, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    plot_accuracy_comparison("RMSE and MAE Comparison")
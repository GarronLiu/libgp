import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import utils

# 配置参数
base_dir = "result/convergence_analysis_algorithms"
algorithms = ["CG", "DE", "GA", "PSO"]
max_run_id = 9
states = [3, 4, 5]
colors = {"CG": "blue", "DE": "green", "GA": "orange", "PSO": "red"}

# 配置图保存参数
save_fig = True
fig_dir = "result/convergence_analysis_figures/algorithms"
save_format = "svg"
dpi = 300

if save_fig and not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

def plot_lml_convergence(title):
    utils.setup_matplotlib_style(single_column=False, figure_height=5)
    fig, axes = plt.subplots(1, 3)

    for idx, state_id in enumerate(states):
        ax = axes[idx]
        ax.set_title(f'State {state_id} Convergence')
        ax.set_xlabel('Generation ID')

        ax2 = ax.twinx()

        for algo in algorithms:
            all_lml = []
            max_len = 0
            
            # Read all runs for the current algorithm and state
            runs_data = []
            runs_time_data = []
            for run_id in range(max_run_id + 1):
                filename = f"KVLCC2_Algorithms_{algo}_RandomID={run_id}_State_{state_id}_LML_TimeCost.csv"
                filepath = os.path.join(base_dir, filename)
                
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    runs_data.append(df['LML'].values)
                    runs_time_data.append(df['TimeCost'].values)
                    max_len = max(max_len, len(df))
                else:
                    print(f"Warning: File not found {filepath}")
            
            if not runs_data or not runs_time_data:
                continue
                
            # Pad sequences with their last value to make them the same length
            # Some algorithms might converge earlier 
            padded_runs = []
            
            for run in runs_data:
                if len(run) < max_len:
                    padded = np.pad(run, (0, max_len - len(run)), mode='edge')
                    padded_runs.append(padded)
                else:
                    padded_runs.append(run)
            
            padded_runs = np.array(padded_runs)

            padded_times = []
            for run_time in runs_time_data:
                if len(run_time) < max_len:
                    padded_time = np.pad(run_time, (0, max_len - len(run_time)), mode='edge')
                    padded_times.append(padded_time)
                else:
                    padded_times.append(run_time)
            padded_times = np.array(padded_times)
            
            # Calculate mean and bounds (e.g., standard deviation or min/max)
            mean_lml = np.mean(padded_runs, axis=0)
            std_lml = np.std(padded_runs, axis=0)

            mean_cost = np.mean(padded_times, axis=0)
            std_cost = np.std(padded_times, axis=0)
            
            generations = np.arange(1, max_len + 1)
            
            # Plot the mean line and the shaded area
            ax.plot(generations, mean_lml, color=colors[algo], label=f"{algo} LML")
            ax.fill_between(generations, mean_lml - std_lml, mean_lml + std_lml, 
                             color=colors[algo], alpha=0.2)
            ax2.plot(generations, mean_cost, color=colors[algo],label=f"{algo} Time Cost", linestyle=':', alpha=0.7)
            ax2.fill_between(generations, mean_cost - std_cost, mean_cost + std_cost, 
                        color=colors[algo], alpha=0.08)
            
            # ax2.tick_params(axis='y')
        
        state_names = {3: '(a)', 4: '(b)', 5: '(c)'}
        ax.set_title(f'{state_names.get(state_id, "")}',fontweight='bold')
        ax.set_xlabel('Iterations')
        if idx == 0:
            ax.set_ylabel('Log Marginal Likelihood (LML)')
            # 只在最右侧显示y轴刻度数值
        if idx != 0:
            ax.set_yticklabels([])
        if idx != 2:
            ax2.set_yticklabels([])
        
        if idx == 2:
            ax.legend(loc='lower right', ncol=1, frameon=True)
            ax2.set_ylabel('Time per Iteration (ms)')

        # 设置统一的y轴范围
        ax.set_ylim(-500, 3500)
        ax2.set_ylim(0, 200)
        # 在右轴绘制每一次迭代的时间耗费             

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
    if save_fig:
        save_path = os.path.join(fig_dir, f"lml_convergence.{save_format}")
        fig.savefig(save_path, format=save_format, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

def plot_accuracy_comparison(title):
    
    # 读取原始的真实数据集
    true_train_filepath = os.path.join(base_dir, "TrainingSet_wo_noise.csv")
    true_test_filepath = os.path.join(base_dir, "TestSet.csv")
    
    if not os.path.exists(true_train_filepath) or not os.path.exists(true_test_filepath):
        print("True dataset files not found!")
        return
        
    true_train_df = pd.read_csv(true_train_filepath)
    true_test_df = pd.read_csv(true_test_filepath)

    # 结果字典
    results = {
        'Train RMSE': {}, 'Test RMSE': {},
        'Train MAE': {}, 'Test MAE': {}
    }

    # 需要对比的所有状态
    all_states = [0, 1, 2, 3, 4, 5]

    for algo in algorithms:
        train_rmse_list, test_rmse_list = [], []
        train_mae_list, test_mae_list = [], []

        for run_id in range(max_run_id + 1):
            train_filename = f"KVLCC2_Algorithms_{algo}_RandomID={run_id}_TrainingSet_prediction.csv"
            test_filename = f"KVLCC2_Algorithms_{algo}_RandomID={run_id}_TestSet_prediction.csv"

            train_filepath = os.path.join(base_dir, train_filename)
            test_filepath = os.path.join(base_dir, test_filename)

            if os.path.exists(train_filepath) and os.path.exists(test_filepath):
                train_df = pd.read_csv(train_filepath)
                test_df = pd.read_csv(test_filepath)

                # RMSE
                train_rmse = np.sqrt(np.mean((train_df[[f'state_{i}' for i in all_states]].values - true_train_df[[f'state_{i}' for i in all_states]].values)**2, axis=0))
                test_rmse = np.sqrt(np.mean((test_df[[f'state_{i}' for i in all_states]].values - true_test_df[[f'state_{i}' for i in all_states]].values)**2, axis=0))

                # MAE
                train_mae = np.mean(np.abs(train_df[[f'state_{i}' for i in all_states]].values - true_train_df[[f'state_{i}' for i in all_states]].values), axis=0)
                test_mae = np.mean(np.abs(test_df[[f'state_{i}' for i in all_states]].values - true_test_df[[f'state_{i}' for i in all_states]].values), axis=0)

                train_rmse_list.append(train_rmse)
                test_rmse_list.append(test_rmse)
                train_mae_list.append(train_mae)
                test_mae_list.append(test_mae)

        
        if train_rmse_list:
            results['Train RMSE'][algo] = np.array(train_rmse_list)
            results['Test RMSE'][algo] = np.array(test_rmse_list)
            results['Train MAE'][algo] = np.array(train_mae_list)
            results['Test MAE'][algo] = np.array(test_mae_list)


    # 打印和制表
    for metric_name, algo_dict in results.items():
        print(f"\n--- {metric_name} Statistics ---")
        for algo, data in algo_dict.items():
            mean_val = np.mean(data, axis=0)
            std_val = np.std(data, axis=0)
            median_val = np.median(data, axis=0)
            
            print(f"{algo}:")
            for i, state in enumerate(all_states):
                 print(f"  State {state}: Mean={mean_val[i]:.4f}, Std={std_val[i]:.4f}, Median={median_val[i]:.4f}")

    # 单独将state 2(yaw角) 和 state 5(yaw rate) 转换为度(°) 和 度每秒(°/s)   1 rad = 180 / pi
    rad_to_deg = 180.0 / np.pi
    
    for metric_name in results:
        for algo in results[metric_name]:
            results[metric_name][algo][:, 2] *= rad_to_deg
            results[metric_name][algo][:, 5] *= rad_to_deg

    utils.setup_matplotlib_style(single_column=False, figure_height=8)
    fig, axes = plt.subplots(2, 6)

    label_list = ["x(m)", "y(m)", "$\\psi$(deg)", "u(m/s)", "v(m/s)", "r(deg/s)"]
    metrics_rows = [('Train RMSE', '(a)'), ('Test RMSE', '(b)')]
    
    for row, (metric_name, row_title) in enumerate(metrics_rows):
        for state_idx in range(6):
            ax = axes[row, state_idx]
            width = 0.6
            for i, algo in enumerate(algorithms):
                if algo in results[metric_name]:
                    data = results[metric_name][algo][:, state_idx]
                    
                    bp = ax.boxplot([data], 
                                    positions=[i], widths=width, patch_artist=False, showfliers=False)
                    
                    for item in ['boxes', 'whiskers', 'medians', 'caps']:
                        for line in bp[item]:
                            line.set_color(colors[algo])
            
            ax.set_xticks(np.arange(len(algorithms)))
            if row == 1:
                ax.set_xticklabels(algorithms)
            else:
                ax.set_xticklabels([])
            if row == 0:
                ax.set_title(f'{label_list[state_idx]}')
            if state_idx == 0:
                ax.set_ylabel(row_title, fontweight='bold')


    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    if save_fig:
        save_path = os.path.join(fig_dir, f"accuracy_comparison.{save_format}")
        fig.savefig(save_path, format=save_format, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

def plot_trajectory_comparison(title, dataset_prefix):
    utils.setup_matplotlib_style(single_column=True, figure_height=5)
    fig, ax = plt.subplots()
    
    # 绘制真实轨迹
    true_test_filepath = os.path.join(base_dir, f"{dataset_prefix}.csv")
    if os.path.exists(true_test_filepath):
        true_df = pd.read_csv(true_test_filepath)
        ax.plot(true_df['state_0'], true_df['state_1'], 'k--', label='True Trajectory')
    else:
        print("True dataset file not found!")

    for algo in algorithms:
        all_x = []
        all_y = []
        max_len = 0

        for run_id in range(max_run_id + 1):
            test_filename = f"KVLCC2_Algorithms_{algo}_RandomID={run_id}_{dataset_prefix}_prediction.csv"
            test_filepath = os.path.join(base_dir, test_filename)

            if os.path.exists(test_filepath):
                df = pd.read_csv(test_filepath)
                all_x.append(df['state_0'].values)
                all_y.append(df['state_1'].values)
                max_len = max(max_len, len(df))
            else:
                print(f"Warning: File not found {test_filepath}")

        if not all_x:
            continue

        # 将不同运行的轨迹对齐（补齐）
        padded_x, padded_y = [], []
        for x_traj, y_traj in zip(all_x, all_y):
            if len(x_traj) < max_len:
                padded_x.append(np.pad(x_traj, (0, max_len - len(x_traj)), mode='edge'))
                padded_y.append(np.pad(y_traj, (0, max_len - len(y_traj)), mode='edge'))
            else:
                padded_x.append(x_traj)
                padded_y.append(y_traj)

        padded_x = np.array(padded_x)
        padded_y = np.array(padded_y)

        # 把X和Y分别求均值以及Y的标准差，构建带状区域
        mean_x = np.mean(padded_x, axis=0)
        mean_y = np.mean(padded_y, axis=0)
        std_y = np.std(padded_y, axis=0)

        # 绘制平均预测轨迹
        ax.plot(mean_x, mean_y, color=colors[algo], label=algo)
        
        if 'true_df' in locals():
            true_x = true_df['state_0'].values
            true_y = true_df['state_1'].values
            
            # Ensure lengths match
            if len(true_x) < max_len:
                true_x = np.pad(true_x, (0, max_len - len(true_x)), mode='edge')
                true_y = np.pad(true_y, (0, max_len - len(true_y)), mode='edge')
            else:
                true_x = true_x[:max_len]
                true_y = true_y[:max_len]
            
            # Calculate combined RMSE for x and y per run
            rmse_per_run = np.sqrt(np.mean((padded_x - true_x)**2 + (padded_y - true_y)**2, axis=1))
            mean_rmse = np.mean(rmse_per_run)
            std_rmse = np.std(rmse_per_run)
            
            # Update the label of the recently plotted line
            ax.lines[-1].set_label(f"{algo} (RMSE: {mean_rmse:.4f} ± {std_rmse:.4f})")
        
        # 绘制Y方向的带状预测区域
        ax.fill_between(mean_x, mean_y - std_y, mean_y + std_y, 
                        color=colors[algo], alpha=0.2)

    # Re-plot true trajectory to ensure it is drawn on top of the shaded areas
    if os.path.exists(true_test_filepath):
        ax.plot(true_df['state_0'], true_df['state_1'], 'k--')
    ax.set_title(title,fontweight='bold')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    if save_fig:
        save_path = os.path.join(fig_dir, f"{dataset_prefix}_trajectory_comparison.{save_format}")
        fig.savefig(save_path, format=save_format, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    plot_lml_convergence("Log Marginal Likelihood (LML) Convergence Comparison")
    plot_accuracy_comparison("RMSE and MAE Comparison")
    plot_trajectory_comparison("(a)", "TrainingSet")
    plot_trajectory_comparison("(b)", "TestSet")
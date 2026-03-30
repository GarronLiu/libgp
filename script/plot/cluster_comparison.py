import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# 配置参数
base_dir = "result/convergence_analysis_clusters/"
clusters = [str(i) for i in range(25, 201, 25)]
prefixes = ["", "wo_"]
max_run_id = 9
states = [3, 4, 5]
cmap = plt.get_cmap('tab10')
colors = {cluster: cmap(i % 10) for i, cluster in enumerate(clusters)}

# 配置图保存参数
save_fig = True
fig_dir = "result/convergence_analysis_figures/clusters"
save_format = "svg"
dpi = 300

if save_fig and not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

def plot_lml_convergence(title):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=16)

    for idx, state_id in enumerate(states):
        ax = axes[idx]
        ax.set_title(f'State {state_id} Convergence')
        ax.set_xlabel('Generation ID')
        ax.set_ylabel('LML')

        for cluster in clusters:
            all_lml = []
            max_len = 0
            
            # Read all runs for the current algorithm and state
            runs_data = []
            for prefix in prefixes:
                filename = f"{prefix}inducing_update/DE_Cluster={cluster}_State_{state_id}_LML_TimeCost_LML_TimeCost.csv"
                filepath = os.path.join(base_dir, filename)
                
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    runs_data.append(df['LML'].values)
                    max_len = max(max_len, len(df))
                else:
                    print(f"Warning: File not found {filepath}")
            
            if not runs_data:
                continue
                
            # Pad sequences with their last value to make them the same length
            # Some clusters might converge earlier 
            padded_runs = []
            for run in runs_data:
                if len(run) < max_len:
                    padded = np.pad(run, (0, max_len - len(run)), mode='edge')
                    padded_runs.append(padded)
                else:
                    padded_runs.append(run)
            
            padded_runs = np.array(padded_runs)
            
            generations = np.arange(1, max_len + 1)
            
            for i, prefix in enumerate(prefixes):
                if i < len(padded_runs):
                    linestyle = '-' if i == 0 else '--'
                    ax.plot(generations, padded_runs[i], color=colors[cluster], 
                            linestyle=linestyle, label=f"{cluster} {prefix}", linewidth=2)
        
        state_names = {3: ' longitudinal velocity (u)', 4: ' sway velocity (v)', 5: ' yaw rate (r)'}
        ax.set_title(f'LML Convergence vs Generation (State {state_id}:{state_names.get(state_id, "")})')
        ax.set_xlabel('Generation ID')
        ax.set_ylabel('Log Marginal Likelihood (LML)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
    if save_fig:
        save_path = os.path.join(fig_dir, f"lml_convergence.{save_format}")
        fig.savefig(save_path, format=save_format, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

def plot_accuracy_comparison(title):

    # 读取原始的真实数据集
    true_train_filepath = os.path.join(base_dir, "inducing_update/TrainingSet_wo_noise.csv")
    true_test_filepath = os.path.join(base_dir, "inducing_update/TestSet.csv")
    
    if not os.path.exists(true_train_filepath) or not os.path.exists(true_test_filepath):
        print("True dataset files not found!")
        return
        
    true_train_df = pd.read_csv(true_train_filepath)
    true_test_df = pd.read_csv(true_test_filepath)

    # 按照前缀保存结果，以绘制对比曲线
    results = {
        prefix: {
            'Train RMSE': [], 'Test RMSE': [],
            'Train MAE': [], 'Test MAE': []
        } for prefix in prefixes
    }

    # 需要对比的所有状态
    all_states = [0, 1, 2, 3, 4, 5]

    for prefix in prefixes:
        for cluster in clusters:
            train_filename = f"{prefix}inducing_update/DE_Cluster={cluster}_TrainingSet_prediction.csv"
            test_filename = f"{prefix}inducing_update/DE_Cluster={cluster}_TestSet_prediction.csv"

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

                results[prefix]['Train RMSE'].append(train_rmse)
                results[prefix]['Test RMSE'].append(test_rmse)
                results[prefix]['Train MAE'].append(train_mae)
                results[prefix]['Test MAE'].append(test_mae)
            else:
                nan_array = np.full(len(all_states), np.nan)
                results[prefix]['Train RMSE'].append(nan_array)
                results[prefix]['Test RMSE'].append(nan_array)
                results[prefix]['Train MAE'].append(nan_array)
                results[prefix]['Test MAE'].append(nan_array)

    for prefix in prefixes:
        for metric in ['Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE']:
            results[prefix][metric] = np.array(results[prefix][metric])

    # 单独将state 2(yaw角) 和 state 5(yaw rate) 转换为度(°) 和 度每秒(°/s)   1 rad = 180 / pi
    rad_to_deg = 180.0 / np.pi
    
    for prefix in prefixes:
        for metric_name in results[prefix]:
            if len(results[prefix][metric_name]) > 0:
                results[prefix][metric_name][:, 2] *= rad_to_deg
                results[prefix][metric_name][:, 5] *= rad_to_deg

    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    fig.suptitle(title, fontsize=16)

    label_list = ["x(m)", "y(m)", "$\\psi$(deg)", "u(m/s)", "v(m/s)", "r(deg/s)"]
    metrics_rows = [('Train RMSE', 'RMSE on Training Dataset'), ('Test RMSE', 'RMSE on Test Dataset')]
    
    x_positions = np.arange(len(clusters))
    
    for row, (metric_name, row_title) in enumerate(metrics_rows):
        for state_idx in range(6):
            ax = axes[row, state_idx]
            
            for prefix in prefixes:
                data = results[prefix][metric_name][:, state_idx]
                label_name = "inducing_update" if prefix == "" else "wo_inducing_update"
                color = 'tab:blue' if prefix == "" else 'tab:orange'
                marker = 'o' if prefix == "" else 's'
                ax.plot(x_positions, data, marker=marker, color=color, label=label_name, linewidth=2, markersize=6)
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels(clusters)
            ax.set_title(f'{label_list[state_idx]}')
            ax.set_xlabel('Clusters')
            if state_idx == 0:
                ax.set_ylabel(row_title)
            ax.grid(True, linestyle='--', alpha=0.7)
            if row == 0 and state_idx == 0:
                ax.legend()

    for state_idx in range(6):
        ylim0 = axes[0, state_idx].get_ylim()
        ylim1 = axes[1, state_idx].get_ylim()
        ymin = min(ylim0[0], ylim1[0])
        ymax = max(ylim0[1], ylim1[1])
        axes[0, state_idx].set_ylim(ymin, ymax)
        axes[1, state_idx].set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
    if save_fig:
        save_path = os.path.join(fig_dir, f"accuracy_comparison.{save_format}")
        fig.savefig(save_path, format=save_format, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

def plot_trajectory_comparison(title, dataset_prefix):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(title, fontsize=16)

    # 绘制真实轨迹
    true_test_filepath = os.path.join(base_dir, f"inducing_update/{dataset_prefix}.csv")
    if os.path.exists(true_test_filepath):
        true_df = pd.read_csv(true_test_filepath)
        ax.plot(true_df['state_0'], true_df['state_1'], 'k-', label='True Trajectory', linewidth=3)
        true_x = true_df['state_0'].values
        true_y = true_df['state_1'].values
    else:
        print("True dataset file not found!")
        true_x, true_y = None, None

    for cluster in clusters:
        for prefix in prefixes:
            test_filename = f"{prefix}inducing_update/DE_Cluster={cluster}_{dataset_prefix}_prediction.csv"
            test_filepath = os.path.join(base_dir, test_filename)

            if os.path.exists(test_filepath):
                df = pd.read_csv(test_filepath)
                pred_x = df['state_0'].values
                pred_y = df['state_1'].values
                
                linestyle = '-' if prefix == "" else '--'
                label_prefix = "inducing_update" if prefix == "" else "wo_inducing_update"
                
                if true_x is not None and true_y is not None:
                    # 获取最小长度以防不匹配
                    min_len = min(len(pred_x), len(true_x))
                    
                    # Calculate combined RMSE for x and y
                    rmse = np.sqrt(np.mean((pred_x[:min_len] - true_x[:min_len])**2 + (pred_y[:min_len] - true_y[:min_len])**2))
                    label = f"{cluster} {label_prefix} (RMSE: {rmse:.4f})"
                else:
                    label = f"{cluster} {label_prefix}"

                # 绘制预测轨迹实线和虚线
                ax.plot(pred_x, pred_y, color=colors[cluster], linestyle=linestyle, label=label, linewidth=2)
            else:
                print(f"Warning: File not found {test_filepath}")

    # Re-plot true trajectory to ensure it is drawn on top
    if os.path.exists(true_test_filepath):
        ax.plot(true_df['state_0'], true_df['state_1'], 'k-', linewidth=3)

    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.legend(loc="lower left", ncol=2)
    ax.grid(True, linestyle='--', alpha=0.7)
    
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
    plot_trajectory_comparison("Trajectory (x, y) Comparison on Test Set(待补充总的积分时长)", "TestSet")
    plot_trajectory_comparison("Trajectory (x, y) Comparison on Train Set(待补充总的积分时长)", "TrainingSet")
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import utils
# 配置参数
base_dir = "result/convergence_analysis_clusters/"
clusters = [str(i) for i in range(25, 200, 25)]
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
    utils.setup_matplotlib_style(single_column=False, figure_height=5)
    fig, axes = plt.subplots(1, 3)

    for idx, state_id in enumerate(states):
        ax = axes[idx]
        ax.set_title(f'State {state_id} Convergence')
        ax.set_xlabel('Generation ID')

        for cluster in clusters:
            for prefix in prefixes:
                runs_data = []
                max_len = 0
                for run_id in range(max_run_id + 1):
                    filename = f"{prefix}inducing_update/KVLCC2_Cluster={cluster}_CG_RandomID={run_id}_State_{state_id}_LML_TimeCost.csv"
                    filepath = os.path.join(base_dir, filename)
                    
                    if os.path.exists(filepath):
                        df = pd.read_csv(filepath)
                        runs_data.append(df['LML'].values)
                        max_len = max(max_len, len(df))
                
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
                mean_lml = np.mean(padded_runs, axis=0)
                std_lml = np.std(padded_runs, axis=0)
                
                generations = np.arange(1, max_len + 1)
                
                linestyle = '-' if prefix == "" else '--'
                label_prefix = "inducing_update" if prefix == "" else "wo_inducing_update"
                
                ax.plot(generations, mean_lml, color=colors[cluster], 
                        linestyle=linestyle, label=f"{cluster} {label_prefix}")
                ax.fill_between(generations, mean_lml - std_lml, mean_lml + std_lml, 
                                color=colors[cluster], alpha=0.2)
        
        state_names = {3: '(a)', 4: '(b)', 5: '(c)'}
        ax.set_title(f'{state_names.get(state_id, "")}',fontweight='bold')
        ax.set_xlabel('Generation ID')
        if idx == 0:
            ax.set_ylabel('Log Marginal Likelihood (LML)')
            # 只在最右侧显示y轴刻度数值
        if idx != 0:
            ax.set_yticklabels([])
        
        if idx == 2:
            ax.legend(loc='lower right', ncol=1, frameon=True)

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

    # 按照前缀和cluster保存结果，由于有十次随机重复实验，因此每个组合保存一个列表
    results = {
        prefix: {
            cluster: {
                'Train RMSE': [], 'Test RMSE': [],
                'Train MAE': [], 'Test MAE': []
            } for cluster in clusters
        } for prefix in prefixes
    }

    # 需要对比的所有状态
    all_states = [0, 1, 2, 3, 4, 5]

    for prefix in prefixes:
        for cluster in clusters:
            for run_id in range(max_run_id + 1):
                train_filename = f"{prefix}inducing_update/KVLCC2_Cluster={cluster}_CG_RandomID={run_id}_TrainingSet_prediction.csv"
                test_filename = f"{prefix}inducing_update/KVLCC2_Cluster={cluster}_CG_RandomID={run_id}_TestSet_prediction.csv"

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

                    results[prefix][cluster]['Train RMSE'].append(train_rmse)
                    results[prefix][cluster]['Test RMSE'].append(test_rmse)
                    results[prefix][cluster]['Train MAE'].append(train_mae)
                    results[prefix][cluster]['Test MAE'].append(test_mae)

    # 将每个cluster的结果转为numpy array (N_runs, states)
    for prefix in prefixes:
        for cluster in clusters:
            for metric in ['Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE']:
                if len(results[prefix][cluster][metric]) > 0:
                    results[prefix][cluster][metric] = np.array(results[prefix][cluster][metric])
                else:
                    results[prefix][cluster][metric] = np.full((1, len(all_states)), np.nan)

    # 单独将state 2(yaw角) 和 state 5(yaw rate) 转换为度(°) 和 度每秒(°/s)   1 rad = 180 / pi
    rad_to_deg = 180.0 / np.pi
    
    for prefix in prefixes:
        for cluster in clusters:
            for metric_name in ['Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE']:
                if not np.isnan(results[prefix][cluster][metric_name]).all():
                    results[prefix][cluster][metric_name][:, 2] *= rad_to_deg
                    results[prefix][cluster][metric_name][:, 5] *= rad_to_deg

    utils.setup_matplotlib_style(single_column=False, figure_height=8)
    fig, axes = plt.subplots(2, 6)

    label_list = ["x(m)", "y(m)", "$\\psi$(deg)", "u(m/s)", "v(m/s)", "r(deg/s)"]
    metrics_rows = [('Train RMSE', '(a)'), ('Test RMSE', '(b)')]
    
    x_positions_base = np.arange(len(clusters)) * 2
    
    for row, (metric_name, row_title) in enumerate(metrics_rows):
        for state_idx in range(6):
            ax = axes[row, state_idx]
            
            for p_idx, prefix in enumerate(prefixes):
                data_to_plot = []
                for cluster in clusters:
                    cluster_data = results[prefix][cluster][metric_name][:, state_idx]
                    cluster_data = cluster_data[~np.isnan(cluster_data)]
                    data_to_plot.append(cluster_data if len(cluster_data) > 0 else [np.nan])
                
                label_name = "inducing_update" if prefix == "" else "wo_inducing_update"
                color = '#1f77b4' if prefix == "" else '#ff7f0e'
                
                # 设置箱线图偏置
                offset = -0.4 if prefix == "" else 0.4
                positions = x_positions_base + offset
                
                bplot = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True, 
                                   showfliers=False,
                                   boxprops=dict(facecolor='white', color=color, alpha=1.0), 
                                   capprops=dict(color=color), whiskerprops=dict(color=color), 
                                   medianprops=dict(color=color))
                                   
                # 为图例创建自定义图形
                if row == 0 and state_idx == 0:
                    ax.plot([], [], color=color, label=label_name)
            
            ax.set_xticks(x_positions_base)
            labels_to_show = {"25", "75", "125", "175"}
            display_labels = [c if c in labels_to_show else "" for c in clusters]
            ax.set_xticklabels(display_labels)
            if row == 0:
                ax.set_title(f'{label_list[state_idx]}')
            if row == 1:
                ax.set_xlabel('Clusters')
            if state_idx == 0:
                ax.set_ylabel(row_title)
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
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    if save_fig:
        save_path = os.path.join(fig_dir, f"accuracy_comparison.{save_format}")
        fig.savefig(save_path, format=save_format, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

def plot_trajectory_comparison(title, dataset_prefix):
    utils.setup_matplotlib_style(single_column=True, figure_height=5)
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制真实轨迹
    true_test_filepath = os.path.join(base_dir, f"inducing_update/{dataset_prefix}.csv")
    if os.path.exists(true_test_filepath):
        true_df = pd.read_csv(true_test_filepath)
        ax.plot(true_df['state_0'], true_df['state_1'], 'k-', label='True Trajectory')
        true_x = true_df['state_0'].values
        true_y = true_df['state_1'].values
    else:
        print("True dataset file not found!")
        true_x, true_y = None, None

    for cluster in clusters:
        for prefix in prefixes:
            test_filename = f"{prefix}inducing_update/KVLCC2_Cluster={cluster}_CG_RandomID=4_{dataset_prefix}_prediction.csv"
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
                ax.plot(pred_x, pred_y, color=colors[cluster], linestyle=linestyle, label=label)
            else:
                print(f"Warning: File not found {test_filepath}")

    # Re-plot true trajectory to ensure it is drawn on top
    if os.path.exists(true_test_filepath):
        ax.plot(true_df['state_0'], true_df['state_1'], 'k-')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend(loc="lower left", ncol=2)
    
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
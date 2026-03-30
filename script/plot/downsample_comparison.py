import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# 配置参数
base_dir = "result/convergence_analysis_downsample/"
downsamples = [str(i) for i in range(1, 11)]
states = [3, 4, 5]
cmap = plt.get_cmap('tab10')
colors = {downsample: cmap(i % 10) for i, downsample in enumerate(downsamples)}

# 配置图保存参数
save_fig = True
fig_dir = "result/convergence_analysis_figures/downsample"
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

        for downsample in downsamples:
            all_lml = []
            max_len = 0
            
            runs_data = []
            filename = f"DEDownsample={downsample}_State_{state_id}_LML_TimeCost_LML_TimeCost.csv"
            filepath = os.path.join(base_dir, filename)
            
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                runs_data.append(df['LML'].values)
                max_len = max(max_len, len(df))
            else:
                print(f"Warning: File not found {filepath}")
            
            if not runs_data:
                continue
                
            padded_runs = []
            for run in runs_data:
                if len(run) < max_len:
                    padded = np.pad(run, (0, max_len - len(run)), mode='edge')
                    padded_runs.append(padded)
                else:
                    padded_runs.append(run)
            
            padded_runs = np.array(padded_runs)
            generations = np.arange(1, max_len + 1)

            for i, ds in enumerate(downsamples):
                if ds == downsample and i < len(padded_runs): # plotting only for the current one
                    ax.plot(generations, padded_runs[0], color=colors[downsample],
                            linestyle='-', label=f"{downsample}", linewidth=2)

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
    true_train_filepath = os.path.join(base_dir, "TrainingSet_wo_noise.csv")
    true_test_filepath = os.path.join(base_dir, "TestSet.csv")
    
    if not os.path.exists(true_train_filepath) or not os.path.exists(true_test_filepath):
        print("True dataset files not found!")
        return
        
    true_train_df = pd.read_csv(true_train_filepath)
    true_test_df = pd.read_csv(true_test_filepath)

    results = {
        'Train RMSE': [], 'Test RMSE': [],
        'Train MAE': [], 'Test MAE': []
    }

    all_states = [0, 1, 2, 3, 4, 5]

    for downsample in downsamples:
        train_filename = f"DEDownsample={downsample}_TrainingSet_prediction.csv"
        test_filename = f"DEDownsample={downsample}_TestSet_prediction.csv"

        train_filepath = os.path.join(base_dir, train_filename)
        test_filepath = os.path.join(base_dir, test_filename)

        if os.path.exists(train_filepath) and os.path.exists(test_filepath):
            train_df = pd.read_csv(train_filepath)
            test_df = pd.read_csv(test_filepath)

            train_rmse = np.sqrt(np.mean((train_df[[f'state_{i}' for i in all_states]].values - true_train_df[[f'state_{i}' for i in all_states]].values)**2, axis=0))
            test_rmse = np.sqrt(np.mean((test_df[[f'state_{i}' for i in all_states]].values - true_test_df[[f'state_{i}' for i in all_states]].values)**2, axis=0))

            train_mae = np.mean(np.abs(train_df[[f'state_{i}' for i in all_states]].values - true_train_df[[f'state_{i}' for i in all_states]].values), axis=0)
            test_mae = np.mean(np.abs(test_df[[f'state_{i}' for i in all_states]].values - true_test_df[[f'state_{i}' for i in all_states]].values), axis=0)

            results['Train RMSE'].append(train_rmse)
            results['Test RMSE'].append(test_rmse)
            results['Train MAE'].append(train_mae)
            results['Test MAE'].append(test_mae)
        else:
            nan_array = np.full(len(all_states), np.nan)
            results['Train RMSE'].append(nan_array)
            results['Test RMSE'].append(nan_array)
            results['Train MAE'].append(nan_array)
            results['Test MAE'].append(nan_array)

    for metric in ['Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE']:
        results[metric] = np.array(results[metric])

    rad_to_deg = 180.0 / np.pi
    for metric_name in results:
        if len(results[metric_name]) > 0:
            results[metric_name][:, 2] *= rad_to_deg
            results[metric_name][:, 5] *= rad_to_deg

    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    fig.suptitle(title, fontsize=16)

    label_list = ["x(m)", "y(m)", "$\\psi$(deg)", "u(m/s)", "v(m/s)", "r(deg/s)"]
    metrics_rows = [('Train RMSE', 'RMSE on Training Dataset'), ('Test RMSE', 'RMSE on Test Dataset')]
    
    x_positions = np.arange(len(downsamples))
    
    for row, (metric_name, row_title) in enumerate(metrics_rows):
        for state_idx in range(6):
            ax = axes[row, state_idx]
            
            data = results[metric_name][:, state_idx]
            ax.plot(x_positions, data, marker='o', color='tab:blue', label="Downsample", linewidth=2, markersize=6)
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels(downsamples)
            ax.set_title(f'{label_list[state_idx]}')
            ax.set_xlabel('Downsample')
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

    true_test_filepath = os.path.join(base_dir, f"{dataset_prefix}.csv")
    if os.path.exists(true_test_filepath):
        true_df = pd.read_csv(true_test_filepath)
        ax.plot(true_df['state_0'], true_df['state_1'], 'k-', label='True Trajectory', linewidth=3)
        true_x = true_df['state_0'].values
        true_y = true_df['state_1'].values
    else:
        print("True dataset file not found!")
        true_x, true_y = None, None

    for downsample in downsamples:
        test_filename = f"DEDownsample={downsample}_{dataset_prefix}_prediction.csv"
        test_filepath = os.path.join(base_dir, test_filename)

        if os.path.exists(test_filepath):
            df = pd.read_csv(test_filepath)
            pred_x = df['state_0'].values
            pred_y = df['state_1'].values
            
            linestyle = '-'
            
            if true_x is not None and true_y is not None:
                min_len = min(len(pred_x), len(true_x))
                rmse = np.sqrt(np.mean((pred_x[:min_len] - true_x[:min_len])**2 + (pred_y[:min_len] - true_y[:min_len])**2))
                label = f"DS={downsample} (RMSE: {rmse:.4f})"
            else:
                label = f"DS={downsample}"

            ax.plot(pred_x, pred_y, color=colors[downsample], linestyle=linestyle, label=label, linewidth=2)
        else:
            print(f"Warning: File not found {test_filepath}")

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
    plot_accuracy_comparison("RMSE and MAE Comparison")
    plot_trajectory_comparison("Trajectory (x, y) Comparison on Test Set", "TestSet")
    plot_trajectory_comparison("Trajectory (x, y) Comparison on Train Set", "TrainingSet")

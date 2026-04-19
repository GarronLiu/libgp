import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import utils

base_dir = "result/8mUSV_Recursive/ZigZag10deg"
conditions = ["wo_rec", 
              "rec", 
              "rec_wo_hyperUpdate", 
              "common_sgp_update",
              "rec_reprop", 
              "rec_wo_hyperUpdate_reprop",
              "common_sgp_update_reprop"]
labels = ["W/O REC", 
          "REC", 
          "REC W/O Hyper", 
          "COMMON SGP UPDATE",
          "REC REPROP", 
          "REC W/O Hyper REPROP",
          "COMMON SGP UPDATE REPROP"]
cmap = plt.get_cmap('tab10')
colors = {condition: cmap(i % 10) for i, condition in enumerate(conditions)}
# 将第2~4个条件的颜色依次分配给第4~7个条件（按1-based理解）
src_idx = [1, 2, 3]  # 2~4
for k, dst in enumerate(range(4, 7)):  # 5~7
    colors[conditions[dst]] = colors[conditions[src_idx[k % len(src_idx)]]]
lines = ["--", "-.", ":","--", "-.", ":","--"]
prediction_files = [os.path.join(base_dir, f"8mUSV_prediction_{condition}.csv") for condition in conditions]

ground_truth_file = os.path.join(base_dir, "TrainSet.csv")

# Read ground truth data
gt_data = pd.read_csv(ground_truth_file)

# Create figure with 6 subplots
utils.setup_matplotlib_style(single_column=False, figure_height=18)
fig, axes = plt.subplots(6, 2)
df_state_names = ['x', 'y', 'psi', 'u', 'v', 'r']
state_names = ['x', 'y', 'psi', 'u', 'v', 'r']
label_list = ["x(m)", "y(m)", "$\\psi$(rad)", "u(m/s)", "v(m/s)", "r(deg/s)"]
# Plot each state
for state_idx in range(6):
    ax = axes[state_idx, 0]
    
    # Plot ground truth
    ax.plot(gt_data['time'], gt_data[f'{df_state_names[state_idx]}'], 'k-', label='GT')
    
    # Plot predictions for first 3 conditions
    for line, condition, pred_file in zip(lines[:4], conditions[:4], prediction_files[:4]):
        pred_data = pd.read_csv(pred_file)
        ax.plot(pred_data['time'], pred_data[f'state_{state_idx}'], 
                color=colors[condition], label=labels[conditions.index(condition)], linestyle=line)
        #统计RMSE
        gt_interp = np.interp(pred_data['time'], gt_data['time'], gt_data[state_names[state_idx]])
        rmse = np.sqrt(np.mean((pred_data[f'state_{state_idx}'] - gt_interp)**2))
        print(f"[{condition}] state {state_names[state_idx]} RMSE: {rmse:.4f}")
        
        if state_idx == 1:
            gt_x_interp = np.interp(pred_data['time'], gt_data['time'], gt_data['x'])
            gt_y_interp = np.interp(pred_data['time'], gt_data['time'], gt_data['y'])
            rmse_pos = np.sqrt(np.mean((pred_data['state_0'] - gt_x_interp)**2 + (pred_data['state_1'] - gt_y_interp)**2))
            print(f"[{condition}] XY Position RMSE: {rmse_pos:.4f}")

    if(state_idx == 5):
        ax.set_xlabel('Time (s)')
    ax.set_ylabel(label_list[state_idx])
    if state_idx == 0:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=5, frameon=True)

# Plot remaining conditions in second column
for state_idx in range(6):
    ax = axes[state_idx, 1]
    
    # Plot ground truth
    ax.plot(gt_data['time'], gt_data[f'{state_names[state_idx]}'], 'k-', label='GT')
    
    # Plot predictions for last 2 conditions
    for line, condition, pred_file in zip(lines[4:], conditions[4:], prediction_files[4:]):
        pred_data = pd.read_csv(pred_file)
        ax.plot(pred_data['time'], pred_data[f'state_{state_idx}'], 
                color=colors[condition], label=labels[conditions.index(condition)], linestyle=line)
        #统计RMSE
        gt_interp = np.interp(pred_data['time'], gt_data['time'], gt_data[state_names[state_idx]])
        rmse = np.sqrt(np.mean((pred_data[f'state_{state_idx}'] - gt_interp)**2))
        print(f"[{condition}] state {state_names[state_idx]} RMSE: {rmse:.4f}")
        
        if state_idx == 1:
            gt_x_interp = np.interp(pred_data['time'], gt_data['time'], gt_data['x'])
            gt_y_interp = np.interp(pred_data['time'], gt_data['time'], gt_data['y'])
            rmse_pos = np.sqrt(np.mean((pred_data['state_0'] - gt_x_interp)**2 + (pred_data['state_1'] - gt_y_interp)**2))
            print(f"[{condition}] XY Position RMSE: {rmse_pos:.4f}")

    if(state_idx == 5):
        ax.set_xlabel('Time (s)')
    ax.set_ylabel(label_list[state_idx])
    if state_idx == 0:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=4, frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'state_comparison.svg'), dpi=150)
plt.show()

# 仅绘制 REC REPROP（3x2 子图 + 不确定性区间）
target_condition = "rec_reprop"
target_label = "REC REPROP"
pred_path = os.path.join(base_dir, f"8mUSV_prediction_{target_condition}.csv")
pred_data = pd.read_csv(pred_path)

utils.setup_matplotlib_style(single_column=False, figure_height=9)  # figure_height=3对于3行2列过于扁平，先改为12等比例
fig, axes = plt.subplots(3, 2)
axes = axes.flatten(order='F')

for i, ax in enumerate(axes):
    t = pred_data["time"].to_numpy()
    y_pred = pred_data[f"state_{i}"].to_numpy()

    ax.plot(gt_data["time"], gt_data[state_names[i]], "k-", label="GT")
    ax.plot(t, y_pred, color=colors[target_condition], label=target_label, linestyle="--")

    # 优先读取上下界；若无则尝试 std/var
    cov_col_names= [f"cov_{i}" for i in range(36)]
    
    # Extract diagonal element from 6x6 covariance matrix
    cov_data = pred_data[[f"cov_{i}" for i in range(36)]].to_numpy()
    cov_matrix = cov_data.reshape(-1, 6, 6)
    cov_sigma = np.sqrt(np.maximum(np.diagonal(cov_matrix, axis1=1, axis2=2), 0.0))

    # Use the covariance-derived sigma for the i-th state
    sigma = cov_sigma[:, i]
    ax.fill_between(t, y_pred - sigma, y_pred + sigma, color=colors[target_condition], alpha=0.2, linewidth=0, label="±1σ")

    if(i == 2 or i == 5):
        ax.set_xlabel('Time (s)')
    ax.set_ylabel(label_list[i])

handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(0.1, 0.95), ncol=3, frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "rec_reprop_state_uncertainty.svg"))
plt.show()




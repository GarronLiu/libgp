import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

base_dir = "result/8mUSV_Recursive/ZigZag10deg"
conditions = ["wo_rec", 
              "rec", 
              "rec_wo_hyperUpdate", 
              "rec_reprop", 
              "rec_wo_hyperUpdate_reprop"]
labels = ["W/O REC", 
          "REC", 
          "REC W/O Hyper", 
          "REC REPROP", 
          "REC W/O Hyper REPROP"]
cmap = plt.get_cmap('tab10')
colors = {condition: cmap(i % 10) for i, condition in enumerate(conditions)}
lines = ["--", "-.", ":", "-.", ":"]
prediction_files = [os.path.join(base_dir, f"8mUSV_prediction_{condition}.csv") for condition in conditions]

ground_truth_file = os.path.join(base_dir, "TrainSet.csv")

# Read ground truth data
gt_data = pd.read_csv(ground_truth_file)

# Create figure with 6 subplots
fig, axes = plt.subplots(6, 2, figsize=(12, 10))
state_names = ['x', 'y', 'psi', 'u', 'v', 'r']


# Plot each state
for state_idx in range(6):
    ax = axes[state_idx, 0]
    
    # Plot ground truth
    ax.plot(gt_data['time'], gt_data[f'{state_names[state_idx]}'], 'k-', linewidth=2, label='GT')
    
    # Plot predictions for first 3 conditions
    for line, condition, pred_file in zip(lines[:3], conditions[:3], prediction_files[:3]):
        pred_data = pd.read_csv(pred_file)
        ax.plot(pred_data['time'], pred_data[f'state_{state_idx}'], 
                color=colors[condition], label=labels[conditions.index(condition)], linestyle=line)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(state_names[state_idx])
    ax.grid(True, alpha=0.3)
    if state_idx == 0:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=4, frameon=True)

# Plot remaining conditions in second column
for state_idx in range(6):
    ax = axes[state_idx, 1]
    
    # Plot ground truth
    ax.plot(gt_data['time'], gt_data[f'{state_names[state_idx]}'], 'k-', linewidth=2, label='GT')
    
    # Plot predictions for last 2 conditions
    for line, condition, pred_file in zip(lines[3:], conditions[3:], prediction_files[3:]):
        pred_data = pd.read_csv(pred_file)
        ax.plot(pred_data['time'], pred_data[f'state_{state_idx}'], 
                color=colors[condition], label=labels[conditions.index(condition)], linestyle=line)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(state_names[state_idx])
    ax.grid(True, alpha=0.3)
    if state_idx == 0:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=3, frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'state_comparison.svg'), dpi=150)
plt.show()

# 仅绘制 REC REPROP（3x2 子图 + 不确定性区间）
target_condition = "rec_reprop"
target_label = "REC REPROP"
pred_path = os.path.join(base_dir, f"8mUSV_prediction_{target_condition}.csv")
pred_data = pd.read_csv(pred_path)

fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    t = pred_data["time"].to_numpy()
    y_pred = pred_data[f"state_{i}"].to_numpy()

    ax.plot(gt_data["time"], gt_data[state_names[i]], "k-", lw=1.8, label="GT")
    ax.plot(t, y_pred, color=colors[target_condition], lw=1.6, label=target_label, linestyle="--")

    # 优先读取上下界；若无则尝试 std/var
    cov_col_names= [f"cov_{i}" for i in range(36)]
    
    # Extract diagonal element from 6x6 covariance matrix
    cov_data = pred_data[[f"cov_{i}" for i in range(36)]].to_numpy()
    cov_matrix = cov_data.reshape(-1, 6, 6)
    cov_sigma = np.sqrt(np.maximum(np.diagonal(cov_matrix, axis1=1, axis2=2), 0.0))

    # Use the covariance-derived sigma for the i-th state
    sigma = cov_sigma[:, i]
    ax.fill_between(t, y_pred - sigma, y_pred + sigma, color=colors[target_condition], alpha=0.2, linewidth=0, label="±1σ")

    ax.set_title(state_names[i])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(state_names[i])
    ax.grid(alpha=0.3)

handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=True)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(base_dir, "rec_reprop_state_uncertainty.svg"), dpi=150)
plt.show()


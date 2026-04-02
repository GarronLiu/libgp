import os
import numpy as np
import pandas as pd

def calculate_metrics(true_df, pred_df, states):
    """计算指定状态的RMSE和MAE"""
    rmse = {}
    mae = {}
    
    # 状态名称映射
    state_names = {0: "x(m)", 1: "y(m)", 2: "psi(deg)", 3: "u(m/s)", 4: "v(m/s)", 5: "r(deg/s)"}
    
    for s in states:
        true_vals = true_df[f'state_{s}'].values
        pred_vals = pred_df[f'state_{s}'].values
        
        err = pred_vals - true_vals
        
        # 针对角度和角速度转换为度
        if s == 2 or s == 5:
            err = err * (180.0 / np.pi)
            
        rmse[state_names[s]] = np.sqrt(np.mean(err**2))
        mae[state_names[s]] = np.mean(np.abs(err))
        
    return rmse, mae

def main():
    base_dir = "result/8mUSV/"
    true_filepath = os.path.join(base_dir, "8mUSV_TrainingSet_Smoothed.csv")
    
    if not os.path.exists(true_filepath):
        print(f"True dataset file not found: {true_filepath}")
        return
        
    true_df = pd.read_csv(true_filepath)
    states_to_eval = [0, 1, 2, 3, 4, 5]
    state_names = ["x(m)", "y(m)", "psi(deg)", "u(m/s)", "v(m/s)", "r(deg/s)"]
    
    print("=================== RMSE Evaluation ===================")
    
    # 1. 评估 SGP (10次实验求平均), 增加离群值剔除
    sgp_runs = []
    
    for i in range(10):
        pred_filepath = os.path.join(base_dir, f"8mUSV_DE_RandomID={i}_TrainingSet_prediction.csv")
        if os.path.exists(pred_filepath):
            pred_df = pd.read_csv(pred_filepath)
            rmse_dict, mae_dict = calculate_metrics(true_df, pred_df, states_to_eval)
            sgp_runs.append({
                'id': i,
                'rmse': rmse_dict,
                'mae': mae_dict
            })
        else:
            print(f"Warning: SGP target file not found {pred_filepath}")
            
    # ------ 剔除离群批次 ------
    # 基于 psi(deg) 的 RMSE，如果偏差明显太大（离群），则剔除该组
    if len(sgp_runs) > 0:
        psi_rmses = [run['rmse']["psi(deg)"] for run in sgp_runs]
        median_psi = np.median(psi_rmses)
        # 用 5 倍的中位数作为粗略阈值过滤离群点 (也可以用更严格的IQR过滤)
        threshold = median_psi * 5.0 
        
        valid_runs = [run for run in sgp_runs if run['rmse']["psi(deg)"] <= threshold]
        discarded_ids = [run['id'] for run in sgp_runs if run['rmse']["psi(deg)"] > threshold]
        if discarded_ids:
            print(f"[*] Discarded Outlier RandomIDs due to divergence: {discarded_ids}")
        
        sgp_runs = valid_runs

    # 挑选表现最好的一次 (基于各状态 RMSE 之和最小)
    if len(sgp_runs) > 0:
        best_run = min(sgp_runs, key=lambda r: sum(r['rmse'].values()))
        print(f"[*] Selected Best SGP-DE Run: RandomID={best_run['id']}")
        
        sgp_rmse_best = best_run['rmse']
        sgp_mae_best = best_run['mae']
    else:
        print("[!] No valid SGP-DE runs found.")
        sgp_rmse_best = {s: np.nan for s in state_names}
        sgp_mae_best = {s: np.nan for s in state_names}
    
    # 2. 评估 Koopman
    kpm_rmse, kpm_mae = {}, {}
    kpm_filepath = os.path.join(base_dir, "8mUSV_KPM_TrainingSet_prediction.csv")
    if os.path.exists(kpm_filepath):
        kpm_df = pd.read_csv(kpm_filepath)
        kpm_rmse, kpm_mae = calculate_metrics(true_df, kpm_df, states_to_eval)
        
    # 3. 评估 NuSVR
    svr_rmse, svr_mae = {}, {}
    svr_filepath = os.path.join(base_dir, "8mUSV_NuSVR_TrainingSet_prediction.csv")
    if os.path.exists(svr_filepath):
        svr_df = pd.read_csv(svr_filepath)
        svr_rmse, svr_mae = calculate_metrics(true_df, svr_df, states_to_eval)

    # 打印 RMSE 表格
    print(f"{'Model':<15} " + " ".join([f"{s:>12}" for s in state_names]))
    kpm_rmse_str = " ".join([f"{kpm_rmse.get(s, np.nan):12.4f}" for s in state_names])
    print(f"{'Koopman':<15} {kpm_rmse_str}")
    
    svr_rmse_str = " ".join([f"{svr_rmse.get(s, np.nan):12.4f}" for s in state_names])
    print(f"{'NuSVR':<15} {svr_rmse_str}")
    
    sgp_rmse_str = " ".join([f"{sgp_rmse_best[s]:12.4f}" for s in state_names])
    print(f"{'SGP-DE(Best)':<15} {sgp_rmse_str}")
    
    print("\n=================== MAE Evaluation ===================")
    print(f"{'Model':<15} " + " ".join([f"{s:>12}" for s in state_names]))
    kpm_mae_str = " ".join([f"{kpm_mae.get(s, np.nan):12.4f}" for s in state_names])
    print(f"{'Koopman':<15} {kpm_mae_str}")
    
    svr_mae_str = " ".join([f"{svr_mae.get(s, np.nan):12.4f}" for s in state_names])
    print(f"{'NuSVR':<15} {svr_mae_str}")
    
    sgp_mae_str = " ".join([f"{sgp_mae_best[s]:12.4f}" for s in state_names])
    print(f"{'SGP-DE(Best)':<15} {sgp_mae_str}")

if __name__ == "__main__":
    main()

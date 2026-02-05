import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import appGaussian as appssian
from app import appMNIST as appNIST

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 設定
    # ---------------------------------------------------------
    Nneuron = 800   
    Nx = 784
    Nclasses = 10        
    
    leak = 50       
    dt = 0.001      
    
    alpha = 0.18    
    beta = 1 / 0.9  
    mu = 0.02 / 0.9
    
    Thresh = 0.5

    Duration = 50

    lr_readout=0.0005
    
    # 保存先設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S(suggestGaussian_exp)")
    base_save_dir = Path("results")
    current_save_dir = base_save_dir / timestamp
    current_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved in: {current_save_dir}")
    
    # ---------------------------------------------------------                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    # データ準備
    # ---------------------------------------------------------
    print("Preparing Data...")

    # まず元のデータを読み込む
    X_train_org, y_train_org = appNIST.load_and_preprocess("identity_test_images.npy", "identity_test_labels.npy")
    X_test_org, y_test_org = appNIST.load_and_preprocess("motionblur_test_images.npy", "motionblur_test_labels.npy")

    # --- 修正箇所: Duration分だけデータを引き延ばす ---
    # axis=0 (サンプル方向) に Duration 回繰り返す
    X_train = np.repeat(X_train_org, Duration, axis=0)
    y_train = np.repeat(y_train_org, Duration, axis=0)
    
    X_test = np.repeat(X_test_org, Duration, axis=0)
    y_test = np.repeat(y_test_org, Duration, axis=0)
    # --------------------------------------------------
    # ---------------------------------------------------------
    # Set 1: Learning(non)
    # ---------------------------------------------------------
    print("--- Phase 1: Learning on Set 1 ---")
    F_initial, C_initial, *_ = appssian.init_weights(Nx, Nneuron, Nclasses)

    # ---------------------------------------------------------
    # Set 1: Learning
    # ---------------------------------------------------------
    print("--- Phase 1: Learning on Set 1 ---")
    F_initial, C_initial, *_ = appssian.init_weights(Nx, Nneuron, Nclasses)

    spk_t_1, spk_i_1, F_set1, C_set1, mem_var_1, acc_his_1, final_states_1, W_1 = appNIST.test_train_continuous_correlated_proposed(
                        F_initial, C_initial, X_train, y_train, # y_dataを追加
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=30,
                        eps=0.00001, 
                        la=0.2, Ucc_scale=2.0, # Figure 5用の追加パラメータ
                        init_states=None,
                        lr_readout=lr_readout, stim_duration=Duration)

    # 戻り値の最後に final_states_1 を受け取る
    nspk_t_1, nspk_i_1, nF_set1, nC_set1, nmem_var_1, nacc_his_1, nfinal_states_1, nW_1 = appNIST.test_train_continuous_correlated(
                        F_initial, C_initial, X_train, y_train,
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=30,
                        epsr=0.00005, epsf=0.000005, 
                        la=0.2, Ucc_scale=2.0, # Figure 5用の追加パラメータ
                        init_states=None,
                        lr_readout=lr_readout, stim_duration=Duration)
    
     # ---------------------------------------------------------
    # Set 2: Learning
    # ---------------------------------------------------------
    print("--- Phase 2: Learning on Set 2 ---")

    # 状態の引き継ぎ忘れないように注意
    spk_t_2, spk_i_2, F_set2, C_set2, mem_var_2, acc_his_2, final_states_2, W_2 = appNIST.test_train_continuous_correlated_proposed(
                        F_set1, C_set1, X_test, y_test, # y_dataを追加
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=30,
                        eps=0.00001, 
                        la=0.2, Ucc_scale=2.0, # Figure 5用の追加パラメータ
                        init_states=final_states_1,
                        lr_readout=lr_readout, stim_duration=Duration)
    
    
    # ---------------------------------------------------------
    # Set 2: Learning(non)
    # ---------------------------------------------------------
    print("--- Phase 2: Learning on Set 2 ---")

    # init_states に Set 1 の終わりの状態を渡す
    nspk_t_2, nspk_i_2, nF_set2, nC_set2, nmem_var_2, nacc_his_2, nfinal_states_2, nW_2 = appNIST.test_train_continuous_correlated(
                        nF_set1, nC_set1, X_test, y_test,
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=30,
                        epsr=0.00005, epsf=0.000005, 
                        la=0.2, Ucc_scale=2.0, # Figure 5用の追加パラメータ
                        init_states=nfinal_states_1,
                        lr_readout=lr_readout, stim_duration=Duration)
    
   

    # ---------------------------------------------------------
    # Data Combination
    # ---------------------------------------------------------
    print("Combining data for plots...")
    
    # Membrane Variance の結合
    full_mem_var = mem_var_1 + mem_var_2

    # 2. Spikeデータの結合
    # 計算用にnumpy配列へ変換
    t1 = np.array(spk_t_1)
    i1 = np.array(spk_i_1)
    t2 = np.array(spk_t_2)
    i2 = np.array(spk_i_2)

    # オフセットの計算 (Set1の最大時間。データが無い場合は0)
    time_offset = np.max(t1) if len(t1) > 0 else 0
    
    # 時刻をシフトして結合
    full_spk_t = np.concatenate([t1, t2 + time_offset])
    full_spk_i = np.concatenate([i1, i2])

    record_interval_steps = 100
    eval_interval = 10000

    # # ---------------------------------------------------------
    # # Plot 1: Raster Plot (Combined) - Modified
    # # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    plt.scatter(full_spk_t, full_spk_i, s=0.5, c='black', marker='.', alpha=0.6)

    plt.axvline(x=time_offset, color='red', linestyle='--', label='End of Set 1')

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title('Spike Raster Plot')
    plt.xlim(left=0, right=np.max(full_spk_t)) # 軸の範囲をデータの最大値に合わせる
    plt.ylim(0, Nneuron)
    plt.legend(loc='upper right')

    raster_plot_path = current_save_dir / "Final_Raster.png"
    plt.savefig(raster_plot_path)
    plt.close()
    print(f"Raster plot saved to: {raster_plot_path}")

    # ---------------------------------------------------------
    # Plot 2: Membrane Potential Variance (新規追加)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # 移動平均
    mem_window_size = 500
    if len(full_mem_var) >= mem_window_size:
        b = np.ones(mem_window_size) / mem_window_size
        full_mem_smooth = np.convolve(full_mem_var, b, mode='valid')
        
        # 生データを薄くプロット
        plt.plot(full_mem_var, label='Raw Variance', color='lightgreen', alpha=0.4)
        # 移動平均を濃くプロット
        plt.plot(np.arange(mem_window_size - 1, len(full_mem_var)), full_mem_smooth, 
                 label=f'Moving Average (window={mem_window_size})', color='green', linewidth=2)
    else:
        plt.plot(full_mem_var, label='Voltage Variance', color='green')

    plt.axvline(x=len(X_train), color='red', linestyle='--', label='Covariate Shift Point')
    
    plt.xlabel('Input Samples')
    plt.ylabel('Voltage Variance')
    
    plt.grid(True)
    plt.legend()
    
    # 対数スケール
    plt.yscale('log') 
    
    mem_plot_path = current_save_dir / "Final_Voltage_Variance.png"
    plt.savefig(mem_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Voltage Variance plot saved to: {mem_plot_path}")

    ##--------------------------------------------------------------------
    # ---------------------------------------------------------
    # Data Combination
    # ---------------------------------------------------------
    print("Combining data for plots...")
    
    # Membrane Variance の結合
    nfull_mem_var = nmem_var_1 + nmem_var_2

    # 2. Spikeデータの結合
    # 計算用にnumpy配列へ変換
    nt1 = np.array(nspk_t_1)
    ni1 = np.array(nspk_i_1)
    nt2 = np.array(nspk_t_2)
    ni2 = np.array(nspk_i_2)

    # オフセットの計算 (Set1の最大時間。データが無い場合は0)
    ntime_offset = np.max(nt1) if len(nt1) > 0 else 0
    
    # 時刻をシフトして結合
    nfull_spk_t = np.concatenate([nt1, nt2 + ntime_offset])
    nfull_spk_i = np.concatenate([ni1, ni2])
    

    nrecord_interval_steps = 100
    neval_interval = 10000

    # # ---------------------------------------------------------
    # # Plot 1: Raster Plot (Combined) - Modified
    # # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    plt.scatter(nfull_spk_t, nfull_spk_i, s=0.5, c='black', marker='.', alpha=0.6)

    plt.axvline(x=ntime_offset, color='red', linestyle='--', label='End of Set 1')

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title('Spike Raster Plot')
    plt.xlim(left=0, right=np.max(nfull_spk_t)) # 軸の範囲をデータの最大値に合わせる
    plt.ylim(0, Nneuron)
    plt.legend(loc='upper right')

    raster_plot_path = current_save_dir / "Final_Raster(non).png"
    plt.savefig(raster_plot_path)
    plt.close()
    print(f"Raster plot saved to: {raster_plot_path}")

    # ---------------------------------------------------------
    # Plot 2: Membrane Potential Variance (新規追加)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # 移動平均
    nmem_window_size = 500
    if len(nfull_mem_var) >= nmem_window_size:
        nb = np.ones(nmem_window_size) / nmem_window_size
        nfull_mem_smooth = np.convolve(nfull_mem_var, nb, mode='valid')
        
        # 生データを薄くプロット
        plt.plot(nfull_mem_var, label='Raw Variance', color='lightgreen', alpha=0.4)
        # 移動平均を濃くプロット
        plt.plot(np.arange(nmem_window_size - 1, len(nfull_mem_var)), nfull_mem_smooth, 
                 label=f'Moving Average (window={nmem_window_size})', color='green', linewidth=2)
    else:
        plt.plot(nfull_mem_var, label='Voltage Variance', color='green')

    plt.axvline(x=len(X_train), color='red', linestyle='--', label='Covariate Shift Point')
    
    plt.xlabel('Input Samples')
    plt.ylabel('Voltage Variance')
    plt.grid(True)
    plt.legend()
    
    # 対数スケール
    plt.yscale('log') 
    
    mem_plot_path = current_save_dir / "Final_Voltage_Variance(non).png"
    plt.savefig(mem_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Voltage Variance plot saved to: {mem_plot_path}")
    ##---------------------------------------------------------------------------------------------------------------

    # =========================================================
    # 追加: Accuracy Plots (精度グラフの作成) - Modified for Reset
    # =========================================================
    print("Plotting Accuracy History...")

    acc_window = 300  # 移動平均のウィンドウサイズ

    # ---------------------------------------------------------
    # Plot 3: Accuracy History (Proposed / Learning)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # --- Set 1 のプロット ---
    if len(acc_his_1) >= acc_window:
        window = np.ones(acc_window) / acc_window
        acc_smooth_1 = np.convolve(acc_his_1, window, mode='valid')
        # x軸: Set 1 の範囲
        x_axis_1 = np.arange(acc_window - 1, len(acc_his_1))
        plt.plot(x_axis_1, acc_smooth_1, label='Set 1', color='blue', linewidth=1.5)
    else:
        plt.plot(acc_his_1, label='Set 1 (Raw)', color='lightblue')

    # --- Set 2 のプロット (リセットして計算) ---
    if len(acc_his_2) >= acc_window:
        window = np.ones(acc_window) / acc_window
        acc_smooth_2 = np.convolve(acc_his_2, window, mode='valid')
        # x軸: Set 2 の範囲 (Set 1 の長さをオフセットとして足す)
        x_axis_2 = np.arange(acc_window - 1, len(acc_his_2)) + len(acc_his_1)
        plt.plot(x_axis_2, acc_smooth_2, label='Set 2 ', color='darkblue', linewidth=1.5)
    else:
        x_axis_raw_2 = np.arange(len(acc_his_2)) + len(acc_his_1)
        plt.plot(x_axis_raw_2, acc_his_2, label='Set 2', color='blue', alpha=0.5)

    # 境界線
    plt.axvline(x=len(acc_his_1), color='red', linestyle='--', label='Boundary')

    plt.xlabel('Input Samples (Images)')
    plt.ylabel('Accuracy (Moving Avg)')
    plt.legend(loc='lower right')
    plt.ylim(0.50, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    acc_plot_path = current_save_dir / "Final_Accuracy.png"
    plt.savefig(acc_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Accuracy plot saved to: {acc_plot_path}")

    # ---------------------------------------------------------
    # Plot 4: Accuracy History (Non-Learning / Control)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # --- Set 1 (Non) のプロット ---
    if len(nacc_his_1) >= acc_window:
        window = np.ones(acc_window) / acc_window
        nacc_smooth_1 = np.convolve(nacc_his_1, window, mode='valid')
        x_axis_n1 = np.arange(acc_window - 1, len(nacc_his_1))
        plt.plot(x_axis_n1, nacc_smooth_1, label='Set 1', color='orange', linewidth=1.5)
    
    # --- Set 2 (Non) のプロット (リセットして計算) ---
    if len(nacc_his_2) >= acc_window:
        window = np.ones(acc_window) / acc_window
        nacc_smooth_2 = np.convolve(nacc_his_2, window, mode='valid')
        # x軸オフセット
        x_axis_n2 = np.arange(acc_window - 1, len(nacc_his_2)) + len(nacc_his_1)
        plt.plot(x_axis_n2, nacc_smooth_2, label='Set 2', color='darkorange', linewidth=1.5)

    # 境界線
    plt.axvline(x=len(nacc_his_1), color='red', linestyle='--', label='Boundary')

    plt.xlabel('Input Samples (Images)')
    plt.ylabel('Accuracy (Moving Avg)')
    plt.legend(loc='lower right')
    plt.ylim(0.40, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    nacc_plot_path = current_save_dir / "Final_Accuracy(non).png"
    plt.savefig(nacc_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Accuracy plot (non) saved to: {nacc_plot_path}")
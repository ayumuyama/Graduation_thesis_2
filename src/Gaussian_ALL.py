import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import appGaussian as appssian

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 設定
    # ---------------------------------------------------------
    Nneuron = 20   
    Nx = 2
    Nclasses = 2        
    
    leak = 50       
    dt = 0.001      
    
    alpha = 0.18    
    beta = 1 / 0.9  
    mu = 0.02 / 0.9
    
    Thresh = 0.5
    
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

    X = appssian.generate_continuous_shift_dataset(n_train=1000000, n_test=1000000, nx=2, sigma=5, seed=30,
                                      train_params={'mean': [0.0, 0.0], 'std': [2.0, 2.0]},
                                      test_params={'mean': [0.0, 0.0], 'std': [6.0, 6.0]})
    
    print(X.shape)
    

    X_train = X[:1000000]
    X_test = X[1000000:]

    # ---------------------------------------------------------
    # Set 1: Learning(non)
    # ---------------------------------------------------------
    print("--- Phase 1: Learning on Set 1 ---")
    F_initial, C_initial, *_ = appssian.init_weights(Nx, Nneuron, Nclasses)

    # 戻り値の最後に final_states_1 を受け取る
    nspk_t_1, nspk_i_1, nF_set1, nC_set1, nmem_var_1, nw_err_1, nd_err_1, nfinal_states_1 = appssian.test_train_continuous_correlated(F_initial, C_initial, X_train,
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain=True, Gain=80,
                          epsr=0.00003, epsf=0.000003, 
                          la=0.2, Ucc_scale=2.0, # Figure 5用の追加パラメータ
                          init_states=None)
    
    # ---------------------------------------------------------
    # Set 1: Learning
    # ---------------------------------------------------------
    print("--- Phase 1: Learning on Set 1 ---")
    F_initial, C_initial, *_ = appssian.init_weights(Nx, Nneuron, Nclasses)

    spk_t_1, spk_i_1, F_set1, C_set1, mem_var_1, w_err_1, d_err_1, final_states_1 = appssian.test_train_continuous_correlated_proposed(
        F_initial, C_initial, X_train,
        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
        alpha, beta, mu, retrain=True, Gain=80,
        la=0.2, Ucc_scale=2.0,
        eps=0.00003, init_states=None)
    
    # ---------------------------------------------------------
    # Set 2: Learning(non)
    # ---------------------------------------------------------
    print("--- Phase 2: Learning on Set 2 ---")

    # init_states に Set 1 の終わりの状態を渡す
    nspk_t_2, nspk_i_2, nF_set2, nC_set2, nmem_var_2, nw_err_2, nd_err_2, nfinal_states_2 = appssian.test_train_continuous_correlated(
                          nF_set1, nC_set1, X_test,
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain=True, Gain=80,
                          epsr=0.00003, epsf=0.000003,
                          la=0.2, Ucc_scale=2.0,
                          init_states=nfinal_states_1)
    
    # ---------------------------------------------------------
    # Set 2: Learning
    # ---------------------------------------------------------
    print("--- Phase 2: Learning on Set 2 ---")

    # 状態の引き継ぎ忘れないように注意
    spk_t_2, spk_i_2, F_set2, C_set2, mem_var_2, w_err_2, d_err_2, final_states_2 = appssian.test_train_continuous_correlated_proposed(
        F_set1, C_set1, X_test,
        Nneuron, Nx, Nclasses, dt, leak, Thresh,
        alpha, beta, mu, retrain=True, Gain=80,
        la=0.2, Ucc_scale=2.0,
        eps=0.00003, init_states=final_states_1
    )

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
    full_weight_error = w_err_1 + w_err_2
    full_dec_err = d_err_1 + d_err_2

    record_interval_steps = 100
    time_axis_error = np.arange(len(full_weight_error)) * dt * record_interval_steps
    eval_interval = 10000
    time_axis_dec = np.arange(len(full_dec_err)) * dt * eval_interval

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
    mem_window_size = 1000
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
    
    plt.xlabel('Input Samples', fontsize=22)
    plt.ylabel('Voltage Variance', fontsize=22)
    # 【追加】軸の数値（刻み目ラベル）のフォントサイズを大きくする
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True)
    plt.legend(loc='lower left', fontsize=18, framealpha=0.8)
    
    # 対数スケール
    plt.yscale('log') 
    plt.ylim(0.0005, 5)
    
    mem_plot_path = current_save_dir / "Final_Voltage_Variance.png"
    plt.savefig(mem_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Voltage Variance plot saved to: {mem_plot_path}")


    # ---------------------------------------------------------
    # ★新規追加: Plot 4: Evolution of Decoding Error
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    plt.plot(time_axis_dec, full_dec_err, 'o-', color='black', label='Decoding Error', markersize=4)
    
    # 境界線
    time_offset_sec = len(X_train) * dt
    plt.axvline(x=time_offset_sec, color='red', linestyle='--', label='Domain Shift Point')

    plt.xlabel('Time (s)', fontsize=24)
    plt.ylabel('Decoding Error', fontsize=24)
    
    plt.tick_params(axis='both', labelsize=20)

    plt.ylim(0.0, 0.2)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.legend(fontsize=20, loc='upper left')

    plt.savefig(current_save_dir / "Final_Decoding_Error.png", bbox_inches='tight')
    plt.close()

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
    nfull_weight_error = nw_err_1 + nw_err_2
    nfull_dec_err = nd_err_1 + nd_err_2

    nrecord_interval_steps = 100
    ntime_axis_error = np.arange(len(nfull_weight_error)) * dt * nrecord_interval_steps
    neval_interval = 10000
    ntime_axis_dec = np.arange(len(nfull_dec_err)) * dt * neval_interval

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
    nmem_window_size = 1000
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
    
    plt.xlabel('Input Samples', fontsize=22)
    plt.ylabel('Voltage Variance', fontsize=22)
    # 【追加】軸の数値（刻み目ラベル）のフォントサイズを大きくする
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True)
   
    plt.legend(loc='lower left', fontsize=18, framealpha=0.8)
    # 対数スケール
    plt.yscale('log') 
    plt.ylim(0.0005, 5)
    
    mem_plot_path = current_save_dir / "Final_Voltage_Variance(non).png"
    plt.savefig(mem_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Voltage Variance plot saved to: {mem_plot_path}")


    # ---------------------------------------------------------
    # ★新規追加: Plot 4: Evolution of Decoding Error
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    plt.plot(ntime_axis_dec, nfull_dec_err, 'o-', color='black', label='Decoding Error', markersize=4)
    
    # 境界線
    ntime_offset_sec = len(X_train) * dt
    plt.axvline(x=ntime_offset_sec, color='red', linestyle='--', label='Domain Shift Point')

    plt.xlabel('Time (s)', fontsize=24)
    plt.ylabel('Decoding Error', fontsize=24)
    
    plt.tick_params(axis='both', labelsize=20)

    plt.ylim(0.0, 0.2)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.legend(fontsize=20, loc='upper left')

    plt.savefig(current_save_dir / "Final_Decoding_Error(non).png", bbox_inches='tight')
    plt.close()
    ##---------------------------------------------------------------------------------------------------------------

    # ---------------------------------------------------------
    # ★新規追加: Plot 5: Combined Decoding Error with Exponential Fit
    # ---------------------------------------------------------
    print("Generating Combined Decoding Error Plot with Fitting...")

    # フィッティング関数の定義: y = a * exp(-lambda * (t - t0)) + b
    def exponential_decay_func(t, a, b, lam, t0):
        # t0はフィッティング開始時刻
        return a * np.exp(-lam * (t - t0)) + b

    # データ準備 (リストの場合があるのでNumPy配列化)
    # Proposed (学習あり)
    dec_err_proposed = np.array(full_dec_err)
    time_proposed = time_axis_dec
    
    # Standard (学習なし/Non)
    dec_err_non = np.array(nfull_dec_err)
    time_non = ntime_axis_dec

    # ドメインシフトの時刻 (Set 1の終了時刻)
    t_shift = 1000.0  # 1,000,000 steps * 0.001 dt

    # プロット作成
    plt.figure(figsize=(12, 7))

    # --- 1. Proposed (学習あり) の描画とフィッティング ---
    # 元データのプロット
    plt.plot(time_proposed, dec_err_proposed, 'o-', color='black', label='Proposed', markersize=3, alpha=0.5)

    # フィッティング (t >= t_shift のデータのみ使用)
    mask_proposed = time_proposed >= t_shift
    if np.any(mask_proposed):
        t_fit_p = time_proposed[mask_proposed]
        y_fit_p = dec_err_proposed[mask_proposed]
        
        # 初期値の推定 [振幅a, オフセットb, 減衰係数lambda, 開始時刻t0]
        # lambdaの初期値は適当な正の値(例: 0.01)を設定
        p0 = [np.max(y_fit_p) - np.min(y_fit_p), np.min(y_fit_p), 0.01, t_shift]
        
        try:
            # t0は固定するため、lambda関数でラップするか、引数を工夫する
            # ここでは t0 を固定パラメータとして関数に埋め込む形で最適化します
            popt_p, pcov_p = curve_fit(lambda t, a, b, lam: exponential_decay_func(t, a, b, lam, t_shift), 
                                       t_fit_p, y_fit_p, p0=p0[:3], maxfev=10000)
            
            a_p, b_p, lam_p = popt_p
            
            # フィッティング曲線の描画
            y_fit_curve_p = exponential_decay_func(t_fit_p, a_p, b_p, lam_p, t_shift)
            plt.plot(t_fit_p, y_fit_curve_p, '--', color='orange', linewidth=2.5, 
                     label=f'Fit Proposed ($\lambda={lam_p:.4f}$)')
            print(f"Proposed Lambda: {lam_p}")
        except Exception as e:
            print(f"Fitting failed for Proposed: {e}")

    # --- 2. Standard (学習なし) の描画とフィッティング ---
    # 元データのプロット
    plt.plot(time_non, dec_err_non, 'o-', color='blue', label='Standard', markersize=3, alpha=0.5)

    # フィッティング
    mask_non = time_non >= t_shift
    if np.any(mask_non):
        t_fit_n = time_non[mask_non]
        y_fit_n = dec_err_non[mask_non]
        
        p0 = [np.max(y_fit_n) - np.min(y_fit_n), np.min(y_fit_n), 0.01, t_shift]
        
        try:
            popt_n, pcov_n = curve_fit(lambda t, a, b, lam: exponential_decay_func(t, a, b, lam, t_shift), 
                                       t_fit_n, y_fit_n, p0=p0[:3], maxfev=10000)
            a_n, b_n, lam_n = popt_n
            
            # フィッティング曲線の描画
            y_fit_curve_n = exponential_decay_func(t_fit_n, a_n, b_n, lam_n, t_shift)
            plt.plot(t_fit_n, y_fit_curve_n, '--', color='cyan', linewidth=2.5, 
                     label=f'Fit Standard ($\lambda={lam_n:.4f}$)')
            print(f"Standard Lambda: {lam_n}")
        except Exception as e:
            print(f"Fitting failed for Standard: {e}")

    # --- グラフの装飾 ---
    plt.axvline(x=t_shift, color='red', linestyle='--', label='Covariate Shift Point (t=1000)')
    
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Decoding Error', fontsize=16)
    
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.ylim(0.0, 0.2) # 必要に応じて範囲を調整してください

    save_path = current_save_dir / "Final_Decoding_Error_Combined_Fit.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Combined fit plot saved to: {save_path}")

# Trainデータのプロット
plt.figure(figsize=(8, 8))
# Trainデータ（X_train）を青い点(c='blue')でプロット
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', alpha=0.5, label='Train Data')
# 軌跡を描画
plt.plot(X_train[:, 0], X_train[:, 1], c='gray', alpha=0.2, linewidth=1)

plt.xlabel("Input Dimension 1", fontsize=20)
plt.ylabel("Input Dimension 2", fontsize=20)
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.xlim(-7.0, 7.0) 
plt.ylim(-7.0, 7.0) 
save_path = current_save_dir / "Set_1_data.png"
plt.savefig(save_path, bbox_inches='tight')
plt.close()

# Testデータのプロット
plt.figure(figsize=(8, 8))
# Testデータ（X_test）を赤い点(c='red')に変更
plt.scatter(X_test[:, 0], X_test[:, 1], c='red', alpha=0.5, label='Test Data')
# 軌跡を描画
plt.plot(X_test[:, 0], X_test[:, 1], c='gray', alpha=0.2, linewidth=1)

plt.xlabel("Input Dimension 1", fontsize=20)
plt.ylabel("Input Dimension 2", fontsize=20)
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.xlim(-7.0, 7.0) 
plt.ylim(-7.0, 7.0) 
save_path = current_save_dir / "Set_2_data.png"
plt.savefig(save_path, bbox_inches='tight')
plt.close()
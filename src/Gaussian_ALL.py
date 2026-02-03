import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

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
                                      train_params={'mean': [0.0, 0.0], 'std': [5.0, 1.0]},
                                      test_params={'mean': [0.0, 0.0], 'std': [1.0, 5.0]})

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
                          alpha, beta, mu, retrain=True, Gain=100,
                          epsr=0.00005, epsf=0.000005, 
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
        alpha, beta, mu, retrain=True, Gain=100,
        la=0.2, Ucc_scale=2.0,
        eps=0.00002, init_states=None)
    
    # ---------------------------------------------------------
    # Set 2: Learning(non)
    # ---------------------------------------------------------
    print("--- Phase 2: Learning on Set 2 ---")

    # init_states に Set 1 の終わりの状態を渡す
    nspk_t_2, nspk_i_2, nF_set2, nC_set2, nmem_var_2, nw_err_2, nd_err_2, nfinal_states_2 = appssian.test_train_continuous_correlated(
                          nF_set1, nC_set1, X_test,
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain=True, Gain=100,
                          epsr=0.00005, epsf=0.000005,
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
        alpha, beta, mu, retrain=True, Gain=100,
        la=0.2, Ucc_scale=2.0,
        eps=0.00002, init_states=final_states_1
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

    plt.axvline(x=len(X_train), color='red', linestyle='--', label='End of Set 1')
    
    plt.xlabel('Input Samples')
    plt.ylabel('Voltage Variance per Neuron')
    plt.title('Evolution of the Variance of the Membrane Potential')
    plt.grid(True)
    plt.legend()
    
    # 対数スケール
    plt.yscale('log') 
    
    mem_plot_path = current_save_dir / "Final_Voltage_Variance.png"
    plt.savefig(mem_plot_path)
    plt.close()
    print(f"Voltage Variance plot saved to: {mem_plot_path}")

    # ---------------------------------------------------------
    # ★新規追加: Plot 3: Distance to Optimal Weights
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis_error, full_weight_error, color='purple', label='Distance to Optimal Weights')
    
    # Set 1 と Set 2 の境界線
    end_time_set1 = len(w_err_1) * dt * record_interval_steps
    plt.axvline(x=end_time_set1, color='red', linestyle='--', label='End of Set 1')

    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Distance')
    plt.title('Convergence to Optimal Recurrent Weights (E-I Balance)')
    plt.yscale('log') # 対数
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    

    weight_plot_path = current_save_dir / "Final_Weight_Convergence.png"
    plt.savefig(weight_plot_path)
    plt.close()
    print(f"Weight Convergence plot saved to: {weight_plot_path}")

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

    plt.axvline(x=len(X_train), color='red', linestyle='--', label='End of Set 1')
    
    plt.xlabel('Input Samples')
    plt.ylabel('Voltage Variance per Neuron')
    plt.title('Evolution of the Variance of the Membrane Potential')
    plt.grid(True)
    plt.legend()
    
    # 対数スケール
    plt.yscale('log') 
    
    mem_plot_path = current_save_dir / "Final_Voltage_Variance(non).png"
    plt.savefig(mem_plot_path)
    plt.close()
    print(f"Voltage Variance plot saved to: {mem_plot_path}")

    # ---------------------------------------------------------
    # ★新規追加: Plot 3: Distance to Optimal Weights
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(ntime_axis_error, nfull_weight_error, color='purple', label='Distance to Optimal Weights')
    
    # Set 1 と Set 2 の境界線
    nend_time_set1 = len(nw_err_1) * dt * nrecord_interval_steps
    plt.axvline(x=nend_time_set1, color='red', linestyle='--', label='End of Set 1')

    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Distance')
    plt.title('Convergence to Optimal Recurrent Weights (E-I Balance)')
    plt.yscale('log') # 対数
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    

    weight_plot_path = current_save_dir / "Final_Weight_Convergence(non).png"
    plt.savefig(weight_plot_path)
    plt.close()
    print(f"Weight Convergence plot saved to: {weight_plot_path}")

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

# Trainデータのプロット
plt.figure(figsize=(8, 8))
# Trainデータ（X_train）を青い点(c='blue')でプロット
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', alpha=0.5, label='Train Data')
# 軌跡を描画
plt.plot(X_train[:, 0], X_train[:, 1], c='gray', alpha=0.2, linewidth=1)

plt.title("Set 1 data (Train)")
plt.xlabel("Input Dimension 1")
plt.ylabel("Input Dimension 2")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig("Set_1_data.png")
plt.close()

# Testデータのプロット
plt.figure(figsize=(8, 8))
# Testデータ（X_test）を赤い点(c='red')に変更
plt.scatter(X_test[:, 0], X_test[:, 1], c='red', alpha=0.5, label='Test Data')
# 軌跡を描画
plt.plot(X_test[:, 0], X_test[:, 1], c='gray', alpha=0.2, linewidth=1)

plt.title("Set 2 data (Test)")
plt.xlabel("Input Dimension 1")
plt.ylabel("Input Dimension 2")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig("Set_2_data.png")
plt.close()

# Totalデータのプロット
plt.figure(figsize=(8, 8))
# 全体の軌跡を描画（背景として先に描画）
plt.plot(X[:, 0], X[:, 1], c='gray', alpha=0.2, linewidth=1)

# Trainデータ（青）とTestデータ（赤）を重ねてプロット
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', alpha=0.5, label='Train Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c='red', alpha=0.5, label='Test Data')

plt.title("Total data")
plt.xlabel("Input Dimension 1")
plt.ylabel("Input Dimension 2")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig("Total_data.png")
plt.close()
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

    X = appssian.generate_continuous_shift_dataset(n_train=400000, n_test=400000, nx=2, sigma=5, seed=23,
                                      train_params={'mean': [0.0, 0.0], 'std': [1.0, 4.0]},
                                      test_params={'mean': [0.0, 0.0], 'std': [4.0, 1.0]})

    X_train = X[:400000]
    X_test = X[400000:]
    # ---------------------------------------------------------
    # Set 1: Learning
    # ---------------------------------------------------------
    print("--- Phase 1: Learning on Set 1 ---")
    F_initial, C_initial, *_ = appssian.init_weights(Nx, Nneuron, Nclasses)

    # 戻り値の最後に final_states_1 を受け取る
    # spk_t_1, spk_i_1, F_set1, C_set1, mem_var_1, weight_error_1, final_states_1 = appssian.test_train_continuous_suggest_nonclass(
    #     F_initial, C_initial, X_train,
    #     Nneuron, Nx, Nclasses, dt, leak, Thresh,
    #     alpha, beta, mu, retrain=True, Gain=200, eps=0.0001, init_states=None)

    spk_t_1, spk_i_1, F_set1, C_set1, mem_var_1, weight_error_1, final_states_1 = appssian.test_train_continuous_nonclass(
                          F_initial, C_initial, X_train,
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain=True, Gain=200,
                          epsr=0.001, epsf=0.0001, init_states=None)
    
    # ---------------------------------------------------------
    # Set 2: Learning
    # ---------------------------------------------------------
    print("--- Phase 2: Learning on Set 2 ---")

    # init_states に Set 1 の終わりの状態を渡す
    spk_t_2, spk_i_2, F_set2, C_set2, mem_var_2, weight_error_2, final_states_2 = appssian.test_train_continuous_suggest_nonclass(
        F_set1, C_set1, X_test,
        Nneuron, Nx, Nclasses, dt, leak, Thresh,
        alpha, beta, mu, retrain=True, Gain=200, eps=0.0001, init_states=final_states_1 # ★ここで引き継ぎ！
    )

    # ---------------------------------------------------------
    # Data Combination
    # ---------------------------------------------------------
    print("Combining data for plots...")
    
    # Membrane Variance の結合
    full_mem_var = mem_var_1 + mem_var_2

    # 2. Spikeデータの結合
    # spk_t_2 の時間は 0 から始まっている可能性が高いため、
    # spk_t_1 の最後の時間をオフセットとして加算して時間を繋げます。
    
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
    full_weight_error = weight_error_1 + weight_error_2

    record_interval_steps = 100
    time_axis_error = np.arange(len(full_weight_error)) * dt * record_interval_steps

    # # ---------------------------------------------------------
    # # Plot 1: Raster Plot (Combined) - Modified
    # # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # スパイクの散布図 (横軸: 時間, 縦軸: ニューロンID)
    # s=0.5 で点を小さくし、alpha=0.6 で密集度を見やすくしています
    plt.scatter(full_spk_t, full_spk_i, s=0.5, c='black', marker='.', alpha=0.6)

    # Set 1 と Set 2 の境界線を描画 (time_offset は Set 1 の終了時刻)
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
    
    # 変動が激しい場合は移動平均を見やすくする
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
    
    # MATLABでは対数グラフを使っていたので、必要に応じて以下をコメントアウト解除して使用してください
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
    end_time_set1 = len(weight_error_1) * dt * record_interval_steps
    plt.axvline(x=end_time_set1, color='red', linestyle='--', label='End of Set 1')

    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Distance')
    plt.title('Convergence to Optimal Recurrent Weights (E-I Balance)')
    plt.yscale('log') # 対数グラフ推奨
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    weight_plot_path = current_save_dir / "Final_Weight_Convergence.png"
    plt.savefig(weight_plot_path)
    plt.close()
    print(f"Weight Convergence plot saved to: {weight_plot_path}")

    # プロット
# Trainデータのプロット
plt.figure(figsize=(8, 8))
# 全データ（X_train）を青い点(c='blue')でプロット
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', alpha=0.5, label='Data Points')
# 軌跡を描画
plt.plot(X_train[:, 0], X_train[:, 1], c='gray', alpha=0.2, linewidth=1)

plt.title("Smoothed Random Walk (Train)")
plt.xlabel("Input Dimension 1")
plt.ylabel("Input Dimension 2")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig("smoothed_dataset_train.png")
plt.close()

# Testデータのプロット
plt.figure(figsize=(8, 8))
# 全データ（X_test）を青い点(c='blue')でプロット
plt.scatter(X_test[:, 0], X_test[:, 1], c='blue', alpha=0.5, label='Data Points')
# 軌跡を描画
plt.plot(X_test[:, 0], X_test[:, 1], c='gray', alpha=0.2, linewidth=1)

plt.title("Smoothed Random Walk (Test)")
plt.xlabel("Input Dimension 1")
plt.ylabel("Input Dimension 2")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig("smoothed_dataset_test.png")
plt.close()
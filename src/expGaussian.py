import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.interpolate import interp1d  # 追加: 補完用

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
    
    alpha = 0.20    
    beta = 1 / 0.9  
    mu = 0.02 / 0.9
    
    Thresh = 0.5
    lr_readout = 0.002
    
    # 保存先設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S(Gaussian_exp)")
    base_save_dir = Path("results")
    current_save_dir = base_save_dir / timestamp
    current_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved in: {current_save_dir}")
    
    # ---------------------------------------------------------
    # データ準備
    # ---------------------------------------------------------
    print("Preparing Data...")

    X, y = appssian.generate_continuous_shift_dataset(n_train=400000, n_test=400000, nx=2, sigma=5, seed=30,
                                      train_params={'mean': [0.0, 0.0], 'std': [1.0, 4.0]},
                                      test_params={'mean': [0.0, 0.0], 'std': [4.0, 1.0]})

    X_train = X[:400000]
    y_train = y[:400000]
    X_test = X[400000:]
    y_test = y[400000:]
    # ---------------------------------------------------------
    # Set 1: Learning
    # ---------------------------------------------------------
    print("--- Phase 1: Learning on Set 1 ---")
    F_initial, C_initial, W_out_initial, b_out_initial = appssian.init_weights(Nx, Nneuron, Nclasses)

    # 戻り値を受け取る (mem_var_times_1, mem_pot_1 を追加)
    acc_hist_1, spk_t_1, spk_i_1, F_set1, C_set1, W_out_set1, b_out_set1, mem_var_1, mem_var_times_1, mem_pot_1, final_states_1 = appssian.test_train_continuous(
        F_initial, C_initial, W_out_initial, b_out_initial, X_train, y_train, Nneuron, Nx, Nclasses, dt, leak, Thresh, alpha, beta, mu,
        retrain=True, Gain=200, lr_readout=lr_readout, epsr=0.0001, epsf=0.00001, init_states=None
    )
    print(len(acc_hist_1))
    
    # ---------------------------------------------------------
    # Set 2: Learning
    # ---------------------------------------------------------
    print("--- Phase 2: Learning on Set 2 ---")

    # 戻り値を受け取る
    acc_hist_2, spk_t_2, spk_i_2, *_, mem_var_2, mem_var_times_2, mem_pot_2, final_states_2 = appssian.test_train_continuous(
        F_set1, C_set1, W_out_set1, b_out_set1, X_test, y_test, Nneuron, Nx, Nclasses, dt, leak, Thresh, alpha, beta, mu,
        retrain=True, Gain=200, lr_readout=lr_readout, epsr=0.001, epsf=0.0001, init_states=final_states_1 
    )
    print(len(acc_hist_2))

    # ---------------------------------------------------------
    # Data Combination
    # ---------------------------------------------------------
    print("Combining data for plots...")
   
    full_acc = acc_hist_1 + acc_hist_2
    
    # Membrane Variance の結合 (値と時間)
    full_mem_var = mem_var_1 + mem_var_2
    
    # Set 2 の時間は Set 1 の長さ分オフセットする
    offset_time = len(X_train)
    mem_var_times_2_shifted = [t + offset_time for t in mem_var_times_2]
    full_mem_var_times = mem_var_times_1 + mem_var_times_2_shifted

    # Membrane Potential の結合 (NumPy配列として結合)
    mem_pot_arr_1 = np.array(mem_pot_1) 
    mem_pot_arr_2 = np.array(mem_pot_2) 
    full_mem_pot = np.concatenate([mem_pot_arr_1, mem_pot_arr_2], axis=0) 

    # Spikeデータの結合
    t1 = np.array(spk_t_1)
    i1 = np.array(spk_i_1)
    t2 = np.array(spk_t_2)
    i2 = np.array(spk_i_2)

    time_offset = np.max(t1) if len(t1) > 0 else 0
    full_spk_t = np.concatenate([t1, t2 + time_offset])
    full_spk_i = np.concatenate([i1, i2])

    # ---------------------------------------------------------
    # Plot 1: Accuracy History
    # ---------------------------------------------------------
    print("Generating Plots...")

    plt.figure(figsize=(10, 6))
    window_size = 1000
    if len(full_acc) >= window_size:
        b = np.ones(window_size) / window_size
        full_acc_smooth = np.convolve(full_acc, b, mode='valid')
        plt.plot(full_acc, label='Raw Accuracy', color='lightblue', alpha=0.4)
        plt.plot(np.arange(window_size - 1, len(full_acc)), full_acc_smooth, 
                 label=f'Moving Average (window={window_size})', color='blue', linewidth=2)
    else:
        plt.plot(full_acc, label='Accuracy', color='blue')

    plt.axvline(x=len(acc_hist_1), color='red', linestyle='--', label='End of Set 1')
    plt.xlabel('Input Samples')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy History')
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.savefig(current_save_dir / "Final_Accuracy_Smoothed.png")
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: Raster Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.scatter(full_spk_t, full_spk_i, s=0.5, c='black', marker='.', alpha=0.6)
    plt.axvline(x=time_offset, color='red', linestyle='--', label='End of Set 1')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title('Spike Raster Plot')
    plt.xlim(left=0, right=np.max(full_spk_t))
    plt.ylim(0, Nneuron)
    plt.legend(loc='upper right')
    plt.savefig(current_save_dir / "Final_Raster.png")
    plt.close()

    # ---------------------------------------------------------
    # Plot 3: Membrane Potential Variance (Interpolated)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # 配列化
    t_arr = np.array(full_mem_var_times)
    v_arr = np.array(full_mem_var)
    
    # 0以下の値を除外 (対数計算のため)
    mask = t_arr > 0
    t_valid = t_arr[mask]
    v_valid = v_arr[mask]

    # データ点がある程度ある場合のみ補完を行う
    if len(t_valid) > 3:
        try:
            # 時間軸を対数変換して補完関数を作成 (log(t) vs value)
            # kind='cubic' で滑らかな曲線、振動が激しい場合は 'linear' に変更してください
            x_log = np.log10(t_valid)
            f_interp = interp1d(x_log, v_valid, kind='cubic')
            
            # 描画用の細かい点を作成 (対数スケール上で等間隔)
            x_log_dense = np.linspace(x_log.min(), x_log.max(), 500)
            t_dense = 10**x_log_dense
            v_dense = f_interp(x_log_dense)
            
            # 補完曲線のプロット
            plt.plot(t_dense, v_dense, color='green', linewidth=2, label='Smoothed Trend')
            
            # 元のデータ点も散布図として表示
            plt.scatter(t_valid, v_valid, color='darkgreen', s=15, alpha=0.6, label='Sampled Points')
            
        except Exception as e:
            print(f"Interpolation failed: {e}. Plotting raw data.")
            plt.plot(t_valid, v_valid, marker='o', color='green', label='Voltage Variance')
    else:
        plt.plot(t_valid, v_valid, marker='o', color='green', label='Voltage Variance')

    plt.axvline(x=len(acc_hist_1), color='red', linestyle='--', label='End of Set 1')
    
    plt.xlabel('Input Samples')
    plt.ylabel('Voltage Variance per Neuron')
    plt.title('Evolution of the Variance (Log-Interpolated)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # 両対数グラフに設定
    plt.xscale('log') 
    plt.yscale('log')
    
    mem_plot_path = current_save_dir / "Final_Voltage_Variance.png"
    plt.savefig(mem_plot_path)
    plt.close()
    print(f"Voltage Variance plot saved to: {mem_plot_path}")

    # ---------------------------------------------------------
    # Plot 4: Membrane Potential Dynamics
    # ---------------------------------------------------------
    # 最初の数ニューロンの時系列 (一部区間)
    plt.figure(figsize=(12, 6))
    steps_to_plot = 2000 # 最初の2000ステップのみ表示
    t_axis = np.arange(steps_to_plot) * dt
    
    for i in range(min(3, Nneuron)): # 見やすさのため3ニューロンだけ
        plt.plot(t_axis, full_mem_pot[:steps_to_plot, i], label=f'Neuron {i}')
        
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (V)')
    plt.title('Membrane Potential Traces (First 2000 steps)')
    plt.legend()
    plt.grid(True)
    plt.savefig(current_save_dir / "Final_Membrane_Potential_Trace.png")
    plt.close()

    # 全期間のヒートマップ (ダウンサンプリング)
    plt.figure(figsize=(12, 6))
    downsample_rate = 100 
    plt.imshow(full_mem_pot[::downsample_rate].T, aspect='auto', cmap='viridis', origin='lower',
               extent=[0, len(full_mem_pot), 0, Nneuron])
    
    plt.colorbar(label='Membrane Potential')
    plt.axvline(x=len(acc_hist_1), color='red', linestyle='--', label='End of Set 1')
    plt.xlabel('Time Step')
    plt.ylabel('Neuron Index')
    plt.title('Membrane Potential Heatmap (Downsampled)')
    plt.savefig(current_save_dir / "Final_Membrane_Potential_Heatmap.png")
    plt.close()
    
    print("All plots generated.")

    # (Dataset plots remain unchanged)
    plt.figure(figsize=(8, 8))
    plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='blue', alpha=0.5, label='Class 0')
    plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='red', alpha=0.5, label='Class 1')
    plt.plot(X_train[:, 0], X_train[:, 1], c='gray', alpha=0.2, linewidth=1)
    plt.title("Smoothed Random Walk with Non-linear Boundary")
    plt.savefig("smoothed_dataset_train.png")
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='blue', alpha=0.5, label='Class 0')
    plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='red', alpha=0.5, label='Class 1')
    plt.plot(X_test[:, 0], X_test[:, 1], c='gray', alpha=0.2, linewidth=1)
    plt.title("Smoothed Random Walk with Non-linear Boundary")
    plt.savefig("smoothed_dataset_test.png")
    plt.close()
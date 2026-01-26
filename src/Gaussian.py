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

    # 戻り値の最後に final_states_1 を受け取る
    acc_hist_1, spk_t_1, spk_i_1, F_set1, C_set1, W_out_set1, b_out_set1, mem_var_1, final_states_1 = appssian.test_train_continuous(
        F_initial, C_initial, W_out_initial, b_out_initial, X_train, y_train, Nneuron, Nx, Nclasses, dt, leak, Thresh, alpha, beta, mu,
        retrain=True, Gain=200, lr_readout=lr_readout, epsr=0.0001, epsf=0.00001, init_states=None # 最初は指定なし
    )
    print(len(acc_hist_1))
    
    # ---------------------------------------------------------
    # Set 2: Learning
    # ---------------------------------------------------------
    print("--- Phase 2: Learning on Set 2 ---")

    # init_states に Set 1 の終わりの状態を渡す
    acc_hist_2, spk_t_2, spk_i_2, *_, mem_var_2, final_states_2 = appssian.test_train_continuous(
        F_set1, C_set1, W_out_set1, b_out_set1, X_test, y_test, Nneuron, Nx, Nclasses, dt, leak, Thresh, alpha, beta, mu,
        retrain=True, Gain=200, lr_readout=lr_readout, epsr=0.001, epsf=0.0001, init_states=final_states_1 # ★ここで引き継ぎ！
    )
    print(len(acc_hist_2))

    # ---------------------------------------------------------
    # Data Combination
    # ---------------------------------------------------------
    print("Combining data for plots...")
   
    full_acc = acc_hist_1 + acc_hist_2
    
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

    # ---------------------------------------------------------
    # Plot 1: Accuracy History (Combined) with Moving Average
    # ---------------------------------------------------------
    print("Generating Plots...")

    plt.figure(figsize=(10, 6))
    
    # --- 移動平均の計算 ---
    window_size = 1000  # 平均を取る範囲 (この数値を大きくするとより滑らかになります)
    if len(full_acc) >= window_size:
        # 移動平均フィルタの作成
        b = np.ones(window_size) / window_size
        # mode='valid' で計算 (端のデータ不足部分はカット)
        full_acc_smooth = np.convolve(full_acc, b, mode='valid')
        
        # --- プロット ---
        # 1. 生データ (薄く表示)
        plt.plot(full_acc, label='Raw Accuracy', color='lightblue', alpha=0.4)
        
        # 2. 平滑化データ (濃く表示)
        # 畳み込みでデータ長が短くなるため、X軸をずらしてプロットします
        plt.plot(np.arange(window_size - 1, len(full_acc)), full_acc_smooth, 
                 label=f'Moving Average (window={window_size})', color='blue', linewidth=2)
    else:
        # データが少なすぎる場合はそのままプロット
        plt.plot(full_acc, label='Accuracy', color='blue')

    # Set 1 と Set 2 の境界線を描画
    plt.axvline(x=len(acc_hist_1), color='red', linestyle='--', label='End of Set 1')
    
    plt.xlabel('Input Samples')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy History')
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    
    acc_plot_path = current_save_dir / "Final_Accuracy_Smoothed.png"
    plt.savefig(acc_plot_path)
    plt.close()
    print(f"Accuracy plot saved to: {acc_plot_path}")

    # # ---------------------------------------------------------
    # # Plot 2: Raster Plot (Combined) - Modified
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
    # Plot 3: Membrane Potential Variance (新規追加)
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

    plt.axvline(x=len(acc_hist_1), color='red', linestyle='--', label='End of Set 1')
    
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

    # プロット
plt.figure(figsize=(8, 8))
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='blue', alpha=0.5, label='Class 0')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='red', alpha=0.5, label='Class 1')
# 軌跡を描画して「時系列的なつながり」を確認
plt.plot(X_train[:, 0], X_train[:, 1], c='gray', alpha=0.2, linewidth=1)

plt.title("Smoothed Random Walk with Non-linear Boundary")
plt.xlabel("Input Dimension 1")
plt.ylabel("Input Dimension 2")
plt.legend()
plt.grid(True)
plt.axis('equal') # アスペクト比を揃える
plt.savefig("smoothed_dataset_train.png")
plt.close()

plt.figure(figsize=(8, 8))
plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='blue', alpha=0.5, label='Class 0')
plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='red', alpha=0.5, label='Class 1')
# 軌跡を描画して「時系列的なつながり」を確認
plt.plot(X_test[:, 0], X_test[:, 1], c='gray', alpha=0.2, linewidth=1)

plt.title("Smoothed Random Walk with Non-linear Boundary")
plt.xlabel("Input Dimension 1")
plt.ylabel("Input Dimension 2")
plt.legend()
plt.grid(True)
plt.axis('equal') # アスペクト比を揃える
plt.savefig("smoothed_dataset_test.png")
plt.close()
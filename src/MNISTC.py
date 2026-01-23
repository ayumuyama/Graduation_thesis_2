import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app_sotsuron import app_MNIST_C as appNIST
from app_sotsuron import brendel_random_mnist_utils as bu

import numpy as np

def load_and_preprocess(image_file_name, label_file_name, data_dir="data", num_classes=10):
    """
    指定された画像とラベルのnpyファイルを読み込み、
    機械学習モデルに入力可能な形式に成形して返します。

    Parameters:
    image_file_path (str): 画像データのファイルパス (.npy)
    label_file_path (str): ラベルデータのファイルパス (.npy)
    num_classes (int): 分類するクラスの数 (デフォルトは10)

    Returns:
    X (numpy.ndarray): (サンプル数, 784) に成形された画像データ
    y (numpy.ndarray): (サンプル数, num_classes) にOne-hot化されたラベルデータ
    """
    image_path = os.path.join(data_dir, image_file_name)
    label_path = os.path.join(data_dir, label_file_name)

    # ファイルの読み込み
    images = np.load(image_path)
    labels = np.load(label_path)

    # 1. 画像データの成形
    # 元の形状 (N, 28, 28, 1) を (N, 784) に変換
    # -1 を指定すると、元の要素数に合わせて自動的に次元サイズが計算されます
    X = images.reshape(images.shape[0], -1)
    
    # 2. ラベルデータの成形 (One-hot Encoding)
    # (N,) の整数ラベルを (N, num_classes) のOne-hotベクトルに変換
    # 例: ラベル 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np.eye(num_classes)[labels]
    
    return X, y

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 設定
    # ---------------------------------------------------------
    Nneuron = 200   
    Nx = 784
    Nclasses = 10        
    
    leak = 50       
    dt = 0.001      
    
    
    epsr = 0.1    
    epsf = 0.005   
    
    alpha = 0.18    
    beta = 1 / 0.9  
    mu = 0.02 / 0.9 
    Gain = 1
    
    
    Thresh = 0.5
    Duration = 30
    lr_readout=0.0002
    
    # 保存先設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S(identity-cannyedges mu=0.02)")
    base_save_dir = Path("results_sotsuron")
    current_save_dir = base_save_dir / timestamp
    current_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved in: {current_save_dir}")
    
    # ---------------------------------------------------------
    # データ準備
    # ---------------------------------------------------------
    print("Preparing Data...")

    X_train, y_train = load_and_preprocess("identity_test_images.npy", "identity_test_labels.npy")
    X_test, y_test = load_and_preprocess("cannyedges_test_images.npy", "cannyedges_test_labels.npy")

    # 確認
    print(f"Set 1 shape: {X_train.shape}") 
    print(f"Set 2 shape: {X_test.shape}")

    # ---------------------------------------------------------
    # Set 1: Learning with Plots
    # ---------------------------------------------------------
    print("--- Phase 1: Learning on Set 1 ---")
    F_initial, C_initial, W_out_initial = bu.init_weights(Nx, Nneuron, Nclasses)

    acc_hist_1, spk_t_1, spk_i_1, F_set1, C_set1, W_out_set1 = bu.train_readout_mnistc_Retrain(
    F_initial, C_initial, W_out_initial, X_train, y_train, Nneuron, Nx, Nclasses, dt, leak, Thresh, Gain, epsf, epsr, alpha, beta, mu, 
    Duration=Duration, lr_readout=lr_readout
    )
    
    # ---------------------------------------------------------
    # Set 2: Learning without Plots (Continue from Set 1)
    # ---------------------------------------------------------
    print("--- Phase 2: Learning on Set 2 ---")

    acc_hist_2, spk_t_2, spk_i_2, *_ = bu.train_readout_mnistc_Retrain(
    F_set1, C_set1, W_out_set1, X_test, y_test, Nneuron, Nx, Nclasses, dt, leak, Thresh, Gain, epsf, epsr, alpha, beta, mu, 
    Duration=Duration, lr_readout=lr_readout
    )

    # ---------------------------------------------------------
    # Data Combination for Plotting
    # ---------------------------------------------------------
    print("Combining data for plots...")
   
    # 1. Accuracyの結合 (リスト同士の結合を想定)
    # もしnumpy配列の場合は np.concatenate([acc_hist_1, acc_hist_2]) を使用してください
    full_acc = acc_hist_1 + acc_hist_2

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
    window_size = 500  # 平均を取る範囲 (この数値を大きくするとより滑らかになります)
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
    plt.title('Classification Accuracy History (identity-cannyedges)')
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    
    acc_plot_path = current_save_dir / "Final_Accuracy_Combined_Smoothed.png"
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
    plt.title('Spike Raster Plot (identity-cannyedges)')
    plt.xlim(left=0, right=np.max(full_spk_t)) # 軸の範囲をデータの最大値に合わせる
    plt.ylim(0, Nneuron)
    plt.legend(loc='upper right')

    raster_plot_path = current_save_dir / "Final_Raster_Combined.png"
    plt.savefig(raster_plot_path)
    plt.close()
    print(f"Raster plot saved to: {raster_plot_path}")
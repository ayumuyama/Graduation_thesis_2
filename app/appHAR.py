import numpy as np
import os
from matplotlib import pyplot as plt
from datetime import datetime
from pathlib import Path

def load_data_for_subject(subject_id, data_dir):
    """
    指定されたsubject_idに対応するデータを読み込み、
    (N, 6, 128)の入力データと(N, 1)のラベルを返す関数。
    
    Args:
        subject_id (int): 抽出したい被験者のID (例: 1)
        data_dir (str): 'data' ディレクトリへのパス
        
    Returns:
        X (np.array): 入力データ。形状は (サンプル数, 6, 128)
        y (np.array): 教師ラベル。形状は (サンプル数, 1)
    """
    
    # 1. subject_train.txt を読み込み、対象の行インデックスを特定
    subject_path = os.path.join(data_dir, 'subject_train.txt')
    subjects = np.loadtxt(subject_path).astype(int)
    
    # subject_id に一致する行のインデックスを取得 (0始まりのインデックス)
    # 例: subject_id=1 の場合、0〜346番目のインデックスが取得される
    target_indices = np.where(subjects == subject_id)[0]
    
    if len(target_indices) == 0:
        print(f"Warning: Subject ID {subject_id} not found in dataset.")
        return None, None

    # 2. y_train.txt (教師ラベル) を読み込み、対象行を抽出
    y_path = os.path.join(data_dir, 'y_train.txt')
    y_all = np.loadtxt(y_path).astype(int)
    y = y_all[target_indices].reshape(-1, 1) # (N, 1) の形に整える

    # 3. Inertial Signals の6つのファイルを読み込み、対象行を抽出して結合
    # 読み込むファイルリスト (Body Acc XYZ, Body Gyro XYZ の6つ)
    signal_files = [
        "body_acc_x_train.txt", "body_acc_y_train.txt", "body_acc_z_train.txt",
        "body_gyro_x_train.txt", "body_gyro_y_train.txt", "body_gyro_z_train.txt"
    ]
    
    signals_dir = os.path.join(data_dir, 'Inertial Signals')
    loaded_signals = []

    for filename in signal_files:
        file_path = os.path.join(signals_dir, filename)
        
        # ファイルを読み込む
        # 各ファイルは (全サンプル数, 128) の形状
        data = np.loadtxt(file_path)
        
        # 対象のsubjectの行だけを抽出
        subject_data = data[target_indices, :] # 形状: (N, 128)
        loaded_signals.append(subject_data)

    # 4. データをスタックして (N, 6, 128) の形状にする
    # axis=1 でスタックすることで、(サンプル数, チャンネル数, 時間) となる
    X = np.stack(loaded_signals, axis=1)

    return X, y

def init_weights(Nx, Nneuron, Nclasses):
    F = 0.5 * np.random.randn(Nx, Nneuron)
    F = F / (np.sqrt(np.sum(F**2, axis=0)) + 1e-8)
    C = -0.2 * np.random.rand(Nneuron, Nneuron) - 0.5 * np.eye(Nneuron)
    return F, C

def online_learning_classifier(F_init, C_init, X_data, y_data,
                               Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                               alpha, beta, mu, retrain, Gain=200,
                               eps=0.005, init_states=None):
    
    # --- 入力データの形状確認 ---
    # X_data: (NumSamples, InputDim, StepsPerSample)
    NumSamples, InputDim, StepsPerSample = X_data.shape
    TotalTime = NumSamples * StepsPerSample
    
    print(f"Phase : Continuous Training/Testing with Classifier")
    print(f"Total Samples: {NumSamples}, Steps per Sample: {StepsPerSample}")

    F = F_init.copy()
    C = C_init.copy()
    
    # 状態の初期化
    if init_states is None:
        V = np.zeros(Nneuron)
        rO = np.zeros(Nneuron)
        x = np.zeros(Nx)
    else:
        V = init_states['V'].copy()
        rO = init_states['rO'].copy()
        x = init_states['x'].copy()

    Id = np.eye(Nneuron)
    O = 0
    k_spike = 0
    spike_count = 0
    
    # --- 線形分類器 (Readout) のための初期化 ---
    W_out = np.zeros((Nclasses, Nneuron))
    
    # 履歴保存用
    spike_times = []
    spike_neurons = []
    accuracy_history = []  # サンプルごとの正解率
    prediction_history = [] # 各ステップの予測クラス
    
    global_step = 0
    
    # --- ループ構造の変更: サンプル -> ステップ ---
    for i in range(NumSamples):
        # 現在のサンプルの正解ラベル (One-hot encoding)
        current_label = y_data[i]
        target = np.zeros(Nclasses)
        target[int(current_label) - 1] = 1.0
        
        # サンプル内での正解数カウント（評価用）
        sample_correct_count = 0
        
        for t in range(StepsPerSample):
            global_step += 1
            if global_step % 1000 == 0:
                print(f'\r  Step: {global_step}/{TotalTime} (Sample {i+1}/{NumSamples})', end='')

            # 入力データの取得 (サンプル i, 次元 :, ステップ t)
            raw_input = X_data[i, :, t]
            img = raw_input * Gain
            
            # --- 既存のSNNダイナミクス (変更なし) ---
            noise = 0.01 * np.random.randn(Nneuron)
            recurrent_input = 0
            if O == 1:
                recurrent_input = C[:, k_spike]

            V = (1 - leak * dt) * V + dt * (F.T @ img) + recurrent_input + noise
            x = (1 - leak * dt) * x + dt * img 
            
            # スパイク判定
            thresh_noise = 0.01 * np.random.randn(Nneuron)
            potentials = V - Thresh - thresh_noise
            k_curr = np.argmax(potentials)

            if potentials[k_curr] >= 0:
                O = 1
                spike_count += 1
                k_spike = k_curr
                spike_times.append(global_step * dt)
                spike_neurons.append(k_spike)
                
                # F, C の学習 (既存ルール)
                if retrain:
                    current_var = np.var(V)
                    current_epsf = eps * current_var
                    current_epsr = 10 * current_epsf
                    
                    F[:, k_spike] += current_epsf * (alpha * x - F[:, k_spike])
                    C[:, k_spike] -= current_epsr * (beta * (V + mu * rO) + C[:, k_spike] + mu * Id[:, k_spike])

                rO[k_spike] += 1.0
            else:
                O = 0

            rO = (1 - leak * dt) * rO
            
            # rO (発火頻度状態) を特徴量として使用
            
            # 1. 推論: y_pred = W_out @ rO
            raw_output = W_out @ rO
            predicted_class = np.argmax(raw_output)
            prediction_history.append(predicted_class)
            
            if predicted_class == (int(current_label) - 1):
                sample_correct_count += 1

            # 2. 学習 (Delta Rule / LMS)
            if retrain:
                # 誤差の計算 (Target - Output)
                error = target - raw_output
                
                # 学習率 (Learning Rate)
                # ※ RLSと異なり、適切な固定値 (0.01など) や減衰する値を設定する必要があります
                learning_rate = 0.005 

                # 重みの更新: W += learning_rate * error * input^T
                # ここでは input が rO になります
                W_out += learning_rate * np.outer(error, rO)

        # サンプル終了時に精度を記録 (サンプル内での平均正解率)
        accuracy_history.append(sample_correct_count / StepsPerSample)

    final_states = {'V': V, 'rO': rO, 'x': x}
    
    print(f"\nTraining Finished. spike count{spike_count}")
    
    # 戻り値に W_out (学習済み分類器) と accuracy_history を追加
    return spike_times, spike_neurons, F, C, W_out, accuracy_history, final_states

def plot_learning_curve(acc_hist, window_size=50):
    """
    acc_hist: 学習関数から返ってきた accuracy_history リスト
    window_size: 移動平均を取る幅（デフォルト50サンプル）
    """

    # 保存先設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S(suggestGaussian_exp)")
    base_save_dir = Path("results")
    current_save_dir = base_save_dir / timestamp
    current_save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    
    # 1. 生の正解率（薄く表示）
    # サンプルごとの変動が激しいので透明度(alpha)を下げて背景にします
    plt.plot(acc_hist, alpha=0.3, color='gray', label='Raw Accuracy (Per Sample)')
    
    # 2. 移動平均（濃く表示）
    # 全体的な学習の傾向（トレンド）を見ます
    if len(acc_hist) >= window_size:
        weights = np.ones(window_size) / window_size
        moving_avg = np.convolve(acc_hist, weights, mode='valid')
        
        # 移動平均はずれるので、X軸を合わせるためのオフセット
        x_axis = np.arange(window_size - 1, len(acc_hist))
        
        plt.plot(x_axis, moving_avg, color='blue', linewidth=2, label=f'Moving Average (Window={window_size})')
    
    plt.title('Online Learning Classification Accuracy', fontsize=16)
    plt.xlabel('Number of Samples', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(-0.05, 1.05) # Y軸を 0~1 に固定（少し余裕を持たせる）
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(current_save_dir / "Final_Decoding_Error.png")

def plot_raster(spike_times, spike_neurons, title="Spike Raster Plot", 
                figsize=(12, 6), marker_size=2, color='black'):
    """
    SNNのスパイク履歴をラスタープロットとして描画する関数
    
    Parameters:
    - spike_times: list or np.array, スパイクが発生した時刻のリスト
    - spike_neurons: list or np.array, スパイクしたニューロンのインデックスのリスト
    - title: str, グラフのタイトル
    - figsize: tuple, グラフのサイズ (幅, 高さ)
    - marker_size: float, プロットする点のサイズ
    - color: str, 点の色
    - save_path: str, 画像を保存するパス (Noneの場合は表示のみ)
    """
    
    plt.figure(figsize=figsize)
    
    # 散布図の描画 (s=サイズ, alpha=透明度)
    plt.scatter(spike_times, spike_neurons, s=marker_size, c=color, alpha=0.6)
    
    # ラベルとタイトルの設定
    plt.xlabel("Time [s]")
    plt.ylabel("Neuron Index")
    plt.title(title)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S(suggestGaussian_exp)")
    base_save_dir = Path("results")
    current_save_dir = base_save_dir / timestamp
    current_save_dir.mkdir(parents=True, exist_ok=True)
    # グリッドとレイアウトの調整
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(current_save_dir / "Final_Decoding_Error.png")
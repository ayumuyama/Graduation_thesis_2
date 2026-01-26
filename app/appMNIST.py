import numpy as np
import os

def init_weights(Nx, Nneuron, Nclasses):
    F = 0.5 * np.random.randn(Nx, Nneuron)
    F = F / (np.sqrt(np.sum(F**2, axis=0)) + 1e-8)
    C = -0.2 * np.random.rand(Nneuron, Nneuron) - 0.5 * np.eye(Nneuron)
    W_out = np.zeros((Nclasses, Nneuron))
    b_out = np.zeros(Nclasses)
    return F, C, W_out, b_out

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

def train_readout_mnistc_Retrain(F_init, C_init, W_out, b_out, X_data, y_data, 
                                 Nneuron, Nx, Nclasses, dt, leak, Thresh, Gain, 
                                 epsf, epsr, alpha, beta, mu, retrain,
                                 Duration=30, 
                                 lr_readout=0.008,
                                 # ★追加: 初期状態を受け取る引数 (デフォルトはNone)
                                 init_states=None):
    
    NumSamples = X_data.shape[0]
    print(f"Phase : Training")
    
    F = F_init.copy()
    C = C_init.copy()
    W_out = W_out.copy()
    b_out = b_out.copy()    
    
    # ★変更: 初期状態のセットアップ
    if init_states is None:
        # 初回（Set 1）や指定がない場合はゼロ初期化
        V = np.zeros(Nneuron)
        rO = np.zeros(Nneuron)
        x = np.zeros(Nx)
    else:
        # 引き継ぎがある場合はそれを使う
        V = init_states['V'].copy()
        rO = init_states['rO'].copy()
        x = init_states['x'].copy()

    Id = np.eye(Nneuron)
    # O と k は瞬時値なので0リセットでOK（あるいは引き継いでも良いが大差なし）
    O = 0
    k = 0

    total_spikes = 0
    acc_history = []
    acc_buffer = []
    spike_times = []
    spike_neurons = []
    membrane_var_history = []
    
    for i in range(NumSamples):
        if i % 100 == 0:
            print(f'\r  Phase Iter: {i}/{NumSamples} (Spikes: {total_spikes})', end='')

        # ★削除: ここにあった x = np.zeros(Nx), rO = np.zeros(Nneuron) を削除！
        # これにより、前の画像の活動（rO）や入力フィルタ（x）が次の画像に影響を与えます。
        
        raw_img = X_data[i]     
        img = raw_img * Gain
        target_vec = y_data[i].copy()
        target_label = np.argmax(target_vec)

        time_offset = i * Duration * dt
        V_temporal_buffer = np.zeros((Nneuron, Duration))
        pred_history = []

        for t in range(Duration):
            # ... (中身の計算処理は変更なし) ...
            noise = 0.02 * np.random.randn(Nneuron)
            recurrent_input = 0
            if O == 1:
                recurrent_input = C[:, k]

            V = (1 - leak * dt) * V + dt * (F.T @ img) + recurrent_input + noise
            x = (1 - leak * dt) * x + dt * img 
            
            V_temporal_buffer[:, t] = V
            
            # ... (スパイク生成、重み更新などの処理) ...
            thresh_noise = 0.01 * np.random.randn(Nneuron)
            potentials = V - Thresh - thresh_noise
            k_curr = np.argmax(potentials)

            if potentials[k_curr] >= 0:
                O = 1
                k = k_curr
                spike_times.append(time_offset + t * dt)
                spike_neurons.append(k)
                total_spikes += 1
                
                if retrain:
                    F[:, k] += epsf * (alpha * x - F[:, k])
                    C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])
            else:
                O = 0

            rO = (1 - leak * dt) * rO
            if O == 1:
                rO[k] += 1.0
            
            # Readout Learning
            y_est_vec = np.dot(W_out, rO) + b_out
            pred_idx = np.argmax(y_est_vec)
            pred_history.append(pred_idx)
            
            error_vec = target_vec - y_est_vec
            W_out += lr_readout * np.outer(error_vec, rO)
            b_out += lr_readout * error_vec
            
            # ... (ループ終了) ...

        # 正誤判定や分散計算
        counts = np.bincount(pred_history, minlength=Nclasses)
        final_prediction = np.argmax(counts)
        is_correct = 1 if final_prediction == target_label else 0
        acc_buffer.append(is_correct)
        if len(acc_buffer) > 200: acc_buffer.pop(0)
        acc_history.append(np.mean(acc_buffer))

        var_per_neuron = np.var(V_temporal_buffer, axis=1)
        mean_var = np.mean(var_per_neuron)
        membrane_var_history.append(mean_var)

    print(f"\nPhase Completed. Final Accuracy: {acc_history[-1]:.4f}")
    
    # ★追加: 最終状態を辞書にまとめる
    final_states = {'V': V, 'rO': rO, 'x': x}

    # 戻り値に final_states を追加
    return acc_history, spike_times, spike_neurons, F, C, W_out, b_out, membrane_var_history, final_states

def train_readout_mnistc_Retrain_fine(F_init, C_init, W_out, b_out, X_data, y_data, 
                                 Nneuron, Nx, Nclasses, dt, leak, Thresh, Gain, 
                                 epsf, epsr, alpha, beta, mu, retrain,
                                 Duration=30, 
                                 lr_readout=0.008,
                                 # ★追加: 初期状態を受け取る引数 (デフォルトはNone)
                                 init_states=None):
    
    NumSamples = X_data.shape[0]
    print(f"Phase : Training")
    
    F = F_init.copy()
    C = C_init.copy()
    W_out = W_out.copy()
    b_out = b_out.copy()    
    
    # ★変更: 初期状態のセットアップ
    if init_states is None:
        # 初回（Set 1）や指定がない場合はゼロ初期化
        V = np.zeros(Nneuron)
        rO = np.zeros(Nneuron)
        x = np.zeros(Nx)
    else:
        # 引き継ぎがある場合はそれを使う
        V = init_states['V'].copy()
        rO = init_states['rO'].copy()
        x = init_states['x'].copy()

    Id = np.eye(Nneuron)
    # O と k は瞬時値なので0リセットでOK（あるいは引き継いでも良いが大差なし）
    O = 0
    k = 0

    total_spikes = 0
    acc_history = []
    acc_buffer = []
    spike_times = []
    spike_neurons = []
    membrane_var_history = []
    
    for i in range(NumSamples):
        if i % 100 == 0:
            print(f'\r  Phase Iter: {i}/{NumSamples} (Spikes: {total_spikes})', end='')

        # ★削除: ここにあった x = np.zeros(Nx), rO = np.zeros(Nneuron) を削除！
        # これにより、前の画像の活動（rO）や入力フィルタ（x）が次の画像に影響を与えます。
        
        raw_img = X_data[i]     
        img = raw_img * Gain
        target_vec = y_data[i].copy()
        target_label = np.argmax(target_vec)

        time_offset = i * Duration * dt
        V_temporal_buffer = np.zeros((Nneuron, Duration))
        pred_history = []

        for t in range(Duration):
            # ... (中身の計算処理は変更なし) ...
            noise = 0.02 * np.random.randn(Nneuron)
            recurrent_input = 0
            if O == 1:
                recurrent_input = C[:, k]

            V = (1 - leak * dt) * V + dt * (F.T @ img) + recurrent_input + noise
            x = (1 - leak * dt) * x + dt * img 
            
            V_temporal_buffer[:, t] = V
            
            # ... (スパイク生成、重み更新などの処理) ...
            thresh_noise = 0.01 * np.random.randn(Nneuron)
            potentials = V - Thresh - thresh_noise
            k_curr = np.argmax(potentials)

            if potentials[k_curr] >= 0:
                O = 1
                k = k_curr
                spike_times.append(time_offset + t * dt)
                spike_neurons.append(k)
                total_spikes += 1
                
                if retrain:
                    F[:, k] += epsf * (alpha * x - F[:, k])
                    C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])
            else:
                O = 0

            rO = (1 - leak * dt) * rO
            if O == 1:
                rO[k] += 1.0
            
            # Readout Learning
            y_est_vec = np.dot(W_out, rO) + b_out
            pred_idx = np.argmax(y_est_vec)
            pred_history.append(pred_idx)
            
            error_vec = target_vec - y_est_vec
            W_out += lr_readout * np.outer(error_vec, rO)
            b_out += lr_readout * error_vec
            
            # ... (ループ終了) ...

        # 正誤判定や分散計算
        counts = np.bincount(pred_history, minlength=Nclasses)
        final_prediction = np.argmax(counts)
        is_correct = 1 if final_prediction == target_label else 0
        acc_buffer.append(is_correct)
        if len(acc_buffer) > 200: acc_buffer.pop(0)
        acc_history.append(np.mean(acc_buffer))

        var_per_neuron = np.var(V_temporal_buffer, axis=1)
        mean_var = np.mean(var_per_neuron)
        membrane_var_history.append(mean_var)

    print(f"\nPhase Completed. Final Accuracy: {acc_history[-1]:.4f}")
    
    # ★追加: 最終状態を辞書にまとめる
    final_states = {'V': V, 'rO': rO, 'x': x}

    # 戻り値に final_states を追加
    return acc_history, spike_times, spike_neurons, F, C, W_out, b_out, membrane_var_history, final_states
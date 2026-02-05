import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

def load_and_preprocess(image_file_name, label_file_name, data_dir="data", num_classes=10):
    """
    指定された画像とラベルのnpyファイルを読み込み，
    正規化および成形して返します．
    """
    image_path = os.path.join(data_dir, image_file_name)
    label_path = os.path.join(data_dir, label_file_name)

    images = np.load(image_path)
    labels = np.load(label_path)

    # 1次元配列への変換と同時に，0.0-1.0に正規化
    # astype(np.float32) を入れることで，精度の維持と計算の高速化を図ります
    X = images.reshape(images.shape[0], -1).astype(np.float32) / 255.0
    
    # ラベルのOne-hotエンコーディング
    # こちらも後続の計算のために float32 にしておくのが一般的です
    y = np.eye(num_classes)[labels].astype(np.float32)
    
    return X, y

def test_train_continuous_correlated_proposed(F_init, C_init, X_data, y_data,
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain, Gain=200,
                          eps=0.005, 
                          la=0.2, Ucc_scale=100.0,
                          init_states=None,
                          lr_readout=0.002,
                          stim_duration=100): # 引数を追加
    
    TotalTime = X_data.shape[0]
    print(f"Phase : Continuous Training (Correlated Input) (Total Steps: {TotalTime})")
    
    F = F_init.copy()
    C = C_init.copy()
    
    if init_states is None:
        V = np.zeros(Nneuron)
        rO = np.zeros(Nneuron)
        x = np.zeros(Nx)
        mI = np.zeros(Nx)
        Ucc = np.zeros((Nx, Nneuron))
        W_out = np.zeros((Nneuron, Nclasses))
    else:
        V = init_states['V'].copy()
        rO = init_states['rO'].copy()
        x = init_states['x'].copy()
        mI = init_states.get('mI', np.zeros(Nx))
        Ucc = init_states.get('Ucc', np.zeros((Nx, Nneuron)))
        W_out = init_states.get('W_out', np.zeros((Nneuron, Nclasses)))

    Id = np.eye(Nneuron)
    O = 0
    k = 0

    spike_times = []
    spike_neurons = []
    membrane_var_history = []
    accuracy_history = [] 
    
    # 1サンプル内の予測ラベルを蓄積するリスト
    current_sample_preds = []

    for t in range(TotalTime):
        # ---------------------------------------------------------
        # サンプル切り替え時のリセット処理
        # ---------------------------------------------------------
        if t > 0 and t % stim_duration == 0:
            V = np.zeros(Nneuron)
            rO = np.zeros(Nneuron)
            x = np.zeros(Nx)
            # mI, Ucc, W_out はリセットしない（学習結果を保持）
            O = 0
            k = 0 
            
        if t % 1000 == 0:
            print(f'\r  Step: {t}/{TotalTime}', end='')

        # --- ダイナミクス計算 ---
        raw_input = X_data[t]
        img = raw_input * Gain
        
        mI = (1 - la * dt) * mI + la * dt * x 
        
        diff = x - mI
        proj = F.T @ diff
        Ucc = (1 - la * dt) * Ucc + la * dt * np.outer(diff, proj)
        
        noise = 0.01 * np.random.randn(Nneuron)
        recurrent_input = 0
        if O == 1:
            recurrent_input = C[:, k]

        V = (1 - leak * dt) * V + dt * (F.T @ img) + recurrent_input + noise
        x = (1 - leak * dt) * x + dt * img 
        
        thresh_noise = 0.01 * np.random.randn(Nneuron)
        potentials = V - Thresh - thresh_noise
        k_curr = np.argmax(potentials)

        if potentials[k_curr] >= 0:
            O = 1
            k = k_curr
            spike_times.append(t * dt)
            spike_neurons.append(k)
            
            if retrain:
                current_var = np.var(V)
                current_epsf = eps * current_var
                current_epsr = 10 * current_epsf
                
                recon_term = Ucc_scale * Ucc[:, k]
                F[:, k] += current_epsf * (x - recon_term)
                C[:, k] -= current_epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])

            rO[k] += 1.0
        else:
            O = 0

        rO = (1 - leak * dt) * rO
        membrane_var_history.append(np.var(V))

        # ---------------------------------------------------------
        # Readout: 推論と学習
        # ---------------------------------------------------------
        pred_logit = W_out.T @ rO
        pred_label = np.argmax(pred_logit)
        
        # 現在の予測をリストに追加（多数決用）
        current_sample_preds.append(pred_label)
        
        # 学習は毎ステップ実施（教師信号は現在のステップのもの）
        error_signal = y_data[t] - pred_logit
        W_out += lr_readout * np.outer(rO, error_signal)
        
        # ---------------------------------------------------------
        # サンプル終了時に多数決で精度判定
        # ---------------------------------------------------------
        if (t + 1) % stim_duration == 0:
            # 最頻値（多数決）を取得
            # np.bincountで各ラベルの出現回数をカウントし、argmaxで最大頻度のラベルを取得
            counts = np.bincount(current_sample_preds, minlength=Nclasses)
            voted_label = np.argmax(counts)
            
            true_label = np.argmax(y_data[t])
            
            is_correct = 1 if voted_label == true_label else 0
            accuracy_history.append(is_correct)
            
            # 次のサンプルのためにクリア
            current_sample_preds = []
    
    final_states = {'V': V, 'rO': rO, 'x': x, 'mI': mI, 'Ucc': Ucc, 'W_out': W_out}

    return spike_times, spike_neurons, F, C, membrane_var_history, accuracy_history, final_states, W_out

def test_train_continuous_correlated(F_init, C_init, X_data, y_data, 
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain, Gain=200,
                          epsr=0.05, epsf=0.005, 
                          la=0.2, Ucc_scale=100.0,
                          init_states=None,
                          lr_readout=0.002,
                          stim_duration=100): # 引数を追加
    
    TotalTime = X_data.shape[0]
    print(f"Phase : Continuous Training (Correlated Input) (Total Steps: {TotalTime})")
    
    F = F_init.copy()
    C = C_init.copy()
    
    if init_states is None:
        V = np.zeros(Nneuron)
        rO = np.zeros(Nneuron)
        x = np.zeros(Nx)
        mI = np.zeros(Nx)
        Ucc = np.zeros((Nx, Nneuron)) 
        W_out = np.zeros((Nneuron, Nclasses))
    else:
        V = init_states['V'].copy()
        rO = init_states['rO'].copy()
        x = init_states['x'].copy()
        mI = init_states.get('mI', np.zeros(Nx))
        Ucc = init_states.get('Ucc', np.zeros((Nx, Nneuron)))
        W_out = init_states.get('W_out', np.zeros((Nneuron, Nclasses)))

    Id = np.eye(Nneuron)
    O = 0
    k = 0

    spike_times = []
    spike_neurons = []
    membrane_var_history = []
    accuracy_history = []
    
    # 1サンプル内の予測ラベルを蓄積するリスト
    current_sample_preds = []
    
    for t in range(TotalTime):
        # ---------------------------------------------------------
        # サンプル切り替え時のリセット処理
        # ---------------------------------------------------------
        if t > 0 and t % stim_duration == 0:
            V = np.zeros(Nneuron)
            rO = np.zeros(Nneuron)
            x = np.zeros(Nx)
            O = 0
            k = 0 

        if t % 1000 == 0:
            print(f'\r  Step: {t}/{TotalTime}', end='')

        # --- ダイナミクス計算 ---
        raw_input = X_data[t]
        img = raw_input * Gain
        
        mI = (1 - la * dt) * mI + la * dt * x 
        
        diff = x - mI
        proj = F.T @ diff
        Ucc = (1 - la * dt) * Ucc + la * dt * np.outer(diff, proj)
        
        noise = 0.01 * np.random.randn(Nneuron)
        recurrent_input = 0
        if O == 1:
            recurrent_input = C[:, k]

        V = (1 - leak * dt) * V + dt * (F.T @ img) + recurrent_input + noise
        x = (1 - leak * dt) * x + dt * img 
        
        thresh_noise = 0.01 * np.random.randn(Nneuron)
        potentials = V - Thresh - thresh_noise
        k_curr = np.argmax(potentials)

        if potentials[k_curr] >= 0:
            O = 1
            k = k_curr
            spike_times.append(t * dt)
            spike_neurons.append(k)
            
            if retrain:
                recon_term = Ucc_scale * Ucc[:, k]
                F[:, k] += epsf * (x - recon_term)
                C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])

            rO[k] += 1.0
        else:
            O = 0

        rO = (1 - leak * dt) * rO
        membrane_var_history.append(np.var(V))

        # ---------------------------------------------------------
        # Readout: 推論と学習
        # ---------------------------------------------------------
        pred_logit = W_out.T @ rO
        pred_label = np.argmax(pred_logit)
        
        current_sample_preds.append(pred_label)
        
        error_signal = y_data[t] - pred_logit
        W_out += lr_readout * np.outer(rO, error_signal)

        # ---------------------------------------------------------
        # サンプル終了時に多数決で精度判定
        # ---------------------------------------------------------
        if (t + 1) % stim_duration == 0:
            counts = np.bincount(current_sample_preds, minlength=Nclasses)
            voted_label = np.argmax(counts)
            
            true_label = np.argmax(y_data[t])
            
            is_correct = 1 if voted_label == true_label else 0
            accuracy_history.append(is_correct)
            
            current_sample_preds = []
    
    final_states = {'V': V, 'rO': rO, 'x': x, 'mI': mI, 'Ucc': Ucc, 'W_out': W_out}

    return spike_times, spike_neurons, F, C, membrane_var_history, accuracy_history, final_states, W_out
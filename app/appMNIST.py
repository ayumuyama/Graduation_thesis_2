import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

def load_and_preprocess(image_file_name, label_file_name, data_dir="data", num_classes=10):
    """
    指定された画像とラベルのnpyファイルを読み込み、
    機械学習モデルに入力可能な形式に成形して返します。
    """
    image_path = os.path.join(data_dir, image_file_name)
    label_path = os.path.join(data_dir, label_file_name)

    images = np.load(image_path)
    labels = np.load(label_path)

    X = images.reshape(images.shape[0], -1)
    y = np.eye(num_classes)[labels]
    
    return X, y

def test_train_continuous_correlated_proposed(F_init, C_init, X_data, y_data, # y_dataを追加
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain, Gain=200,
                          eps=0.005, 
                          la=0.2, Ucc_scale=100.0, # Figure 5用の追加パラメータ
                          init_states=None,
                          lr_readout=0.002): # Readoutの学習率を追加
    
    TotalTime = X_data.shape[0]
    print(f"Phase : Continuous Training (Correlated Input) (Total Steps: {TotalTime})")
    
    F = F_init.copy()
    C = C_init.copy()
    
    if init_states is None:
        V = np.zeros(Nneuron)
        rO = np.zeros(Nneuron)
        x = np.zeros(Nx)
        # 統計量の初期化
        mI = np.zeros(Nx)
        Ucc = np.zeros((Nx, Nneuron))
        # Readout重みの初期化
        W_out = np.zeros((Nneuron, Nclasses))
    else:
        V = init_states['V'].copy()
        rO = init_states['rO'].copy()
        x = init_states['x'].copy()
        # 統計量の引き継ぎ
        mI = init_states.get('mI', np.zeros(Nx))
        Ucc = init_states.get('Ucc', np.zeros((Nx, Nneuron)))
        # Readout重みの引き継ぎ
        W_out = init_states.get('W_out', np.zeros((Nneuron, Nclasses)))

    Id = np.eye(Nneuron)
    O = 0
    k = 0

    spike_times = []
    spike_neurons = []
    membrane_var_history = []
    weight_error_history = []
    accuracy_history = [] # 分類精度記録用
    
    for t in range(TotalTime):
        if t % 1000 == 0:
            print(f'\r  Step: {t}/{TotalTime}', end='')

        # --- ダイナミクス計算 ---
        raw_input = X_data[t]
        img = raw_input * Gain
        
        # ---------------------------------------------------------
        # 統計量の更新
        # ---------------------------------------------------------
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
        
        # スパイク判定
        thresh_noise = 0.01 * np.random.randn(Nneuron)
        potentials = V - Thresh - thresh_noise
        k_curr = np.argmax(potentials)

        if potentials[k_curr] >= 0:
            O = 1
            k = k_curr
            spike_times.append(t * dt)
            spike_neurons.append(k)
            
            if retrain:
                # ---------------------------------------------------------
                # Figure 5 学習則 (Correlated Input)
                # ---------------------------------------------------------
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
        # 線形学習器による分類 (Readout)
        # ---------------------------------------------------------
        # 推論: y = W_out.T @ rO
        pred_logit = W_out.T @ rO
        pred_label = np.argmax(pred_logit)
        true_label = np.argmax(y_data[t])

        # 精度の記録 (1: 正解, 0: 不正解)
        is_correct = 1 if pred_label == true_label else 0
        accuracy_history.append(is_correct)

        # オンライン学習 (Delta Rule / LMS)
        # 誤差信号: e = target - prediction
        error_signal = y_data[t] - pred_logit
        # 重み更新: W += lr * rO * error
        W_out += lr_readout * np.outer(rO, error_signal)
    
    # 最終状態に統計量とReadout重みを含める
    final_states = {'V': V, 'rO': rO, 'x': x, 'mI': mI, 'Ucc': Ucc, 'W_out': W_out}

    # 戻り値に accuracy_history と W_out を追加
    return spike_times, spike_neurons, F, C, membrane_var_history, accuracy_history, final_states, W_out

def test_train_continuous_correlated(F_init, C_init, X_data, y_data, # y_dataを追加
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain, Gain=200,
                          epsr=0.05, epsf=0.005, 
                          la=0.2, Ucc_scale=100.0, # Figure 5用の追加パラメータ
                          init_states=None,
                          lr_readout=0.002): # Readoutの学習率を追加
    
    TotalTime = X_data.shape[0]
    print(f"Phase : Continuous Training (Correlated Input) (Total Steps: {TotalTime})")
    
    F = F_init.copy()
    C = C_init.copy()
    
    if init_states is None:
        V = np.zeros(Nneuron)
        rO = np.zeros(Nneuron)
        x = np.zeros(Nx)
        # 統計量の初期化
        mI = np.zeros(Nx)
        Ucc = np.zeros((Nx, Nneuron)) 
        # Readout重みの初期化
        W_out = np.zeros((Nneuron, Nclasses))
    else:
        V = init_states['V'].copy()
        rO = init_states['rO'].copy()
        x = init_states['x'].copy()
        # 統計量の引き継ぎ
        mI = init_states.get('mI', np.zeros(Nx))
        Ucc = init_states.get('Ucc', np.zeros((Nx, Nneuron)))
        # Readout重みの引き継ぎ
        W_out = init_states.get('W_out', np.zeros((Nneuron, Nclasses)))

    Id = np.eye(Nneuron)
    O = 0
    k = 0

    spike_times = []
    spike_neurons = []
    membrane_var_history = []
    accuracy_history = [] # 分類精度記録用
    
    for t in range(TotalTime):
        if t % 1000 == 0:
            print(f'\r  Step: {t}/{TotalTime}', end='')

        # --- ダイナミクス計算 ---
        raw_input = X_data[t]
        img = raw_input * Gain
        
        # ---------------------------------------------------------
        # 統計量の更新
        # ---------------------------------------------------------
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
        
        # スパイク判定
        thresh_noise = 0.01 * np.random.randn(Nneuron)
        potentials = V - Thresh - thresh_noise
        k_curr = np.argmax(potentials)

        if potentials[k_curr] >= 0:
            O = 1
            k = k_curr
            spike_times.append(t * dt)
            spike_neurons.append(k)
            
            if retrain:
                # ---------------------------------------------------------
                # Figure 5 学習則 (Correlated Input)
                # ---------------------------------------------------------
                recon_term = Ucc_scale * Ucc[:, k]
                F[:, k] += epsf * (x - recon_term)
                
                C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])

            rO[k] += 1.0

        else:
            O = 0

        rO = (1 - leak * dt) * rO
        membrane_var_history.append(np.var(V))

        # ---------------------------------------------------------
        # 線形学習器による分類 (Readout)
        # ---------------------------------------------------------
        # 推論: y = W_out.T @ rO
        pred_logit = W_out.T @ rO
        pred_label = np.argmax(pred_logit)
        true_label = np.argmax(y_data[t])

        # 精度の記録 (1: 正解, 0: 不正解)
        is_correct = 1 if pred_label == true_label else 0
        accuracy_history.append(is_correct)

        # オンライン学習 (Delta Rule / LMS)
        # 誤差信号: e = target - prediction
        error_signal = y_data[t] - pred_logit
        # 重み更新: W += lr * rO * error
        W_out += lr_readout * np.outer(rO, error_signal)
    
    # 最終状態に統計量とReadout重みを含める
    final_states = {'V': V, 'rO': rO, 'x': x, 'mI': mI, 'Ucc': Ucc, 'W_out': W_out}

    # 戻り値に accuracy_history と W_out を追加
    return spike_times, spike_neurons, F, C, membrane_var_history, accuracy_history, final_states, W_out
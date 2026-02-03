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

def test_train_continuous_correlated_proposed(F_init, C_init, X_data,
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain, Gain=200,
                          eps=0.005, 
                          la=0.2, Ucc_scale=100.0, # Figure 5用の追加パラメータ
                          init_states=None):
    
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
    else:
        V = init_states['V'].copy()
        rO = init_states['rO'].copy()
        x = init_states['x'].copy()
        # 統計量の引き継ぎ（もし辞書に含まれていれば）
        mI = init_states.get('mI', np.zeros(Nx))
        Ucc = init_states.get('Ucc', np.zeros((Nx, Nneuron)))

    Id = np.eye(Nneuron)
    O = 0
    k = 0

    spike_times = []
    spike_neurons = []
    membrane_var_history = []
    weight_error_history = []
    decoding_error_history = []
    
    for t in range(TotalTime):
        if t % 1000 == 0:
            print(f'\r  Step: {t}/{TotalTime}', end='')

        # --- ダイナミクス計算 ---
        raw_input = X_data[t]
        img = raw_input * Gain
        
        # ---------------------------------------------------------
        # 統計量の更新 (Figure 5: netadapt.m 準拠)
        # ---------------------------------------------------------
        # mI = (1 - la*dt)*mI + la*dt*In
        # ※本来はIn(高速時定数)を使いますが、簡易的にxを使用します
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
                # F (Feedforward) の更新:
                # MATLAB: Gamma += eps * (O * (In - 100 * Ucc' * O)')
                # Python: F[:, k] += eps * (x - Ucc_scale * Ucc[:, k])

                # 1. 膜電位の分散の計算
                current_var = np.var(V)
                
                # 2. 学習率の都度計算
                # epsf = eps * np.var(V)
                # epsr = 10 * epsf
                current_epsf = eps * current_var
                current_epsr = 10 * current_epsf
                if t < 100:
                    print(f" Step {t}: Var={current_var:.4f}, epsf={current_epsf:.6f}, epsr={current_epsr:.6f}")
                
                recon_term = Ucc_scale * Ucc[:, k]
                F[:, k] += current_epsf * (x - recon_term)
                
                # C (Recurrent) の更新 (通常と同じ)
                C[:, k] -= current_epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])

            rO[k] += 1.0

        else:
            O = 0

        rO = (1 - leak * dt) * rO
        membrane_var_history.append(np.var(V))
    
    # 最終状態に統計量も含める
    final_states = {'V': V, 'rO': rO, 'x': x, 'mI': mI, 'Ucc': Ucc}

    return spike_times, spike_neurons, F, C, membrane_var_history, weight_error_history, decoding_error_history, final_states

def test_train_continuous_correlated(F_init, C_init, X_data,
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain, Gain=200,
                          epsr=0.05, epsf=0.005, 
                          la=0.2, Ucc_scale=100.0, # Figure 5用の追加パラメータ
                          init_states=None):
    
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
    else:
        V = init_states['V'].copy()
        rO = init_states['rO'].copy()
        x = init_states['x'].copy()
        # 統計量の引き継ぎ（もし辞書に含まれていれば）
        mI = init_states.get('mI', np.zeros(Nx))
        Ucc = init_states.get('Ucc', np.zeros((Nx, Nneuron)))

    Id = np.eye(Nneuron)
    O = 0
    k = 0

    spike_times = []
    spike_neurons = []
    membrane_var_history = []
    weight_error_history = []
    decoding_error_history = []
    
    for t in range(TotalTime):
        if t % 1000 == 0:
            print(f'\r  Step: {t}/{TotalTime}', end='')

        # --- ダイナミクス計算 ---
        raw_input = X_data[t]
        img = raw_input * Gain
        
        # ---------------------------------------------------------
        # 統計量の更新 (Figure 5: netadapt.m 準拠)
        # ---------------------------------------------------------
        # mI = (1 - la*dt)*mI + la*dt*In
        # ※本来はIn(高速時定数)を使いますが、簡易的にxを使用します
        mI = (1 - la * dt) * mI + la * dt * x 
        
        # Ucc = (1 - la*dt)*Ucc + la*dt * ((Gamma * diff) * diff')
        # MATLAB: Gamma(Neur x In), diff(In x 1) -> Gamma*diff (Neur x 1)
        # Python: F(In x Neur), diff(In x 1) -> F.T @ diff (Neur x 1)
        # Update: diff (In x 1) * (F.T @ diff).T (1 x Neur) = (In x Neur)
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
                # F (Feedforward) の更新:
                # MATLAB: Gamma += eps * (O * (In - 100 * Ucc' * O)')
                # Python: F[:, k] += eps * (x - Ucc_scale * Ucc[:, k])
                
                recon_term = Ucc_scale * Ucc[:, k]
                F[:, k] += epsf * (x - recon_term)
                
                # C (Recurrent) の更新 (通常と同じ)
                C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])

            rO[k] += 1.0

        else:
            O = 0

        rO = (1 - leak * dt) * rO
        membrane_var_history.append(np.var(V))
    
    # 最終状態に統計量も含める
    final_states = {'V': V, 'rO': rO, 'x': x, 'mI': mI, 'Ucc': Ucc}

    return spike_times, spike_neurons, F, C, membrane_var_history, weight_error_history, decoding_error_history, final_states
import numpy as np
from scipy.signal import convolve

def init_weights(Nx, Nneuron, Nclasses):
    F = 0.5 * np.random.randn(Nx, Nneuron)
    F = F / (np.sqrt(np.sum(F**2, axis=0)) + 1e-8)
    C = -0.2 * np.random.rand(Nneuron, Nneuron) - 0.5 * np.eye(Nneuron)
    W_out = np.zeros((Nclasses, Nneuron))
    b_out = np.zeros(Nclasses)
    return F, C, W_out, b_out

def generate_smooth_dataset(n_time=1000, nx=2, sigma=30, seed=42):
    """
    MATLABコードのロジックに基づいて平滑化された時系列データセットを生成する関数
    """
    np.random.seed(seed)
    
    # 1. ガウス窓(カーネル)の作成
    # MATLAB: w = (1/(sigma*sqrt(2*pi))) * exp(...)
    # 1000ステップ分の窓を作成
    t_kernel = np.arange(1, 1001)
    # 中心を500にずらして計算
    w = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t_kernel - 500)**2) / (2 * sigma**2))
    w = w / np.sum(w) # 合計が1になるように正規化

    # 2. ホワイトノイズの生成
    # Nx次元 x Ntime時間
    white_noise = np.random.randn(nx, n_time)

    # 3. ガウス窓による平滑化 (畳み込み)
    smooth_input = np.zeros_like(white_noise)
    for d in range(nx):
        # mode='same' で出力サイズを入力と同じ長さに保つ
        smooth_input[d, :] = convolve(white_noise[d, :], w, mode='same')
        
    # 転置して (サンプル数, 特徴量数) の形にする => (1000, 2)
    X = smooth_input.T
    
    # 4. ラベル付け（非線形な決定境界の作成）
    radius = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    
    # 半径が一定範囲内ならクラス1 (赤)、それ以外はクラス0 (青)
    y = np.where((radius > 0.1), 1, 0)
    
    return X, y

def generate_smooth_Gaussiandataset(n_time=1000, nx=2, sigma=30, seed=42, 
                                    input_mean=0.0, input_std=1.0):
    """
    平均と分散(標準偏差)を指定可能な、平滑化されたガウス分布データセットを生成する関数
    """
    np.random.seed(seed)
    
    # 1. ガウス窓(カーネル)の作成
    t_kernel = np.arange(1, 1001)
    w = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t_kernel - 500)**2) / (2 * sigma**2))
    w = w / np.sum(w) 

    # 2. ガウス分布に従う点の生成 (★ここを修正)
    # size=(n_time, nx) にすることで、input_mean=[m1, m2] (shape=(2,)) が
    # 最後の次元(nx=2)に対して正しくブロードキャストされます。
    # その後、.T をして (nx, n_time) に戻します。
    gaussian_noise = np.random.normal(loc=input_mean, scale=input_std, size=(n_time, nx)).T

    # 3. ガウス窓による平滑化 (畳み込み)
    smooth_input = np.zeros_like(gaussian_noise)
    for d in range(nx):
        smooth_input[d, :] = convolve(gaussian_noise[d, :], w, mode='same')
        
    # 転置して (サンプル数, 特徴量数) の形にする
    X = smooth_input.T
    
    # 4. ラベル付け
    radius = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    
    # 半径が一定範囲内ならクラス1 (赤)、それ以外はクラス0 (青)
    y = np.where((radius > 0.12), 1, 0)
    
    return X, y

def generate_continuous_shift_dataset(n_train=5000, n_test=5000, nx=2, sigma=30, seed=42,
                                      train_params={'mean': 0.0, 'std': 1.0},
                                      test_params={'mean': 0.0, 'std': 1.0}): # ここでシフトを指定
    """
    前半(Train)と後半(Test)で分布が異なるが、接続部分が滑らかな時系列データを生成する。
    決定境界は前半(Train)のデータに基づいて決定され、後半にも適用される。
    """
    np.random.seed(seed)
    
    # 1. ガウス窓(カーネル)作成
    t_kernel = np.arange(1, 1001)
    w = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t_kernel - 500)**2) / (2 * sigma**2))
    w = w / np.sum(w)

    # 2. ホワイトノイズの生成（前半・後半を別々に作る）
    # Trainパート
    noise_train = np.random.normal(
        loc=train_params['mean'], 
        scale=train_params['std'], 
        size=(n_train, nx)
    ).T
    
    # Testパート（分布を変える）
    noise_test = np.random.normal(
        loc=test_params['mean'], 
        scale=test_params['std'], 
        size=(n_test, nx)
    ).T
    
    # ★重要: ノイズの段階で結合する
    full_noise = np.concatenate([noise_train, noise_test], axis=1) # shape: (nx, n_train+n_test)

    # 3. まとめて平滑化（これでつなぎ目が滑らかになる）
    smooth_input = np.zeros_like(full_noise)
    for d in range(nx):
        smooth_input[d, :] = convolve(full_noise[d, :], w, mode='same')
        
    X = smooth_input.T # shape: (TotalTime, nx)
    
    # 4. ラベル付け
    # 原点(0,0)を基準とするため、データの重心計算は不要です。
    # 単純に各座標の積をとります。
    
    # x * y > 0  --> 第1象限(+,+) または 第3象限(-,-) --> Class 1
    # x * y <= 0 --> 第2象限(-,+) または 第4象限(+,-) --> Class 0
    product = X[:, 0] * X[:, 1]

    y = np.where(product > 0, 1, 0)

    return X, y

def test_train_continuous(F_init, C_init, W_out, b_out, X_data, y_data, 
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain, Gain=1,
                          lr_readout=0.008, epsr=0.005, epsf=0.0005,
                          init_states=None):
    
    # 時間ステップ総数 (Continuous time steps)
    TotalTime = X_data.shape[0]
    print(f"Phase : Continuous Training/Testing (Total Steps: {TotalTime})")
    
    F = F_init.copy()
    C = C_init.copy()
    W_out = W_out.copy()
    b_out = b_out.copy()    
    
    # --- 初期状態のセットアップ ---
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
    k = 0

    total_spikes = 0
    acc_history = []
    acc_buffer = []  # 移動平均用
    spike_times = []
    spike_neurons = []
    membrane_var_history = [] # 各ステップの膜電位分散（空間的）を記録
    
    # --- 単一のメインループ (Time Step) ---
    for t in range(TotalTime):
        if t % 1000 == 0:
            print(f'\r  Step: {t}/{TotalTime} (Spikes: {total_spikes})', end='')

        # 1. 現在の時刻の入力を取得
        raw_input = X_data[t]      # shape: (Nx,)
        img = raw_input * Gain     # ゲインをかける
        
        # 2. 現在の時刻のターゲットを取得 & ワンホット化
        label_scalar = int(y_data[t])
        target_vec = np.zeros(Nclasses)
        target_vec[label_scalar] = 1.0
        
        # --- ネットワークダイナミクス (更新則は変更なし) ---
        noise = 0.01 * np.random.randn(Nneuron)
        
        recurrent_input = 0
        if O == 1:
            recurrent_input = C[:, k] # 前のステップでスパイクしたニューロンkからの入力

        # 膜電位 V の更新
        # 入力 img は毎ステップ変化する連続値になりました
        V = (1 - leak * dt) * V + dt * (F.T @ img) + recurrent_input + noise
        
        # フィルタ済み入力 x の更新
        x = (1 - leak * dt) * x + dt * img 
        
        # --- スパイク生成判定 ---
        thresh_noise = 0.01 * np.random.randn(Nneuron)
        potentials = V - Thresh - thresh_noise
        k_curr = np.argmax(potentials) # 最も閾値を超えているニューロンを探す

        if potentials[k_curr] >= 0:
            O = 1
            k = k_curr
            spike_times.append(t * dt) # 時刻を記録
            spike_neurons.append(k)
            total_spikes += 1
            
            # STDP的な重み更新 (retrain=Trueの場合のみ)
            if retrain:
                F[:, k] += epsf * (alpha * x - F[:, k])
                C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])
        else:
            O = 0

        # フィルタ済みスパイク列 rO の更新
        rO = (1 - leak * dt) * rO
        if O == 1:
            rO[k] += 1.0
        
        # --- Readout Learning (毎ステップ実行) ---
        y_est_vec = np.dot(W_out, rO) + b_out
        
        # 誤差計算と更新
        error_vec = target_vec - y_est_vec
        W_out += lr_readout * np.outer(error_vec, rO)
        b_out += lr_readout * error_vec
        
        # --- パフォーマンス記録 ---
        pred_idx = np.argmax(y_est_vec)
        is_correct = 1 if pred_idx == label_scalar else 0
        
        acc_buffer.append(is_correct)
        if len(acc_buffer) > 200: acc_buffer.pop(0) # 直近200ステップの移動平均精度
        acc_history.append(np.mean(acc_buffer))

        # 膜電位の分散 (ここでは集団全体の分散を記録)
        membrane_var_history.append(np.var(V))

    print(f"\nPhase Completed. Final Accuracy (Moving Avg): {acc_history[-1]:.4f}")
    
    # 最終状態を保存
    final_states = {'V': V, 'rO': rO, 'x': x}

    return acc_history, spike_times, spike_neurons, F, C, W_out, b_out, membrane_var_history, final_states

import numpy as np

def test_train_continuous_soft(F_init, C_init, W_out, b_out, X_data, y_data, 
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          epsf, epsr, alpha, beta, mu, retrain, Gain=1,
                          lr_readout=0.008,
                          init_states=None,
                          sigma_wta=50.0): # 追加: Soft WTAの鋭さを調整するパラメータ
    
    # 時間ステップ総数 (Continuous time steps)
    TotalTime = X_data.shape[0]
    print(f"Phase : Continuous Training/Testing (Total Steps: {TotalTime}) [Soft WTA mode]")
    
    F = F_init.copy()
    C = C_init.copy()
    W_out = W_out.copy()
    b_out = b_out.copy()    
    
    # --- 初期状態のセットアップ ---
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
    k = 0

    total_spikes = 0
    acc_history = []
    acc_buffer = []  # 移動平均用
    spike_times = []
    spike_neurons = []
    membrane_var_history = [] 
    
    # --- 単一のメインループ (Time Step) ---
    for t in range(TotalTime):
        if t % 1000 == 0:
            print(f'\r  Step: {t}/{TotalTime} (Spikes: {total_spikes})', end='')

        # 1. 現在の時刻の入力を取得
        raw_input = X_data[t]      # shape: (Nx,)
        img = raw_input * Gain     # ゲインをかける
        
        # 2. 現在の時刻のターゲットを取得 & ワンホット化
        label_scalar = int(y_data[t])
        target_vec = np.zeros(Nclasses)
        target_vec[label_scalar] = 1.0
        
        # --- ネットワークダイナミクス ---
        noise = 0.01 * np.random.randn(Nneuron)
        
        recurrent_input = 0
        if O == 1:
            recurrent_input = C[:, k] # 前のステップでスパイクしたニューロンkからの入力

        # 膜電位 V の更新
        V = (1 - leak * dt) * V + dt * (F.T @ img) + recurrent_input + noise
        
        # フィルタ済み入力 x の更新
        x = (1 - leak * dt) * x + dt * img 
        
        # --- スパイク生成判定 (Soft WTAへの変更箇所) ---
        thresh_noise = 0.01 * np.random.randn(Nneuron)
        potentials = V - Thresh - thresh_noise
        
        # === Soft WTA Logic Start ===
        # 膜電位(potentials)が高いほど選ばれる確率が高くなる確率分布を作成
        # sigma_wta: 逆温度 (大きいとargmaxに近く、小さいと平坦になる)
        
        # オーバーフロー対策のため最大値を引いてからexpを計算
        scaled_potentials = sigma_wta * potentials
        max_pot = np.max(scaled_potentials)
        exp_p = np.exp(scaled_potentials - max_pot)
        probs = exp_p / np.sum(exp_p) # 確率分布の合計を1にする
        
        # 確率に基づいて勝者ニューロンk_currをサンプリング
        k_curr = np.random.choice(Nneuron, p=probs)
        # === Soft WTA Logic End ===

        # 選ばれたニューロンが実際に閾値を超えているか判定
        # (Softmaxは常に誰かを選ぶため、閾値チェックで発火の有無を決める)
        if potentials[k_curr] >= 0:
            O = 1
            k = k_curr
            spike_times.append(t * dt)
            spike_neurons.append(k)
            total_spikes += 1
            
            # STDP的な重み更新 (retrain=Trueの場合のみ)
            if retrain:
                F[:, k] += epsf * (alpha * x - F[:, k])
                C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])
        else:
            O = 0

        # フィルタ済みスパイク列 rO の更新
        rO = (1 - leak * dt) * rO
        if O == 1:
            rO[k] += 1.0
        
        # --- Readout Learning (毎ステップ実行) ---
        y_est_vec = np.dot(W_out, rO) + b_out
        
        # 誤差計算と更新
        error_vec = target_vec - y_est_vec
        W_out += lr_readout * np.outer(error_vec, rO)
        b_out += lr_readout * error_vec
        
        # --- パフォーマンス記録 ---
        pred_idx = np.argmax(y_est_vec)
        is_correct = 1 if pred_idx == label_scalar else 0
        
        acc_buffer.append(is_correct)
        if len(acc_buffer) > 200: acc_buffer.pop(0)
        acc_history.append(np.mean(acc_buffer))

        membrane_var_history.append(np.var(V))

    print(f"\nPhase Completed. Final Accuracy (Moving Avg): {acc_history[-1]:.4f}")
    
    # 最終状態を保存
    final_states = {'V': V, 'rO': rO, 'x': x}

    return acc_history, spike_times, spike_neurons, F, C, W_out, b_out, membrane_var_history, final_states
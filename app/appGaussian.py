import numpy as np
from scipy.signal import convolve

def init_weights(Nx, Nneuron, Nclasses):
    F = 0.5 * np.random.randn(Nx, Nneuron)
    F = F / (np.sqrt(np.sum(F**2, axis=0)) + 1e-8)
    C = -0.2 * np.random.rand(Nneuron, Nneuron) - 0.5 * np.eye(Nneuron)
    W_out = np.zeros((Nclasses, Nneuron))
    b_out = np.zeros(Nclasses)
    return F, C, W_out, b_out

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

    # 2. ノイズデータ生成
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
    
    return X

def compute_snapshot_decoding_error(F_frozen, C_frozen, input_sequence, 
                                    dt, leak, Thresh, Gain=200, initial_state=None):
    """
    現在の重みを固定し、渡されたデータ配列(input_sequence)に対する
    復号化誤差(Decoding Error)を計算する関数
    """
    test_steps = input_sequence.shape[0]
    Nx = input_sequence.shape[1]
    Nneuron = F_frozen.shape[1]
    
    if initial_state is None:
        V = np.zeros(Nneuron)
        rO = np.zeros(Nneuron)
        x = np.zeros(Nx)
    else:
        # ★メインループの状態をコピーしてスタート地点にする
        V = initial_state['V'].copy()
        rO = initial_state['rO'].copy()
        x = initial_state['x'].copy()
    
    # スパイク等の履歴
    rO_list = []
    x_target_list = []
    
    # 過渡応答を避けるためのバッファ
    transient_steps = min(100, test_steps // 20)
    
    # --- 評価用実行ループ (重み更新なし) ---
    O_vec = np.zeros(Nneuron)
    
    for t in range(test_steps):
        # 入力を取得
        raw_input = input_sequence[t]
        img = raw_input * Gain
        
        # ダイナミクス計算 (学習時と同じだが重み固定)
        # Recurrent Input
        recurrent_input = C_frozen @ O_vec
        
        # ノイズ
        noise = 0.01 * np.random.randn(Nneuron)
        
        # V, x 更新
        V = (1 - leak * dt) * V + dt * (F_frozen.T @ img) + recurrent_input + noise
        x = (1 - leak * dt) * x + dt * img
        
        # スパイク判定
        thresh_noise = 0.01 * np.random.randn(Nneuron)
        potentials = V - Thresh - thresh_noise
        
        k_curr = np.argmax(potentials)
        O_vec = np.zeros(Nneuron)
        
        if potentials[k_curr] >= 0:
            O_vec[k_curr] = 1.0
            rO = rO + O_vec             # 先に足す

        # rO 更新
        rO = (1 - leak * dt) * rO
        
        # 履歴保存 (過渡応答終了後)
        if t >= transient_steps:
            rO_list.append(rO.copy())
            x_target_list.append(x.copy())
            
    # --- 最適デコーダの計算と誤差評価 ---
    if len(rO_list) == 0:
        return 1.0

    X_target = np.array(x_target_list) # (Samples, Nx)
    R_response = np.array(rO_list)     # (Samples, Nneuron)
    
    # Ridge回帰または最小二乗法で最適なデコーダ D を求める (X ~ R @ D.T)
    # R @ D.T = X  =>  D.T = lstsq(R, X)
    try:
        Dec_T, _, _, _ = np.linalg.lstsq(R_response, X_target, rcond=None)
        
        # 再構成信号
        X_est = R_response @ Dec_T
        
        # ★変更点: 平均二乗誤差 (Mean Squared Error) の計算
        # 元のコード: 分散比 (1 - R^2 のような指標)
        # 今回の変更: 単純な二乗誤差の平均 (MSE)
        
        error_sq = (X_target - X_est) ** 2
        decoding_error = np.mean(error_sq)
        
        # もし「平均」ではなく「総和 (Sum of Squared Errors)」が必要な場合は
        # decoding_error = np.sum(error_sq) 
        # としてください。一般的にはステップ数に依存しない mean が使いやすいです。
            
    except np.linalg.LinAlgError:
        # エラー発生時は安全策として大きめの値を返す（文脈に応じて調整してください）
        decoding_error = 1.0
        
    return decoding_error

def test_train_continuous_nonclass(F_init, C_init, X_data,
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain, Gain=200,
                          epsr=0.05, epsf=0.005, init_states=None):
    
    TotalTime = X_data.shape[0]
    print(f"Phase : Continuous Training/Testing (Total Steps: {TotalTime})")
    
    F = F_init.copy()
    C = C_init.copy()
    
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
    rO_prev_step = np.zeros(Nneuron) # 前ステップのrO保持用(必要なら)

    spike_times = []
    spike_neurons = []
    membrane_var_history = []
    weight_error_history = []   # ★ 追加: 重み誤差
    decoding_error_history = [] # ★ 追加: 復号化誤差
    
    # 評価設定
    eval_interval = 10000       # 10000ステップごとに評価
    test_chunk_size = 2000      # 評価に使うデータ長
    
    for t in range(TotalTime):
        if t % 1000 == 0:
            print(f'\r  Step: {t}/{TotalTime}', end='')
            
        # ---------------------------------------------------------
        # 1. 復号化誤差の計算 (Decoding Error)
        # ---------------------------------------------------------
        if t % eval_interval == 0:
            # データセットの「次」の区間を使ってテストする (未来予測的な評価)
            # データの残りが十分にある場合のみ実行
            if t + test_chunk_size < TotalTime:
                # 実際のデータセットからテストデータを切り出す
                test_sequence = X_data[t : t + test_chunk_size]

                current_state_snapshot = {
                    'V': V,
                    'rO': rO,
                    'x': x
                }
                
                d_err = compute_snapshot_decoding_error(
                    F, C, test_sequence, dt, leak, Thresh, Gain,
                    initial_state=current_state_snapshot
                )
                decoding_error_history.append(d_err)
                # print(f" [Eval t={t}] DecErr: {d_err:.4f}") # デバッグ用
            else:
                # データ末尾付近では計算しない、または最後の値をコピー
                if len(decoding_error_history) > 0:
                    decoding_error_history.append(decoding_error_history[-1])

        # ---------------------------------------------------------
        # 2. 最適重みとの距離 (Distance to Optimal Weights)
        # ---------------------------------------------------------
        if t % 100 == 0: # 頻繁に計算しても軽い
            C_opt = -F.T @ F
            C_norm = np.sum(C**2)
            if C_norm > 1e-12:
                # 最適なスケールを合わせる (MATLAB準拠)
                optscale = np.trace(C.T @ C_opt) / np.sum(C_opt**2)
                w_err = np.sum((C - optscale * C_opt)**2) / C_norm
            else:
                w_err = 0.0
            weight_error_history.append(w_err)

        # --- 以下、通常の学習ループ ---
        raw_input = X_data[t]
        img = raw_input * Gain
        
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
                F[:, k] += epsf * (alpha * x - F[:, k])
                C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])

            rO[k] += 1.0

        else:
            O = 0

        rO = (1 - leak * dt) * rO

        membrane_var_history.append(np.var(V))
    
    final_states = {'V': V, 'rO': rO, 'x': x}

    # 戻り値を増やす
    return spike_times, spike_neurons, F, C, membrane_var_history, weight_error_history, decoding_error_history, final_states

def test_train_continuous_suggest_nonclass(F_init, C_init, X_data,
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain, Gain=200,
                          eps=0.005, init_states=None):
    
    TotalTime = X_data.shape[0]
    print(f"Phase : Continuous Training/Testing (Total Steps: {TotalTime})")
    
    F = F_init.copy()
    C = C_init.copy()
    
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
    rO_prev_step = np.zeros(Nneuron) # 前ステップのrO保持用(必要なら)

    spike_times = []
    spike_neurons = []
    membrane_var_history = []
    weight_error_history = []   # ★ 追加: 重み誤差
    decoding_error_history = [] # ★ 追加: 復号化誤差
    
    # 評価設定
    eval_interval = 10000       # 10000ステップごとに評価
    test_chunk_size = 2000      # 評価に使うデータ長
    
    for t in range(TotalTime):
        if t % 1000 == 0:
            print(f'\r  Step: {t}/{TotalTime}', end='')
            
        # ---------------------------------------------------------
        # 1. 復号化誤差の計算 (Decoding Error)
        # ---------------------------------------------------------
        if t % eval_interval == 0:
            # データセットの「次」の区間を使ってテストする (未来予測的な評価)
            # データの残りが十分にある場合のみ実行
            if t + test_chunk_size < TotalTime:
                # 実際のデータセットからテストデータを切り出す
                test_sequence = X_data[t : t + test_chunk_size]

                current_state_snapshot = {
                    'V': V,
                    'rO': rO,
                    'x': x
                }
                
                d_err = compute_snapshot_decoding_error(
                    F, C, test_sequence, dt, leak, Thresh, Gain,
                    initial_state=current_state_snapshot
                )
                decoding_error_history.append(d_err)
                # print(f" [Eval t={t}] DecErr: {d_err:.4f}") # デバッグ用
            else:
                # データ末尾付近では計算しない、または最後の値をコピー
                if len(decoding_error_history) > 0:
                    decoding_error_history.append(decoding_error_history[-1])

        # ---------------------------------------------------------
        # 2. 最適重みとの距離 (Distance to Optimal Weights)
        # ---------------------------------------------------------
        if t % 100 == 0: # 頻繁に計算しても軽い
            C_opt = -F.T @ F
            C_norm = np.sum(C**2)
            if C_norm > 1e-12:
                # 最適なスケールを合わせる (MATLAB準拠)
                optscale = np.trace(C.T @ C_opt) / np.sum(C_opt**2)
                w_err = np.sum((C - optscale * C_opt)**2) / C_norm
            else:
                w_err = 0.0
            weight_error_history.append(w_err)

        # --- 以下、通常の学習ループ ---
        raw_input = X_data[t]
        img = raw_input * Gain
        
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
                # 1. 膜電位の分散の計算
                current_var = np.var(V)
                
                # 2. 学習率の都度計算
                # epsf = eps * np.var(V)
                # epsr = 10 * epsf
                current_epsf = eps * current_var
                current_epsr = 10 * current_epsf
                if t < 100:
                    print(f" Step {t}: Var={current_var:.4f}, epsf={current_epsf:.6f}, epsr={current_epsr:.6f}")
                F[:, k] += current_epsf * (alpha * x - F[:, k])
                C[:, k] -= current_epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])

            rO[k] += 1.0

        else:
            O = 0

        rO = (1 - leak * dt) * rO

        membrane_var_history.append(np.var(V))
    
    final_states = {'V': V, 'rO': rO, 'x': x}

    # 戻り値を増やす
    return spike_times, spike_neurons, F, C, membrane_var_history, weight_error_history, decoding_error_history, final_states

def test_train_continuous_correlated(F_init, C_init, X_data,
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain, Gain=200,
                          epsr=0.05, epsf=0.005, 
                          la=0.2, Ucc_scale=100.0, # Figure 5用の追加パラメータ
                          init_states=None):
    """
    Brendel & Machens (2011) Figure 5に準拠した、相関のある入力(Colored Noise)に対する学習則を実装。
    入力の共分散(Ucc)をオンライン推定し、デコーディング重みFを更新する。
    
    Args:
        la (float): 統計量(mI, Ucc)推定のためのリーク係数 (netadapt.mのla=0.2相当)
        Ucc_scale (float): Ucc項の重み付け係数 (netadapt.mの100相当)
    """
    
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
    
    eval_interval = 10000
    test_chunk_size = 2000
    
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

        # ---------------------------------------------------------
        # 1. 復号化誤差の計算 (Decoding Error)
        # ---------------------------------------------------------
        if t % eval_interval == 0:
            if t + test_chunk_size < TotalTime:
                test_sequence = X_data[t : t + test_chunk_size]
                current_state_snapshot = {'V': V, 'rO': rO, 'x': x}
                d_err = compute_snapshot_decoding_error(
                    F, C, test_sequence, dt, leak, Thresh, Gain,
                    initial_state=current_state_snapshot
                )
                decoding_error_history.append(d_err)
            else:
                if len(decoding_error_history) > 0:
                    decoding_error_history.append(decoding_error_history[-1])

        # ---------------------------------------------------------
        # 2. 重み誤差 (Weight Error)
        # ---------------------------------------------------------
        if t % 100 == 0:
            C_opt = -F.T @ F
            C_norm = np.sum(C**2)
            if C_norm > 1e-12:
                optscale = np.trace(C.T @ C_opt) / np.sum(C_opt**2)
                w_err = np.sum((C - optscale * C_opt)**2) / C_norm
            else:
                w_err = 0.0
            weight_error_history.append(w_err)
        
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

def test_train_continuous_correlated_proposed(F_init, C_init, X_data,
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain, Gain=200,
                          eps=0.005, 
                          la=0.2, Ucc_scale=100.0, # Figure 5用の追加パラメータ
                          init_states=None):
    """
    Brendel & Machens (2011) Figure 5に準拠した、相関のある入力(Colored Noise)に対する学習則を実装。
    入力の共分散(Ucc)をオンライン推定し、デコーディング重みFを更新する。
    
    Args:
        la (float): 統計量(mI, Ucc)推定のためのリーク係数 (netadapt.mのla=0.2相当)
        Ucc_scale (float): Ucc項の重み付け係数 (netadapt.mの100相当)
    """
    
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
    
    eval_interval = 10000
    test_chunk_size = 2000
    
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

        # ---------------------------------------------------------
        # 1. 復号化誤差の計算 (Decoding Error)
        # ---------------------------------------------------------
        if t % eval_interval == 0:
            if t + test_chunk_size < TotalTime:
                test_sequence = X_data[t : t + test_chunk_size]
                current_state_snapshot = {'V': V, 'rO': rO, 'x': x}
                d_err = compute_snapshot_decoding_error(
                    F, C, test_sequence, dt, leak, Thresh, Gain,
                    initial_state=current_state_snapshot
                )
                decoding_error_history.append(d_err)
            else:
                if len(decoding_error_history) > 0:
                    decoding_error_history.append(decoding_error_history[-1])

        # ---------------------------------------------------------
        # 2. 重み誤差 (Weight Error)
        # ---------------------------------------------------------
        if t % 100 == 0:
            C_opt = -F.T @ F
            C_norm = np.sum(C**2)
            if C_norm > 1e-12:
                optscale = np.trace(C.T @ C_opt) / np.sum(C_opt**2)
                w_err = np.sum((C - optscale * C_opt)**2) / C_norm
            else:
                w_err = 0.0
            weight_error_history.append(w_err)
        
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
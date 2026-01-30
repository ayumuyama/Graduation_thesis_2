import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_potential_drift_data(
    n_steps: int = 200000,
    dt: float = 0.1,
    shift_step: int = 100000,
    
    # --- 変更点: 任意の座標を指定するための引数 ---
    # デフォルトはNoneとし、指定がなければ内部で適当な値を設定またはエラー
    centers_phase1: np.ndarray = None,  # Shape: (N_wells, 2)
    centers_phase2: np.ndarray = None,  # Shape: (N_wells, 2)
    
    # --- Phase 1 Parameters (Before Shift) ---
    noise_strength1: float = 0.8,
    well_depth1: float = 0.5,
    well_width1: float = 2.5,
    
    # --- Phase 2 Parameters (After Shift) ---
    noise_strength2: float = 0.8,
    well_depth2: float = 0.5,
    well_width2: float = 2.5,
):
    """
    任意の座標に配置されたポテンシャル井戸の移動・変形による共変量シフトデータを生成
    
    Args:
        centers_phase1: Phase 1 (シフト前) の井戸の中心座標リスト [[x1, y1], [x2, y2], ...]
        centers_phase2: Phase 2 (シフト後) の井戸の中心座標リスト
    """
    
    # デフォルト値の設定（もしNoneが渡された場合の安全策）
    if centers_phase1 is None:
        # デフォルト: 半径4の円周上に4点
        centers_phase1 = np.array([[4, 0], [0, 4], [-4, 0], [0, -4]])
    else:
        centers_phase1 = np.array(centers_phase1)
        
    if centers_phase2 is None:
        # デフォルト: Phase1と同じ（移動なし）
        centers_phase2 = centers_phase1.copy()
    else:
        centers_phase2 = np.array(centers_phase2)

    # 状態の初期化
    x = np.zeros((n_steps, 2))
    labels = np.zeros(n_steps, dtype=int)
    phases = np.zeros(n_steps, dtype=int)
    
    # 初期位置（ランダム）
    x[0] = np.random.uniform(-1, 1, 2)

    # ポテンシャルの勾配計算関数
    def get_gradient(pos, wells, depth, width):
        grad = np.zeros(2)
        for w in wells:
            diff = pos - w
            dist_sq = np.sum(diff**2)
            # Gaussian Potential: U(x) = - A * exp(-|x-mu|^2 / 2sigma^2)
            # Force = - grad U
            coeff = (depth / (width**2)) * np.exp(-dist_sq / (2 * width**2))
            grad += coeff * diff 
        return grad

    # --- 時間発展ループ ---
    for t in range(n_steps - 1):
        # シフト判定とパラメータの切り替え
        if t < shift_step:
            current_wells = centers_phase1
            current_depth = well_depth1
            current_width = well_width1
            current_noise = noise_strength1
            phases[t] = 0
        else:
            current_wells = centers_phase2
            current_depth = well_depth2
            current_width = well_width2
            current_noise = noise_strength2
            phases[t] = 1
            
        # 運動方程式: dx = -grad(U)dt + noise
        force = -get_gradient(x[t], current_wells, current_depth, current_width)
        noise = np.random.normal(0, np.sqrt(dt), 2) * current_noise
        x[t+1] = x[t] + force * dt + noise
        
        # ラベリング (XOR: 第1/3象限=0, 第2/4象限=1)
        # ※井戸の位置に関わらず、座標の正負でラベル付けする場合
        if x[t+1][0] > 0 and x[t+1][1] > 0:   labels[t+1] = 0 # Q1
        elif x[t+1][0] < 0 and x[t+1][1] > 0: labels[t+1] = 1 # Q2
        elif x[t+1][0] < 0 and x[t+1][1] < 0: labels[t+1] = 2 # Q3
        else:                                 labels[t+1] = 3 # Q4
            
    # 最終ステップの処理
    phases[-1] = 1 if (n_steps - 1) >= shift_step else 0
    if x[-1][0] > 0 and x[-1][1] > 0:     labels[-1] = 0
    elif x[-1][0] < 0 and x[-1][1] > 0:   labels[-1] = 1
    elif x[-1][0] < 0 and x[-1][1] < 0:   labels[-1] = 2
    else:                                 labels[-1] = 3

    return x, labels, phases, centers_phase1, centers_phase2

def init_weights(Nx, Nneuron, Nclasses):
    F = 0.5 * np.random.randn(Nx, Nneuron)
    F = F / (np.sqrt(np.sum(F**2, axis=0)) + 1e-8)
    C = -0.2 * np.random.rand(Nneuron, Nneuron) - 0.5 * np.eye(Nneuron)
    W_out = np.zeros((Nclasses, Nneuron))
    b_out = np.zeros(Nclasses)
    return F, C, W_out, b_out

def test_train_continuous_class(F_init, C_init, X_data, Y_data, # Y_dataを追加
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain, Gain=200,
                          epsr=0.05, epsf=0.005, 
                          W_out_init=None, b_out_init=None, lr_out=0.01, # 線形学習器用の引数を追加
                          init_states=None, shift_step=None):
    
    TotalTime = X_data.shape[0]
    print(f"Phase : Continuous Training/Testing with Supervised Readout (Total Steps: {TotalTime})")
    
    F = F_init.copy()
    C = C_init.copy()
    
    # 線形学習器の初期化
    if W_out_init is None:
        W_out = np.zeros((Nclasses, Nneuron))
    else:
        W_out = W_out_init.copy()
        
    if b_out_init is None:
        b_out = np.zeros(Nclasses)
    else:
        b_out = b_out_init.copy()
    
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

    spike_times = []
    spike_neurons = []
    membrane_var_history = []
    
    # 分類精度記録用
    accuracy_history = []
    window_size = 2000 # 移動平均用
    prediction_history = [] # 正誤履歴 (1:正解, 0:不正解)
    decoding_error_history = [] # ★ 追加: 復号化誤差

    # 評価設定
    eval_interval = 10000       # 10000ステップごとに評価
    test_chunk_size = 2000      # 評価に使うデータ長
    
    for t in range(TotalTime):
        if shift_step is not None and t == shift_step:
            print(f"\n[Info] Resetting prediction history at step {t}")
            prediction_history = [] # 履歴をクリア
        if t % 1000 == 0:
            acc = np.mean(prediction_history[-window_size:]) if len(prediction_history) > 0 else 0
            print(f'\r  Step: {t}/{TotalTime} | Acc (last {window_size}): {acc:.4f}', end='')
            accuracy_history.append(acc)

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

        # --- 以下、通常の学習ループ ---
        raw_input = X_data[t]
        img = raw_input * Gain
        
        noise = 0.01 * np.random.randn(Nneuron)
        recurrent_input = 0
        if O == 1:
            recurrent_input = C[:, k]

        V = (1 - leak * dt) * V + dt * (F.T @ img) + recurrent_input + noise
        #V = np.clip(V, -10.0, 10.0)
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
                # F, C の学習則（変更なし）

                F[:, k] += epsf * (alpha * x - F[:, k])
                C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])

            rO[k] += 1.0

        else:
            O = 0

        rO = (1 - leak * dt) * rO

        # ---------------------------------------------------------
        # 線形学習器 (Supervised Online Learning)
        # ---------------------------------------------------------
        # ターゲットの取得
        if Y_data.ndim > 1:
            label = int(Y_data[t, 0])
            
        else:
            label = int(Y_data[t])
            
        target_vec = np.zeros(Nclasses)
        target_vec[label] = 1.0
        
        # 予測 (y = W_out * rO + b)
        # rOはローパスフィルタされたスパイク列なので、これを入出力に使うのが標準的
        pred_vec = np.dot(W_out, rO) + b_out
        
        # 誤差計算 (LMS / Delta Rule)
        error = target_vec - pred_vec
        
        # 重み更新: W += lr * error * input.T
        # inputは rO
        W_out += lr_out * np.outer(error, rO)
        b_out += lr_out * error
        
        # 精度評価
        pred_label = np.argmax(pred_vec)
        is_correct = 1 if pred_label == label else 0
        prediction_history.append(is_correct)
        
        membrane_var_history.append(np.var(V))
    
    final_states = {'V': V, 'rO': rO, 'x': x}

    # 戻り値に W_out, b_out, accuracy_history を追加
    return spike_times, spike_neurons, F, C, W_out, b_out, membrane_var_history, accuracy_history, final_states, decoding_error_history

def test_train_continuous_suggest_class(F_init, C_init, X_data, Y_data, # Y_dataを追加
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain, Gain=200,
                          eps=0.05, 
                          W_out_init=None, b_out_init=None, lr_out=0.01, # 線形学習器用の引数を追加
                          init_states=None, shift_step=None):
    
    TotalTime = X_data.shape[0]
    print(f"Phase : Continuous Training/Testing with Supervised Readout (Total Steps: {TotalTime})")
    
    F = F_init.copy()
    C = C_init.copy()
    
    # 線形学習器の初期化
    if W_out_init is None:
        W_out = np.zeros((Nclasses, Nneuron))
    else:
        W_out = W_out_init.copy()
        
    if b_out_init is None:
        b_out = np.zeros(Nclasses)
    else:
        b_out = b_out_init.copy()
    
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

    spike_times = []
    spike_neurons = []
    membrane_var_history = []
    
    # 分類精度記録用
    accuracy_history = []
    window_size = 2000 # 移動平均用
    prediction_history = [] # 正誤履歴 (1:正解, 0:不正解)
    decoding_error_history = [] # ★ 追加: 復号化誤差

    # 評価設定
    eval_interval = 10000       # 10000ステップごとに評価
    test_chunk_size = 2000      # 評価に使うデータ長
    
    for t in range(TotalTime):
        if shift_step is not None and t == shift_step:
            print(f"\n[Info] Resetting prediction history at step {t}")
            prediction_history = [] # 履歴をクリア
        if t % 1000 == 0:
            acc = np.mean(prediction_history[-window_size:]) if len(prediction_history) > 0 else 0
            print(f'\r  Step: {t}/{TotalTime} | Acc (last {window_size}): {acc:.4f}', end='')
            accuracy_history.append(acc)

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

        # --- 以下、通常の学習ループ ---
        raw_input = X_data[t]
        img = raw_input * Gain
        
        noise = 0.01 * np.random.randn(Nneuron)
        recurrent_input = 0
        if O == 1:
            recurrent_input = C[:, k]

        V = (1 - leak * dt) * V + dt * (F.T @ img) + recurrent_input + noise
        #V = np.clip(V, -10.0, 10.0)
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
                # F, C の学習則（変更なし）
                # 1. 膜電位の分散の計算
                current_var = np.var(V)
                
                # 2. 学習率の都度計算
                # epsf = eps * np.var(V)
                # epsr = 10 * epsf
                epsf = eps * current_var
                epsr = 10 * epsf
                if t < 100:
                    print(f"epsf{epsf}, epsr{epsr}")
                
                F[:, k] += epsf * (alpha * x - F[:, k])
                C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])

            rO[k] += 1.0

        else:
            O = 0

        rO = (1 - leak * dt) * rO

        # ---------------------------------------------------------
        # 線形学習器 (Supervised Online Learning)
        # ---------------------------------------------------------
        # ターゲットの取得
        if Y_data.ndim > 1:
            label = int(Y_data[t, 0])
            
        else:
            label = int(Y_data[t])
            
        target_vec = np.zeros(Nclasses)
        target_vec[label] = 1.0
        
        # 予測 (y = W_out * rO + b)
        # rOはローパスフィルタされたスパイク列なので、これを入出力に使うのが標準的
        pred_vec = np.dot(W_out, rO) + b_out
        
        # 誤差計算 (LMS / Delta Rule)
        error = target_vec - pred_vec
        
        # 重み更新: W += lr * error * input.T
        # inputは rO
        W_out += lr_out * np.outer(error, rO)
        b_out += lr_out * error
        
        # 精度評価
        pred_label = np.argmax(pred_vec)
        is_correct = 1 if pred_label == label else 0
        prediction_history.append(is_correct)
        
        membrane_var_history.append(np.var(V))
    
    final_states = {'V': V, 'rO': rO, 'x': x}

    # 戻り値に W_out, b_out, accuracy_history を追加
    return spike_times, spike_neurons, F, C, W_out, b_out, membrane_var_history, accuracy_history, final_states, decoding_error_history

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

# # --- データ生成の実行 ---
# if __name__ == "__main__":
    
#     # 例：Phase 1 は少し歪んだ四角形
#     # 任意の座標 [x, y] のリストを指定
#     coords_phase1 = [
#         [3.0, 5.0],  # 第1象限寄り
#         [3.0, -5.0],  # 第2象限寄り
#         [4.0, 5.0],  # 第3象限寄り
#         [4.0, -5.0]   # 第4象限寄り
#     ]

#     # 例：Phase 2 は全体的に右上にずれて集まる
#     coords_phase2 = [
#         [5.0, 5.0], 
#         [5.0, -5.0], 
#         [6.0, 5.0], 
#         [6.0, -5.0] 
#     ]

#     X, y, p, w1, w2 = generate_potential_drift_data(
#         n_steps=1000000,
#         dt=0.1,
#         shift_step=500000,
        
#         # ここで任意の座標を渡す
#         centers_phase1=coords_phase1,
#         centers_phase2=coords_phase2,
        
#         # パラメータ設定
#         well_depth1=5.0,
#         well_width1=4.0,
#         noise_strength1=1.0,
        
#         well_depth2=5.0,
#         well_width2=4.0,
#         noise_strength2=1.0
#     )

#     # --- 可視化 ---
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
#     colors = {0: 'blue', 1: 'red', 2: 'orange', 3:'green'}
#     class_names = {0: 'Q1', 1: 'Q2', 2: 'Q3', 3: 'Q4'}

#     def plot_phase_data(ax, X_phase, y_phase, wells, title):
#         for lbl in [0, 1, 2, 3]:
#             mask = y_phase == lbl
#             ax.scatter(X_phase[mask, 0], X_phase[mask, 1], 
#                        c=colors[lbl], alpha=0.4, label=class_names[lbl], s=15)
#         ax.scatter(wells[:, 0], wells[:, 1], c='black', marker='X', s=150, label='Wells', zorder=5)
#         ax.set_title(title)
#         ax.axhline(0, color='gray', linestyle='--')
#         ax.axvline(0, color='gray', linestyle='--')
#         ax.set_xlim(-15, 15)
#         ax.set_ylim(-15, 15)
#         ax.set_aspect('equal')
#         ax.grid(True, alpha=0.3)
#         ax.legend(loc='upper right')

#     mask_p1 = p == 0
#     plot_phase_data(axes[0], X[mask_p1], y[mask_p1], w1, "Phase 1: Arbitrary Coords")

#     mask_p2 = p == 1
#     plot_phase_data(axes[1], X[mask_p2], y[mask_p2], w2, "Phase 2: Shifted Coords")

#     plt.tight_layout()
#     plt.savefig("apotential.png")

#     # データ保存処理（変更なし）
#     df = pd.DataFrame(X, columns=['x1', 'x2'])
#     df['label'] = y
#     df['phase'] = p
#     df.to_csv('xor_shift_data.csv', index=False)
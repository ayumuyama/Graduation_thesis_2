import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_potential_drift_data(
    n_steps: int = 200000,
    dt: float = 0.1,
    noise_strength: float = 0.8,
    well_depth: float = 0.5,
    well_width: float = 2.5,
    shift_step: int = 100000,
    radius = 4.0,
    start_angle = 30,
    theta = 30
):
    """
    ポテンシャル井戸の移動による共変量シフトを含む2次元時系列データを生成
    
    Returns:
        x: 座標 (n_steps, 2)
        labels: XORラベル (n_steps,)
        phases: フェーズ識別子 (n_steps,) 0=Before Shift, 1=After Shift
        wells_phase1: シフト前の井戸座標
        wells_phase2: シフト後の井戸座標
    """
    
    # 状態の初期化
    x = np.zeros((n_steps, 2))
    labels = np.zeros(n_steps, dtype=int)
    phases = np.zeros(n_steps, dtype=int)
    
    # 初期位置（ランダム）
    x[0] = np.random.uniform(-1, 1, 1)
    angles_deg = start_angle + np.array([0, 90, 180, 270])
    angles_rad = np.deg2rad(angles_deg)

    ido_x = radius * np.cos(np.pi / 4)  # または radius / np.sqrt(2)
    ido_y = radius * np.sin(np.pi / 4)  # または radius / np.sqrt(2)

    # --- ポテンシャル井戸の設定 ---
    # フェーズ1: 標準的なXOR配置（各象限の中心）
    wells_phase1 = np.column_stack((
    radius * np.cos(angles_rad),  # X座標
    radius * np.sin(angles_rad)   # Y座標
))
    
    # 回転角の準備
    theta_rad = np.radians(theta)
    c, s = np.cos(theta_rad), np.sin(theta_rad)

    # 回転行列 R
    # [[cos, -sin],
    #  [sin,  cos]]
    R = np.array([
        [c, -s],
        [s,  c]
    ])

    # フェーズ1の座標群 (N, 2) に対して回転を適用
    # 行列積: (N, 2) dot (2, 2).T -> (N, 2)
    wells_phase2 = np.dot(wells_phase1, R.T)
    
    current_wells = wells_phase1
    
    # ポテンシャルの勾配計算関数 (Overdamped Langevin用)
    def get_gradient(pos, wells):
        grad = np.zeros(2)
        for w in wells:
            diff = pos - w
            dist_sq = np.sum(diff**2)
            # Gaussian Potential: U(x) = - A * exp(-|x-mu|^2 / 2sigma^2)
            # Force = - grad U
            coeff = (well_depth / (well_width**2)) * np.exp(-dist_sq / (2 * well_width**2))
            grad += coeff * diff 
        return grad

    # --- 時間発展ループ ---
    for t in range(n_steps - 1):
        # シフトの判定
        if t < shift_step:
            current_wells = wells_phase1
            phases[t] = 0
        else:
            current_wells = wells_phase2
            phases[t] = 1
            
        # 運動方程式: dx = -grad(U)dt + noise
        force = -get_gradient(x[t], current_wells)
        noise = np.random.normal(0, np.sqrt(dt), 2) * noise_strength
        x[t+1] = x[t] + force * dt + noise
        
        # ラベリング (XOR: Q1/Q3=0, Q2/Q4=1)
        if x[t+1][0] * x[t+1][1] > 0:
            labels[t+1] = 0 
        else:
            labels[t+1] = 1 
            
    # 最終ステップの処理
    phases[-1] = 1 if (n_steps - 1) >= shift_step else 0
    if x[-1][0] * x[-1][1] > 0: labels[-1] = 0
    else: labels[-1] = 1

    return x, labels, phases, wells_phase1, wells_phase2

def init_weights(Nx, Nneuron, Nclasses):
    F = 0.5 * np.random.randn(Nx, Nneuron)
    F = F / (np.sqrt(np.sum(F**2, axis=0)) + 1e-8)
    C = -0.2 * np.random.rand(Nneuron, Nneuron) - 0.5 * np.eye(Nneuron)
    W_out = np.zeros((Nclasses, Nneuron))
    b_out = np.zeros(Nclasses)
    return F, C, W_out, b_out

def test_train_continuous_nonclass(F_init, C_init, X_data, Y_data, # Y_dataを追加
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain, Gain=200,
                          epsr=0.05, epsf=0.005, 
                          W_out_init=None, b_out_init=None, lr_out=0.01, # 線形学習器用の引数を追加
                          init_states=None):
    
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
    window_size = 10 # 移動平均用
    prediction_history = [] # 正誤履歴 (1:正解, 0:不正解)
    
    for t in range(TotalTime):
        if t % 1000 == 0:
            acc = np.mean(prediction_history[-window_size:]) if len(prediction_history) > 0 else 0
            print(f'\r  Step: {t}/{TotalTime} | Acc (last {window_size}): {acc:.4f}', end='')
            accuracy_history.append(acc)

        # --- 以下、通常の学習ループ ---
        raw_input = X_data[t]
        img = raw_input * Gain
        if t < 10:
            print(raw_input)
        
        noise = 0.03 * np.random.randn(Nneuron)
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
            if t < 10:
                print(label)
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
    return spike_times, spike_neurons, F, C, W_out, b_out, membrane_var_history, accuracy_history, final_states

# # --- データ生成の実行 ---
# if __name__ == "__main__":
#     X, y, p, w1, w2 = generate_potential_drift_data(
#         n_steps = 200000,
#         dt = 0.1,
#         noise_strength = 0.5,
#         well_depth = 6.0,
#         well_width = 3.0,
#         shift_step = 100000,
#         radius = 5.0,
#         start_angle = 30,
#         theta = 30 
#     )

#     # --- 可視化 ---
# fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
# # ※ figsizeを(14, 6)から(12, 6)程度に調整すると，余白が減りバランスが良くなります

# # 色の設定: クラスごとに色分け
# colors = {0: 'blue', 1: 'red'}
# class_names = {0: 'Class 0 (Q1/Q3)', 1: 'Class 1 (Q2/Q4)'}

# # プロット用ヘルパー関数
# def plot_phase_data(ax, X_phase, y_phase, wells, title):
#     # クラスごとに散布図を描画
#     for lbl in [0, 1]:
#         mask = y_phase == lbl
#         ax.scatter(X_phase[mask, 0], X_phase[mask, 1], 
#                    c=colors[lbl], alpha=0.4, label=class_names[lbl], s=15)
    
#     # 井戸の中心を表示
#     ax.scatter(wells[:, 0], wells[:, 1], c='black', marker='X', s=150, label='Potential Wells', zorder=5)
    
#     # 軸と装飾
#     ax.set_title(title, fontsize=14)
#     ax.axhline(0, color='gray', linestyle='--', linewidth=1)
#     ax.axvline(0, color='gray', linestyle='--', linewidth=1)
#     ax.set_xlim(-6, 6)
#     ax.set_ylim(-6, 6)
    
#     # 【追加】アスペクト比を1:1（正方形）に固定
#     ax.set_aspect('equal')
#     # または ax.set_box_aspect(1) でも可
    
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc='upper right')

# # Phase 1 (Shift前) のプロット
# mask_p1 = p == 0
# plot_phase_data(axes[0], X[mask_p1], y[mask_p1], w1, "Phase 1: Before Shift")

# # Phase 2 (Shift後) のプロット
# mask_p2 = p == 1
# plot_phase_data(axes[1], X[mask_p2], y[mask_p2], w2, "Phase 2: After Covariate Shift")

# plt.suptitle("2D Time Series with Covariate Shift (XOR Labeling)", fontsize=16)
# plt.tight_layout()
# plt.savefig("apotential.png")

# # データ保存処理（変更なし）
# df = pd.DataFrame(X, columns=['x1', 'x2'])
# df['label'] = y
# df['phase'] = p
# df.to_csv('xor_shift_data.csv', index=False)
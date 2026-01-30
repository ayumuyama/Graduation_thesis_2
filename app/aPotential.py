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
    ido_x = 3.0,
    ido_y = 3.0,
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

    # --- ポテンシャル井戸の設定 ---
    # フェーズ1: 標準的なXOR配置（各象限の中心）
    wells_phase1 = np.array([
        [ido_x, ido_y],   # Q1 (+, +)
        [-ido_x, ido_y],  # Q2 (-, +)
        [-ido_x, -ido_y], # Q3 (-, -)
        [ido_x, -ido_y]   # Q4 (+, -)
    ])
    
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

def test_train_continuous_nonclass(F_init, C_init, X_data, labels,
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
            
        # # ---------------------------------------------------------
        # # 1. 復号化誤差の計算 (Decoding Error)
        # # ---------------------------------------------------------
        # if t % eval_interval == 0:
        #     # データセットの「次」の区間を使ってテストする (未来予測的な評価)
        #     # データの残りが十分にある場合のみ実行
        #     if t + test_chunk_size < TotalTime:
        #         # 実際のデータセットからテストデータを切り出す
        #         test_sequence = X_data[t : t + test_chunk_size]

        #         current_state_snapshot = {
        #             'V': V,
        #             'rO': rO,
        #             'x': x
        #         }
                
        #         d_err = compute_snapshot_decoding_error(
        #             F, C, test_sequence, dt, leak, Thresh, Gain,
        #             initial_state=current_state_snapshot
        #         )
        #         decoding_error_history.append(d_err)
        #         # print(f" [Eval t={t}] DecErr: {d_err:.4f}") # デバッグ用
        #     else:
        #         # データ末尾付近では計算しない、または最後の値をコピー
        #         if len(decoding_error_history) > 0:
        #             decoding_error_history.append(decoding_error_history[-1])

        # # ---------------------------------------------------------
        # # 2. 最適重みとの距離 (Distance to Optimal Weights)
        # # ---------------------------------------------------------
        # if t % 100 == 0: # 頻繁に計算しても軽い
        #     C_opt = -F.T @ F
        #     C_norm = np.sum(C**2)
        #     if C_norm > 1e-12:
        #         # 最適なスケールを合わせる (MATLAB準拠)
        #         optscale = np.trace(C.T @ C_opt) / np.sum(C_opt**2)
        #         w_err = np.sum((C - optscale * C_opt)**2) / C_norm
        #     else:
        #         w_err = 0.0
        #     weight_error_history.append(w_err)

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
                if t < 100:
                    print(f" Step {t}: epsf={epsf:.6f}, epsr={epsr:.6f}")

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

def init_weights(Nx, Nneuron, Nclasses):
    F = 0.5 * np.random.randn(Nx, Nneuron)
    F = F / (np.sqrt(np.sum(F**2, axis=0)) + 1e-8)
    C = -0.2 * np.random.rand(Nneuron, Nneuron) - 0.5 * np.eye(Nneuron)
    W_out = np.zeros((Nclasses, Nneuron))
    b_out = np.zeros(Nclasses)
    return F, C, W_out, b_out

# --- データ生成の実行 ---
X, y, p, w1, w2 = generate_potential_drift_data(
    n_steps = 10000,
    dt = 0.1,
    noise_strength = 0.5,
    well_depth = 0.6,
    well_width = 2.0,
    shift_step = 5000,
    ido_x = 3.0,
    ido_y = 3.0,
    theta = 30 
)

# --- 可視化 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# 色の設定: クラスごとに色分け
colors = {0: 'blue', 1: 'red'}
class_names = {0: 'Class 0 (Q1/Q3)', 1: 'Class 1 (Q2/Q4)'}

# プロット用ヘルパー関数
def plot_phase_data(ax, X_phase, y_phase, wells, title):
    # クラスごとに散布図を描画
    for lbl in [0, 1]:
        mask = y_phase == lbl
        ax.scatter(X_phase[mask, 0], X_phase[mask, 1], 
                   c=colors[lbl], alpha=0.4, label=class_names[lbl], s=15)
    
    # 井戸の中心を表示
    ax.scatter(wells[:, 0], wells[:, 1], c='black', marker='X', s=150, label='Potential Wells', zorder=5)
    
    # 軸と装飾
    ax.set_title(title, fontsize=14)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

# Phase 1 (Shift前) のプロット
mask_p1 = p == 0
plot_phase_data(axes[0], X[mask_p1], y[mask_p1], w1, "Phase 1: Before Shift")

# Phase 2 (Shift後) のプロット
mask_p2 = p == 1
plot_phase_data(axes[1], X[mask_p2], y[mask_p2], w2, "Phase 2: After Covariate Shift")

plt.suptitle("2D Time Series with Covariate Shift (XOR Labeling)", fontsize=16)
plt.tight_layout()
plt.savefig("apotential.png")


df = pd.DataFrame(X, columns=['x1', 'x2'])
df['label'] = y
df['phase'] = p
df.to_csv('xor_shift_data.csv', index=False)
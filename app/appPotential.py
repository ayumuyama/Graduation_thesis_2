import numpy as np
import matplotlib.pyplot as plt

class DynamicXORPotential:
    def __init__(self):
        # 初期設定: 4つの井戸を配置 (半径 r=3 の位置)
        # Class 0: 45度(Q1), 225度(Q3)
        # Class 1: 135度(Q2), 315度(Q4)
        self.r = 3.0
        self.base_angles = np.deg2rad([45, 135, 225, 315]) # [Q1, Q2, Q3, Q4]
        self.labels = [0, 1, 0, 1] # XOR配置: [Class 0, Class 1, Class 0, Class 1]
        
        # 現在の回転角（シフト量）
        self.current_rotation = 0.0
        
    def get_wells(self, rotation_angle=None):
        """現在の回転角に基づいて井戸の座標を返す"""
        if rotation_angle is not None:
            self.current_rotation = rotation_angle
            
        theta = self.base_angles + self.current_rotation
        wx = self.r * np.cos(theta)
        wy = self.r * np.sin(theta)
        
        # 形式: [x, y, 深さA, 広がりsigma, ラベル]
        # 井戸の深さと広がりは固定
        wells = []
        for i in range(4):
            wells.append([wx[i], wy[i], 10.0, 0.8, self.labels[i]])
        return wells

    def potential_gradient(self, x, y, wells):
        """ポテンシャルの勾配を計算する（粒子を動かす力）"""
        # 全体を原点に引き寄せる弱い力（発散防止）
        grad_x = 0.05 * x
        grad_y = 0.05 * y
        
        for wx, wy, A, sigma, _ in wells:
            dist_sq = (x - wx)**2 + (y - wy)**2
            exp_term = np.exp(-dist_sq / (2 * sigma**2))
            factor = (A / sigma**2) * exp_term
            
            grad_x -= factor * (x - wx)
            grad_y -= factor * (y - wy)
            
        return np.array([grad_x, grad_y])

    def generate_stream(self, n_samples, rotation_deg=0.0, noise_level=0.5):
        """
        指定された回転角のポテンシャルからデータをサンプリングする
        （ランジュバン動力学で少し動かして分布を作る）
        """
        wells = self.get_wells(np.deg2rad(rotation_deg))
        data = []
        labels = []
        
        # 各井戸から均等にスタートさせる
        samples_per_well = n_samples // 4
        
        for i in range(4):
            wx, wy, _, _, label = wells[i]
            
            # 各井戸周辺にn個の粒子を生成
            for _ in range(samples_per_well):
                # 初期位置: 井戸の中心 + 少しの乱数
                cx, cy = wx + np.random.randn()*0.5, wy + np.random.randn()*0.5
                
                # ポテンシャルに従って少し沈ませる（数ステップ更新）
                for _ in range(10): 
                    grad = self.potential_gradient(cx, cy, wells)
                    cx += -0.1 * grad[0] + np.random.randn() * 0.1
                    cy += -0.1 * grad[1] + np.random.randn() * 0.1
                
                # 最終的な位置にノイズを乗せてデータ点とする
                final_x = cx + np.random.randn() * noise_level
                final_y = cy + np.random.randn() * noise_level
                
                data.append([final_x, final_y])
                labels.append(label)
                
        return np.array(data), np.array(labels)

# --- シミュレーション実行 ---

xor_env = DynamicXORPotential()

# 1. 正常時 (Phase 1): 回転 0度
X1, y1 = xor_env.generate_stream(n_samples=200, rotation_deg=0.0)

# 2. 共変量シフト発生 (Phase 2): ポテンシャルを30度回転させる
# 入力分布 P(X) は変わるが、各クラスタの生成規則（ラベルの意味）は維持される
X2, y2 = xor_env.generate_stream(n_samples=200, rotation_deg=30.0)

# 3. さらにシフト (Phase 3): ポテンシャルを60度回転
X3, y3 = xor_env.generate_stream(n_samples=200, rotation_deg=60.0)


# --- 可視化 ---
plt.figure(figsize=(15, 5))

# Phase 1
plt.subplot(1, 3, 1)
plt.title("Phase 1: Normal (0 deg)")
plt.scatter(X1[y1==0, 0], X1[y1==0, 1], c='blue', marker='o', label='Class 0 (Q1/Q3)')
plt.scatter(X1[y1==1, 0], X1[y1==1, 1], c='red', marker='x', label='Class 1 (Q2/Q4)')
plt.xlim(-5, 5); plt.ylim(-5, 5); plt.grid(True)
plt.axhline(0, color='black', linewidth=1); plt.axvline(0, color='black', linewidth=1)
plt.legend(loc='lower right')

# Phase 2
plt.subplot(1, 3, 2)
plt.title("Phase 2: Covariate Shift (30 deg)")
plt.scatter(X2[y2==0, 0], X2[y2==0, 1], c='blue', marker='o', alpha=0.6)
plt.scatter(X2[y2==1, 0], X2[y2==1, 1], c='red', marker='x', alpha=0.6)
# 元の井戸の位置を点線で表示（比較用）
plt.xlim(-5, 5); plt.ylim(-5, 5); plt.grid(True)
plt.axhline(0, color='black', linewidth=1); plt.axvline(0, color='black', linewidth=1)

# Phase 3
plt.subplot(1, 3, 3)
plt.title("Phase 3: Large Shift (60 deg)")
plt.scatter(X3[y3==0, 0], X3[y3==0, 1], c='blue', marker='o', alpha=0.6)
plt.scatter(X3[y3==1, 0], X3[y3==1, 1], c='red', marker='x', alpha=0.6)
plt.xlim(-5, 5); plt.ylim(-5, 5); plt.grid(True)
plt.axhline(0, color='black', linewidth=1); plt.axvline(0, color='black', linewidth=1)

plt.tight_layout()
plt.savefig("potential.png")
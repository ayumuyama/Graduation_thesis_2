import numpy as np
import matplotlib.pyplot as plt

from app import appGaussian

X, y = appGaussian.generate_continuous_shift_dataset(n_train=5000, n_test=5000, nx=2, sigma=10, seed=42,
                                      train_params={'mean': 0.0, 'std': 1.0},
                                      test_params={'mean': [0.1, 0.1], 'std': 1.0})

# 前半と後半に分割（シームレスにつながっている）
X_train = X[:5000]
y_train = y[:5000]
X_test = X[5000:]
y_test = y[5000:]

print(f"Total Steps: {len(X)}")

# プロット
plt.figure(figsize=(8, 8))
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='blue', alpha=0.5, label='Class 0')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='red', alpha=0.5, label='Class 1')
# 軌跡を描画して「時系列的なつながり」を確認
plt.plot(X_train[:, 0], X_train[:, 1], c='gray', alpha=0.2, linewidth=1)

plt.title("Smoothed Random Walk with Non-linear Boundary")
plt.xlabel("Input Dimension 1")
plt.ylabel("Input Dimension 2")
plt.legend()
plt.grid(True)
plt.axis('equal') # アスペクト比を揃える
plt.savefig("smoothed_dataset_train.png")
plt.close()

plt.figure(figsize=(8, 8))
plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='blue', alpha=0.5, label='Class 0')
plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='red', alpha=0.5, label='Class 1')
# 軌跡を描画して「時系列的なつながり」を確認
plt.plot(X_test[:, 0], X_test[:, 1], c='gray', alpha=0.2, linewidth=1)

plt.title("Smoothed Random Walk with Non-linear Boundary")
plt.xlabel("Input Dimension 1")
plt.ylabel("Input Dimension 2")
plt.legend()
plt.grid(True)
plt.axis('equal') # アスペクト比を揃える
plt.savefig("smoothed_dataset_test.png")
plt.close()
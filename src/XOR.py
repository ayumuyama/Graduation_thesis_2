import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from app import aPotential as aPot

#入力データ，ラベルデータ取得
df = pd.read_csv('xor_shift_data.csv')

target_input = [0, 1]

X = df.iloc[:, target_input].values  #入力の作成

print(X.shape)
print(type(X))

target_labels = [2]

labels = df.iloc[:, target_labels].values

print(labels.shape)

#パラメータ設定
Nx = 2
Nneuron = 10
Nclasses = 2

dt = 0.001
leak = 50
Thresh = 0.5
alpha = 0.18
beta = 1 / 0.9
mu = 0.02 / 0.9

#重みの初期化
F_initial, C_initial, W_out_initial, b_out_initial = aPot.init_weights(Nx, Nneuron, Nclasses)

#学習の開始
spk_t_1, spk_i_1, F_learned, C_learned, W_learned, b_learned, mem_var, acc_hist, final_st = aPot.test_train_continuous_class(
                        F_initial, C_initial, X, labels,
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=80,
                        epsr=0.001, epsf=0.0001, 
                        W_out_init=W_out_initial, b_out_init=b_out_initial,
                        lr_out=0.0001, init_states=None)
spk_t_2, spk_i_2, F_learned_2, C_learned_2, W_learned_2, b_learned_2, mem_var_2, acc_hist_2, final_st_2 = aPot.test_train_continuous_suggest_class(
                        F_initial, C_initial, X, labels,
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=80,
                        eps=0.0001, W_out_init=W_out_initial, b_out_init=b_out_initial,
                        lr_out=0.0001, init_states=None)
# プロット
plt.figure(figsize=(10, 5))
x_axis = np.arange(1, len(acc_hist) + 1) * 1000
plt.plot(x_axis, acc_hist, marker='o', linestyle='-', label='Accuracy (MA)')
   
plt.title('Accuracy History (Moving Average window=1000)')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.grid(True)
plt.ylim(0, 1.1)
    
    # ドリフト地点の表示
plt.axvline(200000, color='r', linestyle='--', label='Drift Start (Step 200000)')
    
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_history_1.png')
plt.close()



# プロット
plt.figure(figsize=(10, 5))
x_axis = np.arange(1, len(acc_hist_2) + 1) * 1000
plt.plot(x_axis, acc_hist_2, marker='o', linestyle='-', label='Accuracy (MA)')
   
plt.title('Accuracy History (Moving Average window=1000)')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.grid(True)
plt.ylim(0, 1.1)
    
    # ドリフト地点の表示
plt.axvline(200000, color='r', linestyle='--', label='Drift Start (Step 200000)')
    
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_history_2.png')
plt.close()

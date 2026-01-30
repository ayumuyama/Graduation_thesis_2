import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from app import aPotential as aPot

#入力データ，ラベルデータ取得
df = pd.read_csv('xor_shift_data.csv')

target_input = [0, 1]

X = df.iloc[:, target_input].values  #入力の作成

target_labels = [2]

labels = df.iloc[:, target_labels].values

#パラメータ設定
Nx = 2
Nneuron = 10
Nclasses = 4

dt = 0.001
leak = 50
Thresh = 0.5
alpha = 0.18
beta = 1 / 0.9
mu = 0.02 / 0.9

#重みの初期化
F_initial, C_initial, W_out_initial, b_out_initial = aPot.init_weights(Nx, Nneuron, Nclasses)

#学習の開始
spk_t_1, spk_i_1, *_, acc_hist, final_st, de_err  = aPot.test_train_continuous_class(
                        F_initial, C_initial, X, labels,
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=15,
                        epsr=0.001, epsf=0.0001, 
                        W_out_init=W_out_initial, b_out_init=b_out_initial,
                        lr_out=0.00001, init_states=None, shift_step=500000)

spk_t_2, spk_i_2, *_, acc_hist_2, final_st_2, de_err_2 = aPot.test_train_continuous_suggest_class(
                        F_initial, C_initial, X, labels,
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=15,
                        eps=0.001, W_out_init=W_out_initial, b_out_init=b_out_initial,
                        lr_out=0.00001, init_states=None, shift_step=500000)
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
plt.axvline(500000, color='r', linestyle='--', label='Drift Start (Step 500000)')
    
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_history_1.png')
plt.close()

# --- ラスタープロット (Method 1) ---
plt.figure(figsize=(12, 6))

# 散布図: 横軸=時間, 縦軸=ニューロンID
# s: 点のサイズ (小さい方が見やすい)
# alpha: 透明度 (密度が高いところが濃くなる)
plt.scatter(spk_t_1, spk_i_1, s=5, color='black', alpha=0.6, label='Spike')

plt.title('Raster Plot (Method 1)')
plt.xlabel('Time [s]')
plt.ylabel('Neuron Index')
plt.yticks(range(Nneuron))  # Y軸の目盛りをニューロン数に合わせる
plt.ylim(-0.5, Nneuron - 0.5) # Y軸の範囲を調整
plt.grid(True, axis='x', linestyle=':', alpha=0.6)

# ドリフト地点の表示
# acc_histのプロットではStep数(200000)で線を引いていますが、
# ラスタープロットの横軸は「時間(s)」なので dt を掛けます。
drift_step = 200000
drift_time = drift_step * dt
plt.axvline(drift_time, color='r', linestyle='--', label=f'Drift Start ({drift_time}s)')

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('raster_plot_1.png')
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
plt.axvline(500000, color='r', linestyle='--', label='Drift Start (Step 500000)')
    
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_history_2.png')
plt.close()

# --- ラスタープロット (Method 2) ---
plt.figure(figsize=(12, 6))
plt.scatter(spk_t_2, spk_i_2, s=5, color='blue', alpha=0.6, label='Spike')

plt.title('Raster Plot (Method 2: Suggest Class)')
plt.xlabel('Time [s]')
plt.ylabel('Neuron Index')
plt.yticks(range(Nneuron))
plt.ylim(-0.5, Nneuron - 0.5)
plt.grid(True, axis='x', linestyle=':', alpha=0.6)

drift_time = 200000 * dt
plt.axvline(drift_time, color='r', linestyle='--', label=f'Drift Start ({drift_time}s)')

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('raster_plot_2.png')
plt.close()

#再構成誤差
plt.figure(figsize=(10, 6))
eval_interval = 10000
time_axis_dec = np.arange(len(de_err)) * dt * eval_interval
    
plt.plot(time_axis_dec, de_err, 'o-', color='black', label='Decoding Error', markersize=4)
    
# 境界線
time_offset_sec = (len(X) * dt) / 2
plt.axvline(x=time_offset_sec, color='red', linestyle='--', label='Domain Shift Point')

plt.xlabel('Time (s)', fontsize=24)
plt.ylabel('Decoding Error', fontsize=24)
    
plt.tick_params(axis='both', labelsize=20)

plt.ylim(0.0, 1.0)
plt.grid(True, which="both", ls="-", alpha=0.5)
    
plt.legend(fontsize=20, loc='upper left')

plt.savefig("Final_Decoding_Error.png", bbox_inches='tight')
plt.close()

eval_interval = 10000
time_axis_dec_2 = np.arange(len(de_err_2)) * dt * eval_interval
plt.figure(figsize=(10, 6))
    
plt.plot(time_axis_dec_2, de_err_2, 'o-', color='black', label='Decoding Error', markersize=4)
    
# 境界線
time_offset_sec = (len(X) * dt) / 2
plt.axvline(x=time_offset_sec, color='red', linestyle='--', label='Domain Shift Point')

plt.xlabel('Time (s)', fontsize=24)
plt.ylabel('Decoding Error', fontsize=24)
    
plt.tick_params(axis='both', labelsize=20)

plt.ylim(0.0, 1.0)
plt.grid(True, which="both", ls="-", alpha=0.5)
    
plt.legend(fontsize=20, loc='upper left')

plt.savefig("Final_Decoding_Error_2.png", bbox_inches='tight')
plt.close()
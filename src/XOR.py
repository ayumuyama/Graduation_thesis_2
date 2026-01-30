import pandas as pd
import numpy as np

from app import aPotential as aPot

#入力データ，ラベルデータ取得
df = pd.read_csv('xor_shift_data.csv')

target_input = [0, 1]

X = df.iloc[:, target_input].values  #入力の作成

print(input.shape)
print(type(input))

target_labels = [2]

labels = df.iloc[:, target_labels].values

print(labels.shape)

#パラメータ設定
Nx = 2
Nneuron = 20
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
spk_t_1, spk_i_1, F_set1, C_set1, mem_var_1, w_err_1, d_err_1, final_states_1 = aPot.test_train_continuous_nonclass(
                          F_initial, C_initial, X, labels,
                          Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                          alpha, beta, mu, retrain=True, Gain=100,
                          epsr=0.001, epsf=0.0001, init_states=None)


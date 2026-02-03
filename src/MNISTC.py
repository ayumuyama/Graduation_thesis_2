import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import appGaussian as appssian
from app import appMNIST as appNIST

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 設定
    # ---------------------------------------------------------
    Nneuron = 1000   
    Nx = 784
    Nclasses = 10        
    
    leak = 50       
    dt = 0.001      
    
    alpha = 0.18    
    beta = 1 / 0.9  
    mu = 0.02 / 0.9
    
    Thresh = 0.5

    Duration = 100

    lr_readout=0.0002
    
    # 保存先設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S(suggestGaussian_exp)")
    base_save_dir = Path("results")
    current_save_dir = base_save_dir / timestamp
    current_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved in: {current_save_dir}")
    
    # ---------------------------------------------------------
    # データ準備
    # ---------------------------------------------------------
    print("Preparing Data...")

    X_train, y_train = appNIST.load_and_preprocess("identity_test_images.npy", "identity_test_labels.npy")
    X_test, y_test = appNIST.load_and_preprocess("shotnoise_test_images.npy", "shotnoise_test_labels.npy")

    print(f"Set 1 shape: {X_train[0]}") 
    print(f"Set 2 shape: {y_test.shape}")
    # ---------------------------------------------------------
    # Set 1: Learning(non)
    # ---------------------------------------------------------
    print("--- Phase 1: Learning on Set 1 ---")
    F_initial, C_initial, *_ = appssian.init_weights(Nx, Nneuron, Nclasses)

    # 戻り値の最後に final_states_1 を受け取る
    nspk_t_1, nspk_i_1, nF_set1, nC_set1, nmem_var_1, nacc_his_1, nfinal_states_1, nW_1 = appNIST.test_train_continuous_correlated(
                        F_initial, C_initial, X_train, y_train,
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=10,
                        epsr=0.00005, epsf=0.000005, 
                        la=0.2, Ucc_scale=2.0, # Figure 5用の追加パラメータ
                        init_states=None,
                        lr_readout=0.002)
    
    # ---------------------------------------------------------
    # Set 1: Learning
    # ---------------------------------------------------------
    print("--- Phase 1: Learning on Set 1 ---")
    F_initial, C_initial, *_ = appssian.init_weights(Nx, Nneuron, Nclasses)

    spk_t_1, spk_i_1, F_set1, C_set1, mem_var_1, acc_his_1, final_states_1, W_1 = appNIST.test_train_continuous_correlated_proposed(
                        F_initial, C_initial, X_train, y_train, # y_dataを追加
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=10,
                        eps=0.00003, 
                        la=0.2, Ucc_scale=2.0, # Figure 5用の追加パラメータ
                        init_states=None,
                        lr_readout=0.002)
    
    # ---------------------------------------------------------
    # Set 2: Learning(non)
    # ---------------------------------------------------------
    print("--- Phase 2: Learning on Set 2 ---")

    # init_states に Set 1 の終わりの状態を渡す
    nspk_t_2, nspk_i_2, nF_set2, nC_set2, nmem_var_2, nacc_his_2, nfinal_states_2, nW_2 = appNIST.test_train_continuous_correlated(
                        nF_set1, nC_set1, X_test, y_test,
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=10,
                        epsr=0.00005, epsf=0.000005, 
                        la=0.2, Ucc_scale=2.0, # Figure 5用の追加パラメータ
                        init_states=nfinal_states_1,
                        lr_readout=0.002)
    
    # ---------------------------------------------------------
    # Set 2: Learning
    # ---------------------------------------------------------
    print("--- Phase 2: Learning on Set 2 ---")

    # 状態の引き継ぎ忘れないように注意
    spk_t_2, spk_i_2, F_set2, C_set2, mem_var_2, acc_his_2, final_states_2, W_2 = appNIST.test_train_continuous_correlated_proposed(
                        F_set1, C_set1, X_test, y_test, # y_dataを追加
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=10,
                        eps=0.00003, 
                        la=0.2, Ucc_scale=2.0, # Figure 5用の追加パラメータ
                        init_states=final_states_1,
                        lr_readout=0.002)

    # ---------------------------------------------------------
    # Data Combination
    # ---------------------------------------------------------
    print("Combining data for plots...")
    
    # Membrane Variance の結合
    full_mem_var = mem_var_1 + mem_var_2

    # 2. Spikeデータの結合
    # 計算用にnumpy配列へ変換
    t1 = np.array(spk_t_1)
    i1 = np.array(spk_i_1)
    t2 = np.array(spk_t_2)
    i2 = np.array(spk_i_2)

    # オフセットの計算 (Set1の最大時間。データが無い場合は0)
    time_offset = np.max(t1) if len(t1) > 0 else 0
    
    # 時刻をシフトして結合
    full_spk_t = np.concatenate([t1, t2 + time_offset])
    full_spk_i = np.concatenate([i1, i2])

    record_interval_steps = 100
    eval_interval = 10000

    # # ---------------------------------------------------------
    # # Plot 1: Raster Plot (Combined) - Modified
    # # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    plt.scatter(full_spk_t, full_spk_i, s=0.5, c='black', marker='.', alpha=0.6)

    plt.axvline(x=time_offset, color='red', linestyle='--', label='End of Set 1')

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title('Spike Raster Plot')
    plt.xlim(left=0, right=np.max(full_spk_t)) # 軸の範囲をデータの最大値に合わせる
    plt.ylim(0, Nneuron)
    plt.legend(loc='upper right')

    raster_plot_path = current_save_dir / "Final_Raster.png"
    plt.savefig(raster_plot_path)
    plt.close()
    print(f"Raster plot saved to: {raster_plot_path}")

    # ---------------------------------------------------------
    # Plot 2: Membrane Potential Variance (新規追加)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # 移動平均
    mem_window_size = 1000
    if len(full_mem_var) >= mem_window_size:
        b = np.ones(mem_window_size) / mem_window_size
        full_mem_smooth = np.convolve(full_mem_var, b, mode='valid')
        
        # 生データを薄くプロット
        plt.plot(full_mem_var, label='Raw Variance', color='lightgreen', alpha=0.4)
        # 移動平均を濃くプロット
        plt.plot(np.arange(mem_window_size - 1, len(full_mem_var)), full_mem_smooth, 
                 label=f'Moving Average (window={mem_window_size})', color='green', linewidth=2)
    else:
        plt.plot(full_mem_var, label='Voltage Variance', color='green')

    plt.axvline(x=len(X_train), color='red', linestyle='--', label='End of Set 1')
    
    plt.xlabel('Input Samples')
    plt.ylabel('Voltage Variance per Neuron')
    plt.title('Evolution of the Variance of the Membrane Potential')
    plt.grid(True)
    plt.legend()
    
    # 対数スケール
    plt.yscale('log') 
    
    mem_plot_path = current_save_dir / "Final_Voltage_Variance.png"
    plt.savefig(mem_plot_path)
    plt.close()
    print(f"Voltage Variance plot saved to: {mem_plot_path}")

    ##--------------------------------------------------------------------
    # ---------------------------------------------------------
    # Data Combination
    # ---------------------------------------------------------
    print("Combining data for plots...")
    
    # Membrane Variance の結合
    nfull_mem_var = nmem_var_1 + nmem_var_2

    # 2. Spikeデータの結合
    # 計算用にnumpy配列へ変換
    nt1 = np.array(nspk_t_1)
    ni1 = np.array(nspk_i_1)
    nt2 = np.array(nspk_t_2)
    ni2 = np.array(nspk_i_2)

    # オフセットの計算 (Set1の最大時間。データが無い場合は0)
    ntime_offset = np.max(nt1) if len(nt1) > 0 else 0
    
    # 時刻をシフトして結合
    nfull_spk_t = np.concatenate([nt1, nt2 + ntime_offset])
    nfull_spk_i = np.concatenate([ni1, ni2])
    

    nrecord_interval_steps = 100
    neval_interval = 10000

    # # ---------------------------------------------------------
    # # Plot 1: Raster Plot (Combined) - Modified
    # # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    plt.scatter(nfull_spk_t, nfull_spk_i, s=0.5, c='black', marker='.', alpha=0.6)

    plt.axvline(x=ntime_offset, color='red', linestyle='--', label='End of Set 1')

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title('Spike Raster Plot')
    plt.xlim(left=0, right=np.max(nfull_spk_t)) # 軸の範囲をデータの最大値に合わせる
    plt.ylim(0, Nneuron)
    plt.legend(loc='upper right')

    raster_plot_path = current_save_dir / "Final_Raster(non).png"
    plt.savefig(raster_plot_path)
    plt.close()
    print(f"Raster plot saved to: {raster_plot_path}")

    # ---------------------------------------------------------
    # Plot 2: Membrane Potential Variance (新規追加)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # 移動平均
    nmem_window_size = 1000
    if len(nfull_mem_var) >= nmem_window_size:
        nb = np.ones(nmem_window_size) / nmem_window_size
        nfull_mem_smooth = np.convolve(nfull_mem_var, nb, mode='valid')
        
        # 生データを薄くプロット
        plt.plot(nfull_mem_var, label='Raw Variance', color='lightgreen', alpha=0.4)
        # 移動平均を濃くプロット
        plt.plot(np.arange(nmem_window_size - 1, len(nfull_mem_var)), nfull_mem_smooth, 
                 label=f'Moving Average (window={nmem_window_size})', color='green', linewidth=2)
    else:
        plt.plot(nfull_mem_var, label='Voltage Variance', color='green')

    plt.axvline(x=len(X_train), color='red', linestyle='--', label='End of Set 1')
    
    plt.xlabel('Input Samples')
    plt.ylabel('Voltage Variance per Neuron')
    plt.title('Evolution of the Variance of the Membrane Potential')
    plt.grid(True)
    plt.legend()
    
    # 対数スケール
    plt.yscale('log') 
    
    mem_plot_path = current_save_dir / "Final_Voltage_Variance(non).png"
    plt.savefig(mem_plot_path)
    plt.close()
    print(f"Voltage Variance plot saved to: {mem_plot_path}")
    ##---------------------------------------------------------------------------------------------------------------
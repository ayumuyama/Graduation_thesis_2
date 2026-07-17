import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit
import pickle
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import appGaussian as appssian
from app import appMNIST as appNIST

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 設定
    # ---------------------------------------------------------
    Nneuron = 800   
    Nx = 784
    Nclasses = 10        
    
    leak = 50       
    dt = 0.001      
    
    alpha = 0.18    
    beta = 1 / 0.9  
    mu = 0.02 / 0.9
    
    Thresh = 0.5

    Duration = 50

    lr_readout=0.0005
    
    # 保存先設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S(stripe)")
    base_save_dir = Path("outputs")
    current_save_dir = base_save_dir / timestamp
    current_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved in: {current_save_dir}")
    
    # ---------------------------------------------------------                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    # データ準備
    # ---------------------------------------------------------
    print("Preparing Data...")

    X_train_org, y_train_org = appNIST.load_and_preprocess("identity_test_images.npy", "identity_test_labels.npy")
    X_test_org, y_test_org = appNIST.load_and_preprocess("stripe_test_images.npy", "stripe_test_labels.npy")

    X_train = np.repeat(X_train_org, Duration, axis=0)
    y_train = np.repeat(y_train_org, Duration, axis=0)

    X_test = np.repeat(X_test_org, Duration, axis=0)
    y_test = np.repeat(y_test_org, Duration, axis=0)

    # ---------------------------------------------------------
    # Set 1: Learning / Loading
    #---------------------------------------------------------
    set1_cache_file = Path("set1_checkpoint.pkl")  

    if set1_cache_file.exists():
        print(f"Loading Set 1 results from {set1_cache_file}...")
        with open(set1_cache_file, "rb") as f:
            data_set1 = pickle.load(f)
        
        spk_t_1, spk_i_1 = data_set1["spk_t_1"], data_set1["spk_i_1"]
        F_set1, C_set1 = data_set1["F_set1"], data_set1["C_set1"]
        mem_var_1, acc_his_1 = data_set1["mem_var_1"], data_set1["acc_his_1"]
        final_states_1, W_1 = data_set1["final_states_1"], data_set1["W_1"]

        nspk_t_1, nspk_i_1 = data_set1["nspk_t_1"], data_set1["nspk_i_1"]
        nF_set1, nC_set1 = data_set1["nF_set1"], data_set1["nC_set1"]
        nmem_var_1, nacc_his_1 = data_set1["nmem_var_1"], data_set1["nacc_his_1"]
        nfinal_states_1, nW_1 = data_set1["nfinal_states_1"], data_set1["nW_1"]

    else:
        print("--- Phase 1: Learning on Set 1 (Running) ---")
        F_initial, C_initial, *_ = appssian.init_weights(Nx, Nneuron, Nclasses)

        spk_t_1, spk_i_1, F_set1, C_set1, mem_var_1, acc_his_1, final_states_1, W_1 = appNIST.test_train_continuous_correlated_proposed(
                        F_initial, C_initial, X_train, y_train,
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=30,
                        eps=0.00005, la=0.2, Ucc_scale=2.0,
                        init_states=None, lr_readout=lr_readout, stim_duration=Duration)

        nspk_t_1, nspk_i_1, nF_set1, nC_set1, nmem_var_1, nacc_his_1, nfinal_states_1, nW_1 = appNIST.test_train_continuous_correlated(
                        F_initial, C_initial, X_train, y_train,
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=30,
                        epsr=0.00005, epsf=0.000005, la=0.2, Ucc_scale=2.0,
                        init_states=None, lr_readout=lr_readout, stim_duration=Duration)
    
        print(f"Saving Set 1 results to {set1_cache_file}...")
        save_data = {
            "spk_t_1": spk_t_1, "spk_i_1": spk_i_1,
            "F_set1": F_set1, "C_set1": C_set1,
            "mem_var_1": mem_var_1, "acc_his_1": acc_his_1,
            "final_states_1": final_states_1, "W_1": W_1,
            "nspk_t_1": nspk_t_1, "nspk_i_1": nspk_i_1,
            "nF_set1": nF_set1, "nC_set1": nC_set1,
            "nmem_var_1": nmem_var_1, "nacc_his_1": nacc_his_1,
            "nfinal_states_1": nfinal_states_1, "nW_1": nW_1
        }
        with open(set1_cache_file, "wb") as f:
            pickle.dump(save_data, f)

    # ---------------------------------------------------------
    # Set 2: Learning
    # ---------------------------------------------------------
    print("--- Phase 2: Learning on Set 2 ---")

    spk_t_2, spk_i_2, F_set2, C_set2, mem_var_2, acc_his_2, final_states_2, W_2 = appNIST.test_train_continuous_correlated_proposed(
                        F_set1, C_set1, X_test, y_test, 
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=30,
                        eps=0.00005, 
                        la=0.2, Ucc_scale=2.0, 
                        init_states=final_states_1,
                        lr_readout=lr_readout, stim_duration=Duration)
    
    
    
    print("--- Phase 2: Learning on Set 2 ---")

   
    nspk_t_2, nspk_i_2, nF_set2, nC_set2, nmem_var_2, nacc_his_2, nfinal_states_2, nW_2 = appNIST.test_train_continuous_correlated(
                        nF_set1, nC_set1, X_test, y_test,
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=30,
                        epsr=0.00005, epsf=0.000005, 
                        la=0.2, Ucc_scale=2.0, 
                        init_states=nfinal_states_1,
                        lr_readout=lr_readout, stim_duration=Duration)
    
   

    
    print("Combining data for plots...")
    
    
    full_mem_var = mem_var_1 + mem_var_2

   
    t1 = np.array(spk_t_1)
    i1 = np.array(spk_i_1)
    t2 = np.array(spk_t_2)
    i2 = np.array(spk_i_2)

  
    time_offset = np.max(t1) if len(t1) > 0 else 0
    
   
    full_spk_t = np.concatenate([t1, t2 + time_offset])
    full_spk_i = np.concatenate([i1, i2])

    record_interval_steps = 100
    eval_interval = 10000

    
    plt.figure(figsize=(12, 6))
    
    plt.scatter(full_spk_t, full_spk_i, s=0.5, c='black', marker='.', alpha=0.6)

    plt.axvline(x=time_offset, color='red', linestyle='--', label='End of Set 1')

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title('Spike Raster Plot')
    plt.xlim(left=0, right=np.max(full_spk_t)) 
    plt.ylim(0, Nneuron)
    plt.legend(loc='upper right')

    raster_plot_path = current_save_dir / "Final_Raster.png"
    plt.savefig(raster_plot_path)
    plt.close()
    print(f"Raster plot saved to: {raster_plot_path}")

    
    plt.figure(figsize=(10, 6))
    
    
    time_axis_mem = np.arange(len(full_mem_var)) * dt

    
    mem_window_size = 500
    if len(full_mem_var) >= mem_window_size:
        b = np.ones(mem_window_size) / mem_window_size
        full_mem_smooth = np.convolve(full_mem_var, b, mode='valid')
        
        plt.plot(time_axis_mem, full_mem_var, label='Raw Variance', color='lightgreen', alpha=0.4)
        
        plt.plot(time_axis_mem[mem_window_size - 1:], full_mem_smooth, 
                 label=f'Moving Average (window={mem_window_size})', color='green', linewidth=2)
    else:
        plt.plot(time_axis_mem, full_mem_var, label='Voltage Variance', color='green')

    
    plt.axvline(x=len(X_train) * dt, color='red', linestyle='--', label='Covariate Shift Point')
    
    
    plt.xlabel('Time (s)', fontsize=22)  
    plt.ylabel('Voltage Variance', fontsize=22)
    
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True)
    plt.legend(loc='lower left', fontsize=18, framealpha=0.8)
    
    
    plt.yscale('log') 
    plt.ylim(0.0005, 0.8)

    mem_plot_path = current_save_dir / "Final_Voltage_Variance.png"
    plt.savefig(mem_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Voltage Variance plot saved to: {mem_plot_path}")

 
    print("Combining data for plots...")
    
    
    nfull_mem_var = nmem_var_1 + nmem_var_2

    nt1 = np.array(nspk_t_1)
    ni1 = np.array(nspk_i_1)
    nt2 = np.array(nspk_t_2)
    ni2 = np.array(nspk_i_2)

    ntime_offset = np.max(nt1) if len(nt1) > 0 else 0
    
    nfull_spk_t = np.concatenate([nt1, nt2 + ntime_offset])
    nfull_spk_i = np.concatenate([ni1, ni2])
    

    nrecord_interval_steps = 100
    neval_interval = 10000

    
    plt.figure(figsize=(12, 6))
    
    plt.scatter(nfull_spk_t, nfull_spk_i, s=0.5, c='black', marker='.', alpha=0.6)

    plt.axvline(x=ntime_offset, color='red', linestyle='--', label='End of Set 1')

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title('Spike Raster Plot')
    plt.xlim(left=0, right=np.max(nfull_spk_t)) 
    plt.ylim(0, Nneuron)
    plt.legend(loc='upper right')

    raster_plot_path = current_save_dir / "Final_Raster(non).png"
    plt.savefig(raster_plot_path)
    plt.close()
    print(f"Raster plot saved to: {raster_plot_path}")

    
    plt.figure(figsize=(10, 6))
    
   
    ntime_axis_mem = np.arange(len(nfull_mem_var)) * dt

    
    nmem_window_size = 500
    if len(nfull_mem_var) >= nmem_window_size:
        nb = np.ones(nmem_window_size) / nmem_window_size
        nfull_mem_smooth = np.convolve(nfull_mem_var, nb, mode='valid')
        
      
        plt.plot(ntime_axis_mem, nfull_mem_var, label='Raw Variance', color='lightgreen', alpha=0.4)
        
       
        plt.plot(ntime_axis_mem[nmem_window_size - 1:], nfull_mem_smooth, 
                 label=f'Moving Average (window={nmem_window_size})', color='green', linewidth=2)
    else:
        plt.plot(ntime_axis_mem, nfull_mem_var, label='Voltage Variance', color='green')

    
    plt.axvline(x=len(X_train) * dt, color='red', linestyle='--', label='Covariate Shift Point')
    
    
    plt.xlabel('Time (s)', fontsize=22) 
    plt.ylabel('Voltage Variance', fontsize=22)
    
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True)
    plt.legend(loc='lower left', fontsize=18, framealpha=0.8)
    

    plt.yscale('log') 
    plt.ylim(0.0005, 0.8)
    
    mem_plot_path = current_save_dir / "Final_Voltage_Variance(non).png"
    plt.savefig(mem_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Voltage Variance plot saved to: {mem_plot_path}")
    

    
    print("Plotting Accuracy History...")

    acc_window = 300  


    plt.figure(figsize=(10, 6))
    
    if len(acc_his_1) >= acc_window:
        window = np.ones(acc_window) / acc_window
        acc_smooth_1 = np.convolve(acc_his_1, window, mode='valid')
        
        x_axis_1 = np.arange(acc_window - 1, len(acc_his_1))
        plt.plot(x_axis_1, acc_smooth_1, label='identity', color='red', linewidth=1.5)
    else:
        plt.plot(acc_his_1, label='identity', color='red')

  
    if len(acc_his_2) >= acc_window:
        window = np.ones(acc_window) / acc_window
        acc_smooth_2 = np.convolve(acc_his_2, window, mode='valid')
       
        x_axis_2 = np.arange(acc_window - 1, len(acc_his_2)) + len(acc_his_1)
        plt.plot(x_axis_2, acc_smooth_2, label='stripe', color='red', linewidth=1.5)
    else:
        x_axis_raw_2 = np.arange(len(acc_his_2)) + len(acc_his_1)
        plt.plot(x_axis_raw_2, acc_his_2, label='stripe', color='red', alpha=0.5)

    plt.axvline(x=len(acc_his_1), color='red', linestyle='--', label='Covariate Shift Point')

    plt.xlabel('Input Samples (Images)', fontsize=22)
    plt.ylabel('Accuracy (Moving Avg)', fontsize=22)
    plt.legend(loc='lower right', fontsize=16, framealpha=0.8)
    plt.ylim(0.50, 1.05)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    acc_plot_path = current_save_dir / "Final_Accuracy.png"
    plt.savefig(acc_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Accuracy plot saved to: {acc_plot_path}")

    plt.figure(figsize=(10, 6))
    
    if len(nacc_his_1) >= acc_window:
        window = np.ones(acc_window) / acc_window
        nacc_smooth_1 = np.convolve(nacc_his_1, window, mode='valid')
        x_axis_n1 = np.arange(acc_window - 1, len(nacc_his_1))
        plt.plot(x_axis_n1, nacc_smooth_1, label='identity', color='red', linewidth=1.5)
    

    if len(nacc_his_2) >= acc_window:
        window = np.ones(acc_window) / acc_window
        nacc_smooth_2 = np.convolve(nacc_his_2, window, mode='valid')
        
        x_axis_n2 = np.arange(acc_window - 1, len(nacc_his_2)) + len(nacc_his_1)
        plt.plot(x_axis_n2, nacc_smooth_2, label='stripe', color='red', linewidth=1.5)

   
    plt.axvline(x=len(nacc_his_1), color='red', linestyle='--', label='Covariate Shift Point')

    plt.xlabel('Input Samples (Images)', fontsize=22)
    plt.ylabel('Accuracy (Moving Avg)', fontsize=22)
    plt.legend(loc='lower right', fontsize=16, framealpha=0.8)
    plt.ylim(0.50, 1.05)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    nacc_plot_path = current_save_dir / "Final_Accuracy(non).png"
    plt.savefig(nacc_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Accuracy plot (non) saved to: {nacc_plot_path}")

    print("Plotting Combined Accuracy History...")

    acc_window = 300  # 移動平均のウィンドウサイズ
    window = np.ones(acc_window) / acc_window

    plt.figure(figsize=(10, 6))

    # --- 変更点1: 背景色の設定 (Shift Point以前と以後) ---
    shift_point = len(acc_his_1)
    total_length = len(acc_his_1) + len(acc_his_2)
    
    # --------------------------------------------------

    all_plotted_data = []

    # --- 従来手法 (黒色) のプロット ---
    # 凡例を統一するため、最初のプロットのみlabelを付け、次はNoneにする
    label_conventional = 'Conventional Model'

    if len(nacc_his_1) >= acc_window:
        nacc_smooth_1 = np.convolve(nacc_his_1, window, mode='valid')
        x_axis_n1 = np.arange(acc_window - 1, len(nacc_his_1))
        # 1つ目の黒線: ラベルあり
        plt.plot(x_axis_n1, nacc_smooth_1, label=label_conventional, 
                 color='black', linewidth=1.0, alpha=0.6, linestyle='-')
        all_plotted_data.extend(nacc_smooth_1)
        label_conventional = None # 次回以降はラベルなし
    
    if len(nacc_his_2) >= acc_window:
        nacc_smooth_2 = np.convolve(nacc_his_2, window, mode='valid')
        x_axis_n2 = np.arange(acc_window - 1, len(nacc_his_2)) + len(nacc_his_1)
        # 2つ目の黒線: ラベルなし (None)
        plt.plot(x_axis_n2, nacc_smooth_2, label=label_conventional, 
                 color='black', linewidth=1.0, alpha=0.6, linestyle='-')
        all_plotted_data.extend(nacc_smooth_2)


    # --- 提案手法 (赤色) のプロット ---
    label_proposed = 'Proposed Model'

    if len(acc_his_1) >= acc_window:
        acc_smooth_1 = np.convolve(acc_his_1, window, mode='valid')
        x_axis_1 = np.arange(acc_window - 1, len(acc_his_1))
        # 1つ目の赤線: ラベルあり
        plt.plot(x_axis_1, acc_smooth_1, label=label_proposed, 
                 color='blue', linewidth=1.0)
        all_plotted_data.extend(acc_smooth_1)
        label_proposed = None # 次回以降はラベルなし
    else:
        plt.plot(acc_his_1, label=label_proposed, color='blue', alpha=0.3)
        all_plotted_data.extend(acc_his_1)
        label_proposed = None

    if len(acc_his_2) >= acc_window:
        acc_smooth_2 = np.convolve(acc_his_2, window, mode='valid')
        x_axis_2 = np.arange(acc_window - 1, len(acc_his_2)) + len(acc_his_1)
        # 2つ目の赤線: ラベルなし
        plt.plot(x_axis_2, acc_smooth_2, label=label_proposed, 
                 color='blue', linewidth=1.0)
        all_plotted_data.extend(acc_smooth_2)
    else:
        x_axis_raw_2 = np.arange(len(acc_his_2)) + len(acc_his_1)
        plt.plot(x_axis_raw_2, acc_his_2, label=label_proposed, color='blue', alpha=0.3)
        all_plotted_data.extend(acc_his_2)


    # 共変量シフト点の垂直線
    plt.axvline(x=len(acc_his_1), color='red', linestyle=':', linewidth=1.5, label='Covariate Shift Point')

    plt.xlabel('Input Samples (Images)', fontsize=22)
    plt.ylabel('Accuracy (Moving Avg)', fontsize=22)
    
    if all_plotted_data:
        data_min = np.min(all_plotted_data)
        data_max = np.max(all_plotted_data)
        y_bottom = min(0.50, data_min - 0.05)
        y_top = max(1.05, data_max + 0.02)
        plt.ylim(y_bottom, y_top)
    else:
        plt.ylim(0.50, 1.05)
    
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    
    # 凡例の表示
    plt.legend(loc='lower right', fontsize=16, framealpha=0.8)

    # 保存
    comb_acc_plot_path = current_save_dir / "Final_Accuracy_Combined.png"
    plt.savefig(comb_acc_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Combined accuracy plot saved to: {comb_acc_plot_path}")

    del X_test, y_test, spk_t_2, mem_var_2, nmem_var_2, spk_i_2, nspk_i_2, nspk_t_2, acc_his_2, nacc_his_2  # 巨大な変数を削除

    gc.collect()  # メモリ解放を強制
    
    # ---------------------------------------------------------                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    # データ準備 8週目
    # ---------------------------------------------------------
    print("Preparing Data...")

    X_test_org, y_test_org = appNIST.load_and_preprocess("fog_test_images.npy", "fog_test_labels.npy")

    X_test = np.repeat(X_test_org, Duration, axis=0)
    y_test = np.repeat(y_test_org, Duration, axis=0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S(fog)")
    current_save_dir = base_save_dir / timestamp
    current_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved in: {current_save_dir}")

    # ---------------------------------------------------------
    # Set 2: Learning
    # ---------------------------------------------------------
    print("--- Phase 2: Learning on Set 2 ---")

    spk_t_2, spk_i_2, F_set2, C_set2, mem_var_2, acc_his_2, final_states_2, W_2 = appNIST.test_train_continuous_correlated_proposed(
                        F_set1, C_set1, X_test, y_test, 
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=30,
                        eps=0.00005, 
                        la=0.2, Ucc_scale=2.0, 
                        init_states=final_states_1,
                        lr_readout=lr_readout, stim_duration=Duration)
    
    print("--- Phase 2: Learning on Set 2 ---")
   
    nspk_t_2, nspk_i_2, nF_set2, nC_set2, nmem_var_2, nacc_his_2, nfinal_states_2, nW_2 = appNIST.test_train_continuous_correlated(
                        nF_set1, nC_set1, X_test, y_test,
                        Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                        alpha, beta, mu, retrain=True, Gain=30,
                        epsr=0.00005, epsf=0.000005, 
                        la=0.2, Ucc_scale=2.0, 
                        init_states=nfinal_states_1,
                        lr_readout=lr_readout, stim_duration=Duration)
    
    print("Combining data for plots...")
    
    full_mem_var = mem_var_1 + mem_var_2

    t1 = np.array(spk_t_1)
    i1 = np.array(spk_i_1)
    t2 = np.array(spk_t_2)
    i2 = np.array(spk_i_2)

  
    time_offset = np.max(t1) if len(t1) > 0 else 0
    
   
    full_spk_t = np.concatenate([t1, t2 + time_offset])
    full_spk_i = np.concatenate([i1, i2])

    record_interval_steps = 100
    eval_interval = 10000

    
    plt.figure(figsize=(12, 6))
    
    plt.scatter(full_spk_t, full_spk_i, s=0.5, c='black', marker='.', alpha=0.6)

    plt.axvline(x=time_offset, color='red', linestyle='--', label='End of Set 1')

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title('Spike Raster Plot')
    plt.xlim(left=0, right=np.max(full_spk_t)) 
    plt.ylim(0, Nneuron)
    plt.legend(loc='upper right')

    raster_plot_path = current_save_dir / "Final_Raster.png"
    plt.savefig(raster_plot_path)
    plt.close()
    print(f"Raster plot saved to: {raster_plot_path}")

    
    plt.figure(figsize=(10, 6))
    
    
    time_axis_mem = np.arange(len(full_mem_var)) * dt

    
    mem_window_size = 500
    if len(full_mem_var) >= mem_window_size:
        b = np.ones(mem_window_size) / mem_window_size
        full_mem_smooth = np.convolve(full_mem_var, b, mode='valid')
        
        plt.plot(time_axis_mem, full_mem_var, label='Raw Variance', color='lightgreen', alpha=0.4)
        
        plt.plot(time_axis_mem[mem_window_size - 1:], full_mem_smooth, 
                 label=f'Moving Average (window={mem_window_size})', color='green', linewidth=2)
    else:
        plt.plot(time_axis_mem, full_mem_var, label='Voltage Variance', color='green')

    
    plt.axvline(x=len(X_train) * dt, color='red', linestyle='--', label='Covariate Shift Point')
    
    
    plt.xlabel('Time (s)', fontsize=22)  
    plt.ylabel('Voltage Variance', fontsize=22)
    
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True)
    plt.legend(loc='lower left', fontsize=18, framealpha=0.8)
    
    
    plt.yscale('log') 
    plt.ylim(0.0005, 0.8)

    mem_plot_path = current_save_dir / "Final_Voltage_Variance.png"
    plt.savefig(mem_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Voltage Variance plot saved to: {mem_plot_path}")

 
    print("Combining data for plots...")
    
    
    nfull_mem_var = nmem_var_1 + nmem_var_2

    nt1 = np.array(nspk_t_1)
    ni1 = np.array(nspk_i_1)
    nt2 = np.array(nspk_t_2)
    ni2 = np.array(nspk_i_2)

    ntime_offset = np.max(nt1) if len(nt1) > 0 else 0
    
    nfull_spk_t = np.concatenate([nt1, nt2 + ntime_offset])
    nfull_spk_i = np.concatenate([ni1, ni2])
    

    nrecord_interval_steps = 100
    neval_interval = 10000

    
    plt.figure(figsize=(12, 6))
    
    plt.scatter(nfull_spk_t, nfull_spk_i, s=0.5, c='black', marker='.', alpha=0.6)

    plt.axvline(x=ntime_offset, color='red', linestyle='--', label='End of Set 1')

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title('Spike Raster Plot')
    plt.xlim(left=0, right=np.max(nfull_spk_t)) 
    plt.ylim(0, Nneuron)
    plt.legend(loc='upper right')

    raster_plot_path = current_save_dir / "Final_Raster(non).png"
    plt.savefig(raster_plot_path)
    plt.close()
    print(f"Raster plot saved to: {raster_plot_path}")

    
    plt.figure(figsize=(10, 6))
    
   
    ntime_axis_mem = np.arange(len(nfull_mem_var)) * dt

    
    nmem_window_size = 500
    if len(nfull_mem_var) >= nmem_window_size:
        nb = np.ones(nmem_window_size) / nmem_window_size
        nfull_mem_smooth = np.convolve(nfull_mem_var, nb, mode='valid')
        
      
        plt.plot(ntime_axis_mem, nfull_mem_var, label='Raw Variance', color='lightgreen', alpha=0.4)
        
       
        plt.plot(ntime_axis_mem[nmem_window_size - 1:], nfull_mem_smooth, 
                 label=f'Moving Average (window={nmem_window_size})', color='green', linewidth=2)
    else:
        plt.plot(ntime_axis_mem, nfull_mem_var, label='Voltage Variance', color='green')

    
    plt.axvline(x=len(X_train) * dt, color='red', linestyle='--', label='Covariate Shift Point')
    
    
    plt.xlabel('Time (s)', fontsize=22) 
    plt.ylabel('Voltage Variance', fontsize=22)
    
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True)
    plt.legend(loc='lower left', fontsize=18, framealpha=0.8)
    

    plt.yscale('log') 
    plt.ylim(0.0005, 0.8)
    
    mem_plot_path = current_save_dir / "Final_Voltage_Variance(non).png"
    plt.savefig(mem_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Voltage Variance plot saved to: {mem_plot_path}")
    

    
    print("Plotting Accuracy History...")

    acc_window = 300  


    plt.figure(figsize=(10, 6))
    
    if len(acc_his_1) >= acc_window:
        window = np.ones(acc_window) / acc_window
        acc_smooth_1 = np.convolve(acc_his_1, window, mode='valid')
        
        x_axis_1 = np.arange(acc_window - 1, len(acc_his_1))
        plt.plot(x_axis_1, acc_smooth_1, label='identity', color='red', linewidth=1.5)
    else:
        plt.plot(acc_his_1, label='identity', color='red')

  
    if len(acc_his_2) >= acc_window:
        window = np.ones(acc_window) / acc_window
        acc_smooth_2 = np.convolve(acc_his_2, window, mode='valid')
       
        x_axis_2 = np.arange(acc_window - 1, len(acc_his_2)) + len(acc_his_1)
        plt.plot(x_axis_2, acc_smooth_2, label='fog', color='red', linewidth=1.5)
    else:
        x_axis_raw_2 = np.arange(len(acc_his_2)) + len(acc_his_1)
        plt.plot(x_axis_raw_2, acc_his_2, label='fog', color='red', alpha=0.5)

   
    plt.axvline(x=len(acc_his_1), color='red', linestyle='--', label='Covariate Shift Point')

    plt.xlabel('Input Samples (Images)', fontsize=22)
    plt.ylabel('Accuracy (Moving Avg)', fontsize=22)
    plt.legend(loc='lower right', fontsize=16, framealpha=0.8)
    plt.ylim(0.50, 1.05)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    acc_plot_path = current_save_dir / "Final_Accuracy.png"
    plt.savefig(acc_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Accuracy plot saved to: {acc_plot_path}")

    plt.figure(figsize=(10, 6))
    
    if len(nacc_his_1) >= acc_window:
        window = np.ones(acc_window) / acc_window
        nacc_smooth_1 = np.convolve(nacc_his_1, window, mode='valid')
        x_axis_n1 = np.arange(acc_window - 1, len(nacc_his_1))
        plt.plot(x_axis_n1, nacc_smooth_1, label='identity', color='red', linewidth=1.5)
    

    if len(nacc_his_2) >= acc_window:
        window = np.ones(acc_window) / acc_window
        nacc_smooth_2 = np.convolve(nacc_his_2, window, mode='valid')
        
        x_axis_n2 = np.arange(acc_window - 1, len(nacc_his_2)) + len(nacc_his_1)
        plt.plot(x_axis_n2, nacc_smooth_2, label='fog', color='red', linewidth=1.5)

   
    plt.axvline(x=len(nacc_his_1), color='red', linestyle='--', label='Covariate Shift Point')

    plt.xlabel('Input Samples (Images)', fontsize=22)
    plt.ylabel('Accuracy (Moving Avg)', fontsize=22)
    plt.legend(loc='lower right', fontsize=16, framealpha=0.8)
    plt.ylim(0.50, 1.05)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    nacc_plot_path = current_save_dir / "Final_Accuracy(non).png"
    plt.savefig(nacc_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Accuracy plot (non) saved to: {nacc_plot_path}")

    print("Plotting Combined Accuracy History...")

    acc_window = 300  # 移動平均のウィンドウサイズ
    window = np.ones(acc_window) / acc_window

    plt.figure(figsize=(10, 6))

    # --- 変更点1: 背景色の設定 (Shift Point以前と以後) ---
    shift_point = len(acc_his_1)
    total_length = len(acc_his_1) + len(acc_his_2)
    
    
    # --------------------------------------------------

    all_plotted_data = []

    # --- 従来手法 (黒色) のプロット ---
    # 凡例を統一するため、最初のプロットのみlabelを付け、次はNoneにする
    label_conventional = 'Conventional Model'

    if len(nacc_his_1) >= acc_window:
        nacc_smooth_1 = np.convolve(nacc_his_1, window, mode='valid')
        x_axis_n1 = np.arange(acc_window - 1, len(nacc_his_1))
        # 1つ目の黒線: ラベルあり
        plt.plot(x_axis_n1, nacc_smooth_1, label=label_conventional, 
                 color='black', linewidth=1.0, alpha=0.6, linestyle='-')
        all_plotted_data.extend(nacc_smooth_1)
        label_conventional = None # 次回以降はラベルなし
    
    if len(nacc_his_2) >= acc_window:
        nacc_smooth_2 = np.convolve(nacc_his_2, window, mode='valid')
        x_axis_n2 = np.arange(acc_window - 1, len(nacc_his_2)) + len(nacc_his_1)
        # 2つ目の黒線: ラベルなし (None)
        plt.plot(x_axis_n2, nacc_smooth_2, label=label_conventional, 
                 color='black', linewidth=1.0, alpha=0.6, linestyle='-')
        all_plotted_data.extend(nacc_smooth_2)


    # --- 提案手法 (赤色) のプロット ---
    label_proposed = 'Proposed Model'

    if len(acc_his_1) >= acc_window:
        acc_smooth_1 = np.convolve(acc_his_1, window, mode='valid')
        x_axis_1 = np.arange(acc_window - 1, len(acc_his_1))
        # 1つ目の赤線: ラベルあり
        plt.plot(x_axis_1, acc_smooth_1, label=label_proposed, 
                 color='blue', linewidth=1.0)
        all_plotted_data.extend(acc_smooth_1)
        label_proposed = None # 次回以降はラベルなし
    else:
        plt.plot(acc_his_1, label=label_proposed, color='blue', alpha=0.3)
        all_plotted_data.extend(acc_his_1)
        label_proposed = None

    if len(acc_his_2) >= acc_window:
        acc_smooth_2 = np.convolve(acc_his_2, window, mode='valid')
        x_axis_2 = np.arange(acc_window - 1, len(acc_his_2)) + len(acc_his_1)
        # 2つ目の赤線: ラベルなし
        plt.plot(x_axis_2, acc_smooth_2, label=label_proposed, 
                 color='blue', linewidth=1.0)
        all_plotted_data.extend(acc_smooth_2)
    else:
        x_axis_raw_2 = np.arange(len(acc_his_2)) + len(acc_his_1)
        plt.plot(x_axis_raw_2, acc_his_2, label=label_proposed, color='blue', alpha=0.3)
        all_plotted_data.extend(acc_his_2)


    # 共変量シフト点の垂直線
    plt.axvline(x=len(acc_his_1), color='red', linestyle=':', linewidth=1.5, label='Covariate Shift Point')

    plt.xlabel('Input Samples (Images)', fontsize=22)
    plt.ylabel('Accuracy (Moving Avg)', fontsize=22)
    
    if all_plotted_data:
        data_min = np.min(all_plotted_data)
        data_max = np.max(all_plotted_data)
        y_bottom = min(0.50, data_min - 0.05)
        y_top = max(1.05, data_max + 0.02)
        plt.ylim(y_bottom, y_top)
    else:
        plt.ylim(0.50, 1.05)
    
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    
    # 凡例の表示
    plt.legend(loc='lower right', fontsize=16, framealpha=0.8)

    # 保存
    comb_acc_plot_path = current_save_dir / "Final_Accuracy_Combined.png"
    plt.savefig(comb_acc_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Combined accuracy plot saved to: {comb_acc_plot_path}")

    
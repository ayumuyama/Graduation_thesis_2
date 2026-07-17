import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit
import gc

# 以前あった pickle のインポートはキャッシュ機能削除に伴い不要になりました

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import appGaussian as appssian
from app import appMNIST as appNIST

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 基本設定
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
    lr_readout = 0.0005
    
    # =========================================================
    # 変更点1: 試したい eps の値をリストで定義
    # =========================================================
    eps_list = [0.00004]  #ここ大事
    
    # 変更点2: 評価するデータセットをリスト化してループ処理できるように整理
    eval_datasets = [
        {"name": "stripe", "img_file": "stripe_test_images.npy", "lbl_file": "stripe_test_labels.npy"},
        {"name": "fog", "img_file": "fog_test_images.npy", "lbl_file": "fog_test_labels.npy"}
    ]

    # Identityデータは全試行で共通なので最初にロード
    print("Preparing Identity Data...")
    X_train_org, y_train_org = appNIST.load_and_preprocess("identity_test_images.npy", "identity_test_labels.npy")
    X_train = np.repeat(X_train_org, Duration, axis=0)
    y_train = np.repeat(y_train_org, Duration, axis=0)

    # ---------------------------------------------------------
    # メインループ (eps と データセット の組み合わせで実行)
    # ---------------------------------------------------------
    for current_eps in eps_list:
        for dataset in eval_datasets:
            ds_name = dataset["name"]
            
            # 保存先設定 (epsとデータセット名をディレクトリ名に含める)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_save_dir = Path("outputs")
            current_save_dir = base_save_dir / f"{timestamp}_{ds_name}_eps_{current_eps}"
            current_save_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n=========================================================")
            print(f"Starting run: Dataset = {ds_name}, eps = {current_eps}")
            print(f"Results will be saved in: {current_save_dir}")
            print(f"=========================================================\n")
            
            # テストデータの準備
            X_test_org, y_test_org = appNIST.load_and_preprocess(dataset["img_file"], dataset["lbl_file"])
            X_test = np.repeat(X_test_org, Duration, axis=0)
            y_test = np.repeat(y_test_org, Duration, axis=0)

            # =========================================================
            # Phase 1: Learning on Set 1 (Identity)
            # キャッシュ(pickle)を廃止し、毎回必ず初期状態から実行
            # =========================================================
            print("--- Phase 1: Learning on Set 1 (Identity) ---")
            F_initial, C_initial, *_ = appssian.init_weights(Nx, Nneuron, Nclasses)

            spk_t_1, spk_i_1, F_set1, C_set1, mem_var_1, acc_his_1, final_states_1, W_1 = appNIST.test_train_continuous_correlated_proposed(
                            F_initial, C_initial, X_train, y_train,
                            Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                            alpha, beta, mu, retrain=True, Gain=30,
                            eps=current_eps, la=0.2, Ucc_scale=2.0, # ここに current_eps を適用
                            init_states=None, lr_readout=lr_readout, stim_duration=Duration)

            nspk_t_1, nspk_i_1, nF_set1, nC_set1, nmem_var_1, nacc_his_1, nfinal_states_1, nW_1 = appNIST.test_train_continuous_correlated(
                            F_initial, C_initial, X_train, y_train,
                            Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                            alpha, beta, mu, retrain=True, Gain=30,
                            epsr=current_eps, epsf=current_eps/10, la=0.2, Ucc_scale=2.0, # 従来手法側も合わせる場合は修正
                            init_states=None, lr_readout=lr_readout, stim_duration=Duration)

            # =========================================================
            # Phase 2: Learning on Set 2 (Stripe or Fog)
            # =========================================================
            print(f"--- Phase 2: Learning on Set 2 ({ds_name}) ---")

            spk_t_2, spk_i_2, F_set2, C_set2, mem_var_2, acc_his_2, final_states_2, W_2 = appNIST.test_train_continuous_correlated_proposed(
                            F_set1, C_set1, X_test, y_test, 
                            Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                            alpha, beta, mu, retrain=True, Gain=30,
                            eps=current_eps, # ここにも current_eps を適用
                            la=0.2, Ucc_scale=2.0, 
                            init_states=final_states_1,
                            lr_readout=lr_readout, stim_duration=Duration)
            
            nspk_t_2, nspk_i_2, nF_set2, nC_set2, nmem_var_2, nacc_his_2, nfinal_states_2, nW_2 = appNIST.test_train_continuous_correlated(
                            nF_set1, nC_set1, X_test, y_test,
                            Nneuron, Nx, Nclasses, dt, leak, Thresh, 
                            alpha, beta, mu, retrain=True, Gain=30,
                            epsr=current_eps, epsf=current_eps/10, 
                            la=0.2, Ucc_scale=2.0, 
                            init_states=nfinal_states_1,
                            lr_readout=lr_readout, stim_duration=Duration)
            
            # === 区間平均精度の計算と出力 ===
            # 各種ノイズなどの条件ごとにこの処理が呼ばれる箇所に配置します

            # 評価区間のインデックス設定 (Pythonのインデックスに合わせて設定)
            phase1_start = 10000
            phase1_end = 12500
            phase2_start = 12500
            phase2_end = 20000

            # 提案手法の区間平均
            prop_phase1_mean = np.mean(acc_his_2[phase1_start:phase1_end])
            prop_phase2_mean = np.mean(acc_his_2[phase2_start:phase2_end])

            # 従来手法の区間平均
            conv_phase1_mean = np.mean(nacc_his_2[phase1_start:phase1_end])
            conv_phase2_mean = np.mean(nacc_his_2[phase2_start:phase2_end])

            # 結果のコンソール出力
            print("\n=== Covariate Shift 後の区間平均精度 ===")
            print(f"初期適応フェーズ (10000 - 12500) | Proposed: {prop_phase1_mean:.3f} | Conventional: {conv_phase1_mean:.3f}")
            print(f"定常・回復フェーズ (12500 - 20000) | Proposed: {prop_phase2_mean:.3f} | Conventional: {conv_phase2_mean:.3f}")
            print("========================================\n")

            # =========================================================
            # プロット処理 (以前と同じロジックを統合)
            # =========================================================
            print("Combining data for plots...")
            
            full_mem_var = mem_var_1 + mem_var_2
            t1 = np.array(spk_t_1)
            i1 = np.array(spk_i_1)
            t2 = np.array(spk_t_2)
            i2 = np.array(spk_i_2)

            time_offset = np.max(t1) if len(t1) > 0 else 0
            full_spk_t = np.concatenate([t1, t2 + time_offset])
            full_spk_i = np.concatenate([i1, i2])

            # 1. Raster Plot
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

            # 2. Voltage Variance Plot
            plt.figure(figsize=(10, 6))
            time_axis_mem = np.arange(len(full_mem_var)) * dt
            mem_window_size = 500
            if len(full_mem_var) >= mem_window_size:
                b = np.ones(mem_window_size) / mem_window_size
                full_mem_smooth = np.convolve(full_mem_var, b, mode='valid')
                plt.plot(time_axis_mem, full_mem_var, label='Raw Variance', color='lightgreen', alpha=0.4)
                plt.plot(time_axis_mem[mem_window_size - 1:], full_mem_smooth, label=f'Moving Average (window={mem_window_size})', color='green', linewidth=2)
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

            # 3. Combined Accuracy Plot
            print("Plotting Combined Accuracy History...")
            acc_window = 300 
            window = np.ones(acc_window) / acc_window
            plt.figure(figsize=(10, 6))
            all_plotted_data = []

            # 従来手法 (黒色) のプロット
            label_conventional = 'Conventional Model'
            if len(nacc_his_1) >= acc_window:
                nacc_smooth_1 = np.convolve(nacc_his_1, window, mode='valid')
                x_axis_n1 = np.arange(acc_window - 1, len(nacc_his_1))
                plt.plot(x_axis_n1, nacc_smooth_1, label=label_conventional, color='gray', linewidth=0.8, alpha=0.7, linestyle='--')
                all_plotted_data.extend(nacc_smooth_1)
                label_conventional = None 
            
            if len(nacc_his_2) >= acc_window:
                nacc_smooth_2 = np.convolve(nacc_his_2, window, mode='valid')
                x_axis_n2 = np.arange(acc_window - 1, len(nacc_his_2)) + len(nacc_his_1)
                plt.plot(x_axis_n2, nacc_smooth_2, label=label_conventional, color='gray', linewidth=0.8, alpha=0.7, linestyle='--')
                all_plotted_data.extend(nacc_smooth_2)

            # 提案手法 (青色) のプロット ※元のコードが赤線指定でしたがCombinedで青を使っていたので青に統一しています
            label_proposed = 'Proposed Model'
            if len(acc_his_1) >= acc_window:
                acc_smooth_1 = np.convolve(acc_his_1, window, mode='valid')
                x_axis_1 = np.arange(acc_window - 1, len(acc_his_1))
                plt.plot(x_axis_1, acc_smooth_1, label=label_proposed, color='black', linewidth=1.0, alpha=1.0)
                all_plotted_data.extend(acc_smooth_1)
                label_proposed = None 
            else:
                plt.plot(acc_his_1, label=label_proposed, color='black', alpha=0.3)
                all_plotted_data.extend(acc_his_1)
                label_proposed = None

            if len(acc_his_2) >= acc_window:
                acc_smooth_2 = np.convolve(acc_his_2, window, mode='valid')
                x_axis_2 = np.arange(acc_window - 1, len(acc_his_2)) + len(acc_his_1)
                plt.plot(x_axis_2, acc_smooth_2, label=label_proposed, color='black', linewidth=1.0, alpha=1.0)
                all_plotted_data.extend(acc_smooth_2)
            else:
                x_axis_raw_2 = np.arange(len(acc_his_2)) + len(acc_his_1)
                plt.plot(x_axis_raw_2, acc_his_2, label=label_proposed, color='black', alpha=0.3)
                all_plotted_data.extend(acc_his_2)

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
            plt.legend(loc='lower right', fontsize=16, framealpha=0.8)
            comb_acc_plot_path = current_save_dir / f"Final_Accuracy_Combined_{ds_name}.png"
            plt.savefig(comb_acc_plot_path, bbox_inches='tight')
            plt.close()

            # メモリ解放
            del X_test, y_test, spk_t_2, mem_var_2, nmem_var_2, spk_i_2, nspk_i_2, nspk_t_2, acc_his_2, nacc_his_2 
            gc.collect() 

    print("All iterations completed.")
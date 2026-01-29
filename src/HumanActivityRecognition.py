from app import appHAR as apar

DATA_DIR = './data' 

target_subject = 1 #subjectを指定
X_sub, y_sub = apar.load_data_for_subject(target_subject, DATA_DIR)

if X_sub is not None:
    print(f"Subject {target_subject} のデータセット作成完了")
    print(f"入力データ X の形状: {X_sub.shape}") # (347, 6, 128) 
    print(f"ラベルデータ y の形状: {y_sub.shape}") # (347, 1)

Nx = 6
Nneuron = 60
Nclasses = 6
Nit = 2

leak = 50       
dt = 0.001      
    
alpha = 0.18    
beta = 1 / 0.9  
mu = 0.02 / 0.9
    
Thresh = 0.5

F_initial, C_initial, *_ = apar.init_weights(Nx, Nneuron, Nclasses)

spike_times, spike_neurons, F, C, W_out, accuracy_history, final_states_1 = apar.online_learning_classifier(
                            F_initial, C_initial, X_sub, y_sub,
                            Nneuron, Nx, Nclasses, Nit, dt, leak, Thresh,
                            alpha, beta, mu, retrain=True, Gain=300,
                            eps=0.005, init_states=None)

apar.plot_learning_curve(accuracy_history, window_size=20)
apar.plot_raster(spike_times, spike_neurons, title="Spike Raster Plot", 
                figsize=(12, 6), marker_size=2, color='black')
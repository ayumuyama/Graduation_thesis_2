import numpy as np
import scipy.io
import scipy.interpolate
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Helper Functions (netrun, netadapt, netruncl)
# ==========================================

def netrun(I, Wref, Gamma, Nneurontot, Ninput, thres):
    """
    Simulate the network without plasticity.
    Equivalent to netrun.m
    """
    Ntime = I.shape[1]
    dt = 0.00005
    lam = 8  # lambda

    x = np.zeros((Ninput, Ntime))
    V = np.zeros((Nneurontot, Ntime))
    O = np.zeros((Nneurontot, Ntime))
    rO = np.zeros((Nneurontot, Ntime))
    
    # Initial conditions
    V[:, 0] = np.random.randn(Nneurontot) * 0.01
    
    # BN noise
    BN = np.random.randn(Nneurontot, Ntime) * 0.005
    
    # Pre-calculate matrix multiplication for speed if Wref is constant
    # But here we follow the loop structure closely
    
    for t in range(Ntime - 1):
        W = -Wref
        
        # Input dynamics
        x[:, t+1] = (1 - lam * dt) * x[:, t] + I[:, t+1]
        
        Input_current = Gamma @ I[:, t+1]
        
        # Membrane potential dynamics
        V[:, t+1] = (1 - lam * dt) * V[:, t] + Input_current + W @ O[:, t]
        
        crit = (V[:, t+1] + BN[:, t+1]) - thres
        
        # Spiking logic
        spikes = (crit > 0).astype(float)
        O[:, t+1] = spikes
        
        # Soft-WTA: Only one spike allowed per step (or simple handling of multiple)
        if np.sum(O[:, t+1]) > 1:
            v_idx = np.argmax(crit)
            O[:, t+1] = 0
            O[v_idx, t+1] = 1
            
        rO[:, t+1] = (1 - lam * dt) * rO[:, t] + O[:, t+1]
        
        # xest is not returned in the main script's netrun call usually, 
        # but calculated inside. We return x (input filtered) and others.
        
    return O, rO, V, x

def netadapt(Il, Wref, Gamma, Nneurontot, Ninput, thres, Ucc, mI, nspeed, nlat):
    """
    Network with plasticity (learning).
    Equivalent to netadapt.m
    """
    Ntime = 49901 # Fixed in original code
    dt = 0.00005
    lam = 8
    la = 0.2
    eps = 0.01
    mu = 0.1
    alpha = 1
    ep = 2000
    
    I = Il
    x = np.zeros((Ninput, Ntime))
    V = np.zeros((Nneurontot, Ntime))
    O = np.zeros((Nneurontot, Ntime))
    rO = np.zeros((Nneurontot, Ntime))
    
    V[:, 0] = np.random.randn(Nneurontot) * 0.01
    In = np.zeros((Ninput, Ntime))
    BN = np.random.randn(Nneurontot, Ntime) * 0.005
    
    # Working copies of weights
    Wref_curr = Wref.copy()
    Gamma_curr = Gamma.copy()
    thres_curr = thres.copy()
    mI_curr = mI.copy()
    Ucc_curr = Ucc.copy()

    for t in range(Ntime - 1):
        W = -Wref_curr
        
        x[:, t+1] = (1 - lam * dt) * x[:, t] + I[:, t+1]
        In[:, t+1] = (1 - ep * dt) * In[:, t] + ep * dt * I[:, t+1]
        
        Input_current = Gamma_curr @ I[:, t+1]
        V[:, t+1] = (1 - lam * dt) * V[:, t] + Input_current + W @ O[:, t]
        
        crit = (V[:, t+1] + BN[:, t+1]) - thres_curr
        O[:, t+1] = (crit > 0).astype(float)
        
        mI_curr = (1 - la * dt) * mI_curr + la * dt * In[:, t] # Note: original uses t
        
        diff = (In[:, t+1] - mI_curr).reshape(-1, 1) # column vector
        term = (Gamma_curr @ diff) @ diff.T
        Ucc_curr = (1 - la * dt) * Ucc_curr + la * dt * term
        
        if np.sum(O[:, t+1]) > 1:
            v_idx = np.argmax(crit)
            O[:, t+1] = 0
            O[v_idx, t+1] = 1
            
        if np.sum(O[:, t+1]) > 0:
            # Plasticity updates
            O_vec = O[:, t+1].reshape(-1, 1)
            
            if t > 0: # t indices are 0-based here, so t=1 is t>0 check roughly
                # Lateral weights update
                term1 = O_vec @ (alpha * (V[:, t+1] + mu * rO[:, t])).reshape(1, -1)
                term2 = (np.ones((Nneurontot, 1)) @ O_vec.T) * (-Wref_curr + mu * np.eye(Nneurontot))
                Wref_curr = Wref_curr + nlat * eps * (term1.T + term2) # Note: Transpose logic carefully checked against MATLAB
                
            if t > 1000:
                # Feedforward weights update
                diff_in = In[:, t+1] - 100 * (Ucc_curr.T @ O_vec).flatten()
                Gamma_curr = Gamma_curr + nspeed * eps * (O_vec @ diff_in.reshape(1, -1))
        
        rO[:, t+1] = (1 - lam * dt) * rO[:, t] + O[:, t+1]

    # Homeostatic regulation
    rate = np.sum(O, axis=1)
    thres_curr = thres_curr + nspeed * eps * (rate > 20).astype(float)
    thres_curr = thres_curr - nspeed * eps * (rate < 1).astype(float)
    
    return O, rO, V, x, None, I, Wref_curr, Gamma_curr, thres_curr, Ucc_curr, mI_curr

def netruncl(I, Wref, Gamma, Nneurontot, Ninput, thres, irec):
    """
    Clamped network run.
    Equivalent to netruncl.m
    """
    Ntime = I.shape[1]
    dt = 0.00005
    lam = 8
    
    x = np.zeros((Ninput, Ntime))
    V = np.zeros((Nneurontot, Ntime))
    O = np.zeros((Nneurontot, Ntime))
    rO = np.zeros((Nneurontot, Ntime))
    
    V[:, 0] = np.random.randn(Nneurontot) * 0.01
    BN = np.random.randn(Nneurontot, Ntime) * 0.005
    
    # Clamp the specific neuron by raising threshold very high
    thres_cl = thres.copy()
    thres_cl[irec] = 1000 
    
    for t in range(Ntime - 1):
        W = -Wref
        x[:, t+1] = (1 - lam * dt) * x[:, t] + I[:, t+1]
        
        Input_current = Gamma @ I[:, t+1]
        V[:, t+1] = (1 - lam * dt) * V[:, t] + Input_current + W @ O[:, t]
        
        crit = (V[:, t+1] + BN[:, t+1]) - thres_cl
        O[:, t+1] = (crit > 0).astype(float)
        
        if np.sum(O[:, t+1]) > 1:
            v_idx = np.argmax(crit)
            O[:, t+1] = 0
            O[v_idx, t+1] = 1
            
        rO[:, t+1] = (1 - lam * dt) * rO[:, t] + O[:, t+1]
        
    return O, rO, V, x


# ==========================================
# 2. Main Script (Equivalent to fig5Anew.m)
# ==========================================

if __name__ == "__main__":
    print("Starting simulation (Python port of fig5Anew.m)...")

    # Parameters
    Nneurontot = 100
    Ninput = 25
    lam = 8
    dt = 0.00005
    Nshow = 16000
    pr = 0.1 
    gain = 0.03
    
    # ---------------------------------------------------------
    # パスの修正: スクリプトと同じフォルダのファイルを探す設定
    # ---------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_icurrent = os.path.join(script_dir, 'Icurrent.mat')
    path_speech = os.path.join(script_dir, 'speech.mat')

    print(f"Looking for files in: {script_dir}")

    # Load Icurrent.mat
    try:
        if not os.path.exists(path_icurrent):
            raise FileNotFoundError(f"File not found: {path_icurrent}")

        icurrent_data = scipy.io.loadmat(path_icurrent)
        Gamma = icurrent_data['Gamma']
        Wref = icurrent_data['Wref']
        
        # MATLABのスカラーや配列の扱いの違いを吸収
        if 'thres' in icurrent_data:
            thres = icurrent_data['thres'].flatten()
        else:
            thres = np.ones(Nneurontot)*0.5
            
        # mI, Ucc の読み込み
        mI = icurrent_data['mI'] if 'mI' in icurrent_data else np.zeros((Ninput, 1))
        Ucc = icurrent_data['Ucc'] if 'Ucc' in icurrent_data else np.zeros((Nneurontot, Nneurontot))
        
        print("Successfully loaded Icurrent.mat")
        
    except Exception as e:
        print(f"Error loading Icurrent.mat: {e}")
        # ファイルがない場合は停止せず、テスト用にランダム初期化（動作確認用）
        print("-> Using random initialization for testing.")
        Gamma = np.random.randn(Nneurontot, Ninput) * 0.025
        Wref = np.random.randn(Nneurontot, Nneurontot) * 0.1 + np.eye(Nneurontot) * 0.8
        thres = np.ones(Nneurontot) * 0.5
        mI = np.zeros(Ninput)
        Ucc = np.zeros((Nneurontot, Nneurontot))

    # Compute Decoder
    Dsp = np.linalg.pinv(Gamma, rcond=pr) @ Wref
    
    irec = 36 
    
    # Load speech stimulus
    try:
        if not os.path.exists(path_speech):
            print(f"Warning: speech.mat not found at {path_speech}")
            raise FileNotFoundError

        speech_data = scipy.io.loadmat(path_speech)
        
        # データ構造の確認と取得
        if 'speech' in speech_data:
            SPu = speech_data['speech']['data'][0][0]
        elif 'data' in speech_data:
            SPu = speech_data['data']
        elif 'SPu' in speech_data:
            SPu = speech_data['SPu']
        else:
            # 構造が不明な場合、キー一覧を表示して確認を促す
            print(f"Keys in speech.mat: {speech_data.keys()}")
            raise ValueError("Could not find data variable in speech.mat")
            
        print("Successfully loaded speech.mat")

    except Exception as e:
        print("-> Generating random noise for testing because speech.mat was not found/readable.")
        SPu = np.random.randn(25, 230500)

    # ----------------------------------------------
    # Create Iex (Speech example)
    # ----------------------------------------------
    # MATLAB: Un=randperm(230500); 
    # Python: Un usually needs to ensure valid range for indexing +500
    Un_start = np.random.randint(0, SPu.shape[1] - 600)
    
    # Logic to recreate the complex interpolation/gradient lines:
    # cI(i,:)=(SPu(i,Un(1)+1)*100+(cumsum(interp1([1:500],gradient(SPu(i,Un(1)+1:Un(1)+500)),[1:0.01:500]))))*0.1;
    
    # x coordinates for interpolation
    x_old = np.arange(1, 501) # 1 to 500
    x_new = np.arange(1, 500.01, 0.01) # 1 to 500 step 0.01 (approx 49901 points)
    
    Iex = np.zeros((Ninput, len(x_new)))
    
    for i in range(Ninput):
        # Extract 500 points
        sp_slice = SPu[i, Un_start : Un_start + 500]
        
        # Gradient
        grad_sp = np.gradient(sp_slice)
        
        # Interpolate gradient
        f_interp = scipy.interpolate.interp1d(x_old, grad_sp, kind='linear', fill_value="extrapolate")
        grad_interp = f_interp(x_new)
        
        # Cumsum
        cum_grad = np.cumsum(grad_interp)
        
        # Combine
        # Note: In MATLAB gradient spacing defaults to 1 unless specified.
        # cI calculation
        val_start = SPu[i, Un_start]
        cI_row = (val_start * 100 + cum_grad) * 0.1
        
        # Final Iex calculation: gradient(cI) + lambda*dt*cI
        Iex[i, :] = np.gradient(cI_row) + lam * dt * cI_row
        
    Iex_short = Iex[:, :20000] # Use first 20000 points

    # ----------------------------------------------
    # Run Network (Optimal weights)
    # ----------------------------------------------
    print("Running network with optimal weights...")
    Osp, rOsp, Vsp, xsp = netrun(Iex_short * gain, Wref, Gamma, Nneurontot, Ninput, thres)

    # Plotting results (Example: Reconstruction)
    plt.figure(figsize=(10, 8))
    
    # Reconstructed input vs Original
    # Equivalent to subplot(7,3,13:14)
    plt.subplot(3, 1, 1)
    reconstruction = Dsp @ rOsp[:, :Nshow]
    plt.imshow(reconstruction, aspect='auto', origin='lower', cmap='viridis')
    plt.title('Reconstruction (Optimal Weights)')
    plt.colorbar()

    # Spikes
    # Equivalent to subplot(7,3,16:17)
    plt.subplot(3, 1, 2)
    rows, cols = np.where(Osp[:, :Nshow])
    plt.scatter(cols, rows, s=1, c='k')
    plt.title('Spikes')
    plt.xlim(0, Nshow)
    plt.ylim(0, Nneurontot)

    # E/I Balance for one neuron
    # Equivalent to subplot(7,3,19:20)
    plt.subplot(3, 1, 3)
    E = Gamma @ xsp
    I_inh = -Wref @ rOsp
    
    # Plot for irec
    plt.plot(E[irec, :Nshow] + I_inh[irec, :Nshow], 'k', label='Total')
    plt.plot(np.maximum(E[irec, :Nshow], 0) + np.maximum(I_inh[irec, :Nshow], 0), 'r', label='Exc') # Simple visual approx
    plt.plot(np.minimum(I_inh[irec, :Nshow], 0) + np.minimum(E[irec, :Nshow], 0), 'b', label='Inh')
    plt.title(f'E/I Balance (Neuron {irec})')
    plt.legend()

    plt.tight_layout()
    plt.savefig("Final_Speech_Results.png")
    plt.close()
    print("Done.")
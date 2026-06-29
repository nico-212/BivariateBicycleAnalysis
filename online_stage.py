import os
import sys
import numpy as np
import scipy.sparse as sp
import multiprocessing as mp
from tqdm import tqdm
from offline_stage import build_stim_circuit 

# kill multithreading conflicts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# settings
error_rate = 0.0001
num_cycles = 12
num_trials = 10000
num_workers = mp.cpu_count() - 1 

n, k, d = 144, 12, 12

# load offline matrices
input_dir = f"{n}_{k}_{d}_cycle_{num_cycles}/p_{error_rate}"

if not os.path.exists(input_dir):
    print(f"error: folder {input_dir} not found. run offline_decoder first.")
    sys.exit(1)

print(f"loading offline matrices from {input_dir}...")
DX = sp.load_npz(f"{input_dir}/DX.npz")
DZ = sp.load_npz(f"{input_dir}/DZ.npz")
DL_X = sp.load_npz(f"{input_dir}/DL_X.npz")
DL_Z = sp.load_npz(f"{input_dir}/DL_Z.npz")
probX = np.load(f"{input_dir}/probX.npy")
probZ = np.load(f"{input_dir}/probZ.npy")

# generate noisy samples using stim
print(f"sampling {num_trials} trials from stim...")
dx_circuit = build_stim_circuit(error_rate,num_cycles,is_dz=False)
dz_circuit = build_stim_circuit(error_rate,num_cycles,is_dz=True)

sampler_X = dx_circuit.compile_detector_sampler()
sampler_Z = dz_circuit.compile_detector_sampler()

det_batch_X, obs_batch_X = sampler_X.sample(shots=num_trials,separate_observables=True)
det_batch_Z, obs_batch_Z = sampler_Z.sample(shots=num_trials, separate_observables=True)

# worker globals
_bpdX, _bpdZ, _DL_X, _DL_Z = None, None, None, None

def init_worker(DX_in, DZ_in, DL_X_in, DL_Z_in, probX_in, probZ_in):
    global _bpdX, _bpdZ, _DL_X, _DL_Z
    from ldpc import BpOsdDecoder  
    
    # original decoder settings
    decoder_args = dict(
        max_iter=1000,             
        bp_method="ms",
        ms_scaling_factor=0,     
        osd_method="osd_cs",
        osd_order=7              
    )
    
    _bpdX = BpOsdDecoder(DX_in, channel_probs=probX_in, **decoder_args)
    _bpdZ = BpOsdDecoder(DZ_in, channel_probs=probZ_in, **decoder_args)
    _DL_X = DL_X_in
    _DL_Z = DL_Z_in

def decode_one_shot(i):
    det_x, obs_x = det_batch_X[i], obs_batch_X[i]
    det_z, obs_z = det_batch_Z[i], obs_batch_Z[i]

    correction_x = _bpdX.decode(det_x)       
    guessed_obs_X = (_DL_X @ correction_x) % 2
    success_X = np.array_equal(guessed_obs_X, obs_x)

    correction_z = _bpdZ.decode(det_z)       
    guessed_obs_Z = (_DL_Z @ correction_z) % 2
    success_Z = np.array_equal(guessed_obs_Z, obs_z)

    return 0 if (success_X and success_Z) else 1

if __name__ == "__main__":
    print(f"decoding with {num_workers} worker processes...")

    # run multiprocessing loop
    with mp.Pool(num_workers, initializer=init_worker, initargs=(DX, DZ, DL_X, DL_Z, probX, probZ)) as pool:
        results = list(tqdm(
            pool.imap_unordered(decode_one_shot, range(num_trials), chunksize=50),
            total=num_trials,
            desc="decoding"
        ))

    logical_errors = sum(results)
    
    # calculate stats
    P_L = logical_errors / num_trials
    pL = 1 - (1 - P_L) ** (1 / num_cycles)

    print("\n=== final results ===")
    print(f"physical error rate (p):         {error_rate}")
    print(f"total trials:                    {num_trials}")
    print(f"logical errors:                  {logical_errors}")
    print(f"P_L (per-experiment):            {P_L:.6f}")
    print(f"pL (per-cycle, paper formula):   {pL:.6f}")

    # save results to csv
    csv_file = "decoder_results.csv"
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, "a") as f:
        if not file_exists:
            f.write("n,k,d,p,num_cycles,trials,logical_errors,P_L,pL_per_cycle\n")
        
        f.write(f"{n},{k},{d},{error_rate},{num_cycles},{num_trials},{logical_errors},{P_L:.6f},{pL:.6f}\n")
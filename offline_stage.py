import os
import numpy as np
import scipy.sparse as sp
import stim
from bposd.css import css_code
from numpy.linalg import matrix_power as pwr


# setup for the [144, 12, 12] bivariate bicycle code
error_rate = 0.0001
num_cycles = 12

l,m = 12,6
n2 = l * m
n = 2 * n2  
k = 12
d = 12

# shift and identity matrices
S_l = np.roll(np.eye(l, dtype=int), 1, axis=1)
S_m = np.roll(np.eye(m, dtype=int), 1, axis=1)
I_l = np.eye(l, dtype=int)
I_m = np.eye(m, dtype=int)

# kronecker product for x and y
x = np.kron(S_l, I_m)
y = np.kron(I_l, S_m)

A1 = pwr(x,3)
A2 = y
A3 = pwr(y,2)

B1 = pwr(y,3)
B2 = x
B3 = pwr(x,2)
"""
[90,8,10] code
A1 = pwr(x,9)
A2 = y
A3 = pwr(y,2)

B1 = pwr(y,0)
B2 = pw(x,2)
B3 = pwr(x,7)
"""
A = (A1 + A2 + A3) % 2
B = (B1 + B2 + B3) % 2

# x and z stabilizers
H_X = np.hstack((A, B)).astype(int)
H_Z = np.hstack((B.T, A.T)).astype(int)

# build logicals using bposd
qcode = css_code(H_X, H_Z)
lx = np.asarray(qcode.lx.todense()).astype(int)
lz = np.asarray(qcode.lz.todense()).astype(int)

# map qubits to 1D integer lists for stim
q_L = list(range(0, n2))
q_R = list(range(n2, 2 * n2))
q_X = list(range(2 * n2, 3 * n2))
q_Z = list(range(3 * n2, 4 * n2))
data_qubits = q_L + q_R

def build_stim_circuit(p_error, cycles, is_dz=False):
    """
    builds the memory experiment. 
    if is_dz is True, we initialize in |0> and measure Z-logicals (tracking X-errors).
    if is_dz is False, we initialize in |+> and measure X-logicals (tracking Z-errors).
    """
    circuit = stim.Circuit()
    p = error_rate

    # initialize data and ancillas
    circuit.append("R" if is_dz else "RX", data_qubits)
    circuit.append("R", q_Z)
    circuit.append("RX", q_X)

    def add_cycle(is_noiseless, cycle_num):
        c = stim.Circuit()

        # round 1: prep x-checks. L is idle.
        c.append("RX", q_X)
        if not is_noiseless and not is_dz: c.append("Z_ERROR", q_X, p)

        cx_r1 = []
        for i in range(n2): cx_r1.extend([q_R[np.where(A1.T[i] == 1)[0][0]], q_Z[i]])
        c.append("CX", cx_r1)
        if not is_noiseless: 
            c.append("DEPOLARIZE2", cx_r1, p)
            c.append("DEPOLARIZE1", q_L, p)

        # round 2
        cx_r2 = []
        for i in range(n2):
            cx_r2.extend([q_X[i], q_L[np.where(A2[i] == 1)[0][0]]])
            cx_r2.extend([q_R[np.where(A3.T[i] == 1)[0][0]], q_Z[i]])
        c.append("CX", cx_r2)
        if not is_noiseless: c.append("DEPOLARIZE2", cx_r2, p)

        # round 3
        cx_r3 = []
        for i in range(n2):
            cx_r3.extend([q_X[i], q_R[np.where(B2[i] == 1)[0][0]]])
            cx_r3.extend([q_L[np.where(B1.T[i] == 1)[0][0]], q_Z[i]])
        c.append("CX", cx_r3)
        if not is_noiseless: c.append("DEPOLARIZE2", cx_r3, p)

        # round 4
        cx_r4 = []
        for i in range(n2):
            cx_r4.extend([q_X[i], q_R[np.where(B1[i] == 1)[0][0]]])
            cx_r4.extend([q_L[np.where(B2.T[i] == 1)[0][0]], q_Z[i]])
        c.append("CX", cx_r4)
        if not is_noiseless: c.append("DEPOLARIZE2", cx_r4, p)

        # round 5
        cx_r5 = []
        for i in range(n2):
            cx_r5.extend([q_X[i], q_R[np.where(B3[i] == 1)[0][0]]])
            cx_r5.extend([q_L[np.where(B3.T[i] == 1)[0][0]], q_Z[i]])
        c.append("CX", cx_r5)
        if not is_noiseless: c.append("DEPOLARIZE2", cx_r5, p)

        # round 6
        cx_r6 = []
        for i in range(n2):
            cx_r6.extend([q_X[i], q_L[np.where(A1[i] == 1)[0][0]]])
            cx_r6.extend([q_R[np.where(A2.T[i] == 1)[0][0]], q_Z[i]])
        c.append("CX", cx_r6)
        if not is_noiseless: c.append("DEPOLARIZE2", cx_r6, p)

        # round 7: R is idle. measure Z.
        cx_r7 = []
        for i in range(n2): cx_r7.extend([q_X[i], q_L[np.where(A3[i] == 1)[0][0]]])
        c.append("CX", cx_r7)
        
        if not is_noiseless: 
            c.append("DEPOLARIZE2", cx_r7, p)
            c.append("DEPOLARIZE1", q_R, p)
            c.append("X_ERROR", q_Z, p)
        
        c.append("M", q_Z)

        # map Z detectors
        if is_dz:
            for i in range(n2):
                if cycle_num == 0: c.append("DETECTOR", [stim.target_rec(-n2 + i)])
                else: c.append("DETECTOR", [stim.target_rec(-n2 + i), stim.target_rec(-3*n2 + i)])

        # round 8: measure X and reset Z
        if not is_noiseless: 
            c.append("DEPOLARIZE1", data_qubits, p)
            c.append("Z_ERROR", q_X, p)
            
        c.append("MX", q_X)

        # map X detectors
        if not is_dz:
            for i in range(n2):
                if cycle_num == 0: c.append("DETECTOR", [stim.target_rec(-n2 + i)])
                else: c.append("DETECTOR", [stim.target_rec(-n2 + i), stim.target_rec(-3*n2 + i)])

        c.append("R", q_Z)
        if not is_noiseless and is_dz: c.append("X_ERROR", q_Z, p)

        return c

    # 1. noisy memory cycles
    for cycle_num in range(num_cycles):
        circuit += add_cycle(is_noiseless=False, cycle_num=cycle_num)

    # 2. noiseless cap to close detectors
    for cycle_num in range(num_cycles, num_cycles + 2):
        circuit += add_cycle(is_noiseless=True, cycle_num=cycle_num)

    # 3. measure logical observables
    circuit.append("M" if is_dz else "MX", data_qubits)
    logical_op = lz if is_dz else lx

    for obs_idx, row in enumerate(logical_op):
        support = np.nonzero(row)[0]
        targets = [stim.target_rec(-len(data_qubits) + i) for i in support]
        if targets:
            circuit.append("OBSERVABLE_INCLUDE", targets, obs_idx)

    return circuit

def extract_matrices(dem: stim.DetectorErrorModel):
    """converts stim's error model into sparse matrices for bp-osd."""
    r_D, c_D, r_L, c_L, probs = [], [], [], [], []
    col_idx = 0

    for instruction in dem:
        if instruction.type == "error":
            probs.append(instruction.args_copy()[0])
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    r_D.append(target.val)
                    c_D.append(col_idx)
                elif target.is_logical_observable_id():
                    r_L.append(target.val)
                    c_L.append(col_idx)
            col_idx += 1

    D = sp.csc_matrix((np.ones(len(r_D), dtype=int), (r_D, c_D)), shape=(dem.num_detectors, col_idx))
    D_L = sp.csc_matrix((np.ones(len(r_L), dtype=int), (r_L, c_L)), shape=(dem.num_observables, col_idx))
    
    return D, D_L, np.array(probs)

if __name__ == "__main__":
    
    # setup simple output folder
    output_dir = f"{n}_{k}_{d}_cycle_{num_cycles}/p_{error_rate}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"building matrices in: {output_dir}/")

    # compile circuits and extract matrices
    dx_circuit = build_stim_circuit(error_rate,num_cycles,is_dz=False)
    dem_X = dx_circuit.detector_error_model(approximate_disjoint_errors=True, ignore_decomposition_failures=True)
    DX, DL_X, probX = extract_matrices(dem_X)

    dz_circuit = build_stim_circuit(error_rate,num_cycles,is_dz=True)
    dem_Z = dz_circuit.detector_error_model(approximate_disjoint_errors=True, ignore_decomposition_failures=True)
    DZ, DL_Z, probZ = extract_matrices(dem_Z)

    # sanity check expected array shapes
    expected_detectors = n2 * (num_cycles + 2)
    assert DX.shape[0] == expected_detectors, "dx detector count mismatch"
    assert DZ.shape[0] == expected_detectors, "dz detector count mismatch"

    # save everything for the online decoder
    sp.save_npz(os.path.join(output_dir, "DX.npz"), DX)
    sp.save_npz(os.path.join(output_dir, "DZ.npz"), DZ)
    sp.save_npz(os.path.join(output_dir, "DL_X.npz"), DL_X)
    sp.save_npz(os.path.join(output_dir, "DL_Z.npz"), DL_Z)

    np.save(os.path.join(output_dir, "probX.npy"), probX)
    np.save(os.path.join(output_dir, "probZ.npy"), probZ)
    np.save(os.path.join(output_dir, "lx.npy"), lx)
    np.save(os.path.join(output_dir, "lz.npy"), lz)

    with open(os.path.join(output_dir, "meta.txt"), "w") as f:
        f.write(f"error_rate={error_rate}\nnum_cycles={num_cycles}\nl={l}\nm={m}\nn={n}\nk={k}\nd={d}\n")

    print("offline build complete.")
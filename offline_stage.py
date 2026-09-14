#!/usr/bin/env python3
"""Generate the offline QEC artifacts for the Bivariate Bicycle codes.

This script builds the circuits, error models (DEMs), parity-check matrices (D/DL), and fault probabilities required for both split and joint decoding.
It also constructs the routing matrices (R_Z.npz, R_X.npz) that safely map joint faults to their split equivalents.
Routing consistency is verified automatically during generation (disable with --no-verify).
"""

import argparse
import os

import numpy as np
import scipy.sparse as sparse
import stim
from bposd.css import css_code
from numpy.linalg import matrix_power

CODE_PARAMS = {
    72: (6, 6, 12, 6),
    144: (12, 6, 12, 12)
}

def artifact_dir(code_id, p, cycles):
    l, m, k, d = CODE_PARAMS[code_id]
    n = 2 * l * m
    return f"{n}_{k}_{d}_cycle_{cycles}/p_{p}"


class BBCode:
    """Bivariate Bicycle (BB) code geometry and parity-check matrices.

    Computes the base check polynomials (A, B), the parity-check matrices
    (H_X, H_Z), and the logical operators (lx, lz). 
    """

    def __init__(self, code_id):
        self.code_id = code_id
        self.l, self.m, self.k, self.d = CODE_PARAMS[code_id]
        
        self.n2 = self.l * self.m
        self.n = 2 * self.n2

        # 1. Base cyclic shift matrices
        S_l = np.roll(np.eye(self.l, dtype=int), 1, axis=1)
        S_m = np.roll(np.eye(self.m, dtype=int), 1, axis=1)
        x = np.kron(S_l, np.eye(self.m, dtype=int))
        y = np.kron(np.eye(self.l, dtype=int), S_m)

        # 2. A/B check polynomials: A = x^3 + y + y^2, B = y^3 + x + x^2.
        self.A1, self.A2, self.A3 = matrix_power(x, 3), y, matrix_power(y, 2)
        self.B1, self.B2, self.B3 = matrix_power(y, 3), x, matrix_power(x, 2)
        
        A = (self.A1 + self.A2 + self.A3) % 2
        B = (self.B1 + self.B2 + self.B3) % 2

        # 3. CSS code matrices
        self.H_X = np.hstack((A, B)).astype(int)
        self.H_Z = np.hstack((B.T, A.T)).astype(int)

        # 4. Logical operators (computed by bposd)
        qcode = css_code(self.H_X, self.H_Z)
        self.lx = qcode.lx.toarray().astype(int)
        self.lz = qcode.lz.toarray().astype(int)

        # 5. Qubit index map for Stim
        self.q_L = list(range(0, self.n2))
        self.q_R = list(range(self.n2, self.n))
        self.q_X = list(range(self.n, self.n + self.n2))
        self.q_Z = list(range(self.n + self.n2, self.n + 2 * self.n2))
        self.data_qubits = self.q_L + self.q_R

    def detector_slices(self, cycles):
        """Calculates row indices for the Z and X halves of the joint DEM.
        
        The joint DEM interleaves detectors cycle by cycle:
        Cycle 0: [ Z_block ] [ X_block ]
        Cycle 1: [ Z_block ] [ X_block ]
        """
        c_total = cycles + 2
        detectors_per_cycle = 2 * self.n2
        
        # Get the absolute starting index for each cycle (shape: c_total x 1)
        cycle_starts = (np.arange(c_total) * detectors_per_cycle)[:, None]
        
        # Z-half: offsets 0 to n2-1
        z_offsets = np.arange(self.n2)
        z_indices_2d = cycle_starts + z_offsets
        
        # X-half: offsets n2 to 2*n2-1
        x_offsets = np.arange(self.n2) + self.n2
        x_indices_2d = cycle_starts + x_offsets
        
        # Flatten the 2D grids back into 1D arrays
        return z_indices_2d.ravel(), x_indices_2d.ravel()

def build_stim_circuit(code, p, cycles, is_z_memory, both_halves=False):
    """Generates a Stim circuit for a given memory basis (Z or X).

    Supports building per-basis (graded half only) or joint (both halves) circuits.
    The schedule applies a 7-round CX pattern with depolarizing and measurement errors.
    In joint mode, a noiseless baseline round is prepended to align the mirror half's 
    deterministic detectors.
    """
    n2, data_qubits = code.n2, code.data_qubits
    q_L, q_R, q_X, q_Z = code.q_L, code.q_R, code.q_X, code.q_Z
    A1, A2, A3 = code.A1, code.A2, code.A3
    B1, B2, B3 = code.B1, code.B2, code.B3
    lx, lz = code.lx, code.lz

    declare_z = declare_x = both_halves
    if not both_halves:
        declare_z, declare_x = is_z_memory, not is_z_memory

    circuit = stim.Circuit()
    circuit.append("R" if is_z_memory else "RX", data_qubits)
    circuit.append("R", q_Z)
    circuit.append("RX", q_X)

    def add_cycle(is_noiseless, cycle_num, declare):
        cycle = stim.Circuit()

        cycle.append("RX", q_X)
        if not is_noiseless and declare_x:
            # X-ancilla reset error.
            cycle.append("Z_ERROR", q_X, p)

        # Round 1: X checks prepared; the L block idles.
        cx_r1 = []
        for i in range(n2):
            cx_r1.extend([q_R[np.where(A1.T[i] == 1)[0][0]], q_Z[i]])
        cycle.append("CX", cx_r1)
        if not is_noiseless:
            cycle.append("DEPOLARIZE2", cx_r1, p)
            cycle.append("DEPOLARIZE1", q_L, p)

        # Round 2.
        cx_r2 = []
        for i in range(n2):
            cx_r2.extend([q_X[i], q_L[np.where(A2[i] == 1)[0][0]]])
            cx_r2.extend([q_R[np.where(A3.T[i] == 1)[0][0]], q_Z[i]])
        cycle.append("CX", cx_r2)
        if not is_noiseless:
            cycle.append("DEPOLARIZE2", cx_r2, p)

        # Round 3.
        cx_r3 = []
        for i in range(n2):
            cx_r3.extend([q_X[i], q_R[np.where(B2[i] == 1)[0][0]]])
            cx_r3.extend([q_L[np.where(B1.T[i] == 1)[0][0]], q_Z[i]])
        cycle.append("CX", cx_r3)
        if not is_noiseless:
            cycle.append("DEPOLARIZE2", cx_r3, p)

        # Round 4.
        cx_r4 = []
        for i in range(n2):
            cx_r4.extend([q_X[i], q_R[np.where(B1[i] == 1)[0][0]]])
            cx_r4.extend([q_L[np.where(B2.T[i] == 1)[0][0]], q_Z[i]])
        cycle.append("CX", cx_r4)
        if not is_noiseless:
            cycle.append("DEPOLARIZE2", cx_r4, p)

        # Round 5.
        cx_r5 = []
        for i in range(n2):
            cx_r5.extend([q_X[i], q_R[np.where(B3[i] == 1)[0][0]]])
            cx_r5.extend([q_L[np.where(B3.T[i] == 1)[0][0]], q_Z[i]])
        cycle.append("CX", cx_r5)
        if not is_noiseless:
            cycle.append("DEPOLARIZE2", cx_r5, p)

        # Round 6.
        cx_r6 = []
        for i in range(n2):
            cx_r6.extend([q_X[i], q_L[np.where(A1[i] == 1)[0][0]]])
            cx_r6.extend([q_R[np.where(A2.T[i] == 1)[0][0]], q_Z[i]])
        cycle.append("CX", cx_r6)
        if not is_noiseless:
            cycle.append("DEPOLARIZE2", cx_r6, p)

        # Round 7: R block idles; the Z ancillas are measured next.
        cx_r7 = []
        for i in range(n2):
            cx_r7.extend([q_X[i], q_L[np.where(A3[i] == 1)[0][0]]])
        cycle.append("CX", cx_r7)
        if not is_noiseless:
            cycle.append("DEPOLARIZE2", cx_r7, p)
            cycle.append("DEPOLARIZE1", q_R, p)
            # Native M-flip channel (pre-M).
            cycle.append("X_ERROR", q_Z, p)

        cycle.append("M", q_Z)

        if declare and declare_z:
            for i in range(n2):
                if is_z_memory and cycle_num == 0:
                    # Z-memory round 0: |0> prep makes first M(q_Z)
                    # deterministic -> single-shot detector.
                    cycle.append("DETECTOR", [stim.target_rec(-n2 + i)])
                else:
                    cycle.append("DETECTOR", [stim.target_rec(-n2 + i),
                                          stim.target_rec(-3 * n2 + i)])

        if not is_noiseless:
            cycle.append("DEPOLARIZE1", data_qubits, p)
            # X-ancilla pre-MX error.
            cycle.append("Z_ERROR", q_X, p)

        cycle.append("MX", q_X)

        if declare and declare_x:
            for i in range(n2):
                if (not is_z_memory) and cycle_num == 0:
                    # X-memory round 0: |+> prep makes first MX(q_X)
                    # deterministic -> single-shot detector.
                    cycle.append("DETECTOR", [stim.target_rec(-n2 + i)])
                else:
                    cycle.append("DETECTOR", [stim.target_rec(-n2 + i),
                                          stim.target_rec(-3 * n2 + i)])

        cycle.append("R", q_Z)
        if not is_noiseless and declare_z:
            # Z-ancilla reset error.
            cycle.append("X_ERROR", q_Z, p)

        return cycle

    if both_halves:
        # Noiseless undeclared round: mirror-half baseline for round-0 pairs.
        circuit += add_cycle(is_noiseless=True, cycle_num=-1, declare=False)
    for cycle_num in range(cycles + 2):
        circuit += add_cycle(is_noiseless=cycle_num >= cycles,
                             cycle_num=cycle_num, declare=True)

    circuit.append("M" if is_z_memory else "MX", data_qubits)
    logical_op = lz if is_z_memory else lx
    for obs_idx, row in enumerate(logical_op):
        support = np.nonzero(row)[0]
        targets = [stim.target_rec(-len(data_qubits) + i) for i in support]
        if targets:
            circuit.append("OBSERVABLE_INCLUDE", targets, obs_idx)

    return circuit


def extract_matrices(dem: stim.DetectorErrorModel):
    """Extracts parity (D), logical (DL) matrices, and priors from a Stim DEM.
    
    Faults with identical detector and observable footprints are merged into a 
    single column using XOR probability to ensure unambiguous routing.
    """
    d_rows, d_cols = [], []
    l_rows, l_cols = [], []
    effect_to_col = {}
    probs = []

    for inst in dem:
        if inst.type != "error":
            continue

        detectors, observables = [], []
        for target in inst.targets_copy():
            if target.is_relative_detector_id():
                detectors.append(target.val)
            elif target.is_logical_observable_id():
                observables.append(target.val)

        effect = (tuple(sorted(detectors)), tuple(sorted(observables)))
        prob = inst.args_copy()[0]

        if effect in effect_to_col:
            col = effect_to_col[effect]
            existing = probs[col]
            probs[col] = existing * (1 - prob) + prob * (1 - existing)
            continue

        col = len(probs)
        effect_to_col[effect] = col
        probs.append(prob)

        for row in effect[0]:
            d_rows.append(row)
            d_cols.append(col)
        for row in effect[1]:
            l_rows.append(row)
            l_cols.append(col)

    n_faults = len(probs)

    D = sparse.csc_matrix((np.ones(len(d_rows), dtype=np.uint8), (d_rows, d_cols)), 
                          shape=(dem.num_detectors, n_faults))
    D.data %= 2
    D.eliminate_zeros()

    DL = sparse.csc_matrix((np.ones(len(l_rows), dtype=np.uint8), (l_rows, l_cols)), 
                           shape=(dem.num_observables, n_faults))
    DL.data %= 2
    DL.eliminate_zeros()

    return D, DL, np.array(probs)


def build_route_projection(d_joint, d_split, split_rows):
    """Builds a sparse projection matrix mapping joint faults to their split equivalents.
    
    Matches faults based on their exact detector footprint. Joint faults with no 
    detectors on the split half are ignored.
    """
    d_split = d_split.tocsc()
    sliced_joint = d_joint[split_rows, :].tocsc()
    
    n_joint_faults = sliced_joint.shape[1]
    n_split_faults = d_split.shape[1]
    
    # Step 1: Catalog the exact detector footprint of every split fault
    split_footprints = {}
    for split_col in range(n_split_faults):
        
        footprint = tuple(d_split[:, split_col].indices)

        assert footprint not in split_footprints, f"Duplicate footprint at col {split_col}"
        split_footprints[footprint] = split_col

    # Step 2: Match every joint fault to its corresponding split fault
    joint_indices = []
    split_indices = []
    
    for joint_col in range(n_joint_faults):
        footprint = tuple(sliced_joint[:, joint_col].indices)
        
        if not footprint:
            continue
            
        if footprint in split_footprints:
            joint_indices.append(joint_col)
            split_indices.append(split_footprints[footprint])

    # Step 3: Construct the mapping matrix
    ones = np.ones(len(joint_indices), dtype=np.uint8)
    return sparse.csc_matrix(
        (ones, (joint_indices, split_indices)),
        shape=(n_joint_faults, n_split_faults)
    )

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=int, choices=sorted(CODE_PARAMS), default=72)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--verify", dest="verify", action="store_true", default=True,
                        help="run the route-consistency checks on built artifacts (default)")
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    args = parser.parse_args(argv)

    code = BBCode(args.code)
    out_dir = artifact_dir(args.code, args.p, args.cycles)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Build | {out_dir} | [[{code.n},{code.k},{code.d}]] | p={args.p} | cyc={args.cycles}")

    # Per-basis artifacts (graded half only)
    dx_circuit = build_stim_circuit(code, args.p, args.cycles, is_z_memory=False)
    dem_X = dx_circuit.detector_error_model(approximate_disjoint_errors=True, ignore_decomposition_failures=True)
    DX, DL_X, probX = extract_matrices(dem_X)

    dz_circuit = build_stim_circuit(code, args.p, args.cycles, is_z_memory=True)
    dem_Z = dz_circuit.detector_error_model(approximate_disjoint_errors=True, ignore_decomposition_failures=True)
    DZ, DL_Z, probZ = extract_matrices(dem_Z)

    expected_detectors = code.n2 * (args.cycles + 2)
    assert DX.shape[0] == expected_detectors, "dx detector count mismatch"
    assert DZ.shape[0] == expected_detectors, "dz detector count mismatch"

    sparse.save_npz(os.path.join(out_dir, "DX.npz"), DX)
    sparse.save_npz(os.path.join(out_dir, "DZ.npz"), DZ)
    sparse.save_npz(os.path.join(out_dir, "DL_X.npz"), DL_X)
    sparse.save_npz(os.path.join(out_dir, "DL_Z.npz"), DL_Z)

    dx_circuit.to_file(os.path.join(out_dir, "circuit_X.stim"))
    dz_circuit.to_file(os.path.join(out_dir, "circuit_Z.stim"))
    dem_X.to_file(os.path.join(out_dir, "dem_X.dem"))
    dem_Z.to_file(os.path.join(out_dir, "dem_Z.dem"))

    np.save(os.path.join(out_dir, "probX.npy"), probX)
    np.save(os.path.join(out_dir, "probZ.npy"), probZ)
    
    print(f"Per-basis: X-mem {DX.shape} | Z-mem {DZ.shape} | circuits/dems/probs saved")

    # Joint artifacts (both halves)
    c_total = args.cycles + 2
    print(f"Joint: halves X {DX.shape} | Z {DZ.shape} | rows/half {code.n2 * c_total}")

    z_half_rows, x_half_rows = code.detector_slices(args.cycles)

    for basis in ("Z", "X"):
        is_z = basis == "Z"
        print(f"Build Joint {basis} | {basis}-memory")
        
        circuit = build_stim_circuit(code, args.p, args.cycles, is_z_memory=is_z, both_halves=True)
        dem = circuit.detector_error_model(approximate_disjoint_errors=True, ignore_decomposition_failures=True)
        
        expected_joint_detectors = 2 * code.n2 * c_total
        assert dem.num_detectors == expected_joint_detectors, (dem.num_detectors, expected_joint_detectors)
        
        circuit.to_file(os.path.join(out_dir, f"circuit_joint_{basis}.stim"))
        dem.to_file(os.path.join(out_dir, f"dem_joint_{basis}.dem"))

        d_joint, _, prob_joint = extract_matrices(dem)
        print(f"Joint {basis} DEM: det {dem.num_detectors} | faults {d_joint.shape[1]} | obs {dem.num_observables}")

        if is_z:
            route_proj = build_route_projection(d_joint, DZ, z_half_rows)
        else:
            route_proj = build_route_projection(d_joint, DX, x_half_rows)

        sparse.save_npz(os.path.join(out_dir, f"D_joint_{basis}.npz"), d_joint)
        np.save(os.path.join(out_dir, f"prob_joint_{basis}.npy"), prob_joint)
        sparse.save_npz(os.path.join(out_dir, f"R_{basis}.npz"), route_proj)

        print(f"Saved joint_{basis}: dem/D/prob/R/circuit")

    if args.verify:
        try:
            from tests.test_offline_stage import check_routes
        except ImportError as exc:
            print(f"WARN: verification skipped (import tests.test_offline_stage failed: {exc})")
        else:
            print("Verify: route checks ...")
            if not check_routes(out_dir, args.code, args.cycles):
                raise SystemExit("artifact verification FAILED -- fix before training")
            print("Verify: OK")
            
    print("Done: per-basis + joint artifacts (Z/X mems)")

if __name__ == "__main__":
    main()

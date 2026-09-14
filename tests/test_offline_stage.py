#!/usr/bin/env python3
"""Integration regression test for the offline artifact pipeline.

Builds the per-basis + joint artifacts for a small case ([[72,12,6]], 2
cycles) into a temporary directory via ``offline_stage.main``, then runs
``check_routes`` on the SAVED files.

Run with:  python tests/test_offline_stage.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import scipy.sparse as sp
import stim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import offline_stage as od  # noqa: E402

CODE, P, CYCLES = 72, 0.003, 2


def _route_identity(joint_side, d_split, route_proj):
    """True iff ``d_split @ route_proj.T == joint_side`` (mod 2)."""
    a = d_split @ route_proj.T
    a.data %= 2
    a.eliminate_zeros()
    return (a != joint_side).nnz == 0


def check_routes(art_dir, code_id, cycles):
    """Exact route consistency over the saved artifacts (moved out of
    offline_stage.py): detector identities for both halves, graded readout
    identities, and per-basis column coverage."""
    path_of = lambda name: os.path.join(art_dir, name)  # noqa: E731
    code = od.BBCode(code_id)
    z_half_rows, x_half_rows = code.detector_slices(cycles)
    d_joint_z = sp.load_npz(path_of("D_joint_Z.npz"))
    d_joint_x = sp.load_npz(path_of("D_joint_X.npz"))
    dz = sp.load_npz(path_of("DZ.npz"))
    dx = sp.load_npz(path_of("DX.npz"))
    dl_z = sp.load_npz(path_of("DL_Z.npz"))
    dl_x = sp.load_npz(path_of("DL_X.npz"))
    rz = sp.load_npz(path_of("R_Z.npz"))
    rx = sp.load_npz(path_of("R_X.npz"))
    # joint observable matrices are not saved; re-extract from the DEMs
    _, dl_joint_z, _ = od.extract_matrices(
        stim.DetectorErrorModel.from_file(path_of("dem_joint_Z.dem")))
    _, dl_joint_x, _ = od.extract_matrices(
        stim.DetectorErrorModel.from_file(path_of("dem_joint_X.dem")))

    checks = [
        ("Z half-syndrome identity",
         _route_identity(d_joint_z[z_half_rows, :], dz, rz)),
        ("X half-syndrome identity",
         _route_identity(d_joint_x[x_half_rows, :], dx, rx)),
        ("Z readout identity",
         _route_identity(dl_joint_z, dl_z, rz)),
        ("X readout identity",
         _route_identity(dl_joint_x, dl_x, rx)),
    ]
    for basis, route_proj, split_matrix in (("Z", rz, dz), ("X", rx, dx)):
        split_faults_reached = np.unique(route_proj.nonzero()[1])
        covered = np.array_equal(split_faults_reached,
                                 np.arange(split_matrix.shape[1]))
        checks.append((f"{basis}: all {split_matrix.shape[1]} "
                       f"per-basis columns covered", covered))

    ok = True
    for name, passed in checks:
        ok &= bool(passed)
        print(f"  [{'OK ' if passed else 'FAIL'}] {name}")
    return ok


def main():
    tmp = tempfile.mkdtemp(prefix="offline_stage_test_")
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp)
        print(f"building artifacts for [[72,12,6]] p={P}, {CYCLES} cycles "
              f"in {tmp}/ ...")
        od.main(["--code", str(CODE), "--p", str(P),
                 "--cycles", str(CYCLES), "--no-verify"])

        art_dir = od.artifact_dir(CODE, P, CYCLES)
        expected = ["DX.npz", "DZ.npz", "DL_X.npz", "DL_Z.npz",
                    "D_joint_X.npz", "D_joint_Z.npz",
                    "probX.npy", "probZ.npy",
                    "prob_joint_X.npy", "prob_joint_Z.npy",
                    "R_Z.npz", "R_X.npz",
                    "dem_X.dem", "dem_Z.dem",
                    "dem_joint_X.dem", "dem_joint_Z.dem",
                    "circuit_X.stim", "circuit_Z.stim",
                    "circuit_joint_X.stim", "circuit_joint_Z.stim"]
        missing = [n for n in expected
                   if not os.path.exists(os.path.join(art_dir, n))]
        if missing:
            raise AssertionError(f"missing artifacts: {missing}")
        print(f"  [OK ] complete artifact set ({len(expected)} files)")

        if not check_routes(art_dir, CODE, CYCLES):
            raise AssertionError("route checks failed")
        print("\nALL OFFLINE-STAGE CHECKS PASSED")
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

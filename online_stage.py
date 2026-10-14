#!/usr/bin/env python3

import argparse
import csv
import itertools
import multiprocessing
import os
import time

# Force single-threaded linear algebra to prevent multiprocessing thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import scipy.sparse as sparse
import stim
from tqdm import tqdm

from offline_stage import BBCode, CODE_PARAMS, artifact_dir, extract_matrices
from decoders import RELAYBP_PRESETS, WorkerConfig, decode_shot_pair, init_worker
from gnn_runtime import (device, load_gnn_models, load_priors_chunk,
                         project_priors, release_memory,
                         run_gnn_inference_pair, save_priors_chunk)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code", type=int, choices=sorted(CODE_PARAMS), default=72)
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--cycles", type=int, default=None)

    parser.add_argument("--decoder", type=str, choices=("gnn", "bposd"), default="gnn")
    parser.add_argument("--backend", type=str, choices=("osd", "lsd", "relay", "joint_bposd"), default="osd")
    parser.add_argument("--osd-order", type=int, default=7)
    parser.add_argument("--lsd-order", type=int, default=0)
    parser.add_argument("--lsd-method", type=str, default="LSD_CS", choices=("LSD_0", "LSD_CS", "LSD_E"))
    parser.add_argument("--ms-scaling-factor", type=float, default=0.0)
    parser.add_argument("--bp-method", type=str, choices=("ms", "ps"), default="ms")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--relay-preset", type=str, default="auto", choices=("auto",) + tuple(sorted(RELAYBP_PRESETS)))

    parser.add_argument("--gnn-batch", type=int, default=(64 if device.type in ("xpu", "cuda") else 128))
    parser.add_argument("--h-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--weights-z", type=str, default=None)
    parser.add_argument("--weights-x", type=str, default=None)
    parser.add_argument("--split", action="store_true")

    parser.add_argument("--save-priors", metavar="PREFIX", default=None)
    parser.add_argument("--load-priors", metavar="PREFIX", default=None)

    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--chunk", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--seed-x", type=int, default=4242)
    parser.add_argument("--seed-z", type=int, default=4141)
    parser.add_argument("--out", type=str, default="results_v2.csv")
    args = parser.parse_args(argv)

    if args.cycles is None:
        args.cycles = 12 if args.code == 144 else 6
    if args.max_iter is None:
        args.max_iter = 1000 if args.decoder == "bposd" else 0

    if args.relay_preset == "auto":
        args.relay_preset = "paper_bb144" if args.code == 144 else "paper_bb72"
    args.relay_preset_dict = dict(RELAYBP_PRESETS[args.relay_preset])

    args.tag = f"{args.decoder}_{args.backend}"

    if args.save_priors and args.load_priors:
        raise SystemExit("--save-priors and --load-priors are mutually exclusive")
    if (args.save_priors or args.load_priors) and args.decoder != "gnn":
        raise SystemExit("--save-priors/--load-priors require --decoder gnn")
    if args.decoder == "gnn" and args.backend == "relay":
        raise SystemExit("FATAL: --backend relay does not support dynamic priors.")
    if args.split and (args.decoder != "gnn" or args.backend == "joint_bposd"):
        raise SystemExit("--split requires --decoder gnn with a split backend")
    return args


def make_run_id(args):
    """Clean, conditional run_id that only includes relevant parameters."""
    # 1. Base problem
    parts = [f"c{args.code}", f"p{args.p}"]
    
    # 2. Decoder & Neural parameters
    if args.decoder == "gnn":
        parts.append(f"gnn_h{args.h_dim}L{args.num_layers}")
        if args.split:
            parts.append("split")
    else:
        parts.append("bposd")
        
    # 3. Backend & specific search parameters
    if args.backend == "osd":
        parts.append(f"osd{args.osd_order}")
    elif args.backend == "lsd":
        method_short = args.lsd_method.replace("LSD_", "").lower() # CS, E, 0
        parts.append(f"lsd{args.lsd_order}_{method_short}")
    elif args.backend == "relay":
        parts.append(f"relay_{args.relay_preset}")
        
    # 4. Shared BP parameters
    if args.backend != "relay":
        parts.append(f"{args.bp_method}{args.ms_scaling_factor}")
        parts.append(f"it{args.max_iter}")
        
    # 5. Simulation metadata
    parts += [f"n{args.trials}", f"s{args.seed_z}-{args.seed_x}"]
    
    return "_".join(parts)


def already_done(args):
    if not os.path.isfile(args.out):
        return False
    rid = make_run_id(args)
    with open(args.out, newline="", encoding="utf-8") as f:
        return any(r.get("run_id") == rid for r in csv.DictReader(f))


def load_artifacts(args):
    data_dir = artifact_dir(args.code, args.p, args.cycles)

    files = {key: f"{data_dir}/{fname}" for key, fname in dict(
        D_joint_Z="D_joint_Z.npz", D_joint_X="D_joint_X.npz",
        joint_dem_Z="dem_joint_Z.dem", joint_dem_X="dem_joint_X.dem",
        graded_dem_Z="dem_Z.dem", graded_dem_X="dem_X.dem",
        R_Z="R_Z.npz", R_X="R_X.npz",
        DZ="DZ.npz", DX="DX.npz",
        DL_Z="DL_Z.npz", DL_X="DL_X.npz",
        prob_Z="probZ.npy", prob_X="probX.npy",
        prob_joint_Z="prob_joint_Z.npy", prob_joint_X="prob_joint_X.npy"
    ).items()}

    missing = [path for path in files.values() if not os.path.exists(path)]
    if missing:
        raise SystemExit(
            f"Missing offline artifacts in {data_dir}.\n"
            f"Build them first with:\n"
            f"  python offline_stage.py --code {args.code} --p {args.p} --cycles {args.cycles}"
        )

    code = BBCode(args.code)
    z_half_rows, x_half_rows = code.detector_slices(args.cycles)

    print(f"Decode {args.tag} ({device.type.upper()}) | [[{code.n},{code.k},{code.d}]] p={args.p} cyc={args.cycles}")

    artifacts = {
        "n": code.n, "k": code.k, "d": code.d,
        "z_half_rows": z_half_rows, "x_half_rows": x_half_rows,
        "D_joint_Z": sparse.load_npz(files["D_joint_Z"]),
        "D_joint_X": sparse.load_npz(files["D_joint_X"]),
        "DZ": sparse.load_npz(files["DZ"]),
        "DX": sparse.load_npz(files["DX"]),
        "DL_Z": sparse.load_npz(files["DL_Z"]),
        "DL_X": sparse.load_npz(files["DL_X"]),
        "route_proj_z": sparse.load_npz(files["R_Z"]),
        "route_proj_x": sparse.load_npz(files["R_X"]),
        "graded_priors_z": np.load(files["prob_Z"]),
        "graded_priors_x": np.load(files["prob_X"]),
        "joint_priors_z": np.load(files["prob_joint_Z"]),
        "joint_priors_x": np.load(files["prob_joint_X"]),
        "joint_shots": args.decoder == "gnn" and args.load_priors is None
    }

    if artifacts["joint_shots"] or args.decoder == "bposd":
        joint_dems = artifacts["joint_shots"] or args.backend == "joint_bposd"
        z_dem = files["joint_dem_Z"] if joint_dems else files["graded_dem_Z"]
        x_dem = files["joint_dem_X"] if joint_dems else files["graded_dem_X"]
        artifacts["sampler_z"] = stim.DetectorErrorModel.from_file(z_dem).compile_sampler(seed=args.seed_z)
        artifacts["sampler_x"] = stim.DetectorErrorModel.from_file(x_dem).compile_sampler(seed=args.seed_x)

    if args.backend == "joint_bposd":
        artifacts["dl_joint"] = {
            "Z": extract_matrices(stim.DetectorErrorModel.from_file(files["joint_dem_Z"]))[1],
            "X": extract_matrices(stim.DetectorErrorModel.from_file(files["joint_dem_X"]))[1]}

    return artifacts


def run_trials(args, artifacts, models):
    metrics = {
        "z_err": 0, "x_err": 0, "union_err": 0,
        "conv_z": 0, "conv_x": 0, "conv_b": 0,
        "fail_z_conv_z": 0,
        "fail_x_conv_x": 0,
        "fail_union_conv_b": 0,
        "gnn_time_s": 0.0, "decode_time_s": 0.0
    }

    all_t_ms = []
    all_fails = []

    n_chunks = (args.trials + args.chunk - 1) // args.chunk
    cfg = WorkerConfig(
        dz=artifacts["DZ"], dx=artifacts["DX"],
        dl_z=artifacts["DL_Z"], dl_x=artifacts["DL_X"],
        backend=args.backend, bp_method=args.bp_method,
        max_iter=args.max_iter, osd_order=args.osd_order,
        lsd_order=args.lsd_order, lsd_method=args.lsd_method, 
        ms_scaling_factor=args.ms_scaling_factor,
        relay_preset=args.relay_preset_dict,
        d_joint={"Z": artifacts["D_joint_Z"], "X": artifacts["D_joint_X"]},
        dl_joint=artifacts.get("dl_joint")
    )

    t_wall = time.perf_counter()

    with multiprocessing.Pool(args.workers, initializer=init_worker, initargs=(cfg,)) as pool:
        for chunk in range(n_chunks):
            chunk_n = min(args.chunk, args.trials - chunk * args.chunk)
            print(f"\nChunk {chunk + 1}/{n_chunks} | {chunk_n} shots")

            if args.load_priors is not None:
                (syndromes_z, observables_z, syndromes_x, observables_x,
                 priors_z, priors_x) = load_priors_chunk(args.load_priors, chunk, chunk_n)
            else:
                syndromes_z, observables_z, _ = artifacts["sampler_z"].sample(chunk_n)
                syndromes_x, observables_x, _ = artifacts["sampler_x"].sample(chunk_n)

                if args.decoder == "gnn":
                    t0 = time.perf_counter()
                    if args.split:
                        syn_in = {"Z": syndromes_z[:, artifacts["z_half_rows"]],
                                  "X": syndromes_x[:, artifacts["x_half_rows"]]}
                    else:
                        syn_in = {"Z": syndromes_z, "X": syndromes_x}
                    priors_joint = run_gnn_inference_pair(models, syn_in, args.gnn_batch)
                    if args.backend == "joint_bposd" or args.split:
                        priors_z = priors_joint["Z"]
                        priors_x = priors_joint["X"]
                    else:
                        priors_z = project_priors(priors_joint["Z"], artifacts["route_proj_z"])
                        priors_x = project_priors(priors_joint["X"], artifacts["route_proj_x"])
                    metrics["gnn_time_s"] += time.perf_counter() - t0

                    if args.save_priors is not None:
                        save_priors_chunk(args.save_priors, chunk, syndromes_z, observables_z,
                                          syndromes_x, observables_x, priors_z, priors_x)
                else:
                    if args.backend == "joint_bposd":
                        prior_z = artifacts["joint_priors_z"]
                        prior_x = artifacts["joint_priors_x"]
                    else:
                        prior_z = artifacts["graded_priors_z"]
                        prior_x = artifacts["graded_priors_x"]
                    priors_z = itertools.repeat(prior_z, chunk_n)
                    priors_x = itertools.repeat(prior_x, chunk_n)

            if args.decoder == "gnn" and args.backend != "joint_bposd":
                syndromes_z = syndromes_z[:, artifacts["z_half_rows"]]
                syndromes_x = syndromes_x[:, artifacts["x_half_rows"]]

            t0 = time.perf_counter()
            dynamic_chunksize = max(1, min(50, chunk_n // (args.workers * 2)))

            t_ms_arr = np.zeros(chunk_n, dtype=np.float32)
            fail_arr = np.zeros(chunk_n, dtype=bool)

            with tqdm(total=chunk_n, desc="Decoding") as dbar:
                for i, res in enumerate(pool.imap(
                    decode_shot_pair,
                    zip(syndromes_z, observables_z, priors_z,
                        syndromes_x, observables_x, priors_x),
                    chunksize=dynamic_chunksize
                )):
                    fail_z, fail_x = bool(res[0]), bool(res[1])
                    conv_z, conv_x = bool(res[2]), bool(res[3])
                    conv_b = conv_z and conv_x

                    if fail_z: metrics["z_err"] += 1
                    if fail_x: metrics["x_err"] += 1
                    if fail_z or fail_x: metrics["union_err"] += 1
                    if conv_z: metrics["conv_z"] += 1
                    if conv_x: metrics["conv_x"] += 1
                    if conv_b: metrics["conv_b"] += 1

                    if fail_z and conv_z: metrics["fail_z_conv_z"] += 1
                    if fail_x and conv_x: metrics["fail_x_conv_x"] += 1
                    if (fail_z or fail_x) and conv_b: metrics["fail_union_conv_b"] += 1

                    t_ms_arr[i] = res[4]
                    fail_arr[i] = fail_z or fail_x
                    dbar.update(1)

            all_t_ms.append(t_ms_arr)
            all_fails.append(fail_arr)
            metrics["decode_time_s"] += time.perf_counter() - t0

            done = min(args.trials, (chunk + 1) * args.chunk)
            print(f"  ↳ Cumulative P_L (Union): {metrics['union_err']}/{done} "
                  f"({(metrics['union_err']/done):.5f}) | Convergence: {(metrics['conv_b']/done):.1%}")

            del (syndromes_z, observables_z, syndromes_x, observables_x, priors_z, priors_x)
            release_memory()

    full_t_ms = np.concatenate(all_t_ms)
    metrics["decode_ms_mean"] = float(np.mean(full_t_ms))
    metrics["decode_ms_p50"] = float(np.median(full_t_ms))
    metrics["decode_ms_p99"] = float(np.percentile(full_t_ms, 99))

    np.save(make_run_id(args) + "_fails.npy", np.concatenate(all_fails))

    report_and_save_results(args, artifacts, metrics, time.perf_counter() - t_wall)


def _cond(fail_and_conv, fail_total, conv, shots):
    p_c = fail_and_conv / conv if conv else float("nan")
    nc = shots - conv
    p_nc = (fail_total - fail_and_conv) / nc if nc else float("nan")
    return p_c, p_nc


def report_and_save_results(args, artifacts, m, total_time_s):
    trials = args.trials
    pL_u = 1 - (1 - (m['union_err'] / trials)) ** (1 / args.cycles)

    pz_c, pz_nc = _cond(m["fail_z_conv_z"], m["z_err"], m["conv_z"], trials)
    px_c, px_nc = _cond(m["fail_x_conv_x"], m["x_err"], m["conv_x"], trials)
    pb_c, pb_nc = _cond(m["fail_union_conv_b"], m["union_err"], m["conv_b"], trials)

    print(f"\nFinal Results: {args.tag}")
    print(f"P_L: Z {m['z_err']/trials:.5f} | X {m['x_err']/trials:.5f} | Union {m['union_err']/trials:.5f}")
    print(f"Logical error per cycle (pL_u): {pL_u:.6f}")
    print(f"Convergence: Z {m['conv_z']/trials:.2%} | X {m['conv_x']/trials:.2%} | Both {m['conv_b']/trials:.2%}")
    print(f"P(fail|conv) Z {pz_c:.4f} X {px_c:.4f} Union|both {pb_c:.4f} | "
          f"P(fail|not conv) Z {pz_nc:.4f} X {px_nc:.4f} Union|both {pb_nc:.4f}")

    throughput = trials / total_time_s
    decode_tp = trials / m['decode_time_s'] if m['decode_time_s'] > 0 else 0
    print(f"Time: GNN {m['gnn_time_s']:.1f}s | Decode {m['decode_time_s']:.1f}s | Total {total_time_s:.1f}s")
    print(f"Throughput: {throughput:.2f} shot/s (Decode only: {decode_tp:.2f} shot/s)")

    row_data = {
        "run_id": make_run_id(args),
        "code_n": artifacts['n'],
        "code_k": artifacts['k'],
        "code_d": artifacts['d'],
        "p": args.p,
        "cycles": args.cycles,
        "decoder": args.decoder,
        "backend": args.backend,
        "bp_method": args.bp_method,
        "ms_scaling_factor": args.ms_scaling_factor,
        "max_iter": args.max_iter,
        "osd_order": args.osd_order,
        "lsd_order": args.lsd_order,
        "lsd_method": args.lsd_method,
        "relay_preset": args.relay_preset if args.backend == 'relay' else '',
        "gnn_layers": args.num_layers if args.decoder == 'gnn' else 0,
        "gnn_hdim": args.h_dim if args.decoder == 'gnn' else 0,
        "gnn_batch": args.gnn_batch if args.decoder == 'gnn' else 0,
        "device": device.type,
        "workers": args.workers,
        "seed_z": args.seed_z,
        "seed_x": args.seed_x,
        "shots": trials,
        "fail_z": m['z_err'],
        "fail_x": m['x_err'],
        "fail_union": m['union_err'],
        "conv_z": m['conv_z'],
        "conv_x": m['conv_x'],
        "conv_both": m['conv_b'],
        "fail_z_conv_z": m['fail_z_conv_z'],
        "fail_x_conv_x": m['fail_x_conv_x'],
        "fail_union_conv_both": m['fail_union_conv_b'],
        "pL_shot_union": m['union_err'] / trials,
        "pL_cycle_union": pL_u,
        "gnn_s": m['gnn_time_s'],
        "gnn_ms_per_shot": m['gnn_time_s'] / trials * 1000,
        "decode_wall_s": m['decode_time_s'],
        "wall_total_s": total_time_s,
        "decode_ms_mean": m['decode_ms_mean'],
        "decode_ms_p50": m['decode_ms_p50'],
        "decode_ms_p99": m['decode_ms_p99'],
        "throughput_shots_s": throughput,
    }

    csv_path = args.out
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def main(argv=None):
    args = parse_args(argv)
    if already_done(args):
        print(f"SKIP (already in {args.out}): {make_run_id(args)}")
        return
    artifacts = load_artifacts(args)

    models = None
    if args.decoder == "gnn":
        if args.load_priors is None:
            weights_z, weights_x = args.weights_z, args.weights_x
            if args.split:
                # The split controls live in split_weights{code}/ under their own name.
                weights_z = weights_z or f"split_weights{args.code}/gnn_split_Z_L{args.num_layers}_H{args.h_dim}_s0_best.pt"
                weights_x = weights_x or f"split_weights{args.code}/gnn_split_X_L{args.num_layers}_H{args.h_dim}_s0_best.pt"
            models = load_gnn_models(
                code_id=args.code, num_layers=args.num_layers,
                h_dim=args.h_dim, weights_z=weights_z, weights_x=weights_x,
                gnn_batch=args.gnn_batch,
                d_joint_z=artifacts["DZ" if args.split else "D_joint_Z"],
                d_joint_x=artifacts["DX" if args.split else "D_joint_X"]
            )

    run_trials(args, artifacts, models)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import argparse
import itertools
import multiprocessing
import os
import time
import numpy as np
import scipy.sparse as sparse
import stim
from tqdm import tqdm

from offline_stage import BBCode, CODE_PARAMS, artifact_dir, extract_matrices
from decoders import RELAYBP_PRESETS, WorkerConfig, decode_shot_pair, init_worker
from gnn_runtime import (device, load_gnn_models, load_priors_chunk,
                         project_priors, release_memory,
                         run_gnn_inference_pair, save_priors_chunk)
                         
# Force single-threaded linear algebra to prevent multiprocessing thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def parse_args(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code", type=int, choices=sorted(CODE_PARAMS), default=72)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--cycles", type=int, default=None)

    parser.add_argument("--decoder", type=str, choices=("gnn", "bposd"), default="gnn")
    parser.add_argument("--backend", type=str, choices=("osd", "lsd", "relay", "joint_bposd"), default="osd")
    parser.add_argument("--osd-order", type=int, default=7)
    parser.add_argument("--lsd-order", type=int, default=0)
    parser.add_argument("--bp-method", type=str, choices=("ms", "ps"), default="ps")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--relay-preset", type=str, default="auto", choices=("auto",) + tuple(sorted(RELAYBP_PRESETS)))

    parser.add_argument("--gnn-batch", type=int, default=32)
    parser.add_argument("--h-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--weights-z", type=str, default=None)
    parser.add_argument("--weights-x", type=str, default=None)

    parser.add_argument("--save-priors", metavar="PREFIX", default=None)
    parser.add_argument("--load-priors", metavar="PREFIX", default=None)

    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--chunk", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--seed-x", type=int, default=4242)
    parser.add_argument("--seed-z", type=int, default=4141)
    parser.add_argument("--out", type=str, default="evaluate_results.csv")
    args = parser.parse_args(argv)

    if args.cycles is None:
        args.cycles = 12 if args.code == 144 else 6
    if args.max_iter is None:
        args.max_iter = 100 if args.decoder == "gnn" else 1000

    if args.relay_preset == "auto":
        args.relay_preset = "paper_bb144" if args.code == 144 else "paper_bb72"
    args.relay_preset_dict = dict(RELAYBP_PRESETS[args.relay_preset])

    args.tag = f"{args.decoder}_{args.backend}"

    if args.save_priors and args.load_priors:
        raise SystemExit("--save-priors and --load-priors are mutually exclusive")
    if (args.save_priors or args.load_priors) and args.decoder != "gnn":
        raise SystemExit("--save-priors/--load-priors require --decoder gnn")
    if args.decoder == "gnn" and args.backend == "relay":
        raise SystemExit(
            "FATAL: --backend relay is configured strictly as a static-prior baseline. "
            "It does not support dynamic per-shot GNN priors. Use --backend osd instead."
        )
    return args


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
        raise SystemExit(f"Missing artifacts in {data_dir}: {missing}")

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
        # joint_bposd decodes the full joint syndrome, so it samples the
        # joint DEMs even for the static bposd baseline.
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
    """Executes the chunked sample -> GNN (optional) -> classical decode pipeline."""
    metrics = {
        "z_err": 0, "x_err": 0, "union_err": 0,
        "conv_z": 0, "conv_x": 0, "conv_b": 0,
        "gnn_time_s": 0.0, "decode_time_s": 0.0,
        "status_z": {"relay": 0, "fail": 0},
        "status_x": {"relay": 0, "fail": 0}
    }

    n_chunks = (args.trials + args.chunk - 1) // args.chunk
    cfg = WorkerConfig(
        dz=artifacts["DZ"], dx=artifacts["DX"],
        dl_z=artifacts["DL_Z"], dl_x=artifacts["DL_X"],
        backend=args.backend, bp_method=args.bp_method,
        max_iter=args.max_iter, osd_order=args.osd_order,
        lsd_order=args.lsd_order, relay_preset=args.relay_preset_dict,
        d_joint={"Z": artifacts["D_joint_Z"], "X": artifacts["D_joint_X"]},
        dl_joint=artifacts.get("dl_joint")
    )

    t_wall = time.perf_counter()
    
    with multiprocessing.Pool(args.workers, initializer=init_worker, initargs=(cfg,)) as pool:
        for chunk in range(n_chunks):
            chunk_n = min(args.chunk, args.trials - chunk * args.chunk)
            print(f"\nChunk {chunk + 1}/{n_chunks} | {chunk_n} shots")

            #Step 1: Acquire Data (Syndromes & Priors)
            if args.load_priors is not None:
                (syndromes_z, observables_z, syndromes_x, observables_x,
                 priors_z, priors_x) = load_priors_chunk(args.load_priors,
                                                         chunk, chunk_n)
            else:
                syndromes_z, observables_z, _ = artifacts["sampler_z"].sample(
                    chunk_n)
                syndromes_x, observables_x, _ = artifacts["sampler_x"].sample(
                    chunk_n)

                if args.decoder == "gnn":
                    t0 = time.perf_counter()
                    priors_joint = run_gnn_inference_pair(
                        models, {"Z": syndromes_z, "X": syndromes_x},
                        args.gnn_batch)
                    if args.backend == "joint_bposd":
                        priors_z = priors_joint["Z"]
                        priors_x = priors_joint["X"]
                    else:
                        priors_z = project_priors(priors_joint["Z"],
                                                   artifacts["route_proj_z"])
                        priors_x = project_priors(priors_joint["X"],
                                                   artifacts["route_proj_x"])
                    metrics["gnn_time_s"] += time.perf_counter() - t0

                    if args.save_priors is not None:
                        save_priors_chunk(args.save_priors, chunk,
                                          syndromes_z, observables_z,
                                          syndromes_x, observables_x,
                                          priors_z, priors_x)
                else:
                    # Classical baseline: use static per-fault priors.
                    # joint_bposd operates on the uncut DEM, requiring joint priors.
                    if args.backend == "joint_bposd":
                        prior_z = artifacts["joint_priors_z"]
                        prior_x = artifacts["joint_priors_x"]
                    else:
                        prior_z = artifacts["graded_priors_z"]
                        prior_x = artifacts["graded_priors_x"]
                    priors_z = itertools.repeat(prior_z, chunk_n)
                    priors_x = itertools.repeat(prior_x, chunk_n)

            # gnn always starts from full joint syndromes (sampled or from
            # cache); bposd samples the graded per-basis DEMs directly.
            if args.decoder == "gnn" and args.backend != "joint_bposd":
                syndromes_z = syndromes_z[:, artifacts["z_half_rows"]]
                syndromes_x = syndromes_x[:, artifacts["x_half_rows"]]

            # Step 2: Classical Decoding
            t0 = time.perf_counter()
            dynamic_chunksize = max(1, min(50, chunk_n // (args.workers * 2)))
            with tqdm(total=chunk_n, desc="Decoding") as dbar:
                for (success_z, status_z, converged_z,
                     success_x, status_x, converged_x) in pool.imap_unordered(
                        decode_shot_pair,
                        zip(syndromes_z, observables_z, priors_z,
                            syndromes_x, observables_x, priors_x),
                        chunksize=dynamic_chunksize):

                    if not success_z:
                        metrics["z_err"] += 1
                    if success_x is False:
                        metrics["x_err"] += 1
                    if not (success_z and success_x is not False):
                        metrics["union_err"] += 1

                    if converged_z:
                        metrics["conv_z"] += 1
                    if converged_x:
                        metrics["conv_x"] += 1
                    if converged_z and converged_x is not False:
                        metrics["conv_b"] += 1

                    if args.backend == "relay":
                        metrics["status_z"][status_z] += 1
                        metrics["status_x"][status_x] += 1

                    dbar.update(1)
            metrics["decode_time_s"] += time.perf_counter() - t0

            done = min(args.trials, (chunk + 1) * args.chunk)
            print(f"  ↳ Cumulative P_L (Union): "
                  f"{metrics['union_err']}/{done} "
                  f"({(metrics['union_err']/done):.5f}) | Convergence: "
                  f"{(metrics['conv_b']/done):.1%}")

            # Free RAM between chunks
            del (syndromes_z, observables_z, syndromes_x, observables_x,
                 priors_z, priors_x)
            release_memory()

    report_and_save_results(args, artifacts, metrics, time.perf_counter() - t_wall)


def report_and_save_results(args, artifacts, m, total_time_s):
    """Calculates final rates, prints to console, and appends to CSV."""
    trials = args.trials
    pL_u = 1 - (1 - (m['union_err'] / trials)) ** (1 / args.cycles)
    
    print(f"\nFinal Results: {args.tag}")
    print(f"P_L: Z {m['z_err']/trials:.5f} | X {m['x_err']/trials:.5f} | Union {m['union_err']/trials:.5f}")
    print(f"Logical error per cycle (pL_u): {pL_u:.6f}")
    print(f"Convergence: Z {m['conv_z']/trials:.2%} | X {m['conv_x']/trials:.2%} | Both {m['conv_b']/trials:.2%}")
    
    if args.backend == "relay":
        print(f"Relay Status -> Z: {m['status_z']} | X: {m['status_x']}")
        
    throughput = trials / total_time_s
    decode_tp = trials / m['decode_time_s'] if m['decode_time_s'] > 0 else 0
    print(f"Time: GNN {m['gnn_time_s']:.1f}s | Decode {m['decode_time_s']:.1f}s | Total {total_time_s:.1f}s")
    print(f"Throughput: {throughput:.2f} shot/s (Decode only: {decode_tp:.2f} shot/s)")

    header = ("decoder,n,k,d,p,num_cycles,trials,osd_order,lsd_order,"
              "relay_preset,bp_max_iter,gnn_layers,gnn_hdim,"
              "logical_errors,P_L,converged_Z,converged_X,converged_both,"
              "conv_rate_Z,conv_rate_X,conv_rate_both,"
              "gnn_inference_s,decode_s,mean_time_ms")
              
    row = (f"{args.tag},{artifacts['n']},{artifacts['k']},{artifacts['d']},{args.p},{args.cycles},{trials},"
           f"{args.osd_order},{args.lsd_order},{args.relay_preset if args.backend == 'relay' else ''},"
           f"{args.max_iter},{args.num_layers if args.decoder == 'gnn' else 0},"
           f"{args.h_dim if args.decoder == 'gnn' else 0},{m['union_err']},{m['union_err']/trials:.6f},"
           f"{m['conv_z']},{m['conv_x']},{m['conv_b']},"
           f"{m['conv_z']/trials:.6f},{m['conv_x']/trials:.6f},{m['conv_b']/trials:.6f},"
           f"{m['gnn_time_s']:.3f},{m['decode_time_s']:.3f},{total_time_s/trials*1000:.4f}")

    new_file = not os.path.exists(args.out)
    with open(args.out, "a") as f:
        if new_file: f.write(header + "\n")
        f.write(row + "\n")


def main(argv=None):
    args = parse_args(argv)
    artifacts = load_artifacts(args)

    models = None
    if args.decoder == "gnn":
        if args.load_priors is None:
            models = load_gnn_models(
                code_id=args.code, num_layers=args.num_layers,
                h_dim=args.h_dim, weights_z=args.weights_z, weights_x=args.weights_x,
                gnn_batch=args.gnn_batch,
                d_joint_z=artifacts["D_joint_Z"], d_joint_x=artifacts["D_joint_X"]
            )
    
    run_trials(args, artifacts, models)

if __name__ == "__main__":
    main()
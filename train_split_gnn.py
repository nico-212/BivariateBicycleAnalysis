#!/usr/bin/env python3
"""Trains a split-graph (per-basis) GNN prior estimator.

Control for the joint-graph GNN: identical recipe to train_joint_gnn.py
(model class, optimizer, scheduler, loss, precision, batch/accumulation,
epochs, noise distribution), but the graph is the per-basis detector matrix
D{basis} instead of D_joint_{basis}. The outputs live directly on the split
fault columns, so no routing matrix is needed at evaluation time.
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import offline_stage as od
from gnn_model import BipartiteGNN

DEFAULT_EPOCHS = 20
DEFAULT_CHECKPOINT_EVERY = 5
DEFAULT_BATCHES_PER_EPOCH = 1200
DEFAULT_BATCH_SIZE = 16
DEFAULT_ACCUM_STEPS = 4
DEFAULT_LR = 2e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_P_MIN, DEFAULT_P_MAX = 0.001, 0.01

# Same device / precision policy as the joint trainer: float32, no autocast.
if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def free_memory_bytes():
    try:
        if device.type == "cuda":
            return int(torch.cuda.mem_get_info()[0])
        if device.type == "xpu":
            if hasattr(torch.xpu, "mem_get_info"):
                return int(torch.xpu.mem_get_info()[0])
            return int(torch.xpu.get_device_properties(0).total_memory * 0.5)
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    except Exception:
        return None


def generate_split_batches(d_mat, base_probs, batch_size, batches_per_epoch,
                           p_artifact, p_min, p_max, rng):
    """Synthesizes (syndromes, faults) on the split graph via Bernoulli sampling."""
    n_faults = len(base_probs)
    d_t = d_mat.T.tocsr()

    for _ in range(batches_per_epoch):
        p_batch = rng.uniform(p_min, p_max)
        batch_probs = np.clip(base_probs * (p_batch / p_artifact), 0.0, 0.5)

        faults = (rng.random((batch_size, n_faults)) < batch_probs).astype(np.float32)
        syndromes = (faults @ d_t) % 2

        yield (torch.tensor(syndromes, dtype=torch.float32),
               torch.tensor(faults, dtype=torch.float32))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", type=int, choices=sorted(od.CODE_PARAMS), default=72)
    ap.add_argument("--p", type=float, default=0.003,
                    help="Rate of the artifact directory = anchor p0 for rescaling.")
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--basis", choices=("Z", "X"), default="Z")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batches-per-epoch", type=int, default=DEFAULT_BATCHES_PER_EPOCH)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--p-min", type=float, default=DEFAULT_P_MIN)
    ap.add_argument("--p-max", type=float, default=DEFAULT_P_MAX)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--h-dim", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    assert 0 < args.p_min < args.p_max <= 0.5, (args.p_min, args.p_max)

    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    data_dir = od.artifact_dir(args.code, args.p, args.cycles)
    l, m, k, d = od.CODE_PARAMS[args.code]
    n = 2 * l * m

    mat_f = f"{data_dir}/D{args.basis}.npz"
    prob_f = f"{data_dir}/prob{args.basis}.npy"
    missing = [f for f in (mat_f, prob_f) if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"Missing split artifacts in {data_dir}: {missing}")

    d_mat = sp.load_npz(mat_f).tocoo()
    base_probs = np.load(prob_f)
    assert d_mat.shape[1] == len(base_probs), (d_mat.shape, len(base_probs))
    n_det, n_fault = d_mat.shape

    if args.batch is not None:
        batch_size, accum_steps = args.batch, DEFAULT_ACCUM_STEPS
    else:
        bytes_per_shot = 4 * args.h_dim * (n_det + n_fault) * 10 * max(1, args.num_layers)
        free_bytes = free_memory_bytes()
        budget = DEFAULT_BATCH_SIZE if free_bytes is None else \
            int((0.5 * free_bytes - 512 * 2**20) // bytes_per_shot)
        batch_size = max([b for b in (DEFAULT_BATCH_SIZE, 8, 4, 2, 1) if b <= budget],
                         default=1)
        accum_steps = (DEFAULT_BATCH_SIZE * DEFAULT_ACCUM_STEPS) // batch_size

    model = BipartiteGNN(n_det=n_det, n_fault=n_fault,
                         edges_det=d_mat.row, edges_fault=d_mat.col,
                         h_dim=args.h_dim, num_layers=args.num_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Train SPLIT {args.basis}-Mem [[{n},{k},{d}]] p0={args.p} cyc={args.cycles} | "
          f"Graph: {n_det}x{n_fault} ({device.type.upper()}) | "
          f"GNN: {args.num_layers}L/{args.h_dim}H, {n_params:,} params | "
          f"Noise: U({args.p_min},{args.p_max}) | Batch: {batch_size}x{accum_steps} | "
          f"seed {args.seed}")

    optimizer = optim.AdamW(model.parameters(), lr=DEFAULT_LR,
                            weight_decay=DEFAULT_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                     factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()  # unweighted, as in the joint recipe

    weights_dir = f"split_weights{args.code}"
    os.makedirs(weights_dir, exist_ok=True)
    tag = f"{args.basis}_L{args.num_layers}_H{args.h_dim}_s{args.seed}"
    best_path = f"{weights_dir}/gnn_split_{tag}_best.pt"
    log_path = f"{weights_dir}/gnn_split_{tag}_log.csv"
    meta_path = f"{weights_dir}/gnn_split_{tag}_meta.json"

    best_loss = float("inf")
    t_start = time.time()
    with open(log_path, "w") as log:
        log.write("epoch,avg_loss,lr,elapsed_s\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        generator = generate_split_batches(
            d_mat, base_probs, batch_size, args.batches_per_epoch,
            p_artifact=args.p, p_min=args.p_min, p_max=args.p_max, rng=np_rng)
        pbar = tqdm(generator, total=args.batches_per_epoch,
                    desc=f"Epoch {epoch}/{args.epochs}")

        optimizer.zero_grad()
        for step, (syndromes, faults) in enumerate(pbar):
            syndromes, faults = syndromes.to(device), faults.to(device)
            logits = model(syndromes)
            loss = criterion(logits, faults) / accum_steps
            loss.backward()
            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accum_steps
            pbar.set_postfix({"loss": f"{loss.item() * accum_steps:.4f}"})

        avg_loss = epoch_loss / args.batches_per_epoch
        scheduler.step(avg_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_start
        print(f"Epoch {epoch} | loss: {avg_loss:.5f} | lr: {lr_now:.6f} | {elapsed/60:.1f} min")
        with open(log_path, "a") as log:
            log.write(f"{epoch},{avg_loss:.6f},{lr_now:.6f},{elapsed:.1f}\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_path)
            print(f"Best {args.basis} -> {best_path}")

        if epoch % DEFAULT_CHECKPOINT_EVERY == 0:
            torch.save(model.state_dict(),
                       f"{weights_dir}/gnn_split_{tag}_epoch_{epoch}.pt")

    total = time.time() - t_start
    meta = dict(vars(args), n_det=n_det, n_fault=n_fault, n_params=n_params,
                batch_size=batch_size, accum_steps=accum_steps,
                best_loss=best_loss, train_seconds=total, device=device.type,
                data_dir=data_dir)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done: split {args.basis}-mem model in {total/60:.1f} min -> {best_path}")


if __name__ == "__main__":
    main()
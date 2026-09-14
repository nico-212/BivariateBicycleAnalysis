#!/usr/bin/env python3
"""Trains a joint GNN model to estimate fault prior probabilities.

Synthesizes training batches on the fly with randomized physical error 
rates (p_batch ~ U(p_min, p_max)). Uses offline artifacts to map full 
joint syndromes to fault priors. 

Uses unweighted BCE Loss to learn true marginal probabilities, providing 
a sparse, high-quality prior that naturally supports Belief Propagation 
convergence without overconfidence.
"""

import argparse
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import offline_stage as od
from gnn_model import BipartiteGNN

try:
    import intel_extension_for_pytorch
except ImportError:
    pass

DEFAULT_EPOCHS = 20
DEFAULT_CHECKPOINT_EVERY = 5
DEFAULT_BATCHES_PER_EPOCH = 1200
DEFAULT_BATCH_SIZE = 16
DEFAULT_ACCUM_STEPS = 4
DEFAULT_LR = 2e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_P_MIN, DEFAULT_P_MAX = 0.001, 0.01


if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
    use_amp = False
    amp_dtype = torch.float32
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
    use_amp = False
    amp_dtype = torch.float32
else:
    device = torch.device("cpu")
    use_amp = False
    amp_dtype = torch.float32


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


def generate_joint_batches(d_joint, base_probs, batch_size, batches_per_epoch,
                           p_artifact, p_min=DEFAULT_P_MIN, p_max=DEFAULT_P_MAX):
    """Synthesizes (syndromes, faults) batches on the fly using Bernoulli sampling."""
    n_faults = len(base_probs)
    d_joint_t = d_joint.T.tocsr()

    for _ in range(batches_per_epoch):
        p_batch = np.random.uniform(p_min, p_max)
        batch_probs = np.clip(base_probs * (p_batch / p_artifact), 0.0, 0.5)

        faults = (np.random.rand(batch_size, n_faults) < batch_probs).astype(np.float32)
        syndromes = (faults @ d_joint_t) % 2

        yield (torch.tensor(syndromes, dtype=torch.float32), 
               torch.tensor(faults, dtype=torch.float32))


def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", type=int, choices=sorted(od.CODE_PARAMS), default=72)
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batches-per-epoch", type=int, default=DEFAULT_BATCHES_PER_EPOCH)
    ap.add_argument("--batch", "--gnn-batch", dest="batch", type=int, default=None,
                    help="Shots per training batch. Auto-scales down for large fault spaces if omitted.")
    ap.add_argument("--p-min", type=float, default=DEFAULT_P_MIN)
    ap.add_argument("--p-max", type=float, default=DEFAULT_P_MAX)
    ap.add_argument("--basis", type=str, choices=("Z", "X"), default="Z")
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--h-dim", type=int, default=16)
    args = ap.parse_args(argv)

    assert 0 < args.p_min < args.p_max <= 0.5, (args.p_min, args.p_max)

    data_dir = od.artifact_dir(args.code, args.p, args.cycles)
    l, m, k, d = od.CODE_PARAMS[args.code]
    n = 2 * l * m
    
    joint_files = {
        "D_joint": f"{data_dir}/D_joint_{args.basis}.npz",
        "prob_joint": f"{data_dir}/prob_joint_{args.basis}.npy"
    }
    
    missing = [v for v in joint_files.values() if not os.path.exists(v)]
    if missing:
        raise SystemExit(f"Missing joint artifacts in {data_dir}: {missing}")

    d_joint = sp.load_npz(joint_files["D_joint"]).tocoo()
    base_probs = np.load(joint_files["prob_joint"])
    assert d_joint.shape[1] == len(base_probs), (d_joint.shape, len(base_probs))

    n_det, n_fault = d_joint.shape

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

    print(f"Train {args.basis}-Mem [[{n},{k},{d}]] p={args.p} cyc={args.cycles} | "
          f"Graph: {n_det}x{n_fault} ({device.type.upper()}) | "
          f"GNN: {args.num_layers}L/{args.h_dim}H | "
          f"Noise: U({args.p_min},{args.p_max}) | Batch: {batch_size}x{accum_steps}")
    
    if args.batch is None and batch_size != DEFAULT_BATCH_SIZE:
        print(f"WARN: batch auto-scaled to {batch_size} (device memory budget)")

    model = BipartiteGNN(n_det=n_det, n_fault=n_fault,
                         edges_det=d_joint.row, edges_fault=d_joint.col,
                         h_dim=args.h_dim, num_layers=args.num_layers).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    # Standard, unweighted BCE Loss
    criterion = nn.BCEWithLogitsLoss()

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    weights_dir = f"weights{args.code}"
    arch_tag = f"{args.basis}_L{args.num_layers}_H{args.h_dim}"
    os.makedirs(weights_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        generator = generate_joint_batches(d_joint, base_probs, batch_size,
                                           args.batches_per_epoch, p_artifact=args.p,
                                           p_min=args.p_min, p_max=args.p_max)
        
        pbar = tqdm(generator, total=args.batches_per_epoch, desc=f"Epoch {epoch}/{args.epochs}")

        optimizer.zero_grad()
        for step, (syndromes, faults) in enumerate(pbar):
            syndromes, faults = syndromes.to(device), faults.to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(syndromes)
                loss = criterion(logits, faults) / accum_steps

            if use_amp:
                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += loss.item() * accum_steps
            pbar.set_postfix({"loss": f"{loss.item() * accum_steps:.4f}"})

        avg_loss = epoch_loss / args.batches_per_epoch
        scheduler.step(avg_loss)
        print(f"Epoch {epoch} | loss: {avg_loss:.5f} | lr: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = f"{weights_dir}/gnn_joint_{arch_tag}_best.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Best {args.basis} -> {save_path}")

        if epoch % DEFAULT_CHECKPOINT_EVERY == 0:
            checkpoint_path = f"{weights_dir}/gnn_joint_{arch_tag}_epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)

    print(f"Done: {args.basis}-mem model")


if __name__ == "__main__":
    main()
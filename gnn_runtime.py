#!/usr/bin/env python3
"""Parent-process GNN runtime for batched inference and prior projection.

Handles model loading, hardware acceleration (IPEX/CUDA), batched inference 
over Z/X memories, and projecting joint faults down to graded per-basis priors.
"""

import gc
import os
import numpy as np
import torch
from tqdm import tqdm

from gnn_model import BipartiteGNN

try:
    import intel_extension_for_pytorch
except ImportError:
    pass

# Restrict BLAS threads to prevent multiprocessing CPU thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Hardware detection and mixed precision setup
if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
    use_amp = True
    amp_data_type = torch.bfloat16
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
    use_amp = False
    amp_data_type = torch.float16
else:
    device = torch.device("cpu")
    use_amp = False
    amp_data_type = torch.bfloat16


def project_priors(joint_priors, route_proj, block=256):
    """Projects joint fault priors onto graded per-basis columns via XOR probability."""
    joint_priors = np.clip(np.asarray(joint_priors, dtype=np.float32), 0.0, 0.5 - 1e-9)
    was_1d = joint_priors.ndim == 1
    if was_1d:
        joint_priors = joint_priors[None, :]

    graded = np.empty((joint_priors.shape[0], route_proj.shape[1]), dtype=np.float32)
    for start in range(0, joint_priors.shape[0], block):
        rows = joint_priors[start:start + block]
        parity_sign = np.clip(1.0 - 2.0 * rows, 1e-12, 1.0)
        graded[start:start + block] = (1.0 - np.exp(np.log(parity_sign) @ route_proj)) * 0.5

    graded = np.clip(graded, 1e-15, 1 - 1e-15)
    return graded[0] if was_1d else graded


def load_gnn_models(code_id, num_layers, h_dim, weights_z, weights_x,
                    gnn_batch, d_joint_z, d_joint_x):
    """Loads and compiles the Z and X memory joint GNNs."""
    if weights_z is None:
        weights_z = f"weights{code_id}/gnn_joint_Z_L{num_layers}_H{h_dim}_best.pt"
    if weights_x is None:
        weights_x = f"weights{code_id}/gnn_joint_X_L{num_layers}_H{h_dim}_best.pt"

    for basis, weights_path in (("Z", weights_z), ("X", weights_x)):
        if not os.path.exists(weights_path):
            raise SystemExit(f"Missing {basis}-memory GNN weights: {weights_path}")

    print(f"Loading GNNs (L{num_layers} H{h_dim}) into {device.type.upper()}...")

    models = {}
    for basis in ("Z", "X"):
        d_joint = d_joint_z if basis == "Z" else d_joint_x
        weights_path = weights_z if basis == "Z" else weights_x
        
        d_joint_coo = d_joint.tocoo()
        model = BipartiteGNN(
            n_det=d_joint.shape[0], n_fault=d_joint.shape[1],
            edges_det=d_joint_coo.row, edges_fault=d_joint_coo.col,
            h_dim=h_dim, num_layers=num_layers
        ).to(device)

        checkpoint_state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint_state)
        model.eval()

        # Build cached sparse adjacency eagerly to avoid first-batch latency
        model._ensure_sparse(device)

        # Apply hardware-specific compiler optimizations
        if device.type == "xpu":
            try:
                import intel_extension_for_pytorch as ipex
                model = ipex.optimize(model, dtype=amp_data_type)
            except Exception as e:
                print(f"[{basis}] IPEX optimization bypassed: {e}")
        elif device.type == "cuda":
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception:
                try:
                    model = torch.compile(model)
                except Exception as e:
                    print(f"[{basis}] torch.compile bypassed: {e}")

        models[basis] = model

    return models


def run_gnn_inference_pair(models, syndrome_batches, batch_size):
    """Executes batched GNN inference over both memory bases."""
    results = {}
    total_batches = sum(int(np.ceil(len(batch) / batch_size)) for batch in syndrome_batches.values())
    
    accel = device.type in ("cuda", "xpu")
    use_pinned = device.type == "cuda"
    old_threads = torch.get_num_threads()
    
    if not accel:
        torch.set_num_threads(min(os.cpu_count() or 1, 64))

    try:
        with tqdm(total=total_batches, desc="GNN Inference", leave=False) as pbar:
            with torch.inference_mode():
                for basis in ("Z", "X"):
                    model = models[basis]
                    syndrome_batch = np.ascontiguousarray(syndrome_batches[basis])
                    n_shots = syndrome_batch.shape[0]

                    batch_parts = []
                    for start in range(0, n_shots, batch_size):
                        end = min(start + batch_size, n_shots)
                        batch_np = syndrome_batch[start:end]

                        if use_pinned:
                            batch_cpu = torch.from_numpy(batch_np).pin_memory()
                            batch_tensor = batch_cpu.to(device, dtype=torch.float32, non_blocking=True)
                        else:
                            batch_tensor = torch.as_tensor(batch_np, dtype=torch.float32, device=device)

                        with torch.autocast(device_type=device.type, dtype=amp_data_type, enabled=use_amp):
                            logits = model(batch_tensor)

                        batch_parts.append(torch.sigmoid(logits).float())
                        pbar.update(1)

                    results[basis] = torch.cat(batch_parts).cpu().numpy()
    finally:
        if not accel:
            torch.set_num_threads(old_threads)
            
    return results


def _cache_path(prefix, chunk):
    return f"{prefix}_c{chunk:04d}.npz"


def save_priors_chunk(prefix, chunk, syndromes_z, observables_z,
                      syndromes_x, observables_x, priors_z, priors_x):
    """Persists a chunk of syndromes, observables, and predicted priors to disk."""
    path = _cache_path(prefix, chunk)
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True) 
    np.savez(path, syndromes_z=syndromes_z, observables_z=observables_z,
             syndromes_x=syndromes_x, observables_x=observables_x,
             priors_z=priors_z, priors_x=priors_x)
    return path


def load_priors_chunk(prefix, chunk, expected_chunk_size):
    """Loads a persisted chunk from disk and validates shapes."""
    path = _cache_path(prefix, chunk)
    if not os.path.exists(path):
        raise SystemExit(f"Missing prior cache file: {path}")
        
    with np.load(path) as cache:
        syn_z, prior_z = cache["syndromes_z"], cache["priors_z"]
        
        # Guard against chunk-size mismatch
        if syn_z.shape[0] != expected_chunk_size:
            raise SystemExit(f"Cache shape mismatch in {path}: expected {expected_chunk_size} shots, got {syn_z.shape[0]}")
            
        return (syn_z, cache["observables_z"], cache["syndromes_x"], 
                cache["observables_x"], prior_z, cache["priors_x"])


def release_memory():
    """Frees device memory between chunks to prevent fragmentation."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
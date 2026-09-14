#!/usr/bin/env python3
"""Per-shot classical decoders for the joint-GNN pipeline.

Applies GNN-predicted fault priors to BP+OSD, or runs static baseline decoding 
using BP+LSD or IBM Relay-BP.
"""

import os
from dataclasses import dataclass, field
import numpy as np

# Restrict BLAS threads to prevent multiprocessing CPU thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

RELAYBP_PRESETS = {
    "default": dict(gamma0=0.1, gamma_dist_interval=(-0.24, 0.66), num_sets=60, pre_iter=60, set_max_iter=60, stop_nconv=5, stopping_criterion="nconv"),
    "full": dict(gamma0=0.1, gamma_dist_interval=(-0.24, 0.66), num_sets=300, pre_iter=80, set_max_iter=60, stop_nconv=5, stopping_criterion="nconv"),
    "paper_bb144": dict(gamma0=0.125, gamma_dist_interval=(-0.161, 0.815), num_sets=601, pre_iter=80, set_max_iter=60, stop_nconv=5, stopping_criterion="nconv"),
    "paper_bb72": dict(gamma0=0.1, gamma_dist_interval=(-0.24, 0.66), num_sets=60, pre_iter=60, set_max_iter=60, stop_nconv=5, stopping_criterion="nconv"),
}


@dataclass
class WorkerConfig:
    dz: object
    dx: object
    dl_z: object
    dl_x: object
    backend: str
    bp_method: str
    max_iter: int
    osd_order: int
    lsd_order: int
    relay_preset: dict = field(default_factory=dict)
    d_joint: object = None
    dl_joint: object = None


class ShotDecoder:
    def __init__(self, cfg: WorkerConfig):
        self.backend = cfg.backend
        self.observable_matrices = {"Z": cfg.dl_z, "X": cfg.dl_x}
        self.check_matrices = {"Z": cfg.dz, "X": cfg.dx}
        
        # Base flat priors for initialization
        init_priors = {basis: np.full(matrix.shape[1], 0.01) 
                       for basis, matrix in self.check_matrices.items()}

        if self.backend == "joint_bposd":
            from ldpc import BpOsdDecoder
            self.joint_observable_matrices = {"Z": cfg.dl_joint["Z"], "X": cfg.dl_joint["X"]}
            self.joint_solvers = {basis: BpOsdDecoder(
                cfg.d_joint[basis], channel_probs=np.full(cfg.d_joint[basis].shape[1], 0.01),
                max_iter=cfg.max_iter, bp_method=cfg.bp_method,
                ms_scaling_factor=0, osd_method="osd_cs", osd_order=cfg.osd_order)
                for basis in ("Z", "X")}

        if self.backend in ("osd", "lsd"):
            if self.backend == "osd":
                from ldpc import BpOsdDecoder as DecoderCls
                kwargs = dict(osd_method="osd_cs", osd_order=cfg.osd_order, ms_scaling_factor=0)
            else:
                from ldpc import BpLsdDecoder as DecoderCls
                kwargs = dict(lsd_order=cfg.lsd_order)

            self.solvers = {
                "Z": DecoderCls(cfg.dz, channel_probs=init_priors["Z"], max_iter=cfg.max_iter, bp_method=cfg.bp_method, **kwargs),
                "X": DecoderCls(cfg.dx, channel_probs=init_priors["X"], max_iter=cfg.max_iter, bp_method=cfg.bp_method, **kwargs),
            }
            
        elif self.backend == "relay":
            try:
                import relay_bp
                self.relay_bp = relay_bp
            except ImportError as exc:
                raise SystemExit("relay_bp not installed. Run: pip install relay-bp") from exc
                
            self.relay_preset = dict(cfg.relay_preset)
            self.relay_runners = None
            self.relay_priors = None

    def _build_relay_runner(self, basis, priors):
        priors = np.clip(np.asarray(priors, dtype=np.float64), 1e-15, 1 - 1e-15)
        decoder = self.relay_bp.RelayDecoderF64(
            self.check_matrices[basis], error_priors=priors, **self.relay_preset)
        return self.relay_bp.ObservableDecoderRunner(
            decoder, self.observable_matrices[basis], include_decode_result=True)

    def _decode_classical(self, basis, syndrome, observed, prior):
        solver = self.solvers[basis]
        
        priors = np.clip(np.asarray(prior, dtype=np.float64), 1e-15, 1 - 1e-15)
        solver.update_channel_probs(priors)
        
        correction = solver.decode(np.asarray(syndrome).ravel())
        predicted = (self.observable_matrices[basis] @ correction) % 2
        ok = bool(np.array_equal(predicted.ravel(), np.asarray(observed).ravel()))
        
        return ok, bool(solver.converge)

    def _decode_joint(self, basis, syndrome, observed, prior):
        solver = self.joint_solvers[basis]
        priors = np.clip(np.asarray(prior, dtype=np.float64), 1e-15, 1 - 1e-15)
        solver.update_channel_probs(priors)
        correction = solver.decode(np.asarray(syndrome).ravel())
        predicted = (self.joint_observable_matrices[basis] @ correction) % 2
        ok = bool(np.array_equal(predicted.ravel(), np.asarray(observed).ravel()))
        return ok, bool(solver.converge)

    def _decode_relay(self, basis, syndrome, observed, prior):
        runner = self.relay_runners[basis]

        syndrome_uint8 = np.array(syndrome, dtype=np.uint8).reshape(1, -1)
        result = runner.decode_observables_detailed_batch(
            np.ascontiguousarray(syndrome_uint8), parallel=False, progress_bar=False)[0]
            
        correction = np.array(result.physical_decode_result.decoding, dtype=np.uint8).ravel()
        predicted = np.array(result.observables, dtype=np.uint8).ravel()
        syndrome_flat = np.array(syndrome).ravel()
        
        syndrome_matched = np.array_equal((self.check_matrices[basis] @ correction) % 2, syndrome_flat)
            
        if syndrome_matched:
            relay_success = bool(np.array_equal(predicted, np.array(observed).ravel()))
            return relay_success, "relay", True
                
        return False, "fail", False

    def decode_shot(self, shot):
        syn_z, obs_z, prior_z, syn_x, obs_x, prior_x = shot

        if self.backend == "joint_bposd":
            succ_z, conv_z = self._decode_joint("Z", syn_z, obs_z, prior_z)
            succ_x, conv_x = self._decode_joint("X", syn_x, obs_x, prior_x)
            return succ_z, "", conv_z, succ_x, "", conv_x

        if self.backend == "relay":
            prior_z, prior_x = np.asarray(prior_z), np.asarray(prior_x)
            if self.relay_runners is None or not (
                    np.array_equal(prior_z, self.relay_priors[0])
                    and np.array_equal(prior_x, self.relay_priors[1])):
                self.relay_runners = {
                    "Z": self._build_relay_runner("Z", prior_z),
                    "X": self._build_relay_runner("X", prior_x)
                }
                self.relay_priors = (prior_z, prior_x)
            
            succ_z, stat_z, conv_z = self._decode_relay("Z", syn_z, obs_z, prior_z)
            succ_x, stat_x, conv_x = self._decode_relay("X", syn_x, obs_x, prior_x)
            return (succ_z, stat_z, conv_z, succ_x, stat_x, conv_x)
            
        succ_z, conv_z = self._decode_classical("Z", syn_z, obs_z, prior_z)
        succ_x, conv_x = self._decode_classical("X", syn_x, obs_x, prior_x)
        return succ_z, "", conv_z, succ_x, "", conv_x


worker_decoder = None

def init_worker(cfg: WorkerConfig):
    global worker_decoder
    worker_decoder = ShotDecoder(cfg)

def decode_shot_pair(shot):
    return worker_decoder.decode_shot(shot)
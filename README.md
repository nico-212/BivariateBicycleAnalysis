# Bivariate Bicycle Codes: Decoding Pipeline and GNN Front-End

Code behind the thesis. It builds syndrome-extraction circuits for Bivariate
Bicycle (BB) codes, extracts a detector error model from the compiled circuit,
decodes with BP-OSD, BP-LSD or Relay-BP, and can decode from a
syndrome-conditioned prior produced by a graph neural network.

Two instances are implemented: `[[72,12,6]]` over six cycles and `[[144,12,12]]`
over twelve, under circuit-level depolarizing noise.

## Files

Entry points, one per stage:

- `offline_stage.py` builds the circuits, the detector error models, the
  decoding matrices, the fault priors and the routing matrices.
- `train_joint_gnn.py` fits one joint-graph GNN per code and memory basis.
- `train_split_gnn.py` fits the split-graph control models.
- `online_stage.py` decodes syndromes and appends one row per run to a CSV.

Shared modules:

- `gnn_model.py` holds the `BipartiteGNN` model class.
- `gnn_runtime.py` runs batched inference and projects the joint marginals onto
  the split fault classes.
- `decoders.py` builds the decoder for the `osd`, `lsd`, `relay` and
  `joint_bposd` backends.
- `generate_plots.py` draws the evaluation figures from `results.csv`.
- `diamonds.py` extracts the reference points of Fig. 2A of Bravyi et al. from
  the vector graphics of the paper. It needs the paper PDF as its argument and
  is not part of the decoding pipeline.

Data and tests:

- `results.csv` holds the decoding runs behind the thesis, one row per run. Its
  last column names the sweep file the row was collected from.
- `requirements.txt` lists the packages the code imports.
- `tests/` holds the two test files described below.

## Running one operating point

The three stages run in sequence. For `[[72,12,6]]` at p = 0.004:

```
python offline_stage.py --code 72 --p 0.004 --cycles 6
python train_joint_gnn.py --code 72 --p 0.004 --cycles 6 --basis Z
python train_joint_gnn.py --code 72 --p 0.004 --cycles 6 --basis X
python online_stage.py --code 72 --p 0.004 --cycles 6 --decoder gnn --backend osd \
    --num-layers 4 --h-dim 16 --max-iter 100 --osd-order 3 --trials 10000
```

That last command runs the configuration of the `h_dim = 16` GNN row of the
operating-point table in Chapter 5, on 10,000 shots rather than the 5,000 behind
that row.

The offline stage runs once per code and rate, and the training once per code
and memory basis. A trained model is not tied to the rate of the artifact it
came from, because every batch is rescaled to its own rate. Once those two
stages are done, the online stage can be repeated on its own.

The baseline is the split BP-OSD decoder with static priors:

```
python online_stage.py --code 72 --p 0.004 --cycles 6 --decoder bposd \
    --backend osd --max-iter 10000 --osd-order 7 --trials 10000
```

`--split` switches the GNN arms to the split-graph model, `--max-iter` and
`--osd-order` set the solver budget, and `--seed-z` and `--seed-x` set the
sampler seeds. Every flag is documented by `--help`.

Pass `--max-iter` and `--osd-order` explicitly on every command. The defaults
are not neutral: `--max-iter` falls back to 1000 for `--decoder bposd` and to 0
for `--decoder gnn`, where 0 means the full column count in BP rounds, and
`--osd-order` falls back to 7. An omitted flag therefore runs a different
configuration than the one you meant.

## Artifacts and checkpoints

`offline_stage.py` writes `{n}_{k}_{d}_cycle_{c}/p_{p}/` with twenty files: the
split circuits and DEMs (`circuit_Z.stim`, `circuit_X.stim`, `dem_Z.dem`,
`dem_X.dem`), the decoding matrices (`DZ.npz`, `DX.npz`, `DL_Z.npz`,
`DL_X.npz`), the priors (`probZ.npy`, `probX.npy`), the joint counterparts
(`circuit_joint_{basis}.stim`, `dem_joint_{basis}.dem`, `D_joint_{basis}.npz`,
`prob_joint_{basis}.npy`) and the routing matrices (`R_Z.npz`, `R_X.npz`).

Checkpoints are written to `weights{code}/`, one per basis and configuration,
named `gnn_joint_{basis}_L{L}_H{h}_s{seed}_best.pt`, with a snapshot every five
epochs. The split controls go to `split_weights{code}/`. A run is identified by
its artifact directory and its seeds, and an identifier that already exists is
skipped, so an interrupted sweep resumes where it stopped. The online stage
reads the seed-0 joint checkpoints unless `--weights-z` and `--weights-x` give
other paths. With `--split` it reads the split checkpoints in `split_weights{code}/`,
named `gnn_split_{basis}_L{L}_H{h}_s0_best.pt`.

## Results and figures

`online_stage.py --out <file>` appends one row per run, so several decoder arms
can be collected into the same file. `generate_plots.py` reads `results.csv`
from the working directory and writes the evaluation figure PDFs next to it:

```
python generate_plots.py
```

The training table and the training-loss figure are not drawn from
`results.csv`. Their numbers come from the per-run `_log.csv` and `_meta.json`
files that the two trainers write next to the checkpoints in `weights{code}/`
and `split_weights{code}/`.

## Tests

```
python tests/test_code_invariants.py
python tests/test_offline_stage.py
```

`test_code_invariants.py` rebuilds the check matrices from the shift operators
and checks that x and y commute, that `AB + BA = 0` and `H_X H_Z^T = 0` over
F2, that `n - rank(H_X) - rank(H_Z) = 12` for both codes, and that both check
graphs have girth 6.

`test_offline_stage.py` builds the `[[72,12,6]]` artifacts at p = 0.003 with two
cycles in a temporary directory and checks that all twenty files are present and
that the routing identities hold. It leaves nothing behind.

## Portability

The pipeline picks its device at import time in the order XPU, CUDA, CPU, so it
runs on an Intel Arc, an NVIDIA card or a plain CPU without configuration
changes. Every stage prints the device it chose, and the results CSV records it
in its `device` column. All runs in this thesis were produced on the XPU.

The two fallback paths were checked as well, on a clean CPython 3.14 environment
on the CPU and on a Colab Tesla T4 under torch 2.11.0+cu130. On the T4 the
device check, both test files, the offline stage and a 200-shot GNN run all
complete on `cuda:0`. The numbers behind this thesis come from the XPU only.

The two test files check a new machine: `tests/test_code_invariants.py` rebuilds
the code invariants, and `tests/test_offline_stage.py` builds an artifact set and
verifies the routing.

```
python tests/test_code_invariants.py
python tests/test_offline_stage.py
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

An accelerator needs the matching PyTorch build: the CUDA wheel from the index
on pytorch.org, or the Intel XPU wheel for Arc. On Windows, `numpy`, `scipy`,
`torch`, `ldpc`, `stim` and `bposd` have wheels but `relay-bp` does not, so the
Relay-BP arm needs WSL2 (or `--backend relay` has to be skipped). The worker
pool also uses the spawn start method there, which re-imports PyTorch in every
worker; WSL2 avoids both issues.

## Requirements

```
pip install -r requirements.txt
```

The pins are the versions of Table A.1 of the thesis, resolved under CPython
3.14.7. Another Python version may not have a wheel for every pin. Any
environment manager works. `uv`, which built the environment behind the thesis,
can also fetch the interpreter for you: `uv venv --python 3.14`.

`torch==2.13.0` as published on PyPI is the CUDA build, so on Linux it also
installs about thirty NVIDIA and Triton packages that a CPU-only machine cannot
use. Install the CPU wheel first, and the pinned line is then already satisfied:

```
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The Intel index gives the XPU wheel for Arc. No Intel extension is needed.

`relay-bp` is commented out in `requirements.txt` because its wheels are
manylinux only. Uncomment it on Linux, or run the Relay-BP arm under WSL2 on
Windows.

# Bivariate Bicycle Code Analysis

Code for simulating and experimenting with Bivariate Bicycle (BB) quantum error correction codes. 

## Overview
This project evaluates the performance of BB codes (like the [[144, 12, 12]] code) under circuit-level noise. To maximize simulation speed, the pipeline is split:

* **`offline_stage.py`**: Uses `Stim` to generate the memory circuits, injects physical noise (depolarizing, X, and Z errors), and extracts the sparse matrices.
* **`online_stage.py`**: Runs parallel Monte Carlo trials, using `BP-OSD` to decode errors and calculate the per-cycle logical error rate ($p_L$).

For each [[n,k,d]]-code and error rate p, offline_stage.py has to be run once. The results get saved in a directory for online_stage.py to access. online_stage.py saves its result in a .csv file.
## Project Goals
1. **Implement**: Build the physical circuits for the BB codes based on the original IBM polynomial constructions.
2. **Analyze & Improve**: Analyze the speed and error rate of the original BB pipeline presented in the paper and compare it to alternative implementations (like MWPM instead of BP-OSD for decoding)

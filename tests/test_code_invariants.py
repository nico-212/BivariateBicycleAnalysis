#!/usr/bin/env python3
"""Code invariants of the two BB instances used in the thesis.

Rebuilds the check matrices from the shift operators and asserts the
properties the pipeline relies on: the shift operators commute, the polynomial
matrices satisfy A B + B A = 0, the CSS condition HX HZ^T = 0 holds, the
dimension comes out as n - rank(HX) - rank(HZ) = k, and both check graphs have
girth six. It also reports the row and column weights
and the largest column-column overlap of the check matrices.

Run with:  python tests/test_code_invariants.py
"""

from collections import deque
from itertools import combinations

import numpy as np

# l, m, n, k, d of the two instances
CASES = [(6, 6, 72, 12, 6), (12, 6, 144, 12, 12)]


def shift(n):
    return np.roll(np.eye(n, dtype=np.uint8), 1, axis=1)


def power(M, k):
    R = np.eye(M.shape[0], dtype=np.uint8)
    for _ in range(k):
        R = (R @ M) % 2
    return R


def build(l, m):
    """Shift operators x, y and the polynomial matrices A, B of one instance."""
    x = np.kron(shift(l), np.eye(m, dtype=np.uint8)) % 2
    y = np.kron(np.eye(l, dtype=np.uint8), shift(m)) % 2
    A = (power(x, 3) ^ power(y, 1) ^ power(y, 2)) % 2
    B = (power(y, 3) ^ power(x, 1) ^ power(x, 2)) % 2
    return x, y, A, B


def rank_gf2(M):
    A = M.copy().astype(np.uint8) % 2
    n_rows, n_cols = A.shape
    r = 0
    for c in range(n_cols):
        pivot = next((i for i in range(r, n_rows) if A[i, c]), None)
        if pivot is None:
            continue
        A[[r, pivot]] = A[[pivot, r]]
        for i in range(n_rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        r += 1
        if r == n_rows:
            break
    return r


def girth(H):
    """Shortest cycle of the Tanner graph of H. Bipartite, so the length is even."""
    H = (H % 2).astype(bool)
    n_rows, n_cols = H.shape
    adj = {("c", i): [("v", j) for j in np.flatnonzero(H[i])] for i in range(n_rows)}
    for j in range(n_cols):
        adj[("v", j)] = [("c", i) for i in np.flatnonzero(H[:, j])]
    best = 10 ** 9
    for start in adj:
        dist = {start: 0}
        parent = {start: None}
        q = deque([start])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    q.append(w)
                elif parent[u] != w:
                    best = min(best, dist[u] + dist[w] + 1)
    return best


def test_code_invariants():
    for l, m, n, k, d in CASES:
        x, y, A, B = build(l, m)
        assert n == 2 * l * m, f"n={n} for l={l}, m={m}"

        HX = np.concatenate([A, B], axis=1) % 2
        HZ = np.concatenate([B.T, A.T], axis=1) % 2

        assert np.array_equal((x @ y) % 2, (y @ x) % 2), "shift operators do not commute"
        assert not np.any((A @ B + B @ A) % 2), "A B + B A is not zero"
        assert not np.any((HX @ HZ.T) % 2), "HX HZ^T is not zero"

        rank_x, rank_z = rank_gf2(HX), rank_gf2(HZ)
        assert 2 * l * m - rank_x - rank_z == k, "code dimension is not k"

        assert girth(HX) == 6, "girth of the X check graph is not six"
        assert girth(HZ) == 6, "girth of the Z check graph is not six"

        overlap_x = max(int(np.count_nonzero(HX[:, i] & HX[:, j]))
                        for i, j in combinations(range(HX.shape[1]), 2))
        overlap_z = max(int(np.count_nonzero(HZ[:, i] & HZ[:, j]))
                        for i, j in combinations(range(HZ.shape[1]), 2))
        print(f"[[{n},{k},{d}]]  rank {rank_x}/{rank_z}  "
              f"row weights {sorted(set(HX.sum(1).tolist()))}  "
              f"col weights {sorted(set(HX.sum(0).tolist()))}  "
              f"max col overlap X {overlap_x} Z {overlap_z}")

    print("ALL CODE-INVARIANT CHECKS PASSED")


if __name__ == "__main__":
    test_code_invariants()

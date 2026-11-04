"""Diagnostic: does a fresh rng state explain the submission's 0.0172 vs the study's 0.0147?"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdmbench import core
import strat_study as ss

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
z = np.load(os.path.join(root, "workloads/sift10m/analytics.npz"))
Q = np.ascontiguousarray(z["queries"][:100].astype(np.float32)); radii = z["radii"].astype(np.float32)
truth = z["truth"][:100]
X, _ = core.load_dataset(os.path.expanduser("~/vdm/data"), "sift10m", 10_000_000)
X = np.ascontiguousarray(X.astype(np.float32), dtype=np.float32)
N = len(X); nlist = int(min(16384, N // 1000)); S = 100_000
C, assign, sizes, order, off, grid, qs = ss.load_or_build_cells(X, nlist, os.path.join(root, "results_remote/cells_sift10m.npz"))
dQX = np.einsum("ij,ij->i", X, X); dQ = np.sqrt(core.pairwise(Q, C))

def neyman(rng, pref=0):
    est = np.zeros_like(truth, dtype=np.float64)
    for _ in range(pref):                      # burn rng like the study's uniform loops do
        rng.choice(N, 40000, replace=False)
    for i, q in enumerate(Q):
        q2 = float(q @ q)
        Ws = []
        for r in radii:
            p = ss.cell_hit_prob(dQ[i], float(r), grid, qs)
            pc = np.clip(p, 1e-3, 1 - 1e-3)
            w = sizes * np.sqrt(pc * (1 - pc)); Ws.append(w / w.sum())
        alloc = ss.allocate(np.mean(Ws, axis=0), S)
        cells, idx = ss.sample_idx(alloc, sizes, order, off, rng)
        dd = np.sqrt(np.maximum(q2 + dQX[idx] - 2.0 * X[idx] @ q, 0.0))
        for j, r in enumerate(radii):
            hit = np.bincount(cells, weights=(dd <= r).astype(np.float64), minlength=nlist)
            est[i, j] = float((sizes * hit / np.maximum(alloc, 1)).sum())
    rel = np.abs(est - truth) / np.maximum(truth, 1)
    return float(rel.mean()), float(np.percentile(rel, 95))

for seed in (0, 1, 2):
    print("fresh rng seed", seed, neyman(np.random.RandomState(seed)), flush=True)
print("study-like burned rng seed0", neyman(np.random.RandomState(0), pref=100), flush=True)

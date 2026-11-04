"""Is the predicted allocation at the ceiling? Oracle allocation across seeds.

`strat_prefix.py --oracle` compares one model-based draw against one oracle-allocation draw, which
is two noisy numbers.  This script computes the true per-cell hit rates ONCE (one pass over the
collection per query, all radii at once), caches them, and then runs the oracle allocation over
several seeds so the comparison has a spread on both sides.

If the model-based Neyman allocation matches allocation at the true hit rates, the surrogate is
not the bottleneck and the remaining error is the irreducible variance of cell-stratified
sampling at that budget -- which is what the paper claims.

Usage: python scripts/strat_oracle_seeds.py --cache results_remote/cells_sift10m.npz --seeds 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdmbench import core  # noqa: E402
import strat_study as ss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift10m")
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--data-root", default=os.path.expanduser("~/vdm/data"))
    ap.add_argument("--budget", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--truth-cache", default=None)
    a = ap.parse_args()

    t0 = time.time()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    z = np.load(os.path.join(root, "workloads", a.dataset, "analytics.npz"))
    Q = np.ascontiguousarray(z["queries"][:100].astype(np.float32))
    radii = z["radii"].astype(np.float32)
    truth = z["truth"][:100]
    X, _ = core.load_dataset(a.data_root, a.dataset, a.n)
    X = np.ascontiguousarray(X.astype(np.float32), dtype=np.float32)
    N, nlist = len(X), int(min(16384, len(X) // 1000))
    C, assign, sizes, order, off, grid, qs = ss.load_or_build_cells(X, nlist, a.cache)
    dQX = np.einsum("ij,ij->i", X, X)
    dQ = np.sqrt(core.pairwise(Q, C))
    nq, nr = len(Q), len(radii)

    tc = a.truth_cache or os.path.join(root, "results_remote", f"oracle_cells_{a.dataset}.npz")
    if os.path.exists(tc):
        P = np.load(tc)["P"]
        print(f"true per-cell hit rates from cache {tc}", flush=True)
    else:
        P = np.zeros((nq, nr, nlist), dtype=np.float32)
        for i, q in enumerate(Q):
            q2 = float(q @ q)
            for s0 in range(0, N, 400_000):
                xc = X[s0:s0 + 400_000]
                d = np.sqrt(np.maximum(q2 + dQX[s0:s0 + len(xc)] - 2.0 * xc @ q, 0.0))
                ac = assign[s0:s0 + len(xc)]
                for j, r in enumerate(radii):
                    P[i, j] += np.bincount(ac, weights=(d <= r).astype(np.float32),
                                           minlength=nlist)
            if i % 10 == 0:
                print(f"  truth cells {i + 1}/{nq}  [{time.time() - t0:.0f}s]", flush=True)
        P /= np.maximum(sizes, 1)[None, None, :].astype(np.float32)
        os.makedirs(os.path.dirname(tc) or ".", exist_ok=True)
        np.savez_compressed(tc, P=P)
        print(f"true per-cell hit rates cached to {tc} [{time.time() - t0:.0f}s]", flush=True)

    S = a.budget
    rows = []
    for seed in range(a.seeds):
        rng = np.random.RandomState(seed)
        est = np.zeros_like(truth, dtype=np.float64)
        for i in range(nq):
            W = P[i].mean(axis=0).astype(np.float64)
            alloc = ss.allocate(W, S)
            cells, idx = ss.sample_idx(alloc, sizes, order, off, rng)
            dd = np.sqrt(np.maximum(float(Q[i] @ Q[i]) + dQX[idx] - 2.0 * X[idx] @ Q[i], 0.0))
            for j, r in enumerate(radii):
                hit = np.bincount(cells, weights=(dd <= r).astype(np.float64), minlength=nlist)
                est[i, j] = float((sizes * hit / np.maximum(alloc, 1)).sum())
        rel = np.abs(est - truth) / np.maximum(truth, 1)
        rows.append((float(rel.mean()), float(np.percentile(rel, 95))))
        print(f"seed {seed}: oracle-alloc rel={rows[-1][0]:.4f} p95={rows[-1][1]:.4f}", flush=True)

    r = np.array([x[0] for x in rows]); p = np.array([x[1] for x in rows])
    out = {"budget": S, "seeds": a.seeds,
           "rel_mean": float(r.mean()), "rel_sd": float(r.std()),
           "p95_mean": float(p.mean()), "p95_sd": float(p.std())}
    json.dump(out, open(os.path.join(root, "results", f"strat_oracle_{a.dataset}.json"), "w"),
              indent=1)
    print(f"\noracle-alloc rel {r.mean():.4f} ± {r.std():.4f}   p95 {p.mean():.4f} ± {p.std():.4f}")


if __name__ == "__main__":
    main()

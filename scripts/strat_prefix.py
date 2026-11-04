"""Unified estimator at scale: exact IVF prefix + Neyman-allocated residual.

Why this experiment exists
--------------------------
At 100k the two-phase estimator (scan the nearest inverted lists exactly, sample the residual
uniformly) beats uniform sampling by 18x at lower cost.  At 10M that same recipe loses at
matched cost, because a fixed 4,000-sample residual covers 0.046% of an 8.75M-point residual
(experiments/range10m_frontier_2026-11-04.md), and the earlier 10M sweep "fixed" the split in
the wrong place: it probed up to 25% of the lists exactly (2.5M distance evaluations) and left
the residual estimator starved.

`scripts/strat_study.py` showed the missing half: with the *same* budget, moving samples from
a uniform residual to a Neyman-allocated one (allocation from predicted per-cell hit rate
p_c ~= F_c(sqrt(r^2 - d_c^2))) is worth 1.6x at matched cost and 2.5x at matched accuracy.

This script measures the two levers *jointly*, so the estimator is one method at every scale
with two cost-controlled parameters:

  * KP  -- how many inverted lists to count exactly (cost ~ KP * list population),
  * S   -- how many residual samples to draw, allocated by Neyman instead of uniformly.

The interesting result is not any single row but the shape: at 10M the optimum moves from
"probe a lot, sample a little" to "probe a little, allocate a lot", i.e. the plan family does
not flip, only its parameters do.

Estimator (unbiased for any positive allocation):

    est = sum_{c in prefix} exact_c  +  sum_{c not probed} size_c * hits_c / s_c

Usage:
  python scripts/strat_prefix.py --dataset sift10m --n 10000000 --budget 100000 \
      --kps 0,16,32,64,128 --cache results_remote/cells_sift10m.npz
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


def log(msg, t0=None):
    print(msg if t0 is None else f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift10m")
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--data-root", default=os.path.expanduser("~/vdm/data"))
    ap.add_argument("--budget", type=int, default=100_000)
    ap.add_argument("--kps", default="0,16,32,64,128")
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache-only", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    z = np.load(os.path.join(root, "workloads", a.dataset, "analytics.npz"))
    Q = np.ascontiguousarray(z["queries"][:a.queries].astype(np.float32))
    radii = z["radii"].astype(np.float32)
    truth = z["truth"][:a.queries]
    X, _ = core.load_dataset(a.data_root, a.dataset, a.n)
    X = np.ascontiguousarray(X.astype(np.float32), dtype=np.float32)
    N, nq, nr = len(X), len(Q), len(radii)
    log(f"n={N} q={nq} radii={np.round(radii,1).tolist()} counts={truth.mean(0).round(0).tolist()}", t0)

    nlist = int(min(16384, max(256, N // 1000)))
    C, assign, sizes, order, off, grid, qs = ss.load_or_build_cells(X, nlist, a.cache)
    if a.cache_only:
        log("cache built; exiting (--cache-only)", t0)
        return
    dQX = np.einsum("ij,ij->i", X, X)
    dQ = np.sqrt(core.pairwise(Q, C))
    rng = np.random.RandomState(0)
    S = a.budget

    def score(est, cost):
        rel = np.abs(est - truth) / np.maximum(truth, 1)
        return {"rel": float(rel.mean()), "p95": float(np.percentile(rel, 95)), "cost": float(cost)}

    res = {}

    # ---- references: uniform, and strat-neyman with no prefix ----
    est = np.zeros_like(truth, dtype=np.float64)
    for i, q in enumerate(Q):
        samp = rng.choice(N, S, replace=False)
        dd = np.sqrt(np.maximum(float(q @ q) + dQX[samp] - 2.0 * X[samp] @ q, 0.0))
        for j, r in enumerate(radii):
            est[i, j] = N * float((dd <= r).sum()) / S
    res[f"uniform S={S}"] = score(est, S)
    log(f"uniform S={S}: {res[f'uniform S={S}']}", t0)

    def run_prefix(KP, alloc="neyman"):
        est = np.zeros_like(truth, dtype=np.float64)
        cost = 0.0
        for i, q in enumerate(Q):
            q2 = float(q @ q)
            if KP > 0:
                order_l = np.argsort(dQ[i])[:KP]
                pid = np.concatenate([order[off[c]:off[c + 1]] for c in order_l])
                prefix_cells = np.zeros(nlist, dtype=bool)
                prefix_cells[order_l] = True
            else:
                pid = np.empty(0, dtype=np.int64)
                prefix_cells = np.zeros(nlist, dtype=bool)
            rest_n = N - len(pid)
            Sr = max(1, S - len(pid))
            ps = [ss.cell_hit_prob(dQ[i], float(r), grid, qs) for r in radii]
            if alloc == "neyman":
                Ws = []
                for p in ps:
                    pc = np.clip(p, 1e-3, 1 - 1e-3)
                    Ws.append(sizes * np.sqrt(pc * (1 - pc)))
                W = np.mean([w / w.sum() for w in Ws], axis=0)
            else:
                W = sizes.astype(np.float64)
            W = np.where(prefix_cells, 0.0, W)
            alloc_arr = ss.allocate(W, Sr)
            cells, idx = ss.sample_idx(alloc_arr, sizes, order, off, rng)
            dp = np.sqrt(np.maximum(q2 + dQX[pid] - 2.0 * X[pid] @ q, 0.0))
            ds = np.sqrt(np.maximum(q2 + dQX[idx] - 2.0 * X[idx] @ q, 0.0))
            cost += len(pid) + len(idx)
            for j, r in enumerate(radii):
                exact = int((dp <= r).sum())
                hit = np.bincount(cells, weights=(ds <= r).astype(np.float64), minlength=nlist)
                est[i, j] = exact + float((sizes * hit / np.maximum(alloc_arr, 1)).sum())
        return est, cost

    for KP in [int(x) for x in a.kps.split(",")]:
        for alloc in ("uniform", "neyman"):
            t1 = time.time()
            est, cost = run_prefix(KP, alloc)
            key = f"prefix KP={KP} + {alloc} residual (S_res={S - KP * int(sizes.mean())})"
            res[key] = score(est, cost)
            log(f"{key}: {res[key]} ({time.time()-t1:.0f}s)", t0)

    if a.oracle:
        # allocation from the TRUE per-cell hit rate, same budget, no prefix: the ceiling that
        # any cell-based allocation could reach (not an upper bound on other estimator families).
        def true_p_all(q):
            """Exact per-cell hit rate for every radius in ONE pass over the collection."""
            q2 = float(q @ q)
            H = np.zeros((nlist, nr), dtype=np.float64)
            for s0 in range(0, N, 400_000):
                xc = X[s0:s0 + 400_000]
                d = np.sqrt(np.maximum(q2 + dQX[s0:s0 + len(xc)] - 2.0 * xc @ q, 0.0))
                ac = assign[s0:s0 + len(xc)]
                for j, r in enumerate(radii):
                    H[:, j] += np.bincount(ac, weights=(d <= r).astype(np.float64),
                                           minlength=nlist)
            return H / np.maximum(sizes, 1)[:, None]

        est = np.zeros_like(truth, dtype=np.float64)
        for i, q in enumerate(Q):
            q2 = float(q @ q)
            W = true_p_all(q).mean(axis=1)
            alloc_arr = ss.allocate(W, S)
            cells, idx = ss.sample_idx(alloc_arr, sizes, order, off, rng)
            dd = np.sqrt(np.maximum(q2 + dQX[idx] - 2.0 * X[idx] @ q, 0.0))
            for j, r in enumerate(radii):
                hit = np.bincount(cells, weights=(dd <= r).astype(np.float64), minlength=nlist)
                est[i, j] = float((sizes * hit / np.maximum(alloc_arr, 1)).sum())
        res[f"oracle-alloc S={S}"] = score(est, S)
        log(f"oracle-alloc S={S}: {res[f'oracle-alloc S={S}']}", t0)

    out = a.out or os.path.join(root, "results", f"strat_prefix_{a.dataset}_{N}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"setup": {"n": N, "nlist": nlist, "budget": S,
                             "radii": radii.tolist(),
                             "counts": truth.mean(0).round(1).tolist()},
                   "results": res}, f, indent=1)
    log(f"wrote {out}", t0)
    print(f"\n{'variant':58s} {'rel.err':>9s} {'p95':>9s} {'cost':>10s}")
    for k, v in res.items():
        print(f"{k:58s} {v['rel']:9.4f} {v['p95']:9.4f} {v['cost']:10.2e}")


if __name__ == "__main__":
    main()

"""Where should the samples go? Index-stratified (Neyman) range estimation at scale.

At 100k the exact IVF prefix is the cheap way to remove estimator variance.  At 10M it is not:
probing 12.5% of the collection costs 1.25M distance evaluations per query and still leaves an
8.75M-point residual, so plain uniform sampling wins at matched cost
(experiments/range10m_frontier_2026-11-04.md).

This script tests the structural alternative: keep the *same* sample budget, but stratify the
sample by IVF cell and allocate it with Neyman's rule using a per-cell hit probability predicted
from information the index already paid for -- the query-to-centroid distance plus the
within-cell radius quantiles.  The estimator is a plain stratified estimator

    est = sum_c size_c * hits_c / s_c

which is unbiased for *any* positive allocation, so allocation is variance-only: a bad model
costs efficiency, never correctness.

Variants (all at the same budget of S distance evaluations per query):
  uniform       -- one uniform sample; all radii read off it (today's 10M plan)
  strat-prop    -- allocation proportional to cell size (sanity check: must look like uniform)
  strat-neyman  -- allocation propto size_c * sqrt(p_c (1-p_c))
  strat-blend a -- mix of size and size*sqrt(p(1-p)); robust to an over-confident model
  strat-perrad  -- per-radius Neyman with S/len(radii) each (total cost is still S)
  strat-oracle  -- allocation from the *true* per-cell hit rate (upper bound on the allocation)

Also reports the calibration of the predicted hit probability and of the oracle gap.

Usage: python scripts/strat_study.py --dataset sift10m --n 10000000
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


def log(msg, t0=None):
    print(msg if t0 is None else f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_cells(X, nlist, seed=0):
    """K-means cells + per-cell within-cell radius quantiles (all paid once at build time)."""
    import faiss
    t = time.time()
    km = faiss.Kmeans(X.shape[1], nlist, niter=10, verbose=False, seed=seed)
    km.train(np.ascontiguousarray(X[: min(400_000, len(X))]))
    C = km.centroids.astype(np.float32)
    qflat = faiss.IndexFlatL2(X.shape[1])
    qflat.add(C)
    N = len(X)
    assign = np.empty(N, dtype=np.int32)
    for s0 in range(0, N, 200_000):
        _, idx = qflat.search(np.ascontiguousarray(X[s0:s0 + 200_000]), 1)
        assign[s0:s0 + len(idx)] = idx.ravel()
    log(f"kmeans nlist={nlist} + assign {time.time() - t:.0f}s", t)

    sizes = np.bincount(assign, minlength=nlist).astype(np.int64)
    order = np.argsort(assign, kind="stable")
    off = np.zeros(nlist + 1, dtype=np.int64)
    np.cumsum(sizes, out=off[1:])

    qs = np.linspace(0.0, 1.0, 33)
    tvec = np.empty(N, dtype=np.float32)
    for s0 in range(0, N, 200_000):
        xc = np.asarray(X[s0:s0 + 200_000], dtype=np.float32)
        dd = xc - C[assign[s0:s0 + len(xc)]]
        tvec[s0:s0 + len(xc)] = np.sqrt(np.einsum("ij,ij->i", dd, dd))
    tord = tvec[order]
    grid = np.zeros((nlist, len(qs)), dtype=np.float32)
    for c in range(nlist):
        seg = tord[off[c]:off[c + 1]]
        grid[c] = np.quantile(seg, qs) if len(seg) > 4 else (seg.mean() if len(seg) else 0.0)
    log(f"within-cell radius quantiles {time.time() - t:.0f}s", t)
    return C, assign, sizes, order, off, grid, qs


def load_or_build_cells(X, nlist, cache, seed=0):
    """Build the cell structure once, then reuse it (k-means on 10M costs ~13 minutes)."""
    if cache and os.path.exists(cache):
        z = np.load(cache)
        log(f"cells loaded from cache {cache}")
        return (z["C"], z["assign"], z["sizes"], z["order"], z["off"], z["grid"], z["qs"])
    out = build_cells(X, nlist, seed=seed)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        (C, assign, sizes, order, off, grid, qs) = out
        np.savez(cache, C=C, assign=assign, sizes=sizes, order=order, off=off,
                 grid=grid, qs=qs)
        log(f"cells cached to {cache}")
    return out


def cell_hit_prob(dq, r, grid, qs):
    """p_c ~= F_c(sqrt(r^2 - d_c^2)) under the spherical-cell approximation d(q,x)^2 ~= d_c^2 + t^2."""
    nc = len(dq)
    rad = np.sqrt(np.maximum(r * r - dq * dq, 0.0))
    idx = np.clip((grid <= rad[:, None]).sum(axis=1), 1, len(qs) - 1)
    ar = np.arange(nc)
    lo, hi = grid[ar, idx - 1], grid[ar, idx]
    w = np.where(hi > lo, (rad - lo) / np.maximum(hi - lo, 1e-9), 1.0)
    p = qs[idx - 1] + np.clip(w, 0.0, 1.0) * (qs[idx] - qs[idx - 1])
    return np.clip(np.where(rad <= 0.0, 0.0, p), 0.0, 1.0)


def cell_hit_prob_angle(dq, r, grid, qs, dim):
    """Angle-aware per-cell hit probability: keeps the cos(theta) term the spherical model drops.

    d(q,x)^2 = d_c^2 + t^2 - 2 d_c t cos(theta).  The spherical model assumes cos(theta) = 0; for
    isotropic points around the centroid cos(theta) has mean 0 and sd 1/sqrt(dim), so

        p_c = E_t[ Phi( (r^2 - d_c^2 - t^2) / (2 d_c t / sqrt(dim)) ) ]

    with t integrated over the cell's own radius quantiles (8-point quadrature).  Measured on
    SIFT10M against the exact per-cell rates: correlation 0.954 vs 0.847, RMSE 0.046 vs 0.081,
    and the model-only count error drops from 48% to 31%.
    """
    from scipy.special import erf
    nc = len(dq)
    qlev = np.linspace(0.05, 0.95, 8)
    k = np.clip(np.searchsorted(qs, qlev) - 1, 0, len(qs) - 2)
    frac = (qlev - qs[k]) / np.maximum(qs[k + 1] - qs[k], 1e-9)
    ts = grid[:, k] * (1.0 - frac)[None, :] + grid[:, k + 1] * frac[None, :]
    d = dq[:, None]
    num = r * r - d * d - ts * ts
    den = np.maximum(2.0 * d * ts / np.sqrt(dim), 1e-6)
    p = 0.5 * (1.0 + erf((num / den) / np.sqrt(2.0)))
    p = np.where(d <= 1e-9, (ts <= r).mean(axis=1)[:, None], p)
    return np.clip(p.mean(axis=1), 0.0, 1.0)


def allocate(w, S, min_per_cell=1):
    w = np.asarray(w, dtype=np.float64)
    s = np.zeros_like(w, dtype=np.int64)
    s[w > 0] = min_per_cell
    rem = S - int(s.sum())
    if rem <= 0:
        return s
    ww = w / w.sum()
    add = np.floor(rem * ww).astype(np.int64)
    s += add
    left = S - int(s.sum())
    if left > 0:
        s[np.argsort(-(rem * ww - add))[:left]] += 1
    return s


def sample_idx(alloc, sizes, order, off, rng):
    """Per-cell draw WITH replacement inside each cell segment (unbiased, and vectorised).

    Finite-population correction is negligible (list population ~1,000, allocation ~10), and
    duplicates cost nothing in the metric: the same distance evaluaton is simply repeated.
    """
    cells = np.flatnonzero(alloc > 0)
    if len(cells) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    ctx = np.repeat(cells, alloc[cells])
    starts = np.repeat(off[cells], alloc[cells])
    lens = np.repeat(sizes[cells], alloc[cells]).astype(np.float64)
    pos = starts + (rng.random_sample(len(ctx)) * lens).astype(np.int64)
    np.minimum(pos, np.repeat(off[cells + 1], alloc[cells]) - 1, out=pos)
    return ctx, order[pos]          # order[pos] maps a cell-segment position to the point id


def run(dataset, n, data_root, seed=0, budgets=(40_000, 100_000), out=None, queries=100,
        cache=None):
    t0 = time.time()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    z = np.load(os.path.join(root, "workloads", dataset, "analytics.npz"))
    Q = np.ascontiguousarray(z["queries"][:queries].astype(np.float32))
    radii = z["radii"].astype(np.float32)
    truth = z["truth"][:queries]
    X, _ = core.load_dataset(data_root, dataset, n)
    X = np.ascontiguousarray(X.astype(np.float32), dtype=np.float32)
    N, nq, nr = len(X), len(Q), len(radii)
    log(f"n={N} q={nq} radii={np.round(radii,1).tolist()} mean counts={truth.mean(0).round(0).tolist()}", t0)

    nlist = int(min(16384, max(256, N // 1000)))
    C, assign, sizes, order, off, grid, qs = load_or_build_cells(X, nlist, cache, seed=seed)
    dQX = np.einsum("ij,ij->i", X, X)
    dQ = np.sqrt(core.pairwise(Q, C))
    rng = np.random.RandomState(seed)

    def err(est):
        rel = np.abs(est - truth) / np.maximum(truth, 1)
        return {"rel": float(rel.mean()), "p95": float(np.percentile(rel, 95))}

    res = {}
    for S in budgets:
        est = np.zeros_like(truth, dtype=np.float64)
        for i, q in enumerate(Q):
            samp = rng.choice(N, S, replace=False)
            dd = np.sqrt(np.maximum(float(q @ q) + dQX[samp] - 2.0 * X[samp] @ q, 0.0))
            for j, r in enumerate(radii):
                est[i, j] = N * float((dd <= r).sum()) / S
        res[f"uniform S={S}"] = {**err(est), "cost": float(S)}
        log(f"uniform S={S}: {res[f'uniform S={S}']}", t0)

    def true_p(q, r):
        """Exact per-cell hit rate: scans the whole collection once per (query, radius)."""
        q2 = float(q @ q)
        h = np.zeros(nlist, dtype=np.float64)
        for s0 in range(0, N, 400_000):
            xc = X[s0:s0 + 400_000]
            d = np.sqrt(np.maximum(q2 + dQX[s0:s0 + len(xc)] - 2.0 * xc @ q, 0.0))
            h += np.bincount(assign[s0:s0 + len(xc)], weights=(d <= r).astype(np.float64),
                             minlength=nlist)
        return h / np.maximum(sizes, 1)

    def variant(kind, S, alloc_alpha):
        est = np.zeros_like(truth, dtype=np.float64)
        for i, q in enumerate(Q):
            q2 = float(q @ q)
            ps = [cell_hit_prob(dQ[i], float(r), grid, qs) for r in radii]
            if kind == "perrad":
                Sr = max(len(radii), S // nr)
                for j, p in enumerate(ps):
                    w = sizes * np.sqrt(np.clip(p, 1e-3, 1 - 1e-3) * (1 - np.clip(p, 1e-3, 1 - 1e-3)))
                    alloc = allocate(w, Sr)
                    cells, idx = sample_idx(alloc, sizes, order, off, rng)
                    dd = np.sqrt(np.maximum(q2 + dQX[idx] - 2.0 * X[idx] @ q, 0.0))
                    hit = np.bincount(cells, weights=(dd <= radii[j]).astype(np.float64),
                                      minlength=nlist)
                    est[i, j] = float((sizes * hit / np.maximum(alloc, 1)).sum())
                continue
            if kind == "prop":
                W = sizes.astype(np.float64)
            else:
                Ws = []
                for p in ps:
                    pc = np.clip(p, 1e-3, 1 - 1e-3)
                    if kind == "neyman":
                        Ws.append(sizes * np.sqrt(pc * (1 - pc)))
                    else:
                        Ws.append(sizes * ((1 - alloc_alpha) + alloc_alpha * 2.0 * np.sqrt(pc * (1 - pc))))
                W = np.mean([w / w.sum() for w in Ws], axis=0)
            alloc = allocate(W, S)
            cells, idx = sample_idx(alloc, sizes, order, off, rng)
            dd = np.sqrt(np.maximum(q2 + dQX[idx] - 2.0 * X[idx] @ q, 0.0))
            for j, r in enumerate(radii):
                hit = np.bincount(cells, weights=(dd <= r).astype(np.float64), minlength=nlist)
                est[i, j] = float((sizes * hit / np.maximum(alloc, 1)).sum())
        return est

    calib = []
    for S in budgets:
        for kind, alpha in (("prop", 0.0), ("neyman", 0.0), ("blend", 0.25), ("blend", 0.5),
                            ("blend", 0.75), ("perrad", 0.0)):
            name = f"strat-{kind}" + (f"{alpha}" if kind == "blend" else "") + f" S={S}"
            t1 = time.time()
            est = variant(kind, S, alpha)
            res[name] = {**err(est), "cost": float(S if kind != "perrad" else max(len(radii), S // nr) * nr)}
            log(f"{name}: rel={res[name]['rel']:.4f} p95={res[name]['p95']:.4f} ({time.time()-t1:.0f}s)", t0)

    # calibration of the surrogate + oracle-allocation bound, on a few queries
    t1 = time.time()
    for i in range(min(3, nq)):
        po = cell_hit_prob(dQ[i], float(radii[-1]), grid, qs)
        pt = true_p(Q[i], float(radii[-1]))
        calib.append({"query": i, "radius": float(radii[-1]),
                      "corr": float(np.corrcoef(po, pt)[0, 1]),
                      "pred_mean": float(po.mean()), "true_mean": float(pt.mean()),
                      "true_inside_cells": int((pt > 0.999).sum()),
                      "true_empty_cells": int((pt <= 1e-9).sum())})
    S = budgets[-1]
    est = np.zeros_like(truth, dtype=np.float64)
    for i, q in enumerate(Q):
        q2 = float(q @ q)
        W = np.mean([true_p(q, float(r)) for r in radii], axis=0)
        alloc = allocate(W, S)
        cells, idx = sample_idx(alloc, sizes, order, off, rng)
        dd = np.sqrt(np.maximum(q2 + dQX[idx] - 2.0 * X[idx] @ q, 0.0))
        for j, r in enumerate(radii):
            hit = np.bincount(cells, weights=(dd <= r).astype(np.float64), minlength=nlist)
            est[i, j] = float((sizes * hit / np.maximum(alloc, 1)).sum())
    res[f"strat-oracle S={S}"] = {**err(est), "cost": float(S)}
    log(f"strat-oracle S={S}: {res[f'strat-oracle S={S}']} ({time.time()-t1:.0f}s)", t0)

    out = out or os.path.join(root, "results", f"strat_study_{dataset}_{N}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"setup": {"n": N, "nlist": nlist, "radii": radii.tolist(),
                             "counts": truth.mean(0).round(1).tolist()},
                   "calibration": calib, "results": res}, f, indent=1)
    log(f"wrote {out}", t0)
    print()
    print(f"{'variant':30s} {'rel.err':>9s} {'p95':>9s} {'cost':>10s}")
    for k, v in res.items():
        print(f"{k:30s} {v['rel']:9.4f} {v['p95']:9.4f} {v['cost']:10.2e}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift10m")
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--data-root", default=os.path.expanduser("~/vdm/data"))
    ap.add_argument("--budgets", default="40000,100000")
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.dataset, a.n, a.data_root, budgets=tuple(int(x) for x in a.budgets.split(",")),
        out=a.out, queries=a.queries, cache=a.cache)

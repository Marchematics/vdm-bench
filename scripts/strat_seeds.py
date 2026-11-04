"""Multi-seed check of the stratified estimator (removes the Monte-Carlo ambiguity).

The 500 (query, radius) estimates give a mean relative error whose run-to-run spread is a few
percent, so every claim in the paper is reported as mean +/- spread over seeds rather than a
single draw.  Uniform sampling, proportional stratification and Neyman allocation all use the
same budget and the same cached cell structure.

Usage: python scripts/strat_seeds.py --cache results_remote/cells_sift10m.npz --seeds 5
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdmbench import core
import strat_study as ss

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="sift10m")
ap.add_argument("--n", type=int, default=10_000_000)
ap.add_argument("--data-root", default=os.path.expanduser("~/vdm/data"))
ap.add_argument("--budget", type=int, default=100_000)
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--cache", default=None)
a = ap.parse_args()

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
z = np.load(os.path.join(root, "workloads", a.dataset, "analytics.npz"))
Q = np.ascontiguousarray(z["queries"][:100].astype(np.float32))
radii = z["radii"].astype(np.float32); truth = z["truth"][:100]
X, _ = core.load_dataset(a.data_root, a.dataset, a.n)
X = np.ascontiguousarray(X.astype(np.float32), dtype=np.float32)
N, nlist = len(X), int(min(16384, len(X) // 1000))
C, assign, sizes, order, off, grid, qs = ss.load_or_build_cells(X, nlist, a.cache)
dQX = np.einsum("ij,ij->i", X, X); dQ = np.sqrt(core.pairwise(Q, C))
S = a.budget
out = {"uniform": [], "prop": [], "neyman": [], "neyman_angle": []}
for seed in range(a.seeds):
    for kind in ("uniform", "prop", "neyman", "neyman_angle"):
        # independent stream per (seed, estimator): a shared stream would couple the three
        # estimators' draws and made the ratios depend on consumption order in the first
        # version of this script (one rng state gave 0.0146 where a fresh one gives 0.0172).
        rng = np.random.RandomState(1000 * seed + {"uniform": 0, "prop": 1, "neyman": 2,
                                                   "neyman_angle": 3}[kind])
        est = np.zeros_like(truth, dtype=np.float64)
        for i, q in enumerate(Q):
            q2 = float(q @ q)
            if kind == "uniform":
                idx = rng.choice(N, S, replace=False)
                dd = np.sqrt(np.maximum(q2 + dQX[idx] - 2.0 * X[idx] @ q, 0.0))
                for j, r in enumerate(radii):
                    est[i, j] = N * float((dd <= r).sum()) / S
                continue
            if kind == "prop":
                W = sizes.astype(np.float64)
            else:
                Ws = []
                for r in radii:
                    if kind == "neyman_angle":
                        p = ss.cell_hit_prob_angle(dQ[i], float(r), grid, qs, X.shape[1])
                    else:
                        p = ss.cell_hit_prob(dQ[i], float(r), grid, qs)
                    pc = np.clip(p, 1e-3, 1 - 1e-3)
                    w = sizes * np.sqrt(pc * (1 - pc))
                    Ws.append(w / w.sum())
                W = np.mean(Ws, axis=0)
            alloc = ss.allocate(W, S)
            cells, idx = ss.sample_idx(alloc, sizes, order, off, rng)
            dd = np.sqrt(np.maximum(q2 + dQX[idx] - 2.0 * X[idx] @ q, 0.0))
            for j, r in enumerate(radii):
                hit = np.bincount(cells, weights=(dd <= r).astype(np.float64), minlength=nlist)
                est[i, j] = float((sizes * hit / np.maximum(alloc, 1)).sum())
        rel = np.abs(est - truth) / np.maximum(truth, 1)
        out[kind].append((float(rel.mean()), float(np.percentile(rel, 95))))
    print(f"seed {seed}: " + "  ".join(f"{k}={out[k][-1][0]:.4f}/{out[k][-1][1]:.4f}" for k in out), flush=True)

print()
for k, v in out.items():
    r = np.array([x[0] for x in v]); p = np.array([x[1] for x in v])
    print(f"{k:9s} rel {r.mean():.4f} ± {r.std():.4f}   p95 {p.mean():.4f} ± {p.std():.4f}   (n={len(v)} seeds)")
res = {k: {"rel_mean": float(np.mean([x[0] for x in v])), "rel_sd": float(np.std([x[0] for x in v])),
           "p95_mean": float(np.mean([x[1] for x in v])), "p95_sd": float(np.std([x[1] for x in v]))}
       for k, v in out.items()}
json.dump({"budget": S, "seeds": a.seeds, "results": res},
          open(os.path.join(root, "results", f"strat_seeds_{a.dataset}.json"), "w"), indent=1)

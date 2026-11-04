"""Diagnose the 1.20x gap between the predicted and the true per-cell hit rate.

`strat_oracle_seeds.py` caches the TRUE per-cell hit rate P[q, r, c] for SIFT10M. With the cell
structure also cached, the surrogate can be scored offline against P without any sampling, which
is how the 1.20x gap left by allocation (0.0160 model-based vs 0.0133 oracle) is attributed
before anything is changed.

Two models are compared:

  spherical   p_c = F_c(sqrt(r^2 - d_c^2))                       (what ships today)
  angle-aware p_c = E_t[ Phi( (r^2 - d_c^2 - t^2) / (2 d_c t / sqrt(d)) ) ]
              i.e. the spherical identity is replaced by
              d(q,x)^2 ~= d_c^2 + t^2 - 2 d_c t cos(theta) with cos(theta) ~ N(0, 1/d) for
              isotropic points around the centroid, integrated over the cell's own t quantiles.
              The spherical model is the cos(theta) == 0 special case, so any improvement here is
              attributable to the angle term alone.

Reported per model: correlation with P over all (q, r, c), RMSE in hit-rate units, and -- the
quantity the estimator actually cares about -- the relative error of the *predicted per-radius
count* sum_c size_c p_c, which is the model-only error floor.

Usage: python scripts/diag_predictor.py --cache results_remote/cells_sift10m.npz \
           --truth-cache results_remote/oracle_cells_sift10m.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdmbench import core  # noqa: E402
import strat_study as ss  # noqa: E402


def predict_spherical(dq, r, grid, qs):
    return ss.cell_hit_prob(dq, r, grid, qs)


def predict_angle(dq, r, grid, qs, d):
    """p_c = E_t Phi((r^2 - d_c^2 - t^2) / (2 d_c t / sqrt(d))), t drawn from F_c."""
    nc = len(dq)
    # 8-point quantile quadrature over the cell's own t distribution, vectorised over cells:
    # every cell is interpolated at the same quantile levels, so this is one gather per level.
    qlev = np.linspace(0.05, 0.95, 8)
    k = np.clip(np.searchsorted(qs, qlev) - 1, 0, len(qs) - 2)
    frac = (qlev - qs[k]) / np.maximum(qs[k + 1] - qs[k], 1e-9)
    ts = grid[:, k] * (1.0 - frac)[None, :] + grid[:, k + 1] * frac[None, :]
    dq_ = dq[:, None]
    num = r * r - dq_ * dq_ - ts * ts
    den = np.maximum(2.0 * dq_ * ts / np.sqrt(d), 1e-6)
    from scipy.special import erf  # noqa: E402
    p = 0.5 * (1.0 + erf((num / den) / np.sqrt(2.0)))
    p = np.where(dq_ <= 1e-9, (ts <= r).mean(axis=1)[:, None], p)
    return p.mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift10m")
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--data-root", default=os.path.expanduser("~/vdm/data"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--truth-cache", default=None)
    ap.add_argument("--queries", type=int, default=25)
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    z = np.load(os.path.join(root, "workloads", a.dataset, "analytics.npz"))
    Q = np.ascontiguousarray(z["queries"][:100].astype(np.float32))
    radii = z["radii"].astype(np.float32)
    truth = z["truth"][:100]
    X, _ = core.load_dataset(a.data_root, a.dataset, a.n)
    X = np.ascontiguousarray(X.astype(np.float32), dtype=np.float32)
    nlist = int(min(16384, len(X) // 1000))
    C, assign, sizes, order, off, grid, qs = ss.load_or_build_cells(X, nlist, a.cache)
    tc = a.truth_cache or os.path.join(root, "results_remote", f"oracle_cells_{a.dataset}.npz")
    P = np.load(tc)["P"]
    dQ = np.sqrt(core.pairwise(Q, C))
    d = X.shape[1]
    nq = min(a.queries, len(Q))

    rows = {"spherical": [], "angle": []}
    cnt_err = {"spherical": [], "angle": []}
    for i in range(nq):
        for j, r in enumerate(radii):
            pt = P[i, j].astype(np.float64)
            ps = predict_spherical(dQ[i], float(r), grid, qs)
            pa = predict_angle(dQ[i], float(r), grid, qs, d)
            for name, p in (("spherical", ps), ("angle", pa)):
                rows[name].append((float(np.corrcoef(p, pt)[0, 1]),
                                   float(np.sqrt(np.mean((p - pt) ** 2))),
                                   float(np.mean(p - pt))))
                cnt_err[name].append(abs(float((sizes * p).sum()) - float(truth[i, j]))
                                     / max(truth[i, j], 1))
    print(f"queries={nq} radii={len(radii)} cells={nlist}\n")
    print(f"{'model':10s} {'corr(P)':>8s} {'rmse':>8s} {'bias':>9s} {'count err':>10s}")
    for name in rows:
        a_ = np.array(rows[name])
        print(f"{name:10s} {a_[:,0].mean():8.3f} {a_[:,1].mean():8.4f} "
              f"{a_[:,2].mean():+9.4f} {np.mean(cnt_err[name]):10.4f}")


if __name__ == "__main__":
    main()

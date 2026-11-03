"""Estimator study: uniform vs two-phase vs certified-list IS on real SIFT.

Reproduce with:  python scripts/est_study.py --data-root /path/to/data
Numbers land in experiments/est_study_2026-11-04.md (run on joysuch, CPU only).
"""
import argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--nlist", type=int, default=256)
    a = ap.parse_args()
    import faiss
    X = np.ascontiguousarray(np.load(f"{a.data_root}/sift/train.npy", mmap_mode="r")[:a.n]).astype(np.float32)
    Q = np.load(f"{a.data_root}/sift/test.npy").astype(np.float32)[:100]
    n = len(X); rng = np.random.RandomState(0)
    d0 = np.sqrt(np.maximum((Q[:10] ** 2).sum(1)[:, None] + (X[:20000] ** 2).sum(1)[None, :]
                            - 2 * Q[:10] @ X[:20000].T, 0))
    radii = np.quantile(d0.ravel(), [0.005, 0.02, 0.05, 0.2, 0.5]).astype(np.float32)
    truth = np.array([[(np.sqrt(np.maximum(((X - q) ** 2).sum(1), 0)) <= r).sum() for r in radii]
                      for q in Q])
    km = faiss.Kmeans(X.shape[1], a.nlist, niter=10, verbose=False, seed=0)
    km.train(X[:40000]); C = km.centroids.astype(np.float32)
    assign = (((X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * X @ C.T)).argmin(1)
    members = [np.flatnonzero(assign == c) for c in range(a.nlist)]
    rmax = np.array([float(np.sqrt(np.maximum(((X[m] - C[c]) ** 2).sum(1), 0)).max())
                     if len(m) else 0.0 for c, m in enumerate(members)])
    dQ = np.sqrt(np.maximum((Q ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * Q @ C.T, 0))

    def score(est, cost):
        rel = np.abs(est - truth) / np.maximum(truth, 1)
        return {"rel": float(rel.mean()), "p95": float(np.percentile(rel, 95)), "cost": float(cost)}

    res = {}
    for S in (4000, 12000, 20000):
        est = np.zeros_like(truth, dtype=float)
        for i, q in enumerate(Q):
            s = rng.choice(n, S, replace=False)
            dd = np.sqrt(np.maximum(((X[s] - q) ** 2).sum(1), 0))
            for j, r in enumerate(radii):
                est[i, j] = n * (dd <= r).sum() / S
        res[f"uniform S={S}"] = score(est, S * len(Q)); print(f"uniform S={S}", res[f"uniform S={S}"], flush=True)
    for kp, S in ((16, 4000), (32, 4000), (64, 6000)):
        est = np.zeros_like(truth, dtype=float); cost = 0
        for i, q in enumerate(Q):
            order = np.argsort(dQ[i])[:kp]
            pid = np.concatenate([members[c] for c in order])
            mask = np.ones(n, bool); mask[pid] = False; rest = np.flatnonzero(mask)
            samp = rng.choice(rest, min(S, len(rest)), replace=False)
            cost += len(pid) + len(samp)
            dp = np.sqrt(np.maximum(((X[pid] - q) ** 2).sum(1), 0))
            ds = np.sqrt(np.maximum(((X[samp] - q) ** 2).sum(1), 0))
            for j, r in enumerate(radii):
                est[i, j] = int((dp <= r).sum()) + len(rest) * int((ds <= r).sum()) / max(1, len(samp))
        res[f"two-phase k={kp} S={S}"] = score(est, cost); print(f"two-phase k={kp} S={S}", res[f"two-phase k={kp} S={S}"], flush=True)
    json.dump({"radii": radii.tolist(), "mean_counts": truth.mean(0).tolist(), "res": res},
              open("est_study.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()

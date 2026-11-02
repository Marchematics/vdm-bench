"""VDM-Bench core: deterministic workloads, ground truth, and scoring.

The benchmark intentionally keeps vectors out of the repository: workloads store queries,
indices, injections and ground truth; evaluators reload the raw dataset from --data-root.
"""
from __future__ import annotations

import json
import os
import heapq

import numpy as np


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _load(path):
    return np.load(path)


def load_sift(root, n=None):
    X = np.load(os.path.join(root, "sift", "train.npy"), mmap_mode="r")
    Q = np.load(os.path.join(root, "sift", "test.npy")).astype(np.float32)
    if n is not None:
        X = np.asarray(X[:n])
    else:
        X = np.asarray(X)
    return X.astype(np.float32), Q


def load_mnist(root, n=None):
    X = np.load(os.path.join(root, "mnist", "train.npy"), mmap_mode="r")
    Q = np.load(os.path.join(root, "mnist", "test.npy")).astype(np.float32)
    X = np.asarray(X[:n] if n else X).astype(np.float32)
    return X, Q


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sqdist(q, X):
    xn = np.einsum("ij,ij->i", X, X)
    return np.maximum(xn + q @ q - 2.0 * (X @ q), 0.0)


def exact_range_mask(q, X, r, chunk=100_000):
    out = np.empty(len(X), dtype=bool)
    for s in range(0, len(X), chunk):
        out[s:s + chunk] = sqdist(q, X[s:s + chunk]) <= r * r
    return out


def pairwise(Q, X):
    xn = np.einsum("ij,ij->i", X, X)
    qn = np.einsum("ij,ij->i", Q, Q)
    return np.maximum(qn[:, None] + xn[None, :] - 2.0 * Q @ X.T, 0.0)


# --------------------------------------------------------------------------- #
# Analytics workload
# --------------------------------------------------------------------------- #
def build_analytics(root, n=100_000, nq=100, seed=0):
    X, Q = load_sift(root, n)
    Q = Q[:nq]
    rng = np.random.RandomState(seed)
    # pooled distance quantiles
    probe = Q[:50]
    vals = []
    for i in range(0, len(probe), 10):
        vals.append(np.sqrt(np.maximum(
            (probe[i:i + 10] ** 2).sum(1)[:, None]
            + (X ** 2).sum(1)[None, :]
            - 2 * probe[i:i + 10] @ X.T, 0)).ravel())
    pooled = np.concatenate(vals)
    sels = [0.005, 0.02, 0.05, 0.2, 0.5]
    radii = np.array([np.quantile(pooled, s) for s in sels], dtype=np.float32)
    truth = np.zeros((len(Q), len(radii)), dtype=np.int64)
    for i, q in enumerate(Q):
        for j, r in enumerate(radii):
            truth[i, j] = int(exact_range_mask(q, X, float(r)).sum())
    # coverage
    anchors = Q[:min(100, len(Q))].copy()
    k = 30
    # r_cov from 1-NN distances
    d1 = np.sqrt(pairwise(anchors[:50], X[: min(5000, len(X))]).min(axis=1))
    r_cov = float(np.median(d1) * 1.5)
    D = np.sqrt(pairwise(anchors, X))
    covered = np.zeros(len(anchors), dtype=bool)
    chosen = []
    for _ in range(k):
        gain = ((D <= r_cov) & (~covered)[:, None]).sum(axis=0)
        j = int(gain.argmax())
        chosen.append(j)
        covered |= D[:, j] <= r_cov
    exact_covered = int(covered.sum())
    # join on a 10k slice
    Xj = X[:10_000]
    p = (5.0 * len(Xj) / 2.0) / (len(Xj) * (len(Xj) - 1) / 2.0)
    dj = np.sqrt(pairwise(Xj[:100], Xj)).ravel()
    r_join = float(np.quantile(dj, p))
    pairs = []
    for i in range(len(Xj)):
        m = sqdist(Xj[i], Xj) <= r_join * r_join
        js = np.where(m)[0]
        for j in js:
            if j > i:
                pairs.append([i, int(j)])
    pairs = np.asarray(pairs, dtype=np.int64)
    return {
        "queries": Q, "radii": radii, "truth": truth,
        "anchors": anchors, "r_cov": r_cov, "k": k, "exact_covered": exact_covered,
        "coverage_chosen": np.asarray(chosen, dtype=np.int64),
        "X_join_n": len(Xj), "r_join": r_join, "join_pairs": pairs,
        "n": len(X),
    }


def eval_analytics(pred, wl, root):
    X, _ = load_sift(root, wl["n"])
    out = {}
    # range counts
    rc = np.asarray(pred.get("range_count", []), dtype=np.float64)
    truth = wl["truth"]
    if rc.shape == truth.shape:
        rel = np.abs(rc - truth) / np.maximum(truth, 1.0)
        out["range_mean_rel_err"] = float(rel.mean())
        out["range_p95_rel_err"] = float(np.percentile(rel, 95))
    else:
        out["range_mean_rel_err"] = float("nan")
    # coverage
    sel = np.asarray(pred.get("coverage", []), dtype=np.int64)
    if len(sel):
        A = wl["anchors"]
        cov = int((np.sqrt(pairwise(A, X[sel])).min(axis=1) <= wl["r_cov"]).sum())
        out["coverage_ratio"] = cov / max(1, wl["exact_covered"])
    # join
    got = set(map(tuple, np.asarray(pred.get("join", []), dtype=np.int64).tolist()))
    gt = set(map(tuple, wl["join_pairs"].tolist()))
    tp = len(got & gt)
    out["join_recall"] = tp / max(1, len(gt))
    out["join_precision"] = tp / max(1, len(got))
    out["cost"] = float(pred.get("cost", float("nan")))
    return out


# --------------------------------------------------------------------------- #
# Quality workload
# --------------------------------------------------------------------------- #
def exact_pairs_faiss(X, eps):
    """Exact epsilon-pairs via faiss flat range search (ground truth, no approximation)."""
    import faiss
    idx = faiss.IndexFlatL2(X.shape[1])
    idx.add(X.astype(np.float32))
    lims, D, I = idx.range_search(X.astype(np.float32), eps * eps)
    pairs = set()
    for i in range(len(X)):
        js = I[lims[i]:lims[i + 1]]
        for j in js:
            if j > i:
                pairs.add((i, int(j)))
    return pairs


def make_quality_sets(root, n=50_000, seed=0):
    """Deterministic corrupted collections (vectors are regenerated, never committed)."""
    X, _ = load_sift(root, n)
    rng = np.random.RandomState(seed)
    sets = [("clean", "-", X.copy())]
    nd = int(0.05 * len(X))
    dup_idx = rng.choice(len(X), nd, replace=False)
    Xd = np.vstack([X, X[dup_idx] + rng.normal(0, 1.0, (nd, X.shape[1])).astype(np.float32)])
    sets.append(("dup", "5%", Xd))
    no = int(0.01 * len(X)); mu, sd = X.mean(0), X.std(0)
    Xo = np.vstack([X, mu + rng.normal(0, 1, (no, X.shape[1])) * 6 * sd])
    sets.append(("outlier", "1%", Xo))
    Xr = X.copy(); mask = rng.choice(len(X), int(0.2 * len(X)), replace=False)
    Xr[mask] += 0.5 * X.std(0)
    sets.append(("drift", "20%", Xr))
    a = X[rng.randint(len(X))]
    keep = np.argsort(np.linalg.norm(X - a, axis=1))[int(0.01 * len(X)):]
    sets.append(("hole", "1%", X[keep]))
    return sets


def build_quality(root, n=50_000, seed=0):
    eps = 25.0
    instances = []
    for family, level, X in make_quality_sets(root, n, seed):
        pairs = exact_pairs_faiss(X, eps)
        verts = set(sum(([a, b] for a, b in pairs), [])) if pairs else set()
        instances.append({"family": family, "level": level, "n": len(X),
                          "dup_rate": len(verts) / len(X),
                          "n_outliers": int(0.01 * n) if family == "outlier" else 0,
                          "drift_mmd": 0.5 if family == "drift" else 0.0,
                          "coverage": None})
    return {"instances": instances, "eps": eps}


def eval_quality(pred, wl):
    out = {}
    gt = wl["instances"]
    preds = pred.get("instances", [])
    if len(preds) != len(gt):
        return {"error": "instances length mismatch"}
    errs = []
    for p, g in zip(preds, gt):
        e = {}
        e["dup_abs_err"] = abs(float(p.get("dup_rate", 0)) - g["dup_rate"]) / max(1e-9, g["dup_rate"])
        e["outlier_abs_err"] = abs(float(p.get("n_outliers", 0)) - g["n_outliers"])
        e["drift_abs_err"] = abs(float(p.get("drift", 0)) - g["drift_mmd"])
        # detection: is the instance corrupt?
        truth_flag = g["family"] in ("dup", "outlier", "drift")
        pred_flag = (p.get("dup_rate", 0) > 0.05) or (p.get("n_outliers", 0) > 0) or (p.get("drift", 0) > 0.1)
        e["detect_ok"] = int(pred_flag == truth_flag)
        errs.append(e)
    out["dup_rel_err_mean"] = float(np.mean([x["dup_abs_err"] for x in errs]))
    out["outlier_abs_err_mean"] = float(np.mean([x["outlier_abs_err"] for x in errs]))
    out["drift_abs_err_mean"] = float(np.mean([x["drift_abs_err"] for x in errs]))
    out["detect_acc"] = float(np.mean([x["detect_ok"] for x in errs]))
    out["cost"] = float(pred.get("cost", float("nan")))
    return out


# --------------------------------------------------------------------------- #
# Serving workload
# --------------------------------------------------------------------------- #
SURFACE_NPROBE = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
SURFACE_RECALL = np.array([0.4836, 0.6707, 0.8276, 0.9328, 0.9840, 0.9979, 0.9999, 1.0000, 1.0000])
SURFACE_P50 = np.array([0.220, 0.397, 0.549, 0.959, 2.048, 3.323, 6.240, 16.887, 47.259])


def build_serving(seed=0, n_tenants=8, n_per_tenant=2000, rho=1.6, cores=8):
    rng = np.random.RandomState(seed)
    classes = [(3.3, 0.60), (6.5, 0.70), (10.9, 0.80), (21.8, 0.85),
               (43.7, 0.90), (87.4, 0.90), (218.0, 0.80), (218.0, 0.90)]
    tenants = [{"id": i, "L": classes[i % len(classes)][0], "R": classes[i % len(classes)][1]}
               for i in range(n_tenants)]
    hard = rng.rand(n_tenants, n_per_tenant) < 0.3
    score = 0.6 * hard.astype(np.float32) + 0.4 * rng.rand(n_tenants, n_per_tenant).astype(np.float32)
    # base nprobe for R (bigger than needed for hard queries)
    def nprobe_for(R):
        return int(SURFACE_NPROBE[np.argmax(SURFACE_RECALL >= R)])
    arrival = np.zeros((n_tenants, n_per_tenant), dtype=np.float32)
    for i in range(n_tenants):
        rate = 40.0  # queries/s offered per tenant
        arrival[i] = np.cumsum(rng.exponential(1.0 / rate, n_per_tenant))
    return {"tenants": tenants, "hard": hard, "score": score, "arrival": arrival,
            "rho": rho, "cores": cores, "surface": {"nprobe": SURFACE_NPROBE.tolist(),
                                                     "recall": SURFACE_RECALL.tolist(),
                                                     "p50": SURFACE_P50.tolist()}}


def eval_serving(pred, wl):
    efforts = np.asarray(pred.get("efforts", []), dtype=np.int64)
    nT, nq = wl["hard"].shape
    if efforts.shape != (nT, nq):
        return {"error": f"efforts shape {efforts.shape} != {(nT,nq)}"}
    # per-query service and recall
    def lookup(arr, npb):
        idx = np.argmin(np.abs(SURFACE_NPROBE - npb))
        return arr[idx]
    service = np.zeros((nT, nq), dtype=np.float32)
    recall = np.zeros((nT, nq), dtype=np.float32)
    for i in range(nT):
        for j in range(nq):
            npb = int(efforts[i, j])
            service[i, j] = lookup(SURFACE_P50, npb)
            r = float(lookup(SURFACE_RECALL, npb))
            if wl["hard"][i, j]:
                r2 = float(lookup(SURFACE_RECALL, max(1, npb // 2)))
                r = min(r, r2)
            recall[i, j] = r
    # FCFS simulation across cores
    events = []
    for i in range(nT):
        for j in range(nq):
            events.append((float(wl["arrival"][i, j]), i, j))
    events.sort()
    cores = [0.0] * wl["cores"]  # next free time (ms)
    lat = np.zeros((nT, nq), dtype=np.float32)
    for t, i, j in events:
        k = int(np.argmin(cores))
        start = max(t * 1000.0, cores[k])  # ms
        cores[k] = start + service[i, j]
        lat[i, j] = cores[k] - t * 1000.0
    L = np.array([[wl["tenants"][i]["L"]] * nq for i in range(nT)])
    R = np.array([[wl["tenants"][i]["R"]] * nq for i in range(nT)])
    ok = (lat <= L) & (recall >= R)
    return {"slo_pass": float(ok.mean()), "p95_latency": float(np.percentile(lat, 95)),
            "p99_latency": float(np.percentile(lat, 99)),
            "recall_slo": float((recall >= R).mean()),
            "goodput_qps": float(ok.sum() / (wl["arrival"].max() - wl["arrival"].min() + 1e-9))}

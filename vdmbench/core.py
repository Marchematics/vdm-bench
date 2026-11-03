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


def load_dataset(root, name="sift", n=None):
    """Load (X, Q) for a supported dataset: sift | mnist | scifact."""
    if name == "sift":
        return load_sift(root, n)
    if name == "mnist":
        return load_mnist(root, n)
    if name == "scifact":
        X = np.load(os.path.join(root, "scifact", "train.npy")).astype(np.float32)
        Q = np.load(os.path.join(root, "scifact", "test.npy")).astype(np.float32)
        return (X[:n] if n else X), Q
    raise ValueError(f"unknown dataset {name}")


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
def build_analytics(root, dataset="sift", n=None, nq=100, seed=0):
    if n is None:
        n = {"sift": 100_000, "mnist": 60_000, "scifact": None}[dataset]
    X, Q = load_dataset(root, dataset, n)
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
    Xj = X[: min(10_000, len(X) // 5)]
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


def eval_analytics(pred, wl, root, dataset="sift"):
    X, _ = load_dataset(root, dataset, wl["n"])
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
    return quality_sets_from(X, seed=seed)


def quality_sets_from(X, seed=0):
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


def _pairdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T, 0.0)


def _knn_dists(X, k=10, idx=None):
    """Exact k-th nearest-neighbour distance (excluding self) via faiss."""
    import faiss
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(np.ascontiguousarray(X, dtype=np.float32))
    Q = np.ascontiguousarray(X if idx is None else X[idx], dtype=np.float32)
    D, _ = index.search(Q, k + 1)
    D = np.sqrt(np.maximum(D, 0.0))
    return D[:, -1] if idx is None else D[:, -1]


def _mmd2(A, B, gamma, block=1024):
    """Unbiased MMD^2 with a Gaussian kernel, chunked."""
    def ksum(P, Q, same):
        tot = 0.0
        for i in range(0, len(P), block):
            kk = np.exp(-gamma * _pairdist(P[i:i + block], Q))
            if same:
                kk[np.arange(len(kk)), np.arange(i, i + len(kk))] = 0.0
            tot += float(kk.sum())
        return tot
    m, n = len(A), len(B)
    return (ksum(A, A, True) / (m * (m - 1))
            + ksum(B, B, True) / (n * (n - 1))
            - 2.0 * ksum(A, B, False) / (m * n))


def build_quality(root, n=50_000, seed=0, eps=25.0, k_nn=10, ref_n=8000, sample_n=4000):
    """Quality workload with *published rules* and exact ground truth.

    Every number a submission reports is defined by a rule shipped with the workload,
    so the task is "compute these definitions cheaply", not "guess our convention":

    * ``dup_rate``: fraction of points that are the endpoint of some pair at distance
      <= ``eps`` (exact epsilon-graph vertex rate).
    * ``n_outliers``: number of points whose exact ``k_nn``-th neighbour distance exceeds
      ``thr``, where ``thr = median + 6*(p75-p25)`` of the k-NN distances on the *clean
      reference* set (a fixed, published threshold - not re-derived per instance).
    * ``drift``: unbiased MMD^2 between the collection sample and the clean reference
      sample under a Gaussian kernel with bandwidth ``gamma`` (published, from the median
      pairwise distance of the reference).
    * ``drift_threshold``: 99th percentile of MMD^2 under the null (clean vs clean split),
      published so detection is decidable.
    """
    # The reference must be *exchangeable* with the profiled rows and disjoint from them.
    # Taking a contiguous block (rows n..n+ref_n) is not exchangeable on real SIFT, which
    # makes the clean instance read as drifted; a seeded random split of the same rows is
    # exchangeable by construction and is published so submissions can reproduce it.
    X, _ = load_sift(root, n)
    split_seed = 20260101 + seed
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(len(X))
    ref_n = int(min(ref_n, max(2, len(X) // 4)))
    # keep the *permutation order* for the reference: ref[:sample_n] must be a random
    # subset. Sorting ref_idx made refA the lowest-index rows, and on real SIFT row order
    # correlates with the data, which made every clean instance read as drifted (48 sigma).
    ref_idx = perm[:ref_n]
    inst_idx = np.sort(perm[ref_n:])
    ref = np.ascontiguousarray(np.asarray(X)[ref_idx])
    # the null band needs two disjoint halves, so the comparison sample cannot exceed
    # half the reference set
    sample_n = int(min(sample_n, ref_n // 2))
    refA = np.ascontiguousarray(ref[:sample_n])
    # bandwidth: median pairwise distance among a subsample of the clean reference
    sub = ref[rng.choice(len(ref), min(800, len(ref)), replace=False)]
    pd = np.sqrt(_pairdist(sub, sub))
    sigma = float(np.median(pd[np.triu_indices(len(sub), 1)]))
    gamma = 1.0 / (2.0 * max(sigma, 1e-9) ** 2)
    # outlier threshold from the clean reference
    dk_ref = _knn_dists(ref, k=k_nn)
    q75, q25 = np.percentile(dk_ref, 75), np.percentile(dk_ref, 25)
    thr = float(np.median(dk_ref) + 6.0 * (q75 - q25))
    # null band of MMD^2, computed at the *same* sample sizes the instances use
    # (4,000 vs 4,000) so the detection threshold is calibrated correctly
    nulls = []
    for t in range(20):
        r2 = np.random.RandomState(1000 + t)
        perm = r2.permutation(len(ref))
        nulls.append(_mmd2(ref[perm[:sample_n]], ref[perm[sample_n:2 * sample_n]], gamma))
    drift_thr = float(np.quantile(nulls, 0.99))

    instances = []
    for family, level, Xi in quality_sets_from(np.asarray(X)[inst_idx], seed):
        pairs = exact_pairs_faiss(Xi, eps)
        verts = set()
        for u, v in pairs:
            verts.add(u); verts.add(v)
        sidx = np.random.RandomState(seed).choice(len(Xi), min(sample_n, len(Xi)), replace=False)
        sample = np.ascontiguousarray(Xi[sidx])
        dk = _knn_dists(Xi, k=k_nn, idx=sidx)
        instances.append({"family": family, "level": level, "n": int(len(Xi)),
                          "dup_rate": float(len(verts) / len(Xi)),
                          "n_outliers": int((dk > thr).sum() * len(Xi) / len(sidx)),
                          "drift": float(_mmd2(sample, refA, gamma)),
                          "coverage": None})
    return {"instances": instances,
            "rules": {"eps": eps, "k_nn": k_nn, "outlier_thr": thr,
                      "gamma": gamma, "drift_threshold": drift_thr,
                      "ref_n": ref_n, "split_seed": split_seed, "sample_n": sample_n,
                      "ref_order": "permutation",
                      "dataset_n": int(len(X)), "base_n": int(len(inst_idx)),
                      "dup_threshold": 0.01, "outlier_rate_threshold": 0.005},
            "null_mmd2_q99": drift_thr}


def eval_quality(pred, wl):
    """Score against the published rules; every metric is relative to its ground truth."""
    gt = wl["instances"]
    preds = pred.get("instances", [])
    if len(preds) != len(gt):
        return {"error": f"instances length mismatch: {len(preds)} vs {len(gt)}"}
    rules = wl.get("rules", {})
    drift_thr = float(rules.get("drift_threshold", 0.0))
    dup_thr = float(rules.get("dup_threshold", 0.01))
    out_thr = float(rules.get("outlier_rate_threshold", 0.005))
    errs = []
    for pr, g in zip(preds, gt):
        n = max(1, int(g["n"]))
        e = {}
        # absolute errors: a clean collection has dup_rate == 0 exactly, so dividing by
        # the ground truth would make any non-zero estimate catastrophically penalised
        e["dup_abs_err"] = abs(float(pr.get("dup_rate", 0)) - g["dup_rate"])
        e["outlier_abs_err"] = abs(float(pr.get("n_outliers", 0)) - g["n_outliers"])
        # an unbiased MMD^2 estimate can be negative, so relative error is unstable;
        # report drift error in units of the published detection threshold instead
        e["drift_err"] = abs(float(pr.get("drift", 0)) - g["drift"]) / max(1e-12, drift_thr)
        truth_flag = g["family"] in ("dup", "outlier", "drift")
        pred_flag = (float(pr.get("dup_rate", 0)) > dup_thr
                     or float(pr.get("n_outliers", 0)) / n > out_thr
                     or float(pr.get("drift", 0)) > drift_thr)
        # "hole" is a *coverage* defect, decided by the COVERAGE clause rather than by any
        # of the three scalar rules, so scoring scalar detection on it is off-task
        e["scored"] = int(g["family"] != "hole")
        e["detect_ok"] = int(pred_flag == truth_flag)
        errs.append(e)
    return {"dup_abs_err_mean": float(np.mean([x["dup_abs_err"] for x in errs])),
            "outlier_abs_err_mean": float(np.mean([x["outlier_abs_err"] for x in errs])),
            "drift_err_thr_units": float(np.mean([x["drift_err"] for x in errs])),
            "detect_acc": float(np.mean([e["detect_ok"] for e in errs if e["scored"]])),
            "cost": float(pred.get("cost", float("nan")))}


# --------------------------------------------------------------------------- #
# Serving workload
# --------------------------------------------------------------------------- #
SURFACE_NPROBE = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
SURFACE_RECALL = np.array([0.4836, 0.6707, 0.8276, 0.9328, 0.9840, 0.9979, 0.9999, 1.0000, 1.0000])
SURFACE_P50 = np.array([0.220, 0.397, 0.549, 0.959, 2.048, 3.323, 6.240, 16.887, 47.259])


def build_serving(seed=0, n_tenants=8, n_per_tenant=2000, rhos=(0.70, 1.00, 1.15),
                  cores=8, skew=None, eval_seed=None):
    """Multi-tenant serving workload under a joint latency+recall contract.

    Offered load is defined *relative to capacity*: ``rho`` is the aggregate utilisation
    of ``cores`` cores at the per-tenant base effort. ``skew`` re-weights the per-tenant
    arrival rates (a hot tenant); with ``skew=None`` load is uniform. The interesting
    regime for an SLO contract is aggregate ``rho < 1`` *with* skew: nothing is globally
    overloaded, but one tenant's demand exceeds its fair share and, without isolation,
    starves the others.

    A submission returns a single per-query *policy*; it is scored across every load
    regime in ``rhos`` (mean and worst-case ``slo_pass``). Two independent arrival
    traces are emitted per regime: ``arrival`` (pilot, visible to submissions) and
    ``arrival_eval`` (held-out realisation used for scoring). A submission must therefore
    return a policy that generalises, not a replay of one trace.
    """
    rng = np.random.RandomState(seed)
    classes = [(3.3, 0.60), (6.5, 0.70), (10.9, 0.80), (21.8, 0.85),
               (43.7, 0.90), (87.4, 0.90), (218.0, 0.80), (218.0, 0.90)]
    tenants = [{"id": i, "L": classes[i % len(classes)][0], "R": classes[i % len(classes)][1]}
               for i in range(n_tenants)]
    hard = rng.rand(n_tenants, n_per_tenant) < 0.3
    score = 0.6 * hard.astype(np.float32) + 0.4 * rng.rand(n_tenants, n_per_tenant).astype(np.float32)

    def nprobe_for(R):
        return int(SURFACE_NPROBE[np.argmax(SURFACE_RECALL >= R)])

    def p50_for(npb):
        return float(SURFACE_P50[int(np.argmin(np.abs(SURFACE_NPROBE - npb)))])

    base_ms = np.array([0.7 * p50_for(nprobe_for(t["R"]))
                        + 0.3 * p50_for(min(256, 2 * nprobe_for(t["R"])))
                        for t in tenants], dtype=np.float64)
    mean_base_ms = float(base_ms.mean())
    if skew is None:
        skew = np.ones(n_tenants, dtype=np.float64)
    skew = np.asarray(skew, dtype=np.float64)
    rhos = np.atleast_1d(np.asarray(rhos, dtype=np.float64))

    def trace(rate_vec, seed_i):
        r = np.random.RandomState(seed_i)
        a = np.zeros((n_tenants, n_per_tenant), dtype=np.float32)
        for i in range(n_tenants):
            a[i] = np.cumsum(r.exponential(1.0 / rate_vec[i], n_per_tenant))
        return a

    rates = np.zeros((len(rhos), n_tenants), dtype=np.float64)
    arrival = np.zeros((len(rhos), n_tenants, n_per_tenant), dtype=np.float32)
    arrival_eval = np.zeros_like(arrival)
    ev_seed = 1000 + 7 if eval_seed is None else int(eval_seed)
    for k, rho in enumerate(rhos):
        unit = rho * cores * 1000.0 / float((skew * base_ms).sum())
        rates[k] = unit * skew
        arrival[k] = trace(rates[k], seed + 100 * k)
        arrival_eval[k] = trace(rates[k], ev_seed + 100 * k)
    return {"tenants": tenants, "hard": hard, "score": score,
            "arrival": arrival, "arrival_eval": arrival_eval,
            "rhos": rhos.tolist(), "cores": cores, "skew": skew.tolist(),
            "rates_qps": rates.tolist(), "mean_base_service_ms": mean_base_ms,
            "surface": {"nprobe": SURFACE_NPROBE.tolist(),
                        "recall": SURFACE_RECALL.tolist(),
                        "p50": SURFACE_P50.tolist()}}


def _simulate(arr, service, cores, priority=None):
    """Non-preemptive multi-core queueing simulation.

    ``arr`` (T, nq) arrival times in seconds; ``service`` (T, nq) service times in ms;
    ``priority`` (T, nq) optional, *higher is served first* (e.g. EDF gives higher
    priority to tighter deadlines). Without ``priority`` this is plain FCFS.
    Returns per-query latency in ms.
    """
    T, nq = arr.shape
    lat = np.zeros((T, nq), dtype=np.float32)
    pri = np.zeros((T, nq), dtype=np.float64) if priority is None else np.asarray(priority, dtype=np.float64)
    flat = [(float(arr[i, j]), int(i), int(j)) for i in range(T) for j in range(nq)]
    flat.sort()
    free = [0.0] * cores
    import heapq
    waiting = []          # (-priority, seq, i, j, arrival_ms)
    seq = 0
    k = 0
    n = len(flat)
    while k < n or waiting:
        if not waiting:
            t_next = free[int(np.argmin(free))]
            a_ms = flat[k][0] * 1000.0
            if a_ms > t_next:
                free[int(np.argmin(free))] = a_ms
        # admit every arrival that is already waiting at the earliest free core
        t_min = min(free)
        while k < n and flat[k][0] * 1000.0 <= t_min + 1e-9:
            at, i, j = flat[k]
            heapq.heappush(waiting, (-pri[i, j], seq, at, i, j))
            seq += 1
            k += 1
        if not waiting:
            continue
        negp, _, at, i, j = heapq.heappop(waiting)
        c = int(np.argmin(free))
        start = max(at * 1000.0, free[c])
        free[c] = start + float(service[i, j])
        lat[i, j] = free[c] - at * 1000.0
    return lat


def eval_serving(pred, wl):
    """Score a serving submission across every load regime in the workload.

    A submission may return ``efforts`` (T,nq) and, optionally, ``priority`` (T,nq)
    where *higher* priority is served first (deadline-aware scheduling, e.g. EDF).
    Omitting ``priority`` gives plain FCFS. Scoring uses the held-out arrival traces.
    """
    efforts = np.asarray(pred.get("efforts", []), dtype=np.int64)
    nT, nq = wl["hard"].shape
    if efforts.shape != (nT, nq):
        return {"error": f"efforts shape {efforts.shape} != {(nT,nq)}"}
    pri = pred.get("priority")
    if pri is not None:
        pri = np.asarray(pri, dtype=np.float64)
        if pri.shape != (nT, nq):
            return {"error": f"priority shape {pri.shape} != {(nT,nq)}"}

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

    L = np.array([[wl["tenants"][i]["L"]] * nq for i in range(nT)], dtype=np.float32)
    R = np.array([[wl["tenants"][i]["R"]] * nq for i in range(nT)], dtype=np.float32)
    arr_all = np.asarray(wl.get("arrival_eval", wl["arrival"]))
    if arr_all.ndim == 2:
        arr_all = arr_all[None, :, :]

    passes, p95s, p99s, recalls, goodputs = [], [], [], [], []
    for arr in arr_all:
        lat = _simulate(arr, service, wl["cores"], pri)
        ok = (lat <= L) & (recall >= R)
        passes.append(float(ok.mean()))
        p95s.append(float(np.percentile(lat, 95)))
        p99s.append(float(np.percentile(lat, 99)))
        recalls.append(float((recall >= R).mean()))
        goodputs.append(float(ok.sum() / (arr.max() - arr.min() + 1e-9)))
    return {"slo_pass": float(np.mean(passes)),
            "slo_pass_worst": float(np.min(passes)),
            "p95_latency": float(np.mean(p95s)),
            "p99_latency": float(np.mean(p99s)),
            "recall_slo": float(np.mean(recalls)),
            "goodput_qps": float(np.mean(goodputs))}

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
    if name == "sift10m":
        return load_sift10m(root, n)
    if name == "mnist":
        return load_mnist(root, n)
    if name == "scifact":
        X = np.load(os.path.join(root, "scifact", "train.npy")).astype(np.float32)
        Q = np.load(os.path.join(root, "scifact", "test.npy")).astype(np.float32)
        return (X[:n] if n else X), Q
    raise ValueError(f"unknown dataset {name}")


def load_sift10m(root, n=None):
    """SIFT10M from ModelScope sift1b shards: 100k x 128 float64, values normalised to 0..1.

    Shards must be rescaled by 255 to match SIFT1M's raw integer scale, otherwise every
    radius/threshold in a SIFT-calibrated workload is off by that factor.
    """
    import glob
    files = sorted(glob.glob(os.path.join(root, "sift10m", "binary_128d_*.npy")))
    need = (n or 10_000_000)
    Xs = []
    got = 0
    for f in files:
        if got >= need:
            break
        if os.path.getsize(f) < 102_000_000:
            continue                      # partially downloaded shard
        a = np.load(f, mmap_mode="r")
        if a.shape[1] != 128:
            continue
        Xs.append(np.asarray(a, dtype=np.float32) * 255.0)
        got += len(a)
    if not Xs:
        raise FileNotFoundError(f"no sift10m shards under {root}/sift10m")
    X = np.concatenate(Xs, axis=0)
    Q = np.load(os.path.join(root, "sift", "test.npy")).astype(np.float32)
    return X[:need], Q


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
def build_analytics(root, dataset="sift", n=None, nq=100, seed=0, chunk=200_000):
    """Analytics workload. Chunked over the collection so it scales to 10M+ vectors.

    Ground truth is exact; every pass over X is blocked, so memory is O(nq*chunk) rather than
    O(nq*n) (the naive pairwise version would need 8 GB at 10M x 128 for 100 queries).
    """
    if n is None:
        n = {"sift": 100_000, "sift10m": 10_000_000, "mnist": 60_000, "scifact": None}[dataset]
    X, Q = load_dataset(root, dataset, n)
    Q = Q[:nq]
    N = len(X)
    rng = np.random.RandomState(seed)
    # pooled distance quantiles over a sample of the collection
    probe = Q[:50]
    samp = np.sort(rng.choice(N, min(200_000, N), replace=False))
    Xs = np.asarray(X[samp], dtype=np.float32)
    xs2 = np.einsum("ij,ij->i", Xs, Xs)
    vals = []
    for i in range(len(probe)):
        d2 = np.maximum((probe[i] ** 2).sum() + xs2 - 2.0 * Xs @ probe[i], 0.0)
        vals.append(np.sqrt(d2))
    pooled = np.concatenate(vals)
    sels = [0.005, 0.02, 0.05, 0.2, 0.5]
    radii = np.array([np.quantile(pooled, s) for s in sels], dtype=np.float32)
    # exact truth: one blocked pass per query, all radii at once
    truth = np.zeros((len(Q), len(radii)), dtype=np.int64)
    for i, q in enumerate(Q):
        q2 = float(q @ q)
        cnt = np.zeros(len(radii), dtype=np.int64)
        for s0 in range(0, N, chunk):
            xc = np.asarray(X[s0:s0 + chunk], dtype=np.float32)
            d2 = np.maximum(q2 + np.einsum("ij,ij->i", xc, xc) - 2.0 * xc @ q, 0.0)
            d = np.sqrt(d2)
            for j, r in enumerate(radii):
                cnt[j] += int((d <= r).sum())
        truth[i] = cnt
    # coverage: blocked greedy maximisation (never materialises anchors x N)
    anchors = Q[:min(100, len(Q))].copy()
    k = 30
    # r_cov = 1.5 x median exact 1-NN distance. This MUST be computed over the whole
    # collection: sampling the first 5,000 rows gave a 1-NN estimate that grows with n
    # (at 10M those rows are 0.05% of the data), which inflated r_cov to 566 and made the
    # coverage task trivially satisfiable by 30 random points (ratio 1.00 for every
    # submission). Exact 1-NN over 10M costs ~15 s of BLAS.
    d1 = []
    for a in anchors[:50]:
        a2 = float(a @ a)
        best = np.inf
        for s0 in range(0, N, chunk):
            xc = np.asarray(X[s0:s0 + chunk], dtype=np.float32)
            d = np.sqrt(np.maximum(a2 + np.einsum("ij,ij->i", xc, xc) - 2.0 * xc @ a, 0.0))
            best = min(best, float(d.min()))
        d1.append(best)
    # 1.5x the median 1-NN was loose enough that a single greedy point covered every
    # anchor, so the ground truth saturated at covered == len(anchors) with all k picks
    # being the same point. 1.0x keeps the task partial and therefore discriminating.
    r_cov = float(np.median(d1) * 1.0)
    # Coverage ground truth: compute the anchors x collection mask ONCE, then run the greedy
    # over it. The natural loop recomputes all distances on each of the k iterations (30 passes
    # over 10M = 3e10 evaluations); one pass is 1e9 and the greedy then costs 30 passes over a
    # uint8 matrix. Same result, ~30x less work.
    A = len(anchors)
    mask = np.zeros((N, A), dtype=np.uint8)
    a2 = (anchors ** 2).sum(1)
    for s0 in range(0, N, chunk):
        xc = np.asarray(X[s0:s0 + chunk], dtype=np.float32)
        d = np.sqrt(np.maximum(a2[:, None] + np.einsum("ij,ij->i", xc, xc)[None, :]
                               - 2.0 * anchors @ xc.T, 0.0))
        mask[s0:s0 + len(xc)] = (d <= r_cov).T
    rem = mask.copy()
    covered = np.zeros(A, dtype=bool)
    chosen = []
    for _ in range(k):
        if covered.all():
            break
        # gain per COLLECTION POINT (axis=1), not per anchor: the greedy selects points.
        # Summing axis=0 selected anchors, so the "chosen point" was really anchor index j and
        # the greedy only ever looked at the first len(anchors) rows of X -- at 10M that reported
        # exact_covered = 0 for a task that is in fact coverable.
        gain = rem.sum(axis=1, dtype=np.int64)
        if gain.max() <= 0:
            break
        j = int(gain.argmax())
        chosen.append(j)
        cov_j = mask[j].astype(bool)
        covered |= cov_j
        rem[:, cov_j] = 0
    exact_covered = int(covered.sum())
    # join on a slice
    Xj = np.asarray(X[: min(10_000, N // 5)], dtype=np.float32)
    p = (5.0 * len(Xj) / 2.0) / (len(Xj) * (len(Xj) - 1) / 2.0)
    dj0 = np.sqrt(np.maximum((Xj[:100] ** 2).sum(1)[:, None] + (Xj ** 2).sum(1)[None, :]
                             - 2.0 * Xj[:100] @ Xj.T, 0.0)).ravel()
    r_join = float(np.quantile(dj0, p))
    # Join ground truth via an exact FAISS radius search instead of the O(n^2) python loop
    # (the loop was 41.7 s of a 42 s build at 200k, and it is O(1e4 x 1e4 x d) regardless of n).
    import faiss
    flat = faiss.IndexFlatL2(Xj.shape[1])
    flat.add(np.ascontiguousarray(Xj))
    lims, _, II = flat.range_search(np.ascontiguousarray(Xj), float(r_join * r_join))
    pairs = []
    for i in range(len(Xj)):
        for j in II[lims[i]:lims[i + 1]]:
            if j > i:
                pairs.append([i, int(j)])
    pairs = np.asarray(pairs, dtype=np.int64)
    return {
        "queries": Q, "radii": radii, "truth": truth,
        "anchors": anchors, "r_cov": r_cov, "k": k, "exact_covered": exact_covered,
        "coverage_chosen": np.asarray(chosen, dtype=np.int64),
        "X_join_n": len(Xj), "r_join": r_join, "join_pairs": pairs,
        "n": N,
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
# The quality task ships a contract per instance. This is the *published* contract
# (mirrored by ``vdmbench.planner.QualityContract``); ``eval_quality`` enforces it per
# instance. Because the prediction interface reports one scalar per rule (not the
# epsilon-pair set), each clause is mapped onto the measurable quantity as documented
# here, and that mapping is part of the task definition:
#
#   integrity_recall    >= 1.0   no epsilon-pair may be missed. The interface reports the
#                                epsilon-graph *vertex rate*, so the clause is enforced as
#                                the one-sided rate clause  reported_rate >= true_rate
#                                (equivalently a measured recall of 1.0). Under-reporting
#                                is a missed-pair violation; over-reporting is not caught by
#                                this clause (``detect_acc`` catches it on clean instances).
#                                Verifying *set* recall needs pair-level output, which the
#                                interface does not ship (still unenforced).
#   outlier_rel_tol     <= 0.25  relative error of the outlier count:
#                                |pred - truth| / max(1, truth).
#   drift_tol_thr_units <= 2.0   drift error in units of the published detection
#                                threshold: |pred - truth| / drift_threshold.
QUALITY_CONTRACT = {
    "integrity_recall": 1.0,
    "outlier_rel_tol": 0.25,
    "drift_tol_thr_units": 2.0,
}

# Measured sampling sd of the *defined-sample* drift ground truth across 20 collection-sample
# seeds, in units of the published drift threshold (SIFT workload, seed 0).  Published as
# workload metadata because it is the reason the drift clause is a defined-sample clause: the
# ground truth is one specific 4,000-point sample statistic whose own noise is this large, so a
# submission must target the published statistic (reproduce the sample rule) rather than a
# population functional.  See experiments/quality_drift_contract_2026-11-05.md.
QUALITY_DRIFT_GT_SD_THR_UNITS = {
    "clean": 1.071, "dup": 1.808, "outlier": 0.598, "drift": 2.176, "hole": 1.081,
}


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
                      "dup_threshold": 0.01, "outlier_rate_threshold": 0.005,
                      # --- defined-sample drift clause (metadata only) ----------------
                      # The drift GT is the MMD^2 of one *specific* sample_n-point collection
                      # sample against the published reference sample.  The sample is drawn by
                      # this rule; publish it so third-party submissions can reproduce the
                      # defined statistic.  The clause is a defined-sample clause, not a
                      # population functional (see QUALITY_DRIFT_GT_SD_THR_UNITS above).
                      "instance_sample_seed": int(seed),
                      "instance_sample_rule": "RandomState(instance_sample_seed).choice("
                                              "len(collection), min(sample_n, len(collection)), "
                                              "replace=False)",
                      "drift_clause": "defined-sample",
                      "drift_clause_note": "MMD^2 of the specific sample_n-point collection "
                                           "sample (instance_sample_rule) against the published "
                                           "reference sample; a defined sample statistic, not a "
                                           "population functional",
                      "drift_gt_sampling_sd_thr_units": dict(QUALITY_DRIFT_GT_SD_THR_UNITS)},
            "null_mmd2_q99": drift_thr}


def quality_contract_clauses(pr, g, rules, contract=None):
    """Check one instance against the published quality contract.

    Returns ``(met, failures)``. Clause mapping (documented in ``QUALITY_CONTRACT``):

      * ``integrity_recall`` (``>=``): the reported epsilon-graph vertex rate must not
        under-report the exact rate, ``recall = min(1, pred_rate / true_rate)``. The scalar
        interface cannot distinguish a set that has the right rate but misses pairs; that
        part needs pair-level output and is still unenforced.
      * ``outlier_rel_tol`` (``<=``): ``|pred - truth| / max(1, truth)``.
      * ``drift_tol_thr_units`` (``<=``): ``|pred - truth| / drift_threshold``.

    Each failure records the clause, its operator, the measured value, the limit and the
    margin by which it was missed (in the clause's own units).
    """
    contract = QUALITY_CONTRACT if contract is None else contract
    drift_thr = float(rules.get("drift_threshold", 0.0))
    gt_dup = float(g["dup_rate"])
    pred_dup = float(pr.get("dup_rate", 0.0))
    integrity = min(1.0, pred_dup / gt_dup) if gt_dup > 0.0 else 1.0
    outlier_rel = (abs(float(pr.get("n_outliers", 0)) - g["n_outliers"])
                   / max(1.0, float(g["n_outliers"])))
    drift_err = abs(float(pr.get("drift", 0.0)) - g["drift"]) / max(1e-12, drift_thr)
    checks = [
        ("integrity_recall", ">=", integrity, float(contract["integrity_recall"])),
        ("outlier_rel_tol", "<=", outlier_rel, float(contract["outlier_rel_tol"])),
        ("drift_tol_thr_units", "<=", drift_err, float(contract["drift_tol_thr_units"])),
    ]
    failures = []
    for clause, op, value, limit in checks:
        ok = value >= limit - 1e-12 if op == ">=" else value <= limit + 1e-12
        if not ok:
            margin = (limit - value) if op == ">=" else (value - limit)
            failures.append({"clause": clause, "op": op, "value": float(value),
                             "limit": float(limit), "margin": float(margin)})
    return (len(failures) == 0), failures


def eval_quality(pred, wl):
    """Score against the published rules; every metric is relative to its ground truth.

    Also enforces the published per-instance *contract* (``QUALITY_CONTRACT``): returns the
    per-instance ``contract_met`` flags, the aggregate ``contract_rate`` and every failure
    with its clause and margin. A submission that looks best on a mean error column but
    violates a clause is reported as failing the contract.
    """
    gt = wl["instances"]
    preds = pred.get("instances", [])
    if len(preds) != len(gt):
        return {"error": f"instances length mismatch: {len(preds)} vs {len(gt)}"}
    rules = wl.get("rules", {})
    contract = wl.get("contract", QUALITY_CONTRACT)
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
        met, failures = quality_contract_clauses(pr, g, rules, contract)
        e["contract_met"] = bool(met)
        e["contract_failures"] = failures
        errs.append(e)
    contract_failures = []
    for idx, (e, g) in enumerate(zip(errs, gt)):
        for f in e["contract_failures"]:
            contract_failures.append({"instance": idx, "family": g["family"],
                                      "level": g["level"], **f})
    return {"dup_abs_err_mean": float(np.mean([x["dup_abs_err"] for x in errs])),
            "outlier_abs_err_mean": float(np.mean([x["outlier_abs_err"] for x in errs])),
            "drift_err_thr_units": float(np.mean([x["drift_err"] for x in errs])),
            "detect_acc": float(np.mean([e["detect_ok"] for e in errs if e["scored"]])),
            "contract_rate": float(np.mean([e["contract_met"] for e in errs])),
            "contract_met_count": int(sum(e["contract_met"] for e in errs)),
            "cost": float(pred.get("cost", float("nan"))),
            "_contract": dict(contract),
            "_contract_met": [e["contract_met"] for e in errs],
            "_contract_failures": contract_failures}


# --------------------------------------------------------------------------- #
# Serving workload
# --------------------------------------------------------------------------- #
SURFACE_NPROBE = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
SURFACE_RECALL = np.array([0.4836, 0.6707, 0.8276, 0.9328, 0.9840, 0.9979, 0.9999, 1.0000, 1.0000])
SURFACE_P50 = np.array([0.220, 0.397, 0.549, 0.959, 2.048, 3.323, 6.240, 16.887, 47.259])

# nearest published probe budget for any integer nprobe (submissions may ask for any value)
_NP_NEAREST = np.array([int(np.argmin(np.abs(SURFACE_NPROBE - v))) for v in range(257)],
                       dtype=np.int64)

# Published contract set: a *mixed* hard/soft set, not one uniform contract. Every tenant
# draws one (latency deadline L in ms, recall target R) class. Within every tenant 30% of
# the queries are "hard": their deadline is 0.6x and their recall at a given probe budget is
# the recall at half that budget, so they need 2x the probes for the same recall. Both rules
# are published in serving_meta.json and the resulting per-query contract is shipped with the
# workload (``deadline``, ``recall_req``), so a submission can reproduce it exactly.
#
# The set is deliberately heterogeneous: one hot tenant is cheap and tolerant of queueing;
# two tenants have tight deadlines and expensive (high-recall) probes, and the hard queries of
# "tight-expensive" are *infeasible* (6.24 ms of service against a 3.6 ms deadline) -- the
# correct action for those is to shed them, not to serve them. Two tenants have loose
# deadlines (they are the ones a pure deadline order starves under overload).
SERVING_CONTRACTS = [
    ("hot-loose-cheap",  140.0, 0.65),   # hot tenant: cheap probes, tolerant of queueing
    ("tight-expensive",    6.0, 0.99),   # tight deadline + expensive probes; hard queries infeasible
    ("tight-high",         9.0, 0.95),
    ("tight-medium",       7.0, 0.85),
    ("medium-high",       25.0, 0.95),
    ("medium-low",        30.0, 0.80),
    ("loose-expensive",  160.0, 0.99),   # loose deadline + expensive probes: EDF starves it
    ("tight-low",          5.0, 0.70),
]
SERVING_HARD_FRAC = 0.30
SERVING_HARD_DEADLINE_SCALE = 0.6
SERVING_SIGNAL_NOISE = 0.6
SERVING_OVERLOAD_RHO = 1.60
SERVING_DEFAULT_RHOS = (0.70, 1.00, 1.15, 1.60, 2.00, 2.50, 3.00)


def surface_idx(nprobe):
    """Index of the nearest published probe budget for an arbitrary integer nprobe."""
    a = np.asarray(nprobe, dtype=np.int64)
    return _NP_NEAREST[np.clip(a, 0, 256)]


def surface_lookup(nprobe):
    """(p50 service ms, recall) at the nearest published probe budget."""
    idx = int(_NP_NEAREST[int(np.clip(nprobe, 0, 256))])
    return float(SURFACE_P50[idx]), float(SURFACE_RECALL[idx])


def build_serving(seed=0, n_tenants=8, n_per_tenant=2000, rhos=SERVING_DEFAULT_RHOS,
                  cores=8, skew=None, eval_seed=None, hard_frac=SERVING_HARD_FRAC,
                  contracts=None):
    """Multi-tenant serving workload under a *mixed* set of joint latency+recall contracts.

    Offered load is defined relative to capacity: ``rho`` is the aggregate utilisation of
    ``cores`` cores at the per-tenant recall-minimal effort. ``skew`` re-weights the
    per-tenant arrival rates (a hot tenant). The sweep covers sub-capacity (0.70, 1.00, 1.15)
    and *overload* (1.60, 2.00, 2.50, 3.00) regimes: the interesting regime for a
    multi-tenant SLO contract is where demand exceeds capacity, because that is where
    ordering, effort allocation and admission stop being interchangeable.

    Each tenant gets one published (L, R) contract class (``SERVING_CONTRACTS``); 30% of
    its queries are hard (0.6x deadline, recall at half the probe budget). The per-query
    contract is shipped as ``deadline`` and ``recall_req`` so a submission sees exactly
    what it is being scored against. ``score`` is a *noisy* per-query difficulty signal
    (truth + Gaussian noise), not a perfect detector of ``hard``.

    Two independent arrival traces are emitted per regime: ``arrival`` (pilot, visible to
    submissions) and ``arrival_eval`` (held-out realisation used for scoring). A submission
    therefore returns a *policy*, not a replay. A policy may be load-blind (arrays shaped
    (T, nq), applied to every regime) or load-indexed (arrays shaped (R, T, nq), one policy
    per regime); both are policies over the held-out trace.
    """
    rng = np.random.RandomState(seed)
    pool = list(SERVING_CONTRACTS if contracts is None else contracts)
    tenants = []
    for i in range(n_tenants):
        name, L, R = pool[i % len(pool)]
        tenants.append({"id": i, "L": float(L), "R": float(R), "contract": str(name)})

    hard = rng.rand(n_tenants, n_per_tenant) < hard_frac
    # noisy per-query difficulty signal: overlaps the truth, so a signal-driven method is
    # a real (imperfect) baseline rather than a perfect detector of the hard flag
    score = hard.astype(np.float32) + SERVING_SIGNAL_NOISE * rng.randn(
        n_tenants, n_per_tenant).astype(np.float32)
    deadline = np.empty((n_tenants, n_per_tenant), dtype=np.float32)
    recall_req = np.empty_like(deadline)
    for i, t in enumerate(tenants):
        deadline[i] = t["L"]
        recall_req[i] = t["R"]
    deadline[hard] *= SERVING_HARD_DEADLINE_SCALE

    def nprobe_for(R):
        return int(SURFACE_NPROBE[np.argmax(SURFACE_RECALL >= R)])

    def p50_for(npb):
        return float(SURFACE_P50[int(np.argmin(np.abs(SURFACE_NPROBE - npb)))])

    base_nprobe = np.array([nprobe_for(t["R"]) for t in tenants], dtype=np.int64)
    base_ms = np.array([(1.0 - hard_frac) * p50_for(base_nprobe[i])
                        + hard_frac * p50_for(min(256, 2 * base_nprobe[i]))
                        for i in range(n_tenants)], dtype=np.float64)
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
            "deadline": deadline, "recall_req": recall_req,
            "arrival": arrival, "arrival_eval": arrival_eval,
            "rhos": rhos.tolist(), "cores": cores, "skew": skew.tolist(),
            "rates_qps": rates.tolist(), "mean_base_service_ms": mean_base_ms,
            "base_nprobe": base_nprobe.tolist(),
            "contracts": [{"name": str(n), "L": float(L), "R": float(R)}
                          for (n, L, R) in pool],
            "hard_frac": float(hard_frac),
            "hard_deadline_scale": float(SERVING_HARD_DEADLINE_SCALE),
            "overload_rho": float(SERVING_OVERLOAD_RHO),
            "surface": {"nprobe": SURFACE_NPROBE.tolist(),
                        "recall": SURFACE_RECALL.tolist(),
                        "p50": SURFACE_P50.tolist()}}


def _simulate_partitioned(arr, service, cores, priority=None, reservation=None, admit=None):
    """Per-tenant core allocation + non-preemptive priority within each tenant.

    Isolation is the default: cores are *partitioned* by the submission's ``reservation``
    vector (fractions, summing to <= 1). A tenant can therefore not steal another tenant's
    cores by submitting more work — the only cross-tenant lever is re-allocating cores, which
    is exactly the decision a multi-tenant vector DB operator actually owns.

    With ``reservation=None`` the partition is equal (``1/T`` per tenant), so a hot tenant is
    confined to its fair share and will visibly violate its SLO. A submission owns the entire
    allocation: fractions must sum to 1, leftover integer cores go to the tenants with the
    largest unmet need, and a tenant allocated zero cores is simply unserved (it fails).

    Returns per-query latency in ms and the integer core allocation actually used.
    """
    T, nq = arr.shape
    if reservation is None:
        res = np.full(T, 1.0 / T)
    else:
        res = np.clip(np.asarray(reservation, dtype=np.float64), 0.0, 1.0).ravel()
        if res.shape[0] != T:
            res = np.full(T, 1.0 / T)
        if res.sum() > 1.0:
            res = res / res.sum()
    alloc = np.floor(res * cores).astype(int)
    left = int(cores - alloc.sum())
    # unmet need in cores: demand (q/s) x mean service (ms) / 1000
    span = arr.max(axis=1) - arr.min(axis=1) + 1e-9
    demand = nq / span
    need = demand * np.asarray(service).mean(axis=1) / 1000.0
    while left > 0:
        deficit = need - alloc
        i = int(np.argmax(deficit))
        if deficit[i] <= 0:
            i = int(np.argmax(demand))
        alloc[i] += 1
        left -= 1
    # NOTE: no "everyone gets at least one core" repair. The submission owns the whole
    # allocation; a tenant it gives 0 cores to simply cannot be served and will fail its
    # contract. Repairing that silently would flatten exactly the allocation being tested
    # (it did: a deliberate 3-core skew collapsed back to 1 core each).
    pri = None if priority is None else np.asarray(priority, dtype=np.float64)
    adm = None if admit is None else np.asarray(admit, dtype=bool)
    lat = np.full((T, nq), 1e6, dtype=np.float32)   # unserved = 1e6 ms, fails any SLO
    for i in range(T):
        if alloc[i] == 0:
            continue
        order = [j for j in range(nq) if adm is None or bool(adm[i, j])]
        if pri is not None:
            order.sort(key=lambda j: (-pri[i, j], arr[i, j]))
        else:
            order.sort(key=lambda j: arr[i, j])
        free = [0.0] * int(alloc[i])
        for j in order:
            c = int(np.argmin(free))
            at = float(arr[i, j]) * 1000.0
            start = at if at > free[c] else free[c]
            free[c] = start + float(service[i, j])
            lat[i, j] = free[c] - at
    return lat, alloc


def _simulate(arr, service, cores, priority=None, admit=None):
    """Non-preemptive multi-core queue, FCFS or priority order, shared core pool.

    ``arr`` (T, nq) arrival seconds; ``service`` (T, nq) ms; ``priority`` (T, nq) higher is
    served first (omitted = FCFS); ``admit`` (T, nq) bool mask of queries that are served at
    all — a shed query consumes no capacity (its latency is handled by the caller).

    The shared pool is deliberate: a tenant *can* affect another tenant's latency by
    submitting work, which is what makes isolation a real problem rather than a partition
    exercise. Hard partitioning was measured and rejected — with total load near capacity it
    cannot create capacity (see the benchmark README).
    """
    T, nq = arr.shape
    lat = np.zeros((T, nq), dtype=np.float32)
    pri = np.zeros((T, nq), dtype=np.float64) if priority is None else np.asarray(priority, dtype=np.float64)
    flat = [(float(arr[i, j]), int(i), int(j)) for i in range(T) for j in range(nq)
            if admit is None or bool(admit[i][j])]
    flat.sort()
    free = [0.0] * cores
    waiting = []
    seq = 0
    k = 0
    n = len(flat)
    while k < n or waiting:
        if not waiting:
            a_ms = flat[k][0] * 1000.0
            c = int(np.argmin(free))
            if a_ms > free[c]:
                free[c] = a_ms
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


def _serving_contract(wl):
    """Per-query (deadline ms, recall requirement) arrays, with a fallback for old workloads."""
    hard = np.asarray(wl["hard"], dtype=bool)
    if "deadline" in wl and wl["deadline"] is not None:
        deadline = np.asarray(wl["deadline"], dtype=np.float32)
        recall_req = np.asarray(wl["recall_req"], dtype=np.float32)
        return deadline, recall_req
    T, nq = hard.shape
    scale = float(wl.get("hard_deadline_scale", SERVING_HARD_DEADLINE_SCALE))
    deadline = np.array([[t["L"]] * nq for t in wl["tenants"]], dtype=np.float32)
    deadline[hard] *= scale
    recall_req = np.array([[t["R"]] * nq for t in wl["tenants"]], dtype=np.float32)
    return deadline, recall_req


def _serving_service_recall(wl, efforts):
    """(service ms, effective recall) for a (T, nq) effort matrix; hard queries are harder."""
    hard = np.asarray(wl["hard"], dtype=bool)
    eff = np.asarray(efforts, dtype=np.int64)
    idx = surface_idx(eff)
    service = SURFACE_P50[idx].astype(np.float32)
    rec = SURFACE_RECALL[idx].astype(np.float32)
    hidx = surface_idx(np.maximum(1, eff // 2))
    recall = np.where(hard, np.minimum(rec, SURFACE_RECALL[hidx].astype(np.float32)), rec)
    return service, recall


def _serving_traces(wl):
    traces = wl.get("arrival_eval", wl["arrival"])
    traces = np.asarray(traces)
    if traces.ndim == 2:
        traces = traces[None, :, :]
    return traces


def eval_serving_regime(wl, efforts, priority=None, admission=None, r=0, reservation=None):
    """Score one load regime of a serving policy.

    Public helper: submissions use it to tune a policy on the *pilot* trace (pass ``wl``
    without ``arrival_eval``). Returns aggregate pass, per-tenant pass, latency tails,
    recall-SLO rate and goodput for regime ``r``.

    ``reservation`` (optional, per-tenant core fractions summing to <= 1) switches from the
    shared core pool to the *partitioned* simulator: each tenant only gets its reserved cores,
    so a tenant cannot take capacity from another. Without it the shared pool is used, which
    is the default the reference baselines are scored under.
    """
    hard = np.asarray(wl["hard"], dtype=bool)
    nT, nq = hard.shape
    eff = np.asarray(efforts, dtype=np.int64)
    if eff.shape != (nT, nq):
        raise ValueError(f"efforts shape {eff.shape} != {(nT, nq)}")
    service, recall = _serving_service_recall(wl, eff)
    pri = None if priority is None else np.asarray(priority, dtype=np.float64)
    adm = None if admission is None else np.asarray(admission, dtype=bool)
    deadline, recall_req = _serving_contract(wl)
    arr = _serving_traces(wl)[int(r)]
    if reservation is None:
        lat = _simulate(arr, service, int(wl["cores"]), pri, adm)
    else:
        lat, _ = _simulate_partitioned(arr, service, int(wl["cores"]), pri, reservation, adm)
    served = lat if adm is None else lat[adm]      # latency tails over *served* queries only
    if adm is not None:
        lat = np.where(adm, lat, 1e6)      # shed queries are not served -> they fail
    ok = (lat <= deadline) & (recall >= recall_req)
    span = float(arr.max() - arr.min()) + 1e-9
    if served.size:
        p95 = float(np.percentile(served, 95))
        p99 = float(np.percentile(served, 99))
    else:
        p95 = p99 = 1e6
    return {"slo_pass": float(ok.mean()),
            "tenant_pass": [float(x) for x in ok.mean(axis=1)],
            "p95_latency": p95,
            "p99_latency": p99,
            "recall_slo": float((recall >= recall_req).mean()),
            "goodput_qps": float(ok.sum() / span),
            "served_frac": float(adm.mean()) if adm is not None else 1.0}


def _policy_vector(pred, key, nT, nR):
    """Read a per-tenant vector field that may be load-blind (T,) or load-indexed (R, T)."""
    if key not in pred or pred[key] is None:
        return None
    a = np.asarray(pred[key], dtype=np.float64)
    if a.ndim == 1 and a.shape == (nT,):
        return a
    if a.ndim == 2 and a.shape == (nR, nT):
        return a
    raise ValueError(f"{key} must be 1-D (T,) or 2-D (R,T), got shape {a.shape}")


def _policy_array(pred, key, dtype, nT, nq, nR):
    """Read a policy field that may be load-blind (T, nq) or load-indexed (R, T, nq)."""
    if key not in pred or pred[key] is None:
        return None
    a = np.asarray(pred[key])
    if a.ndim == 2:
        if a.shape != (nT, nq):
            raise ValueError(f"{key} shape {a.shape} != {(nT, nq)} (or (R,{nT},{nq}))")
        return a.astype(dtype)
    if a.ndim == 3:
        if a.shape != (nR, nT, nq):
            raise ValueError(f"{key} shape {a.shape} != {(nR, nT, nq)}")
        return a.astype(dtype)
    raise ValueError(f"{key} must be 2-D (T,nq) or 3-D (R,T,nq), got shape {a.shape}")


def eval_serving(pred, wl):
    """Score a serving submission across every load regime in the workload.

    A submission returns ``efforts`` (T, nq) and, optionally, ``priority`` (T, nq) where
    *higher* priority is served first (deadline-aware scheduling, e.g. EDF),
    ``admission`` (T, nq) of queries that are served at all (shed queries fail their
    contract) and ``reservation`` (T,) of per-tenant core fractions summing to <= 1 (opt in
    to the partitioned simulator). The array fields may instead be shaped (R, T, nq) and
    the reservation (R, T), i.e. one policy per load regime — the load regime is public (the
    pilot trace ships with the workload), the held-out trace realisation is not.

    Metrics: ``slo_pass`` (mean over regimes), ``slo_pass_worst`` (worst regime),
    ``slo_pass_overload`` (mean over regimes with rho >= 1.6), ``tenant_pass_worst``
    (worst tenant in the worst regime — the isolation metric), latency tails, recall-SLO
    rate and goodput. Scoring uses the held-out arrival traces.
    """
    hard = np.asarray(wl["hard"], dtype=bool)
    nT, nq = hard.shape
    rhos = [float(x) for x in wl.get("rhos", [1.0])]
    nR = len(rhos)
    try:
        efforts = _policy_array(pred, "efforts", np.int64, nT, nq, nR)
        if efforts is None:
            return {"error": "missing efforts"}
        pri = _policy_array(pred, "priority", np.float64, nT, nq, nR)
        adm = _policy_array(pred, "admission", bool, nT, nq, nR)
        res = _policy_vector(pred, "reservation", nT, nR)
    except ValueError as e:
        return {"error": str(e)}

    passes, per_tenant, p95s, p99s, recalls, goodputs, served = [], [], [], [], [], [], []
    for r in range(nR):
        e = efforts[r] if efforts.ndim == 3 else efforts
        p = None if pri is None else (pri[r] if pri.ndim == 3 else pri)
        a = None if adm is None else (adm[r] if adm.ndim == 3 else adm)
        rv = None if res is None else (res[r] if res.ndim == 2 else res)
        m = eval_serving_regime(wl, e, p, a, r, rv)
        passes.append(m["slo_pass"])
        per_tenant.append(m["tenant_pass"])
        p95s.append(m["p95_latency"])
        p99s.append(m["p99_latency"])
        recalls.append(m["recall_slo"])
        goodputs.append(m["goodput_qps"])
        served.append(m["served_frac"])
    overload = [i for i, rho in enumerate(rhos) if rho >= SERVING_OVERLOAD_RHO]
    overload_pass = float(np.mean([passes[i] for i in overload])) if overload else float("nan")
    tenant_worst = float(min(min(pt) for pt in per_tenant))
    return {"slo_pass": float(np.mean(passes)),
            "slo_pass_worst": float(np.min(passes)),
            "slo_pass_overload": overload_pass,
            "tenant_pass_worst": tenant_worst,
            "p95_latency": float(np.mean(p95s)),
            "p99_latency": float(np.mean(p99s)),
            "recall_slo": float(np.mean(recalls)),
            "goodput_qps": float(np.mean(goodputs)),
            "served_frac": float(np.mean(served)),
            "_rhos": rhos,
            "_per_regime_slo_pass": [float(x) for x in passes],
            "_per_regime_tenant_pass": per_tenant,
            "_per_regime_served_frac": [float(x) for x in served]}

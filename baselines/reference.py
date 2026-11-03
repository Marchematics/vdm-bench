"""Reference submissions for VDM-Bench (define the tasks; not the final word)."""
import math

import numpy as np

from vdmbench import core


def analytics_exact(wl):
    return {"range_count": wl["truth"].tolist(),
            "coverage": wl["coverage_chosen"].tolist(),
            "join": wl["join_pairs"].tolist(),
            "cost": float(wl["n"] * len(wl["queries"]) * len(wl["radii"]))}


def analytics_sampling(data_root, wl, s=2000, seed=0, dataset="sift"):
    X, _ = core.load_dataset(data_root, dataset, wl["n"])
    Q = wl["queries"]; rng = np.random.RandomState(seed)
    est = np.zeros_like(wl["truth"], dtype=np.float64)
    for i, q in enumerate(Q):
        idx = rng.randint(0, len(X), s)
        d = np.sqrt(core.sqdist(q, X[idx]))
        for j, r in enumerate(wl["radii"]):
            est[i, j] = len(X) * float((d <= r).mean())
    k = int(wl["k"])
    return {"range_count": est.tolist(),
            "coverage": rng.choice(len(X), k, replace=False).tolist(),
            "join": [],
            "cost": float(s * len(Q) * len(wl["radii"]))}


def analytics_ivf(data_root, wl, nlist=256, nprobe=32, seed=0, dataset="sift"):
    import faiss
    X, _ = core.load_dataset(data_root, dataset, wl["n"])
    Q = wl["queries"]
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(X.shape[1]), X.shape[1], nlist)
    ivf.train(X[np.random.RandomState(seed).choice(len(X), min(50000, len(X)), replace=False)])
    ivf.add(X); ivf.nprobe = nprobe
    est = np.zeros_like(wl["truth"], dtype=np.float64)
    for i, q in enumerate(Q):
        for j, r in enumerate(wl["radii"]):
            lims, _, _ = ivf.range_search(q[None, :].astype(np.float32), float(r * r))
            est[i, j] = lims[1] - lims[0]
    rng = np.random.RandomState(seed)
    k = int(wl["k"])
    return {"range_count": est.tolist(),
            "coverage": rng.choice(len(X), k, replace=False).tolist(),
            "join": [],
            "cost": float(len(Q) * len(wl["radii"]) * nprobe / nlist * len(X))}


def quality_exact(wl):
    out = []
    for g in wl["instances"]:
        out.append({"family": g["family"], "level": g["level"],
                    "dup_rate": g["dup_rate"], "n_outliers": g["n_outliers"],
                    "drift": g["drift"]})
    return {"instances": out, "cost": 1e12}


def quality_naive(wl):
    out = [{"family": g["family"], "level": g["level"], "dup_rate": 0.0,
            "n_outliers": 0, "drift": 0.0} for g in wl["instances"]]
    return {"instances": out, "cost": 0.0}


def _nprobe_for(R):
    arr = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    rec = np.array([0.4836, 0.6707, 0.8276, 0.9328, 0.9840, 0.9979, 0.9999, 1.0, 1.0])
    return int(arr[np.argmax(rec >= R)])


def serving_static(wl):
    T = len(wl["tenants"]); nq = wl["hard"].shape[1]
    eff = np.zeros((T, nq), dtype=np.int64)
    for i, t in enumerate(wl["tenants"]):
        eff[i, :] = 2 * _nprobe_for(t["R"])
    return {"efforts": eff.tolist()}


def serving_adaptive(wl):
    T = len(wl["tenants"]); nq = wl["hard"].shape[1]
    eff = np.zeros((T, nq), dtype=np.int64)
    for i, t in enumerate(wl["tenants"]):
        eff[i, :] = _nprobe_for(t["R"])
    return {"efforts": eff.tolist()}


def serving_oracle(wl):
    T = len(wl["tenants"]); nq = wl["hard"].shape[1]
    eff = np.zeros((T, nq), dtype=np.int64)
    for i, t in enumerate(wl["tenants"]):
        base = _nprobe_for(t["R"])
        for j in range(nq):
            eff[i, j] = 2 * base if wl["hard"][i, j] else base
    return {"efforts": eff.tolist()}


def analytics_anavec(data_root, wl, seed=0, dataset="sift", n_probe_lists=32, sample_n=4000):
    """Reference AnaVec submission: two-phase range counts + HNSW candidate coverage.

    The planner's range estimator is *two-phase* rather than plain sampling:

      1. Probe the `n_probe_lists` inverted lists nearest to the query and count their
         members within the radius **exactly** (cost: the size of those lists).
      2. Sample `sample_n` points uniformly from everything else and scale.

    On real clustered embeddings the near lists hold most of the hits for selective radii, so
    the exact part removes most of the estimator's variance and only the small residual is
    sampled. Measured on real SIFT at 100k (``scripts`` in this repo, 100 queries x 5 radii):
    uniform sampling costs 1.2M distance evaluations for 0.091 mean relative error, while the
    two-phase estimator costs 1.07M for **0.033**, and 1.68M for **0.015** -- i.e. 2.8x to 4.8x
    more accurate at *lower* cost. A certified-list importance-sampling variant was also
    measured and rejected (0.16-0.22): the certified set over-covers, so reweighting costs more
    variance than the exact prefix removes.
    """
    import faiss
    X, _ = core.load_dataset(data_root, dataset, wl["n"])
    n = len(X)
    Q = wl["queries"]
    radii = wl["radii"]
    rng = np.random.RandomState(seed)

    # nlist must grow with n, otherwise at 10M each list holds ~39k vectors and probing 32
    # lists exactly costs 1.25M distance computations per query. n//1000 keeps list size near
    # 1000 and the two-phase cost at tens of thousands per query at any scale.
    # ---- plan selection (this is the point of the paper) -------------------
    # Two-phase wins at small n and LOSES at 10M: measured on SIFT10M the 4,000-sample residual
    # estimate covers 0.046% of the residual, so matched-cost uniform sampling wins at every
    # point of the frontier (experiments/range10m_frontier_2026-11-04.md). The planner therefore
    # picks by scale: two-phase below 2M vectors, budgeted uniform sampling above it.
    est = np.zeros_like(wl["truth"], dtype=np.float64)
    cost = 0.0
    if n >= 2_000_000:
        S = max(10_000, int(0.01 * n))
        for i, q in enumerate(Q):
            samp = rng.choice(n, S, replace=False)
            dd = np.sqrt(np.maximum(float(q @ q) + np.einsum("ij,ij->i", X[samp], X[samp])
                                    - 2.0 * X[samp] @ q, 0.0))
            cost += S
            for j, r in enumerate(radii):
                est[i, j] = n * float((dd <= r).sum()) / S
    else:
        nlist = int(min(16384, max(256, n // 1000)))
        km = faiss.Kmeans(X.shape[1], nlist, niter=10, verbose=False, seed=seed)
        km.train(np.ascontiguousarray(X[: min(80_000, n)]))
        C = km.centroids.astype(np.float32)
        qflat = faiss.IndexFlatL2(X.shape[1])
        qflat.add(C)
        assign = np.empty(n, dtype=np.int64)
        for s0 in range(0, n, 100_000):
            _, idx = qflat.search(np.ascontiguousarray(X[s0:s0 + 100_000]), 1)
            assign[s0:s0 + len(idx)] = idx.ravel()
        members = [np.flatnonzero(assign == c) for c in range(nlist)]
        dQ = np.sqrt(core.pairwise(Q, C))
        dQX = np.einsum("ij,ij->i", X, X)
        for i, q in enumerate(Q):
            order = np.argsort(dQ[i])[:n_probe_lists]
            pid = np.concatenate([members[c] for c in order])
            mask = np.ones(n, dtype=bool)
            mask[pid] = False
            rest = np.flatnonzero(mask)
            samp = rest if len(rest) <= sample_n else rng.choice(rest, sample_n, replace=False)
            cost += len(pid) + len(samp)
            q2 = float(q @ q)
            dp = np.sqrt(np.maximum(q2 + dQX[pid] - 2.0 * X[pid] @ q, 0.0))
            ds = np.sqrt(np.maximum(q2 + dQX[samp] - 2.0 * X[samp] @ q, 0.0))
            for j, r in enumerate(radii):
                exact = int((dp <= r).sum())
                hit = int((ds <= r).sum()) if len(ds) else 0
                est[i, j] = exact + (len(rest) * hit / len(samp) if len(samp) else 0.0)

    # coverage candidates
    hnsw = faiss.IndexHNSWFlat(X.shape[1], 32)
    hnsw.hnsw.efConstruction = 100
    hnsw.add(X); hnsw.hnsw.efSearch = 64
    A = wl["anchors"]
    _, I = hnsw.search(A.astype(np.float32), 20)
    cand = set(I.ravel().tolist())
    cand.update(rng.choice(n, 5000, replace=False).tolist())
    pi = rng.randint(0, len(A), size=(200, 2))
    mids = (A[pi[:, 0]] + A[pi[:, 1]]) / 2.0
    _, I2 = hnsw.search(mids.astype(np.float32), 2)
    cand.update(I2.ravel().tolist())
    Cc = np.asarray(sorted(cand), dtype=np.int64)
    D = np.sqrt(core.pairwise(A, X[Cc]))
    covered = np.zeros(len(A), dtype=bool); sel = []
    for _ in range(int(wl["k"])):
        gain = ((D <= wl["r_cov"]) & (~covered)[:, None]).sum(axis=0)
        j = int(gain.argmax()); sel.append(int(Cc[j])); covered |= D[:, j] <= wl["r_cov"]
    # join: IVF range search on the 10k slice
    Xj = X[: int(wl["X_join_n"])]
    ivj = faiss.IndexIVFFlat(faiss.IndexFlatL2(Xj.shape[1]), Xj.shape[1], 128)
    ivj.train(Xj); ivj.add(Xj); ivj.nprobe = 16
    lims, _, II = ivj.range_search(Xj.astype(np.float32), float(wl["r_join"] ** 2))
    pairs = []
    for i in range(len(Xj)):
        for j in II[lims[i]:lims[i + 1]]:
            if j > i:
                pairs.append([i, int(j)])
    return {"range_count": est.tolist(), "coverage": sel, "join": pairs,
            "cost": float(cost + len(A) * len(Cc) + len(Xj) * 16 / 128 * len(Xj))}


def quality_exact(wl):
    out = []
    for g in wl["instances"]:
        out.append({"family": g["family"], "level": g["level"],
                    "dup_rate": g["dup_rate"], "n_outliers": g["n_outliers"],
                    "drift": g["drift"]})
    return {"instances": out, "cost": 1e12}


def quality_naive(wl):
    out = [{"family": g["family"], "level": g["level"], "dup_rate": 0.0,
            "n_outliers": 0, "drift": 0.0} for g in wl["instances"]]
    return {"instances": out, "cost": 0.0}


def _nprobe_for(R):
    arr = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    rec = np.array([0.4836, 0.6707, 0.8276, 0.9328, 0.9840, 0.9979, 0.9999, 1.0, 1.0])
    return int(arr[np.argmax(rec >= R)])


def serving_static(wl):
    T = len(wl["tenants"]); nq = wl["hard"].shape[1]
    eff = np.zeros((T, nq), dtype=np.int64)
    for i, t in enumerate(wl["tenants"]):
        eff[i, :] = 2 * _nprobe_for(t["R"])
    return {"efforts": eff.tolist()}


def serving_adaptive(wl):
    T = len(wl["tenants"]); nq = wl["hard"].shape[1]
    eff = np.zeros((T, nq), dtype=np.int64)
    for i, t in enumerate(wl["tenants"]):
        eff[i, :] = _nprobe_for(t["R"])
    return {"efforts": eff.tolist()}


def serving_oracle(wl):
    T = len(wl["tenants"]); nq = wl["hard"].shape[1]
    eff = np.zeros((T, nq), dtype=np.int64)
    for i, t in enumerate(wl["tenants"]):
        base = _nprobe_for(t["R"])
        for j in range(nq):
            eff[i, j] = 2 * base if wl["hard"][i, j] else base
    return {"efforts": eff.tolist()}


def _rff_features(V, gamma, m=256, seed=0):
    """Random Fourier features for a Gaussian kernel; returns (n, m) and the op cost."""
    rng = np.random.RandomState(seed)
    W = rng.normal(0.0, np.sqrt(2.0 * gamma), (V.shape[1], m)).astype(np.float32)
    b = rng.uniform(0.0, 2.0 * np.pi, m).astype(np.float32)
    Z = np.sqrt(2.0 / m) * np.cos(V @ W + b)
    return Z, float(len(V) * m)


def _mmd2_from_features(ZA, ZB):
    return float(np.sum((ZA.mean(0) - ZB.mean(0)) ** 2))


def _certified_dup_rate(X, eps, nlist=256, seed=0, batch=512):
    """Certified epsilon-graph vertex rate.

    Probes every list c with ||x - c|| <= eps + R_max(c). That predicate is necessary
    for completeness (if ||x-y|| <= eps and y is in c then ||x-c|| <= eps + R_max(c)),
    so no epsilon-pair is missed regardless of a probe budget.
    """
    import faiss
    d = X.shape[1]
    km = faiss.Kmeans(d, nlist, niter=10, verbose=False, seed=seed)
    ntr = min(40_000, len(X))
    km.train(np.ascontiguousarray(X[np.random.RandomState(seed).choice(len(X), ntr, replace=False)]))
    C = km.centroids.astype(np.float32)
    D2 = core.pairwise(X, C)
    assign = D2.argmin(1)
    r_max = np.zeros(nlist, dtype=np.float32)
    for c in range(nlist):
        ids = np.flatnonzero(assign == c)
        if len(ids):
            r_max[c] = float(np.sqrt(D2[ids, c].max()))
    members = [np.flatnonzero(assign == c) for c in range(nlist)]
    thr2 = (eps + r_max) ** 2
    verts = np.zeros(len(X), dtype=bool)
    evals = 0
    for s in range(0, len(X), batch):
        B = X[s:s + batch]
        sel = core.pairwise(B, C) <= thr2[None, :]
        cols = np.flatnonzero(sel.any(0))
        for c in cols:
            qi = np.flatnonzero(sel[:, c])
            mem = members[c]
            if not len(mem):
                continue
            dd = np.sqrt(np.maximum(
                (B[qi] ** 2).sum(1)[:, None] + (X[mem] ** 2).sum(1)[None, :]
                - 2.0 * B[qi] @ X[mem].T, 0.0))
            evals += dd.size
            r, cc = np.nonzero(dd <= eps)
            gq, gc = s + qi[r], mem[cc]
            keep = gq != gc
            verts[gq[keep]] = True
            verts[gc[keep]] = True
    return float(verts.mean()), float(evals)


def quality_vecprof(data_root, wl, seed=0, m=256, n_bias=12):
    """Reference VecProf submission: certified duplicates + debiased RFF-MMD + sampled kNN.

    All three quantities are the workload's *published* rules; the submission's claim is
    that they can be computed far below exhaustive cost. The drift estimate is
    *debiased*: an RFF MMD^2 estimator has a positive O(1/m) bias, so the sketch's own
    null (clean vs clean at the same sample sizes) is measured and subtracted. Without
    this the clean null sits above the exact threshold and every clean collection is a
    false positive.
    """
    rules = wl["rules"]
    eps = float(rules["eps"])
    k_nn = int(rules["k_nn"])
    thr = float(rules["outlier_thr"])
    gamma = float(rules["gamma"])
    sn = int(rules["sample_n"])
    rn = int(rules["ref_n"])

    # reproduce the published random split: reference = perm[:ref_n], instances = the rest
    dn = int(rules["dataset_n"])
    X_all, _ = core.load_sift(data_root, dn)
    Xa = np.asarray(X_all)
    perm = np.random.RandomState(int(rules["split_seed"])).permutation(len(Xa))
    ref_idx = perm[:rn]          # permutation order: ref[:sample_n] must be a random subset
    inst_idx = np.sort(perm[rn:])
    refall = np.ascontiguousarray(Xa[ref_idx])
    sets = core.quality_sets_from(Xa[inst_idx], seed=seed)
    Zref, c_ref = _rff_features(refall, gamma, m=m, seed=seed)
    total_cost = c_ref
    # sketch null at exactly the sizes the instances use -> the bias to subtract
    biases = []
    for t in range(n_bias):
        p = np.random.RandomState(700 + t).permutation(len(refall))
        biases.append(_mmd2_from_features(Zref[p[:sn]], Zref[p[sn:2 * sn]]))
    bias = float(np.mean(biases))

    out = []
    for family, level, X in sets:
        dup_rate, dup_cost = _certified_dup_rate(X, eps, seed=seed)
        total_cost += dup_cost
        # outliers: k-NN distance for *every* point via the store's IVF index.
        # Sampling 2k of 42k gave 25 counts of error purely from sampling noise (the true
        # rate is ~1.1%, so 2k points see ~22 hits); the index makes a full pass affordable
        # (n * nprobe/nlist * n instead of n * n).
        import faiss
        nlist = int(min(256, max(1, len(X) // 64)))
        quant = faiss.IndexFlatL2(X.shape[1])
        ivf = faiss.IndexIVFFlat(quant, X.shape[1], nlist)
        ivf.train(np.ascontiguousarray(X[: min(40_000, len(X))]))
        ivf.add(np.ascontiguousarray(X))
        ivf.nprobe = 32   # k-NN recall matters: nprobe=8 under-scans and over-counts outliers
        Xc = np.ascontiguousarray(X)
        Dk, _ = ivf.search(Xc, k_nn + 1)
        dk = np.sqrt(np.maximum(Dk[:, -1], 0.0))
        total_cost += float(len(X) * (32.0 / nlist) * len(X))
        n_out = int((dk > thr).sum())
        # drift: debiased RFF-MMD against the clean reference
        ss = np.random.RandomState(seed + 1).choice(len(X), sn, replace=False)
        Zi, c_i = _rff_features(np.ascontiguousarray(X[ss]), gamma, m=m, seed=seed)
        total_cost += c_i
        drift = _mmd2_from_features(Zi, Zref[:sn]) - bias
        out.append({"family": family, "level": level,
                    "dup_rate": dup_rate, "n_outliers": n_out, "drift": drift})
    return {"instances": out, "cost": float(total_cost)}


def serving_signal(wl):
    T = len(wl["tenants"]); nq = wl["hard"].shape[1]
    eff = np.zeros((T, nq), dtype=np.int64)
    sc = wl["score"]
    for i, t in enumerate(wl["tenants"]):
        base = _nprobe_for(t["R"])
        for j in range(nq):
            eff[i, j] = 2 * base if sc[i, j] > 0.55 else base
    return {"efforts": eff.tolist()}


def _surface(npb):
    idx = int(np.argmin(np.abs(core.SURFACE_NPROBE - npb)))
    return float(core.SURFACE_P50[idx]), float(core.SURFACE_RECALL[idx])


def serving_edf(wl):
    """Ablation: deadline-aware ordering only, per-query minimal effort (no quotas).

    Isolates the scheduling lever from the effort-allocation lever. Shipped as a
    first-class baseline because it is the comparison that matters: published SLO
    methods tune *effort* and leave service order at FCFS.
    """
    tenants, hard = wl["tenants"], wl["hard"]
    T, nq = hard.shape
    L = np.array([t["L"] for t in tenants], dtype=np.float64)
    pri = (-L[:, None]).repeat(nq, axis=1) + 0.5 * hard
    efforts = np.zeros((T, nq), dtype=np.int64)
    for i in range(T):
        base = _nprobe_for(tenants[i]["R"])
        efforts[i] = base
        rec_h = _surface(max(1, base // 2))[1]
        if rec_h < tenants[i]["R"]:
            efforts[i][hard[i]] = min(256, 2 * base)
    return {"efforts": efforts.tolist(), "priority": np.asarray(pri).tolist()}


def serving_vecslo(wl, n_rounds=1, verbose=False):
    """Reference VecSLO submission: deadline order + per-tenant effort quota.

    Two further mechanisms were implemented and measured on this workload and are **not**
    shipped, because both lose to plain deadline ordering here (measurements in the README):
    hard per-tenant core partitions (cannot create capacity when total load ~= capacity) and
    admission/shedding (removing 5-6% of queries costs more in failed contracts than it buys
    in latency). The evaluator still accepts both (`reservation`, `admission`) so later work
    can revisit them; the reference submission deliberately only uses what wins.

    The planner sees only the pilot traces; scoring uses the held-out traces across regimes.
    """
    tenants, hard = wl["tenants"], wl["hard"]
    T, nq = hard.shape
    pilot = dict(wl)
    pilot.pop("arrival_eval", None)

    L = np.array([t["L"] for t in tenants], dtype=np.float64)
    pri = (-L[:, None]).repeat(nq, axis=1) + 0.5 * hard     # EDF, hard queries first

    def score(efforts):
        return core.eval_serving({"efforts": efforts.tolist(), "priority": pri.tolist()}, pilot)

    def nprobe_for(rec):
        return int(core.SURFACE_NPROBE[np.argmax(core.SURFACE_RECALL >= rec)])

    R = np.array([t["R"] for t in tenants], dtype=np.float64)
    efforts = np.zeros((T, nq), dtype=np.int64)
    for i in range(T):
        base = nprobe_for(R[i])
        efforts[i] = base
        if _surface(max(1, base // 2))[1] < R[i]:
            efforts[i][hard[i]] = min(256, 2 * base)
    best = (score(efforts)["slo_pass"], efforts.copy())
    for _ in range(n_rounds):
        improved = False
        for i in range(T):
            for lv in [int(x) for x in core.SURFACE_NPROBE]:
                cand = best[1].copy()
                cand[i] = lv
                m = score(cand)
                if m["slo_pass"] > best[0] + 1e-6:
                    best = (m["slo_pass"], cand)
                    improved = True
        if not improved:
            break
    if verbose:
        print("vecslo slo_pass", best[0], flush=True)
    return {"efforts": best[1].tolist(), "priority": pri.tolist()}

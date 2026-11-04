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


# --------------------------------------------------------------------------- #
# Index-stratified range estimation (the plan that wins at 10M)
# --------------------------------------------------------------------------- #
def _cell_hit_prob(dq, r, grid, qs):
    """Predicted fraction of a cell inside the query ball.

    d(q, x)^2 ~= d_c^2 + t^2 for a point x at distance t from its centroid c, so a cell can
    hold a point within r only if t <= sqrt(r^2 - d_c^2); the cell's within-cell radius
    quantiles give that probability without touching any of its members.
    """
    nc = len(dq)
    rad = np.sqrt(np.maximum(r * r - dq * dq, 0.0))
    idx = np.clip((grid <= rad[:, None]).sum(axis=1), 1, len(qs) - 1)
    ar = np.arange(nc)
    lo, hi = grid[ar, idx - 1], grid[ar, idx]
    w = np.where(hi > lo, (rad - lo) / np.maximum(hi - lo, 1e-9), 1.0)
    p = qs[idx - 1] + np.clip(w, 0.0, 1.0) * (qs[idx] - qs[idx - 1])
    return np.clip(np.where(rad <= 0.0, 0.0, p), 0.0, 1.0)


def _allocate(w, budget, min_per_cell=1):
    """Integer per-cell sample allocation summing to `budget` (largest remainder)."""
    w = np.asarray(w, dtype=np.float64)
    s = np.zeros_like(w, dtype=np.int64)
    s[w > 0] = min_per_cell
    rem = budget - int(s.sum())
    if rem <= 0:
        return s
    ww = w / w.sum()
    add = np.floor(rem * ww).astype(np.int64)
    s += add
    left = budget - int(s.sum())
    if left > 0:
        s[np.argsort(-(rem * ww - add))[:left]] += 1
    return s


def _sample_within(alloc, sizes, order, off, rng):
    """Draw each sample from inside its own cell (with replacement: unbiased, vectorised)."""
    cells = np.flatnonzero(alloc > 0)
    if len(cells) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    ctx = np.repeat(cells, alloc[cells])
    starts = np.repeat(off[cells], alloc[cells])
    lens = np.repeat(sizes[cells], alloc[cells]).astype(np.float64)
    pos = starts + (rng.random_sample(len(ctx)) * lens).astype(np.int64)
    np.minimum(pos, np.repeat(off[cells + 1], alloc[cells]) - 1, out=pos)
    return ctx, order[pos]


def range_stratified(X, Q, radii, nlist=None, sample_n=None, seed=0, batch=200_000):
    """Stratified range-count estimator: stratify by IVF cell, allocate by Neyman's rule.

    The estimator is ``sum_c size_c * hits_c / s_c`` (plus the exact part when a prefix of
    cells is scanned), which is unbiased for any positive allocation, so a mispredicted
    allocation costs variance but never correctness.  Measured on SIFT10M, 100 queries x 5
    radii at 100k distance evaluations per query: mean relative error 0.0147 against 0.0297
    for equally-budgeted uniform sampling (2.0x), p95 0.054 vs 0.122 (2.3x), and 4.1x less
    compute than uniform sampling needs to reach the same accuracy
    (experiments/stratification_10m.md).  Spending part of the budget on an exact IVF prefix
    instead *loses* at this scale (0.0173 at 16 lists), which is why the planner switches the
    lever, not the family, when the list population becomes small relative to the ball.
    """
    import faiss
    N = len(X)
    nlist = int(nlist or min(16384, max(256, N // 1000)))
    sample_n = int(sample_n or max(10_000, int(0.01 * N)))
    rng = np.random.RandomState(seed)
    t0 = 0
    km = faiss.Kmeans(X.shape[1], nlist, niter=10, verbose=False, seed=seed)
    km.train(np.ascontiguousarray(X[: min(400_000, N)]))
    C = km.centroids.astype(np.float32)
    qflat = faiss.IndexFlatL2(X.shape[1])
    qflat.add(C)
    assign = np.empty(N, dtype=np.int32)
    for s0 in range(0, N, batch):
        _, idx = qflat.search(np.ascontiguousarray(X[s0:s0 + batch]), 1)
        assign[s0:s0 + len(idx)] = idx.ravel()
    sizes = np.bincount(assign, minlength=nlist).astype(np.int64)
    order = np.argsort(assign, kind="stable")
    off = np.zeros(nlist + 1, dtype=np.int64)
    np.cumsum(sizes, out=off[1:])
    qs = np.linspace(0.0, 1.0, 33)
    tvec = np.empty(N, dtype=np.float32)
    for s0 in range(0, N, batch):
        xc = np.asarray(X[s0:s0 + batch], dtype=np.float32)
        dd = xc - C[assign[s0:s0 + len(xc)]]
        tvec[s0:s0 + len(xc)] = np.sqrt(np.einsum("ij,ij->i", dd, dd))
    tord = tvec[order]
    grid = np.zeros((nlist, len(qs)), dtype=np.float32)
    for c in range(nlist):
        seg = tord[off[c]:off[c + 1]]
        grid[c] = np.quantile(seg, qs) if len(seg) > 4 else (seg.mean() if len(seg) else 0.0)

    dQX = np.einsum("ij,ij->i", X, X)
    dQ = np.sqrt(core.pairwise(Q, C))
    est = np.zeros((len(Q), len(radii)), dtype=np.float64)
    cost = 0.0
    for i, q in enumerate(Q):
        q2 = float(q @ q)
        Ws = []
        for r in radii:
            p = _cell_hit_prob(dQ[i], float(r), grid, qs)
            pc = np.clip(p, 1e-3, 1 - 1e-3)
            w = sizes * np.sqrt(pc * (1 - pc))
            Ws.append(w / w.sum())
        alloc = _allocate(np.mean(Ws, axis=0), sample_n)
        cells, idx = _sample_within(alloc, sizes, order, off, rng)
        dd = np.sqrt(np.maximum(q2 + dQX[idx] - 2.0 * X[idx] @ q, 0.0))
        cost += float(len(idx)) + float(nlist)          # samples + query-to-centroid distances
        for j, r in enumerate(radii):
            hit = np.bincount(cells, weights=(dd <= r).astype(np.float64), minlength=nlist)
            est[i, j] = float((sizes * hit / np.maximum(alloc, 1)).sum())
    return est, cost


def analytics_anavec(data_root, wl, seed=0, dataset="sift", n_probe_lists=32, sample_n=4000,
                     range_plan=None):
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
    plan = range_plan or ("stratified" if n >= 2_000_000 else "two-phase")
    if plan == "uniform":
        S = max(10_000, int(0.01 * n)) if n >= 2_000_000 else wl["n"]
        for i, q in enumerate(Q):
            samp = rng.choice(n, S, replace=False)
            dd = np.sqrt(np.maximum(float(q @ q) + np.einsum("ij,ij->i", X[samp], X[samp])
                                    - 2.0 * X[samp] @ q, 0.0))
            cost += S
            for j, r in enumerate(radii):
                est[i, j] = n * float((dd <= r).sum()) / S
    elif plan == "stratified":
        est, cost = range_stratified(X, Q, radii, seed=seed)
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
            if n_probe_lists > 0:
                order = np.argsort(dQ[i])[:n_probe_lists]
                pid = np.concatenate([members[c] for c in order])
            else:
                pid = np.empty(0, dtype=np.int64)   # pure-sampling ablation
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


# --------------------------------------------------------------------------- #
# Serving baselines
# --------------------------------------------------------------------------- #
SERVING_SIGNAL_THRESHOLD = 0.5


def _nprobe_for(R):
    return int(core.SURFACE_NPROBE[np.argmax(core.SURFACE_RECALL >= R)])


def _base_efforts(wl):
    """Recall-minimal effort for the *soft* contract of every tenant."""
    T, nq = wl["hard"].shape
    base = np.array([_nprobe_for(t["R"]) for t in wl["tenants"]], dtype=np.int64)
    return np.repeat(base[:, None], nq, axis=1)


def _minimal_efforts(wl):
    """Recall-minimal effort per query: base for soft queries, 2x base for hard queries."""
    eff = _base_efforts(wl).copy()
    hard = np.asarray(wl["hard"], dtype=bool)
    eff[hard] = np.minimum(256, 2 * eff[hard])
    return eff


def _edf_priority(wl):
    """Higher priority = smaller deadline (earliest deadline first)."""
    return -np.asarray(wl["deadline"], dtype=np.float64)


def serving_static(wl):
    """Fixed over-provisioned effort (2x recall-minimal), FCFS: the effort-only baseline."""
    return {"efforts": (2 * _base_efforts(wl)).tolist()}


def serving_adaptive(wl):
    """Recall-minimal effort per tenant, FCFS: effort-only without over-provisioning."""
    return {"efforts": _base_efforts(wl).tolist()}


def serving_signal(wl):
    """Effort boost where the *noisy* difficulty signal fires, FCFS."""
    base = _base_efforts(wl)
    sc = np.asarray(wl["score"], dtype=np.float64)
    eff = np.where(sc > SERVING_SIGNAL_THRESHOLD, 2 * base, base)
    return {"efforts": eff.astype(np.int64).tolist()}


def serving_edf(wl):
    """Deadline ordering only: EDF priority + recall-minimal effort, no admission.

    The ablation that isolates ordering from effort allocation and admission. It is shipped
    as a first-class baseline because it is the comparison that matters: deadline order fixes
    latency compliance under load, but it is *not* a multi-tenant SLO policy — it starves
    tenants with loose deadlines and serves expensive hard queries that cannot make their
    deadline.
    """
    return {"efforts": _minimal_efforts(wl).tolist(),
            "priority": _edf_priority(wl).tolist()}


def _admission_mask(nq, frac_soft, frac_hard, hard_row):
    """Admit the first round(frac*n) queries of each class.

    The subset is fixed (trace-independent): on the held-out trace the query indices are an
    exchangeable sample of arrivals, so this is a *policy* ("admit a fraction of each
    tenant's contract classes"), not a replay of the pilot arrivals.
    """
    adm = np.zeros(nq, dtype=bool)
    soft = np.flatnonzero(~hard_row)
    hardi = np.flatnonzero(hard_row)
    ks = int(round(float(frac_soft) * len(soft)))
    kh = int(round(float(frac_hard) * len(hardi)))
    if ks > 0:
        adm[soft[:ks]] = True
    if kh > 0:
        adm[hardi[:kh]] = True
    return adm


def _serving_quota_search(wl, mode="quota", n_rounds=2, warm=None, verbose=False):
    """Coordinate ascent over per-tenant effort quotas and admission quotas.

    ``mode="quota"`` (the reference VecSLO-style policy): per tenant, one effort multiplier
    for its hard contract class and one admission fraction. Deadline order + per-tenant
    effort quota; the quota is the only cross-query lever.

    ``mode="oracle"`` (upper bound): per tenant, a soft and a hard effort multiplier and a
    soft and a hard admission fraction, so the oracle can shed the hard class of a tenant
    without shedding its soft class. This space contains the quota space, so the oracle
    cannot score below it.

    Tuning uses the *pilot* trace only (``wl`` without ``arrival_eval``). The objective is
    lexicographic: maximise the worst tenant's pass rate first, then the aggregate pass rate.
    Returns load-indexed (R, T, nq) effort and admission arrays plus the (shared) EDF priority.
    """
    hard = np.asarray(wl["hard"], dtype=bool)
    T, nq = hard.shape
    rhos = [float(x) for x in wl.get("rhos", [1.0])]
    nR = len(rhos)
    pilot = {k: v for k, v in wl.items() if k != "arrival_eval"}
    pri = _edf_priority(wl)
    base = np.array([_nprobe_for(t["R"]) for t in wl["tenants"]], dtype=np.int64)
    f_grid = np.round(np.arange(0.0, 1.0001, 0.1), 2)
    s_grid = np.round(np.arange(0.1, 1.0001, 0.1), 2)
    out_eff = np.zeros((nR, T, nq), dtype=np.int64)
    out_adm = np.ones((nR, T, nq), dtype=bool)

    warm_eff = None if warm is None else np.asarray(warm["efforts"], dtype=np.int64)
    warm_adm = None if warm is None else np.asarray(warm["admission"], dtype=bool)

    for r in range(nR):
        rho = rhos[r]
        # Analytic fair-share start: at recall-minimal effort the offered load is exactly
        # rho x capacity, so admitting a fraction ~0.95/rho of every tenant puts the admitted
        # load at 0.95 of capacity -- a stable operating point for EDF. Without this start the
        # coordinate ascent gets stuck reducing one tenant at a time while the rest stay
        # overloaded (measured: worst-tenant 0.005 instead of 0.30 at rho=3).
        f0 = min(1.0, 0.95 / rho)
        ms = np.ones(T)
        mh = np.full(T, 2.0)
        fs = np.full(T, f0)
        fh = np.ones(T)
        if warm_eff is not None and warm_eff.ndim == 3:
            for i in range(T):
                hrow = hard[i]
                srow = ~hrow
                if srow.any():
                    ms[i] = max(1, int(round(float(warm_eff[r, i, srow].mean()) / base[i])))
                    fs[i] = float(warm_adm[r, i, srow].mean())
                if hrow.any():
                    mh[i] = max(1, int(round(float(warm_eff[r, i, hrow].mean()) / base[i])))
                    fh[i] = float(warm_adm[r, i, hrow].mean())

        def build():
            eff = np.empty((T, nq), dtype=np.int64)
            adm = np.empty((T, nq), dtype=bool)
            for i in range(T):
                soft_eff = np.minimum(256, int(base[i] * int(ms[i])))
                hard_eff = np.minimum(256, int(base[i] * int(mh[i])))
                eff[i] = np.where(hard[i], hard_eff, soft_eff)
                if mode == "quota":
                    adm[i] = _admission_mask(nq, fs[i], fs[i], hard[i])
                else:
                    adm[i] = _admission_mask(nq, fs[i], fh[i], hard[i])
            return eff, adm

        def snapshot():
            return (ms.copy(), mh.copy(), fs.copy(), fh.copy())

        def score(eff, adm):
            m = core.eval_serving_regime(pilot, eff, pri, adm, r)
            return (min(m["tenant_pass"]), m["slo_pass"])

        best_eff, best_adm = build()
        best = score(best_eff, best_adm)
        best_params = snapshot()
        # Global scan over a uniform admission scale: escapes the coordinate-ascent local
        # optimum where every tenant stays admitted and the system stays overloaded.
        if mode == "quota":
            for fv in np.round(np.arange(0.1, 1.0001, 0.05), 2):
                for mhv in (1.0, 2.0):
                    fs[:] = fv
                    fh[:] = fv
                    mh[:] = mhv
                    ms[:] = 1.0
                    eff, adm = build()
                    obj = score(eff, adm)
                    if obj > best:
                        best, best_eff, best_adm = obj, eff, adm
                        best_params = snapshot()
        else:
            for fsv in np.round(np.arange(0.1, 1.0001, 0.1), 2):
                for fhv in (0.0, 0.5, 1.0):
                    fs[:] = fsv
                    fh[:] = fhv
                    mh[:] = 2.0
                    ms[:] = 1.0
                    eff, adm = build()
                    obj = score(eff, adm)
                    if obj > best:
                        best, best_eff, best_adm = obj, eff, adm
                        best_params = snapshot()
        ms, mh, fs, fh = [a.copy() for a in best_params]
        if mode == "quota":
            grids = [("mh", [1, 2]), ("fs", s_grid)]     # fs is the uniform per-tenant quota
        else:
            grids = [("ms", [1, 2]), ("mh", [1, 2]), ("fs", s_grid), ("fh", f_grid)]
        for _ in range(n_rounds):
            improved = False
            for i in range(T):
                for name, grid in grids:
                    cur = {"ms": ms, "mh": mh, "fs": fs, "fh": fh}[name]
                    old = cur[i]
                    for v in grid:
                        if v == old:
                            continue
                        cur[i] = v
                        if mode == "quota" and name == "fs":
                            fh[i] = v
                        eff, adm = build()
                        obj = score(eff, adm)
                        if obj > best:
                            best, best_eff, best_adm = obj, eff, adm
                            best_params = snapshot()
                            improved = True
                        else:
                            cur[i] = old
                            if mode == "quota" and name == "fs":
                                fh[i] = old
            if not improved:
                break
        out_eff[r] = best_eff
        out_adm[r] = best_adm
        if verbose:
            print(f"  serving[{mode}] rho={rho}: worst_tenant={best[0]:.4f} "
                  f"pass={best[1]:.4f}", flush=True)
    return {"efforts": out_eff.tolist(), "priority": pri.tolist(),
            "admission": out_adm.tolist()}


def serving_vecslo(wl, n_rounds=2, verbose=False):
    """Reference VecSLO submission: deadline order + per-tenant effort quota.

    Per load regime it chooses, for every tenant, the effort multiplier of its hard contract
    class and an admission fraction (the quota). Both are tuned on the pilot trace only;
    scoring uses the held-out traces across regimes.
    """
    return _serving_quota_search(wl, mode="quota", n_rounds=n_rounds, verbose=verbose)


def serving_oracle(wl, warm=None, n_rounds=2, verbose=False):
    """Full-information oracle: deadline order + per-query effort and admission.

    The oracle knows each query's true difficulty (the published ``hard`` flag), so it can
    set the soft and hard effort independently and shed the hard class of a tenant without
    shedding its soft class. That space contains the per-tenant quota space, so the oracle
    is an upper bound on it. Tuned on the pilot trace only.

    ``warm`` (optional) is the VecSLO prediction; it initialises the search so the oracle is
    guaranteed not to fall below the quota submission. Without it the oracle computes its
    own quota warm start.
    """
    if warm is None:
        warm = _serving_quota_search(wl, mode="quota", n_rounds=1)
    return _serving_quota_search(wl, mode="oracle", n_rounds=n_rounds, warm=warm,
                                 verbose=verbose)

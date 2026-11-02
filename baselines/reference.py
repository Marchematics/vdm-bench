"""Reference submissions for VDM-Bench (define the tasks; not the final word)."""
import numpy as np

from vdmbench import core


def analytics_exact(wl):
    return {"range_count": wl["truth"].tolist(),
            "coverage": wl["coverage_chosen"].tolist(),
            "join": wl["join_pairs"].tolist(),
            "cost": float(wl["n"] * len(wl["queries"]) * len(wl["radii"]))}


def analytics_sampling(data_root, wl, s=2000, seed=0):
    X, _ = core.load_sift(data_root, wl["n"])
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


def analytics_ivf(data_root, wl, nlist=256, nprobe=32, seed=0):
    import faiss
    X, _ = core.load_sift(data_root, wl["n"])
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
                    "drift": g["drift_mmd"]})
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


def analytics_anavec(data_root, wl, seed=0):
    """Reference AnaVec-style submission: IVF range counts + HNSW candidate coverage."""
    import faiss
    X, _ = core.load_sift(data_root, wl["n"])
    Q = wl["queries"]
    nlist, nprobe = 256, 256
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(X.shape[1]), X.shape[1], nlist)
    ivf.train(X[np.random.RandomState(seed).choice(len(X), min(50000, len(X)), replace=False)])
    ivf.add(X); ivf.nprobe = nprobe
    est = np.zeros_like(wl["truth"], dtype=np.float64)
    for i, q in enumerate(Q):
        for j, r in enumerate(wl["radii"]):
            lims, _, _ = ivf.range_search(q[None, :].astype(np.float32), float(r * r))
            est[i, j] = lims[1] - lims[0]
    # coverage candidates
    hnsw = faiss.IndexHNSWFlat(X.shape[1], 32)
    hnsw.hnsw.efConstruction = 100
    hnsw.add(X); hnsw.hnsw.efSearch = 64
    A = wl["anchors"]
    _, I = hnsw.search(A.astype(np.float32), 20)
    rng = np.random.RandomState(seed)
    cand = set(I.ravel().tolist())
    cand.update(rng.choice(len(X), 5000, replace=False).tolist())
    pi = rng.randint(0, len(A), size=(200, 2))
    mids = (A[pi[:, 0]] + A[pi[:, 1]]) / 2.0
    _, I2 = hnsw.search(mids.astype(np.float32), 2)
    cand.update(I2.ravel().tolist())
    C = np.asarray(sorted(cand), dtype=np.int64)
    D = np.sqrt(core.pairwise(A, X[C]))
    covered = np.zeros(len(A), dtype=bool); sel = []
    for _ in range(int(wl["k"])):
        gain = ((D <= wl["r_cov"]) & (~covered)[:, None]).sum(axis=0)
        j = int(gain.argmax()); sel.append(int(C[j])); covered |= D[:, j] <= wl["r_cov"]
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
            "cost": float(len(Q) * len(wl["radii"]) * nprobe / nlist * len(X))}


def quality_vecprof(data_root, wl, seed=0):
    """Reference VecProf-style submission: certified epsilon-pairs + neighbourhood outliers."""
    import faiss
    n = int(wl["instances"][0]["n"])
    sets = core.make_quality_sets(data_root, n=n, seed=seed)
    out = []
    total_cost = 0.0
    eps = 25.0
    for family, level, X in sets:
        nlist = 256
        km = faiss.Kmeans(X.shape[1], nlist, niter=10, verbose=False, seed=seed)
        km.train(X[np.random.RandomState(seed).choice(len(X), min(10000, len(X)), replace=False)].astype(np.float32))
        C = km.centroids.astype(np.float32)
        D = np.sqrt(core.pairwise(X, C))  # n x nlist
        I = D.argmin(1)
        lists = {}
        for i, c in enumerate(I):
            lists.setdefault(int(c), []).append(i)
        Rmax = np.zeros(nlist, dtype=np.float32)
        for c, ids in lists.items():
            Rmax[c] = np.linalg.norm(X[ids] - C[c], axis=1).max()
        cand = set()
        probes = 0
        for i in range(len(X)):
            cs = np.where(D[i] <= eps + Rmax)[0]
            for c in cs:
                ids = lists.get(int(c), [])
                probes += len(ids)
                for j in ids:
                    if j > i:
                        cand.add((i, j))
        cand = np.asarray(list(cand), dtype=np.int64) if cand else np.zeros((0, 2), np.int64)
        if len(cand):
            dd = np.linalg.norm(X[cand[:, 0]] - X[cand[:, 1]], axis=1)
            cand = cand[dd <= eps]
        verts = set(cand.ravel().tolist()) if len(cand) else set()
        dcl = D.min(1)
        thr = np.median(dcl) + 6 * (np.percentile(dcl, 75) - np.percentile(dcl, 25))
        n_out = int((dcl > thr).sum())
        drift = float(np.linalg.norm(X.mean(0) - sets[0][2].mean(0)))
        out.append({"family": family, "level": level,
                    "dup_rate": len(verts) / len(X), "n_outliers": n_out, "drift": drift})
        total_cost += probes
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

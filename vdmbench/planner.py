"""One cost-based planner for every VDM-Bench contract.

The benchmark ships *per-task reference submissions*: each one is tuned inside its own task
(analytics: exact / uniform / IVF plan / stratified two-phase; quality: naive / certified
epsilon-graph; serving: static / adaptive / signal / EDF / quotas).  This module is the other
half of that comparison: a **single** planner that, given only

    (a) a per-query cost budget (distance evaluations for analytics/quality, CPU capacity for
        serving), and
    (b) the contract being asked for (analytics: a per-query relative-error target at
        confidence 1 - delta; quality: the integrity clause "epsilon-graph recall = 1.0";
        serving: a per-tenant latency + recall SLO under a measured offered load),

returns a plan plus the budget it will spend, using a *cost model* rather than a per-task
constant.  The plan space is the same primitive set the reference submissions use, so the
comparison is fair:

    analytics  uniform sample | IVF probe (exact prefix) | Neyman-allocated stratified sample |
               two-phase (exact IVF prefix + Neyman residual) | exact scan
    quality    certified probe rule | heuristic nprobe graph | sampled kNN | RFF-MMD | exact
    serving    recall-minimal effort | deadline order (EDF) | admission by cost | quotas

The one structural fact the cost model has to exploit is already measured in this repository:
the winning analytics lever *inverts* with scale.  At 100k the exact IVF prefix wins because a
handful of lists is a large fraction of the collection and holds most of the ball mass; at 10M
the same prefix loses because a handful of 1,000-point lists is a tiny fraction of the
collection, so the budget is better spent on Neyman-allocated samples.  The rule the planner
implements is a *mass comparison* (see :meth:`AnalyticsPlanner.mass_diagnostics`): spend budget
on exactness only while the probed list population is a meaningful fraction of the ball mass a
probe can capture; otherwise allocate samples.

Nothing here calls an LLM or a network service; everything is CPU-only and seeded.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from vdmbench import core


# =========================================================================== #
# Contracts
# =========================================================================== #
@dataclass
class AnalyticsContract:
    """A per-query relative-error contract: P(|est-truth|/truth <= eps) >= 1 - delta.

    ``eps`` is a relative error (0.05 = 5%) and ``delta`` the tolerated violation rate over
    (query, radius) pairs.  ``eps=None`` means "maximise accuracy within the budget" (the
    matched-cost comparison); a finite ``eps`` lets the planner return the *cheapest* plan
    that satisfies the contract instead of spending the whole budget.
    """

    eps: float | None = 0.05
    delta: float = 0.05

    def to_json(self):
        return {"kind": "analytics", "eps": self.eps, "delta": self.delta}


@dataclass
class QualityContract:
    """The quality task's integrity clause plus tolerances on the scalar rules.

    ``integrity_recall`` is 1.0 by construction of the task: no epsilon-pair may be missed.
    That is a *certification* requirement, not a statistical target, so the planner can only
    satisfy it with the certified probe rule.  The remaining budget is spent on the scalar
    rules (outlier count, drift) to the tolerances the published decision thresholds need.
    """

    # defaults come from the single published source of truth in vdmbench.core
    integrity_recall: float = core.QUALITY_CONTRACT["integrity_recall"]
    outlier_rel_tol: float = core.QUALITY_CONTRACT["outlier_rel_tol"]  # 25% of a 1.1% rate is 12 sigma
    # Drift is a two-sample statistic: with the published 4,000-vs-4,000 protocol its sampling
    # noise is ~1 threshold unit (the reference submission measures 0.99 on the shipped
    # workload), so no plan -- including the exact one -- can certify 0.5 units.  A 2-unit
    # tolerance is what the protocol can actually certify; the decision rule is still exact
    # because the drift instance's signal is 37x the threshold.
    drift_tol_thr_units: float = core.QUALITY_CONTRACT["drift_tol_thr_units"]

    def to_json(self):
        return {"kind": "quality", "integrity_recall": self.integrity_recall,
                "outlier_rel_tol": self.outlier_rel_tol,
                "drift_tol_thr_units": self.drift_tol_thr_units}


@dataclass
class ServingContract:
    """Per-tenant latency+recall SLO, read directly off the shipped per-query contract.

    ``deadline`` (ms) and ``recall_req`` are arrays shaped (T, nq).  The hard/soft split is
    *not* read from the workload's ``hard`` flag (that is ground truth); it is read off the
    contract's own deadline: a query whose deadline is 0.6x its tenant's class deadline is a
    hard query and needs the doubled probe budget the published hard-query rule implies.
    """

    deadline: np.ndarray
    recall_req: np.ndarray
    cores: int
    rhos: tuple | None = None

    def to_json(self):
        return {"kind": "serving", "cores": int(self.cores),
                "rhos": None if self.rhos is None else [float(x) for x in self.rhos],
                "mean_deadline_ms": float(self.deadline.mean()),
                "mean_recall_req": float(self.recall_req.mean())}


# =========================================================================== #
# Shared cost-model helpers (copied from baselines/reference.py, not imported: the reference
# implementation is another submission and must not be edited or refactored underneath it)
# =========================================================================== #
def _cell_hit_prob_angle(dq, r, grid, qs, dim):
    """Angle-aware per-cell hit probability, the predictor that ships in the reference.

    d(q,x)^2 = d_c^2 + t^2 - 2 d_c t cos(theta); for isotropic points around a centroid
    cos(theta) has sd 1/sqrt(dim), which is not negligible at 128 dims.  Integrating t over
    the cell's own within-cell radius quantiles:

        p_c = E_t[ Phi( (r^2 - d_c^2 - t^2) / (2 d_c t / sqrt(dim)) ) ]
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


def _allocate(w, budget, min_per_cell=1):
    """Integer per-cell sample allocation summing to ``budget`` (largest remainder)."""
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


def _normal_cdf(x):
    from scipy.special import erf
    return 0.5 * (1.0 + erf(np.asarray(x, dtype=np.float64) / math.sqrt(2.0)))


# =========================================================================== #
# Analytics
# =========================================================================== #
#: Prefix sizes the planner considers (number of nearest IVF lists scanned exactly).
PREFIX_LADDER = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

#: Spend levels the planner considers as fractions of the per-query budget.  A finite-eps
#: contract may be satisfiable for less than the budget, and a planner that can only ever spend
#: the whole budget cannot answer "what will this contract cost?".
SPEND_FRACTIONS = (1.0, 0.5, 0.25)

#: Calibration of the cost model, taken from the repository's own predictor study
#: (experiments/stratification_10m.md, "Predictor calibration"): the angle-aware per-cell
#: hit-rate model has a model-only count error of 31%, and allocating from the model is within
#: 1.11x of allocating from the true per-cell rates.  Both make the *true* error larger than
#: the model's prediction, and plan selection compounds it (the planner picks the candidate
#: whose prediction is most optimistic).  A planner that predicts its own compliance from an
#: uncalibrated variance is therefore over-confident and spends too little to satisfy a
#: statistical contract; carrying this uncertainty explicitly is what makes the contract
#: prediction honest.  Measured on the 100k analytics sweep: without the correction the
#: eps=0.05 contract predicted 0.965 compliance but measured 0.906.
PRED_COUNT_REL_ERR = 0.31
PRED_ALLOC_GAP = 1.11
PRED_INFLATION = (1.0 + PRED_COUNT_REL_ERR) * PRED_ALLOC_GAP     # ~1.454

#: The cost model's own resolution.  Allocating from the model is within 1.11x of allocating
#: from the true per-cell rates, and the predicted relative error scales like sqrt(variance),
#: so predicted errors are only meaningful to ~sqrt(1.11) - 1 = 5%.  A plan that wins by less
#: than this is a *tie*: the planner must not switch to the exactness lever (whose value rests
#: entirely on the predicted ball-mass share, the model quantity with the largest error) on a
#: difference it cannot resolve.  Measured on SIFT10M: the model preferred an 8-list prefix by
#: 2% (0.0193 vs 0.0198) where the measurement prefers none by 6% (0.0160 vs 0.0151); with the
#: band the planner picks allocation and matches the best fixed plan instead of losing to it.
PRED_TIE_BAND = 0.05

#: Finite-sample margin on a statistical contract.  Contract compliance is measured over the
#: benchmark's (query, radius) observations (500 in the analytics task), so even a correctly
#: calibrated predicted compliance is itself known to only ~sqrt(delta(1-delta)/n) = 1% (one
#: standard error at delta = 0.05).  A planner that accepts a plan predicted at exactly
#: 1 - delta sits on the boundary and misses the measured contract about half the time
#: (measured: eps=0.05 at a 20k budget, predicted 0.951, measured 0.941).  Require one standard
#: error of margin, so the plan it returns is robust to the estimate's own noise.
COMPLIANCE_OBS = 500
COMPLIANCE_MARGIN = 0.01


class AnalyticsIndex:
    """IVF cell structure + within-cell radius quantiles (the index both plan families share).

    Building it is amortised exactly like every reference submission's index build; on SIFT10M
    it is loaded from the cached structure instead of being rebuilt so a planner comparison
    does not pay a 13-minute k-means bill per run.
    """

    def __init__(self, C, sizes, order, off, grid, qs, N, dim):
        self.C = np.ascontiguousarray(C, dtype=np.float32)
        self.sizes = np.asarray(sizes, dtype=np.int64)
        self.order = np.asarray(order, dtype=np.int64)
        self.off = np.asarray(off, dtype=np.int64)
        self.grid = np.asarray(grid, dtype=np.float32)
        self.qs = np.asarray(qs, dtype=np.float64)
        self.nlist = len(self.sizes)
        self.N = int(N)
        self.dim = int(dim)

    # -- construction ------------------------------------------------------
    @classmethod
    def build(cls, X, nlist=None, seed=0, batch=200_000):
        import faiss
        N = len(X)
        if nlist is None:
            nlist = int(min(16384, max(256, N // 1000)))
        prev = faiss.omp_get_max_threads()
        # Pin the builder's thread count: FAISS k-means parallelises the distance reduction,
        # so the centroids differ in the last bits between thread counts and 10M numbers
        # drift in the third digit (see experiments/reproducibility_2026-11-04.md).
        faiss.omp_set_num_threads(int(os.environ.get("VDMBENCH_BUILD_THREADS", "8")))
        try:
            km = faiss.Kmeans(X.shape[1], nlist, niter=10, verbose=False, seed=seed)
            km.train(np.ascontiguousarray(X[: min(400_000, N)]))
            C = km.centroids.astype(np.float32)
            qflat = faiss.IndexFlatL2(X.shape[1])
            qflat.add(C)
            assign = np.empty(N, dtype=np.int32)
            for s0 in range(0, N, batch):
                _, idx = qflat.search(np.ascontiguousarray(X[s0:s0 + batch]), 1)
                assign[s0:s0 + len(idx)] = idx.ravel()
        finally:
            faiss.omp_set_num_threads(prev)
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
        return cls(C, sizes, order, off, grid, qs, N, X.shape[1])

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["C"], z["sizes"], z["order"], z["off"], z["grid"], z["qs"],
                   int(z["assign"].shape[0]) if "assign" in z.files else len(z["order"]),
                   z["C"].shape[1])

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, C=self.C, sizes=self.sizes, order=self.order, off=self.off,
                 grid=self.grid, qs=self.qs)

    # -- query-to-index distances -----------------------------------------
    def dQ(self, Q, batch=64):
        out = np.empty((len(Q), self.nlist), dtype=np.float32)
        for s in range(0, len(Q), batch):
            out[s:s + batch] = np.sqrt(core.pairwise(np.asarray(Q[s:s + batch], np.float32),
                                                    self.C))
        return out

    def hit_probs(self, dq, radii):
        """(n_radii, nlist) matrix of predicted per-cell hit probabilities."""
        return np.stack([_cell_hit_prob_angle(dq, float(r), self.grid, self.qs, self.dim)
                         for r in radii])

    # -- the cost model ----------------------------------------------------
    def _predict(self, sizes, p, radii, cand):
        """Predicted (count, sd) per radius for one plan candidate; fills cost/alloc in place."""
        nr = len(radii)
        N = self.N
        H = sizes @ p.T                       # (nr,) predicted ball mass, i.e. the true count
        kind = cand["kind"]
        if kind == "exact":
            cand["alloc"] = None
            return H, np.zeros(nr), np.zeros(nr)
        if kind == "uniform":
            S = cand["S"]
            pbar = np.clip(H / max(N, 1), 0.0, 1.0)
            var = (N * N) * pbar * (1.0 - pbar) / max(S, 1)
            cand["alloc"] = None
            return H, np.sqrt(var), None
        if kind == "ivf":
            P = cand["P"]
            Hp = sizes[P] @ p[:, P].T
            bias = np.abs(H - Hp) / np.maximum(H, 1.0)
            cand["alloc"] = None
            return H, np.zeros(nr), bias
        if kind == "two-phase":
            P = cand["P"]
            S = cand["S"]
            live = np.ones(self.nlist, dtype=bool)
            live[P] = False
            if S < int(live.sum()):
                return None, None, None          # not enough residual budget for one sample/cell
            # Neyman weights averaged (after normalisation) over radii, as the reference does
            Ws = []
            for j in range(nr):
                pc = np.clip(p[j], 1e-3, 1.0 - 1e-3)
                w = np.where(live, sizes * np.sqrt(pc * (1.0 - pc)), 0.0)
                tot = w.sum()
                if tot <= 0:
                    return None, None, None
                Ws.append(w / tot)
            alloc = _allocate(np.mean(Ws, axis=0), S)
            cand["alloc"] = alloc
            var = np.zeros(nr)
            for j in range(nr):
                # raw p for the variance (a cell with p == 0 contributes no variance and no
                # sample); the 1e-3 clip is only ever used to build allocation weights
                pc = np.clip(p[j], 0.0, 1.0)
                v = np.where(live, sizes * sizes * pc * (1.0 - pc)
                             / np.maximum(alloc, 1), 0.0)
                var[j] = v.sum()
            return H, np.sqrt(var), None
        raise ValueError(f"unknown plan kind {kind}")

    def _candidates(self, dq, radii, budget):
        """Enumerate the plan space at this budget (deterministic given the index and query)."""
        nlist, N = self.nlist, self.N
        ord_cells = np.argsort(dq, kind="stable")
        sizes_sorted = self.sizes[ord_cells]
        cum = np.cumsum(sizes_sorted)
        out = []

        seen = set()

        def add(c):
            key = (c["kind"], c.get("k"), c.get("S"), c.get("m"))
            if key not in seen:
                seen.add(key)
                out.append(c)

        # 1. plain uniform sampling of the collection, at every spend level
        for f in SPEND_FRACTIONS:
            S = int(min(budget * f, N))
            if S >= 1:
                add({"kind": "uniform", "S": S})

        # 2. stratified / two-phase for every prefix size and spend level the budget allows
        for k in PREFIX_LADDER:
            if k > nlist:
                break
            P = ord_cells[:k]
            prefix_cost = int(cum[k - 1]) if k else 0
            for f in SPEND_FRACTIONS:
                S = int(budget * f - nlist - prefix_cost)
                if S <= 0:
                    continue
                if k and S < nlist - k:  # one sample per live cell is the estimator's floor
                    continue
                add({"kind": "two-phase", "k": int(k), "P": P,
                     "prefix_cost": prefix_cost, "S": S})

        # 3. IVF probe plan: the largest exact prefix the budget can afford, nothing sampled
        m = int(np.searchsorted(cum, max(budget - nlist, 0), side="right"))
        m = min(m, nlist)
        if m > 0:
            out.append({"kind": "ivf", "m": m, "P": ord_cells[:m],
                        "prefix_cost": int(cum[m - 1])})

        # 4. exact scan
        if budget >= N:
            out.append({"kind": "exact", "cost": N})
        return out

    @staticmethod
    def _candidate_cost(cand, nlist, N):
        kind = cand["kind"]
        if kind == "uniform":
            return float(cand["S"])
        if kind == "ivf":
            return float(nlist + cand["prefix_cost"])
        if kind == "two-phase":
            return float(nlist + cand["prefix_cost"] + cand["S"])
        if kind == "exact":
            return float(N)
        raise ValueError(kind)

    def plan(self, dq, radii, contract, budget):
        """Return the cheapest (or best, if the contract is infeasible) plan at this budget.

        The decision is made *per query* from the cost model only: predicted variance of each
        candidate family (uniform / IVF probe / two-phase with k exact lists) is compared at
        equal spend, and the plan that meets the contract at the lowest predicted cost wins.
        """
        sizes, N, nlist = self.sizes, self.N, self.nlist
        p = self.hit_probs(dq, radii)
        cands = self._candidates(dq, radii, budget)
        for c in cands:
            c["cost"] = self._candidate_cost(c, nlist, N)
            c["_pred"] = self._predict(sizes, p, radii, c)
        js = []
        for c in cands:
            H, sd, bias = c["_pred"]
            if H is None:
                continue
            c["count_hat"] = H
            if bias is not None:                       # deterministic bias (IVF probe)
                rel = bias * PRED_INFLATION
                c["pred_rel"] = rel
                c["pred_compliance"] = (float(np.mean(rel <= contract.eps))
                                        if contract.eps is not None else None)
            else:
                c["pred_rel"] = sd / np.maximum(H, 1.0)
                if contract.eps is None:
                    c["pred_compliance"] = None
                else:
                    z = contract.eps * H / np.maximum(PRED_INFLATION * sd, 1e-12)
                    c["pred_compliance"] = float(np.mean(2.0 * _normal_cdf(z) - 1.0))
            js.append(c)
        if contract.eps is not None:
            need = 1.0 - contract.delta + COMPLIANCE_MARGIN
            feas = [c for c in js if c["pred_compliance"] is not None
                    and c["pred_compliance"] >= need]
            if feas:
                # cheapest plan that certifies the contract; on a cost tie prefer the plan
                # that relies least on the ball-mass prediction (fewest exact lists)
                return min(feas, key=lambda c: (c["cost"],
                                                c.get("prefix_cost", 0),
                                                -c["pred_compliance"]))
        # No plan certifies the contract inside the budget (or the contract asks for maximum
        # accuracy): spend up to the budget on the lowest predicted error.  Candidate plans
        # whose predicted error is within the model's own resolution of the best one are a
        # tie, and the tie is broken towards allocation (fewest exact lists) because that is
        # the decision the mass predictor is least able to resolve.
        best = min(float(np.mean(c["pred_rel"])) for c in js)
        tied = [c for c in js if float(np.mean(c["pred_rel"])) <= best * (1.0 + PRED_TIE_BAND)]
        return min(tied, key=lambda c: (c.get("prefix_cost", 0), c["cost"],
                                        float(np.mean(c["pred_rel"]))))

    # -- execution ---------------------------------------------------------
    def prepare(self, cand, dq, radii):
        """Populate the per-query allocation of a candidate built only from ``dq`` (fixed plans)."""
        if cand.get("alloc") is None and cand["kind"] == "two-phase":
            p = self.hit_probs(dq, radii)
            cand["_pred"] = self._predict(self.sizes, p, radii, cand)
        return cand

    def estimate(self, X, dQX, q, radii, cand, rng):
        """Run one plan candidate for one query; returns (counts, distance evaluations used)."""
        kind = cand["kind"]
        q2 = float(q @ q)
        N = self.N
        kind_cost = 0
        if kind == "exact":
            cnt = np.zeros(len(radii), dtype=np.float64)
            for s0 in range(0, N, 200_000):
                xc = np.asarray(X[s0:s0 + 200_000], dtype=np.float32)
                dd = np.sqrt(np.maximum(q2 + dQX[s0:s0 + len(xc)] - 2.0 * xc @ q, 0.0))
                for j, r in enumerate(radii):
                    cnt[j] += float((dd <= r).sum())
            return cnt, float(N)
        if kind == "uniform":
            S = int(cand["S"])
            idx = rng.choice(N, S, replace=False)
            dd = np.sqrt(np.maximum(q2 + dQX[idx] - 2.0 * X[idx] @ q, 0.0))
            cnt = np.array([N * float((dd <= r).mean()) for r in radii], dtype=np.float64)
            return cnt, float(S)
        if kind == "ivf":
            P = cand["P"]
            idx = np.concatenate([self.order[self.off[c]:self.off[c + 1]] for c in P])
            dd = np.sqrt(np.maximum(q2 + dQX[idx] - 2.0 * X[idx] @ q, 0.0))
            cnt = np.array([float((dd <= r).sum()) for r in radii], dtype=np.float64)
            return cnt, float(self.nlist + len(idx))
        if kind == "two-phase":
            P = cand["P"]
            ix = np.concatenate([self.order[self.off[c]:self.off[c + 1]] for c in P]) \
                if len(P) else np.empty(0, dtype=np.int64)
            cells_s, idx_s = _sample_within(cand["alloc"], self.sizes, self.order,
                                            self.off, rng)
            idx = np.concatenate([ix, idx_s]) if len(ix) else idx_s
            dd = np.sqrt(np.maximum(q2 + dQX[idx] - 2.0 * X[idx] @ q, 0.0))
            dd_e, dd_s = dd[:len(ix)], dd[len(ix):]
            cnt = np.empty(len(radii), dtype=np.float64)
            for j, r in enumerate(radii):
                exact = float((dd_e <= r).sum()) if len(dd_e) else 0.0
                hit = (np.bincount(cells_s, weights=(dd_s <= r).astype(np.float64),
                                   minlength=self.nlist) if len(cells_s) else 0.0)
                cnt[j] = exact + float((self.sizes * hit / np.maximum(cand["alloc"], 1)).sum())
            return cnt, float(self.nlist + len(ix) + len(idx_s))
        raise ValueError(f"unknown plan kind {kind}")

    # -- diagnostics -------------------------------------------------------
    def mass_diagnostics(self, dq, radii, budget, k_list=PREFIX_LADDER):
        """The mass rule, quantified: for each prefix size, cost share vs ball-mass share.

        ``cost_share`` is the fraction of the collection a probe of k nearest lists touches.
        ``mass_share`` is the fraction of the expected in-ball count those lists hold (mean
        over radii).  ``pred_rel`` is the cost model's predicted relative error of the
        two-phase plan with that prefix at this budget, so the rule's prediction can be
        compared with the measured error directly.
        """
        p = self.hit_probs(dq, radii)
        sizes = self.sizes
        H = sizes @ p.T
        tot = np.maximum(H, 1.0)                      # per-radius total ball mass
        ord_cells = np.argsort(dq, kind="stable")
        rows = []
        for k in k_list:
            if k > self.nlist:
                break
            P = ord_cells[:k]
            prefix_cost = int(sizes[P].sum())
            cost_share = prefix_cost / self.N
            mass_share = float(np.mean((sizes[P] @ p[:, P].T) / tot)) if k else 0.0
            S = int(budget - self.nlist - prefix_cost)
            pred = None
            if S > max(0, self.nlist - k):
                cand = {"kind": "two-phase", "P": P, "prefix_cost": prefix_cost, "S": S}
                Hp, sd, _ = self._predict(sizes, p, radii, cand)
                if Hp is not None:
                    pred = float(np.mean(sd / np.maximum(Hp, 1.0)))
            rows.append({"k": int(k), "cost_share": float(cost_share),
                         "mass_share": mass_share,
                         "mass_per_cost": float(mass_share / cost_share) if cost_share else 0.0,
                         "pred_rel": pred, "predicted_cost_per_query": int(
                             self.nlist + prefix_cost + max(S, 0))})
        return rows


class AnalyticsPlanner:
    """Plan selection for the analytics contract; one planner, no per-task constants."""

    def __init__(self, index: AnalyticsIndex):
        self.index = index

    def plan(self, dq, radii, contract, budget):
        return self.index.plan(dq, radii, contract, budget)

    def run(self, X, dQX, Q, radii, contract, budget, seed=0, plan_callback=None):
        """Plan *and execute*: returns (estimates, costs, chosen-plan descriptors)."""
        rng = np.random.RandomState(seed)
        dQ = self.index.dQ(np.asarray(Q, dtype=np.float32))
        est = np.zeros((len(Q), len(radii)), dtype=np.float64)
        costs = np.zeros(len(Q), dtype=np.float64)
        plans = []
        for i, q in enumerate(Q):
            cand = self.plan(dQ[i], radii, contract, budget)
            cnt, cost = self.index.estimate(X, dQX, q, radii, cand, rng)
            est[i] = cnt
            costs[i] = cost
            plans.append(_describe(cand))
            if plan_callback is not None:
                plan_callback(i, cand)
        return est, costs, plans


def _describe(cand):
    kind = cand["kind"]
    d = {"kind": kind, "cost": float(cand["cost"])}
    if kind == "uniform":
        d["S"] = int(cand["S"])
    elif kind == "two-phase":
        d["k"] = int(cand["k"])
        d["S"] = int(cand["S"])
    elif kind == "ivf":
        d["m"] = int(cand["m"])
    if "pred_rel" in cand:                     # cost-model prediction for this contract
        d["pred_rel"] = float(np.mean(cand["pred_rel"]))
        d["pred_compliance"] = (None if cand.get("pred_compliance") is None
                                else float(cand["pred_compliance"]))
    return d


# =========================================================================== #
# Serving
# =========================================================================== #
def _nprobe_for(R):
    return int(core.SURFACE_NPROBE[np.argmax(core.SURFACE_RECALL >= float(R))])


class ServingPlanner:
    """Cost-based serving policy: effort from the contract, EDF, admission by cost.

    The reference VecSLO submission tunes per-tenant effort multipliers *and* per-tenant
    admission fractions by coordinate ascent plus subset scans on the pilot trace.  This
    planner instead derives the policy:

    * effort: the recall-minimal probe budget of the tenant's contract; hard queries (the ones
      whose contract deadline is the 0.6x class) need 2x, because the published hard-query
      rule makes their recall at ``n`` equal to recall at ``n/2``;
    * feasibility: a query whose service time already exceeds its deadline is shed (serving it
      cannot pass and would burn capacity);
    * admission: under overload, admit the *cheapest* contract classes first (each query is
      worth the same in the pass metric, so maximising served contracts subject to the
      capacity budget is a unit-value knapsack, solved by the greedy cost order), then refine
      it with the **queue state**: the pilot trace is simulated with the candidate admission,
      admitted queries the queue predicts will miss their deadline (their debt) are shed, and
      the freed budget is refilled with the cheapest remaining candidates;
    * scheduling: earliest-deadline-first, which is the contract's own ordering (priority =
      -deadline), not a tuned parameter.

    The only calibrated scalar is the utilisation target ``u``; it is calibrated **once** on
    the visible pilot trace, shared by every tenant and every regime (not per-task tuned), and
    the experiment doc reports the sensitivity of the result to it.
    """

    U_GRID = (0.70, 0.80, 0.85, 0.90, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99)

    def __init__(self, u_target=None):
        self.u_target = u_target

    # -- policy construction ----------------------------------------------
    @staticmethod
    def _effort_and_service(wl):
        """(effort, service_ms, effective_recall, feasible) from the shipped contract only."""
        T, nq = wl["hard"].shape
        tenants = wl["tenants"]
        deadline = np.asarray(wl["deadline"], dtype=np.float64)
        recall_req = np.asarray(wl["recall_req"], dtype=np.float64)
        L = np.array([t["L"] for t in tenants], dtype=np.float64)
        base = np.array([_nprobe_for(t["R"]) for t in tenants], dtype=np.int64)
        effort = np.repeat(base[:, None], nq, axis=1).astype(np.int64)
        # a query is "hard" iff its contract deadline is the 0.6x class deadline
        hard = deadline < L[:, None] * (1.0 - 1e-3)
        effort[hard] = np.minimum(256, 2 * effort[hard])
        idx = core.surface_idx(effort)
        service = core.SURFACE_P50[idx].astype(np.float64)
        rec = core.SURFACE_RECALL[idx].astype(np.float64)
        hidx = core.surface_idx(np.maximum(1, effort // 2))
        eff_recall = np.where(hard, np.minimum(rec, core.SURFACE_RECALL[hidx]), rec)
        feasible = (service <= deadline) & (eff_recall >= recall_req)
        return effort, service, eff_recall, feasible, deadline, recall_req

    @staticmethod
    def _knapsack(w, feasible, budget, service):
        """Cheapest-*service*-first unit-value knapsack (the pre-debt admission rule)."""
        T, nq = w.shape
        adm = np.zeros((T, nq), dtype=bool)
        if w.sum() <= budget:
            return feasible.copy()
        ii, jj = np.nonzero(feasible)
        order = np.argsort(service[ii, jj], kind="stable")
        cum = np.cumsum(w[ii[order], jj[order]])
        keep = cum <= budget
        adm[ii[order][keep], jj[order][keep]] = True
        return adm

    @staticmethod
    def _debt_refine(arr, service, cores, deadline, w, feasible, adm, budget, rounds=2):
        """Queue-state (debt) admission: use the pilot queue to shed predicted misses.

        The static knapsack prices admission as long-run load and assumes an admitted query
        passes.  The pilot trace shows the *transient* queue state instead: some admitted
        queries are predicted to miss their deadline (their "debt"), and their capacity is
        better spent on next-cheapest queries.  One round = simulate the pilot queue with the
        current admission, shed the admitted queries the queue predicts will miss, then refill
        the freed budget from the cheapest remaining candidates; the debt-shed set is kept
        excluded so the refill cannot re-admit it.  Returns the admission with the best pilot
        per-regime pass over the rounds.
        """
        T, nq = service.shape
        pri = -deadline
        ii, jj = np.nonzero(feasible)
        order = np.argsort(service[ii, jj], kind="stable")
        best = adm.copy()
        best_m = -1.0
        excluded = np.zeros((T, nq), dtype=bool)
        for _ in range(max(1, rounds)):
            lat = core._simulate(np.asarray(arr, dtype=np.float64), service, cores, pri, adm)
            passed = adm & (lat <= deadline)
            m = float(passed.mean())
            if m > best_m:
                best, best_m = adm.copy(), m
            excluded |= adm & ~(lat <= deadline)
            new = adm & (lat <= deadline)
            load = float(w[new].sum())
            for k in order:
                i, j = ii[k], jj[k]
                if new[i, j] or excluded[i, j]:
                    continue
                if load + w[i, j] <= budget:
                    new[i, j] = True
                    load += w[i, j]
            if np.array_equal(new, adm):
                break
            adm = new
        lat = core._simulate(np.asarray(arr, dtype=np.float64), service, cores, pri, adm)
        m = float((adm & (lat <= deadline)).mean())
        if m > best_m:
            best = adm.copy()
        return best

    def _build(self, wl, rates, u_target, debt_rounds=2):
        T, nq = wl["hard"].shape
        rhos = [float(x) for x in wl.get("rhos", [1.0])]
        nR = len(rhos)
        cores = int(wl["cores"])
        cap = cores * 1000.0                     # capacity in ms of service per second
        effort, service, _, feasible, deadline, _ = self._effort_and_service(wl)
        out_eff = np.zeros((nR, T, nq), dtype=np.int64)
        out_pri = np.zeros((nR, T, nq), dtype=np.float64)
        out_adm = np.zeros((nR, T, nq), dtype=bool)
        arrival = wl.get("arrival")
        for r in range(nR):
            lam = np.asarray(rates[r], dtype=np.float64)          # queries/sec per tenant
            w = lam[:, None] * service / nq                       # ms/sec per admitted query
            w = np.where(feasible, w, 0.0)
            adm = self._knapsack(w, feasible, u_target * cap, service)
            if debt_rounds and arrival is not None:
                adm = self._debt_refine(arrival[r], service, cores, deadline, w, feasible,
                                        adm, u_target * cap, rounds=debt_rounds)
            out_eff[r] = effort
            out_pri[r] = -deadline
            out_adm[r] = adm
        return {"efforts": out_eff, "priority": out_pri, "admission": out_adm}

    def plan(self, wl, u_target=None, calibrate=True, debt_rounds=2):
        """Return a serving policy plus the calibration it chose.

        ``wl`` must be the *pilot* workload (``arrival``, no ``arrival_eval``) when
        ``calibrate`` is true: the utilisation target is chosen by evaluating this planner's
        own policy on the visible trace.  The returned policy is then applied unchanged to the
        held-out trace by the caller.
        """
        pilot = {k: v for k, v in wl.items() if k != "arrival_eval"}
        T, nq = pilot["hard"].shape
        nR = len([float(x) for x in pilot.get("rhos", [1.0])])
        arr = np.asarray(pilot["arrival"], dtype=np.float64)
        span = arr.max(axis=2) - arr.min(axis=2) + 1e-9
        rates = nq / span
        if u_target is None and self.u_target is not None:
            u_target = self.u_target
        calib = []
        if u_target is None:
            if not calibrate:
                raise ValueError("u_target not given and calibrate=False")
            best = None
            for u in self.U_GRID:
                pol = self._build(pilot, rates, u, debt_rounds=debt_rounds)
                m = core.eval_serving(pol, pilot)
                calib.append({"u": float(u), "pilot_slo_pass": float(m["slo_pass"]),
                              "pilot_worst": float(m["slo_pass_worst"])})
                key = (m["slo_pass"], m["slo_pass_worst"])
                if best is None or key > best[0]:
                    best = (key, u)
            u_target = best[1]
        pol = self._build(pilot, rates, u_target, debt_rounds=debt_rounds)
        pol["u_target"] = float(u_target)
        pol["calibration"] = calib
        return pol


# =========================================================================== #
# Quality
# =========================================================================== #
def _certified_dup_rate_costed(X, eps, nlist, seed=0, batch=512):
    """Certified epsilon-graph vertex rate, with the plan's own distance-evaluation count.

    Copied from the reference submission (the certification predicate is what the task
    requires); returns (dup_rate, evals, vertex_mask).
    """
    import faiss
    d = X.shape[1]
    km = faiss.Kmeans(d, nlist, niter=10, verbose=False, seed=seed)
    ntr = min(40_000, len(X))
    km.train(np.ascontiguousarray(X[np.random.RandomState(seed).choice(len(X), ntr,
                                                                      replace=False)]))
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
            r_, cc = np.nonzero(dd <= eps)
            gq, gc = s + qi[r_], mem[cc]
            keep = gq != gc
            verts[gq[keep]] = True
            verts[gc[keep]] = True
    return float(verts.mean()), float(evals), verts


def certified_cost_model(X, eps, nlists=(256, 512, 1024, 2048, 4096), seed=0):
    """Predict the certified rule's cost for candidate nlist values, without running it.

    The certification predicate is necessary and sufficient: a cell c must be probed for a
    point x iff ``||x - c|| <= eps + R_max(c)``.  The work has *two* terms and the exchange rate
    between them is the plan-selection problem:

      1. an eligibility pass of ``N * nlist`` distances (every point against every centroid,
         once) -- this term grows with the cell count;
      2. ``sum_c (#points eligible for c) * size_c`` pair evaluations -- this term shrinks with
         the cell count, because a compact cell has a small R_max(c) and therefore a small
         eligible region.

    The reference submission fixes nlist=256 by hand; the planner minimises the sum instead
    (build-time k-means is amortised, like every baseline's index build, and is excluded from
    the reported query cost).  ``n_elig`` is exact, so term (2) is an *identity*, not a model.
    """
    import faiss
    rows = []
    N = len(X)
    for nlist in nlists:
        nlist = int(nlist)
        ntr = min(40_000, N)
        km = faiss.Kmeans(X.shape[1], nlist, niter=10, verbose=False, seed=seed)
        km.train(np.ascontiguousarray(X[np.random.RandomState(seed).choice(N, ntr,
                                                                          replace=False)]))
        C = km.centroids.astype(np.float32)
        D2 = core.pairwise(X, C)
        assign = D2.argmin(1)
        sizes = np.bincount(assign, minlength=nlist).astype(np.int64)
        r_max = np.zeros(nlist, dtype=np.float32)
        for c in range(nlist):
            ids = np.flatnonzero(assign == c)
            if len(ids):
                r_max[c] = float(np.sqrt(D2[ids, c].max()))
        thr2 = (eps + r_max) ** 2
        n_elig = (D2 <= thr2[None, :]).sum(axis=0)
        pair = float((n_elig * sizes).sum())
        elig = float(N * nlist)
        rows.append({"nlist": nlist, "predicted_pair_evals": pair,
                     "predicted_eligibility_evals": elig,
                     "predicted_evals": pair + elig,
                     "mean_list_size": float(sizes.mean()),
                     "mean_elig": float(n_elig.mean())})
    return rows


def _ivf_dup_rate(X, eps, nprobe, nlist, seed=0, batch=512):
    """Heuristic (uncertified) epsilon-graph vertex rate: probe nprobe nearest lists only."""
    import faiss
    d = X.shape[1]
    km = faiss.Kmeans(d, nlist, niter=10, verbose=False, seed=seed)
    ntr = min(40_000, len(X))
    km.train(np.ascontiguousarray(X[np.random.RandomState(seed).choice(len(X), ntr,
                                                                      replace=False)]))
    C = km.centroids.astype(np.float32)
    D2 = core.pairwise(X, C)
    assign = D2.argmin(1)
    members = [np.flatnonzero(assign == c) for c in range(nlist)]
    order = np.argsort(D2, axis=1)[:, :nprobe]
    verts = np.zeros(len(X), dtype=bool)
    evals = 0
    for s in range(0, len(X), batch):
        B = X[s:s + batch]
        q = np.arange(len(B))
        idx = set()
        for t in range(nprobe):
            for c in order[s + q, t]:
                idx.update(members[c].tolist())
        if not idx:
            continue
        mem = np.asarray(sorted(idx), dtype=np.int64)
        dd = np.sqrt(np.maximum((B ** 2).sum(1)[:, None] + (X[mem] ** 2).sum(1)[None, :]
                                - 2.0 * B @ X[mem].T, 0.0))
        evals += dd.size
        r_, cc = np.nonzero(dd <= eps)
        gq, gc = s + r_, mem[cc]
        keep = gq != gc
        verts[gq[keep]] = True
        verts[gc[keep]] = True
    return float(verts.mean()), float(evals), verts

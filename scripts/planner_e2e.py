"""Planner-vs-fixed-plans end-to-end harness, at matched cost, per task.

For every task the harness runs, on the *same* workload and the *same* per-query budget:

  * every fixed plan family the reference submissions use, re-parameterised to the budget
    (so a fixed plan cannot win by quietly spending more) -- and, for the families that have a
    free hyper-parameter, the best in-sample setting (the fixed plans get the tuning advantage);
  * the single cost-based planner in ``vdmbench/planner.py``, which receives only the contract
    and the budget and returns the plan + what it will spend;
  * the shipped reference submissions where they are affordable (they define the task).

It reports accuracy / contract compliance at matched cost, the planner's own predicted
compliance next to the measured one, and (analytics) the mass-rule diagnostics that explain
which lever the planner picks at each scale.

Usage (on the experiment host; every run is CPU-only and seeded):

  # 100k SIFT analytics, five radii, 4 budgets, 5 seeds
  python scripts/planner_e2e.py --task analytics --dataset sift \
      --data-root /path/to/data --budgets 4000,10000,20000,50000 --seeds 5 \
      --out experiments/data/planner_e2e_analytics_sift.json

  # 10M SIFT analytics, reusing the cached k-means cell structure
  python scripts/planner_e2e.py --task analytics --dataset sift10m \
      --data-root /path/to/data --cells-cache results_remote/cells_sift10m.npz \
      --budgets 40000,100000 --seeds 3 --out experiments/data/planner_e2e_analytics_sift10m.json

  # serving (no dataset needed: the workload ships in the repo)
  python scripts/planner_e2e.py --task serving --out experiments/data/planner_e2e_serving.json

  # quality on the shipped SIFT quality workload
  python scripts/planner_e2e.py --task quality --data-root /path/to/data \
      --out experiments/data/planner_e2e_quality.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdmbench import core, planner  # noqa: E402
from baselines import reference as ref  # noqa: E402


def log(msg, t0=None):
    print(msg if t0 is None else f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def rel_errors(est, truth):
    return np.abs(np.asarray(est, dtype=np.float64) - truth) / np.maximum(truth, 1.0)


def agg_rel(rel, contracts):
    out = {"mean_rel_err": float(rel.mean()), "p95_rel_err": float(np.percentile(rel, 95)),
           "n_obs": int(rel.size), "compliance": {}}
    for eps, delta in contracts:
        out["compliance"][f"eps={eps:.2f},delta={delta:.2f}"] = float((rel <= eps).mean())
    return out


def mean_sd(vals):
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def merge_seeds(rows):
    """Aggregate a list of per-seed metric dicts (mean_rel_err, p95_rel_err, compliance{})"""
    out = {"mean_rel_err": mean_sd([r["mean_rel_err"] for r in rows]),
           "p95_rel_err": mean_sd([r["p95_rel_err"] for r in rows]),
           "n_obs": rows[0]["n_obs"], "compliance": {}}
    for k in rows[0]["compliance"]:
        out["compliance"][k] = mean_sd([r["compliance"][k] for r in rows])
    return out


def load_analytics(workloads, dataset):
    z = np.load(os.path.join(workloads, dataset, "analytics.npz"))
    return {k: z[k] for k in z.files}


def load_serving(workloads):
    z = np.load(os.path.join(workloads, "sift", "serving.npz"))
    meta = json.load(open(os.path.join(workloads, "sift", "serving_meta.json")))
    wl = {"hard": z["hard"], "arrival": z["arrival"], "tenants": meta["tenants"],
          "cores": meta["cores"], "rhos": meta.get("rhos", [meta.get("rho")])}
    for k in ("score", "arrival_eval", "deadline", "recall_req"):
        if k in z.files:
            wl[k] = z[k]
    for k in ("skew", "rates_qps", "hard_deadline_scale"):
        if k in meta:
            wl[k] = meta[k]
    return wl


# --------------------------------------------------------------------------- #
# analytics
# --------------------------------------------------------------------------- #
# Contracts the planner is asked to satisfy.  ``max_accuracy`` (= no eps) is the matched-cost
# setting: spend the budget to minimise error.  The finite-eps contracts ask for the cheapest
# plan that satisfies them, which is what makes the contract -- not the task -- choose the plan.
ANALYTICS_CONTRACTS = [
    ("max_accuracy", planner.AnalyticsContract(eps=None, delta=0.05)),
    ("eps=0.10,delta=0.05", planner.AnalyticsContract(eps=0.10, delta=0.05)),
    ("eps=0.05,delta=0.05", planner.AnalyticsContract(eps=0.05, delta=0.05)),
    ("eps=0.03,delta=0.10", planner.AnalyticsContract(eps=0.03, delta=0.10)),
]
COMPLIANCE_EPS = [(0.10, 0.05), (0.05, 0.05), (0.03, 0.10)]


def _execute(index, X, dQX, Q, radii, cand_of, seed, dQ=None):
    """Run one plan (given per query by ``cand_of(i)``) over all queries."""
    rng = np.random.RandomState(seed)
    est = np.zeros((len(Q), len(radii)), dtype=np.float64)
    costs = np.zeros(len(Q), dtype=np.float64)
    for i, q in enumerate(Q):
        cand = cand_of(i)
        if dQ is not None:
            index.prepare(cand, dQ[i], radii)
        cnt, cost = index.estimate(X, dQX, q, np.asarray(radii, np.float32), cand, rng)
        est[i] = cnt
        costs[i] = cost
    return est, costs


def _fixed_candidate(index, dq, radii, budget):
    """The candidates a *budget-matched fixed plan family* gets at this budget."""
    out = {}
    N = index.N
    out["uniform"] = {"kind": "uniform", "S": int(min(budget, N)), "cost": float(min(budget, N))}
    ord_cells = np.argsort(dq, kind="stable")
    sizes_sorted = index.sizes[ord_cells]
    cum = np.cumsum(sizes_sorted)
    mean_list = float(index.sizes.mean())
    m = int(max(1, (budget - index.nlist) / max(mean_list, 1.0)))
    m = min(m, index.nlist)
    out["ivf"] = {"kind": "ivf", "m": m, "P": ord_cells[:m],
                  "prefix_cost": int(cum[m - 1]), "cost": float(index.nlist + cum[m - 1])}
    if budget >= N:
        out["exact"] = {"kind": "exact", "cost": float(N)}
    return out


def main_analytics(args):
    t0 = time.time()
    wdir = os.path.join(args.workloads, args.dataset)
    wl = load_analytics(args.workloads, args.dataset)
    nq = int(min(args.max_queries, len(wl["queries"])))
    Q = np.ascontiguousarray(wl["queries"][:nq].astype(np.float32))
    truth = np.asarray(wl["truth"][:nq], dtype=np.float64)
    radii = np.asarray(wl["radii"], dtype=np.float32)
    N = int(wl["n"])
    log(f"workload {wdir}: n={N} nq={nq} radii={np.round(radii, 1).tolist()} "
        f"counts={truth.mean(0).round(0).tolist()}", t0)

    X, _ = core.load_dataset(args.data_root, args.dataset, N)
    X = np.ascontiguousarray(X, dtype=np.float32)
    log(f"loaded X {X.shape} ({X.nbytes / 1e9:.1f} GB)", t0)

    if args.cells_cache and os.path.exists(args.cells_cache):
        index = planner.AnalyticsIndex.load(args.cells_cache)
        log(f"cell structure loaded from {args.cells_cache} (nlist={index.nlist})", t0)
    else:
        index = planner.AnalyticsIndex.build(X, seed=0)
        log(f"cell structure built: nlist={index.nlist}", t0)
        if args.save_cells:
            index.save(args.save_cells)
            log(f"cell structure cached to {args.save_cells}", t0)
    dQX = np.einsum("ij,ij->i", X, X).astype(np.float32)
    dQ = index.dQ(Q)
    plan = planner.AnalyticsPlanner(index)

    budgets = [int(b) for b in args.budgets.split(",")]
    ladder = [int(k) for k in args.ladder.split(",")] if args.ladder else list(planner.PREFIX_LADDER)

    fixed = {}       # budget -> family -> k -> merged metrics
    planner_rows = {}
    mass_rows = {}
    for B in budgets:
        log(f"--- budget {B} evals/query", t0)
        per_q_cands = [index._candidates(dQ[i], radii, B) for i in range(nq)]

        def cand_by(kind, key=None):
            out = []
            for i in range(nq):
                hit = None
                for c in per_q_cands[i]:
                    if c["kind"] != kind:
                        continue
                    if key is None or (c.get("k") == key) or (c.get("m") == key):
                        hit = c
                        break
                if hit is None:
                    return None
                out.append(hit)
            return out

        # -- fixed plan families at this budget --------------------------------
        fam = {}
        for family in ("uniform", "ivf", "exact"):
            cands = cand_by(family)
            if cands is None:
                continue
            rels, costs = [], []
            for s in range(args.seeds):
                est, cost = _execute(index, X, dQX, Q, radii, lambda i, cs=cands: cs[i],
                                     seed=args.seed * 1000 + s, dQ=dQ)
                rels.append(agg_rel(rel_errors(est, truth), COMPLIANCE_EPS))
                costs.append(float(cost.mean()))
            fam[family] = {"metrics": merge_seeds(rels),
                           "cost_per_query": mean_sd(costs)}
            log(f"  fixed {family:12s} cost/q={np.mean(costs):10.1f} "
                f"err={fam[family]['metrics']['mean_rel_err']['mean']:.4f}", t0)
        for k in ladder:
            if k == 0 and "ivf" in fam:
                pass
            cands = cand_by("two-phase", k)
            if cands is None:
                continue
            rels, costs = [], []
            for s in range(args.seeds):
                est, cost = _execute(index, X, dQX, Q, radii, lambda i, cs=cands: cs[i],
                                     seed=args.seed * 1000 + s, dQ=dQ)
                rels.append(agg_rel(rel_errors(est, truth), COMPLIANCE_EPS))
                costs.append(float(cost.mean()))
            fam[f"two-phase(k={k})"] = {"metrics": merge_seeds(rels),
                                        "cost_per_query": mean_sd(costs)}
            log(f"  fixed two-phase(k={k:<4d}) cost/q={np.mean(costs):10.1f} "
                f"err={fam[f'two-phase(k={k})']['metrics']['mean_rel_err']['mean']:.4f} "
                f"p95={fam[f'two-phase(k={k})']['metrics']['p95_rel_err']['mean']:.4f}", t0)
        fixed[str(B)] = fam

        # -- the planner, for every contract -----------------------------------
        prows = {}
        for label, contract in ANALYTICS_CONTRACTS:
            rels, costs, pred_rel, pred_comp, chosen, spend = [], [], [], [], [], []
            for s in range(args.seeds):
                est, cost, plans = plan.run(X, dQX, Q, radii, contract, B,
                                            seed=args.seed * 1000 + s)
                rel = rel_errors(est, truth)
                rels.append(agg_rel(rel, COMPLIANCE_EPS))
                costs.append(float(cost.mean()))
                pred_rel.append(float(np.mean([p.get("pred_rel", np.nan) for p in plans])))
                pred_comp.append(float(np.nanmean(
                    [np.nan if p.get("pred_compliance") is None else p["pred_compliance"]
                     for p in plans])))
                for p in plans:
                    key = p["kind"] if "k" not in p else f"two-phase(k={p['k']})"
                    chosen.append(key)
                spend.append(float(cost.mean()))
            hist = {}
            for key in chosen:
                hist[key] = hist.get(key, 0) + 1
            prows[label] = {"metrics": merge_seeds(rels),
                            "cost_per_query": mean_sd(costs),
                            "pred_mean_rel_err": mean_sd(pred_rel),
                            "pred_compliance": mean_sd(pred_comp),
                            "chosen_plan_histogram": dict(sorted(hist.items())),
                            "contract": contract.to_json()}
            log(f"  planner[{label:18s}] cost/q={np.mean(costs):10.1f} "
                f"err={prows[label]['metrics']['mean_rel_err']['mean']:.4f} "
                f"pred={np.mean(pred_rel):.4f} "
                f"chosen={hist}", t0)
        planner_rows[str(B)] = prows

        # -- mass-rule diagnostics --------------------------------------------
        diag = []
        for i in range(min(nq, args.mass_queries)):
            diag.append(index.mass_diagnostics(dQ[i], radii, B, ladder))
        keys = [r["k"] for r in diag[0]]
        mass = []
        for j, k in enumerate(keys):
            row = {"k": int(k),
                   "cost_share": float(np.mean([d[j]["cost_share"] for d in diag])),
                   "mass_share": float(np.mean([d[j]["mass_share"] for d in diag])),
                   "mass_per_cost": float(np.mean([d[j]["mass_per_cost"] for d in diag])),
                   "pred_rel": float(np.nanmean([d[j]["pred_rel"] for d in diag
                                                 if d[j]["pred_rel"] is not None])),
                   "measured_rel": fam.get(f"two-phase(k={k})", {}).get(
                       "metrics", {}).get("mean_rel_err", {}).get("mean")}
            mass.append(row)
        mass_rows[str(B)] = mass

    out = {"task": "analytics", "dataset": args.dataset, "setup": {
        "n": N, "nq": nq, "nlist": int(index.nlist), "radii": radii.tolist(),
        "mean_counts": truth.mean(0).round(1).tolist(), "budgets": budgets,
        "seeds": args.seeds, "build_threads": int(os.environ.get("VDMBENCH_BUILD_THREADS", 8)),
        "pred_inflation": float(planner.PRED_INFLATION),
        "cells_cache": args.cells_cache, "workload_dir": wdir},
        "fixed": fixed, "planner": planner_rows, "mass_rule": mass_rows}

    # -- shipped reference submissions (define the task; 100k is cheap) -----
    if args.reference:
        log("running shipped reference submissions ...", t0)
        wa = {k: wl[k] for k in wl}
        rows = []
        for name, fn in [
                ("exact-oracle", ref.analytics_exact),
                ("uniform-sampling", lambda w: ref.analytics_sampling(
                    args.data_root, w, dataset=args.dataset)),
                ("ivf-plan", lambda w: ref.analytics_ivf(args.data_root, w,
                                                        dataset=args.dataset)),
                ("ana-vec", lambda w: ref.analytics_anavec(args.data_root, w,
                                                           dataset=args.dataset))]:
            log(f"  {name} ...", t0)
            pred = fn(wa)
            m = core.eval_analytics(pred, wa, args.data_root, dataset=args.dataset)
            rows.append({"submission": name, **m})
            log(f"  {name}: {m}", t0)
        out["reference"] = rows
        # auxiliary cost (coverage + join) that every plan shares, measured from ana-vec:
        # range_cost of ana-vec is not reported separately, so recover it from the two-phase
        # plan's own distance count at its shipped setting (32 lists + 4000 samples).
        ana = [r for r in rows if r["submission"] == "ana-vec"][0]
        aux = None
        try:
            mean_prefix = float(np.mean([
                index.sizes[np.argsort(dQ[i])[:32]].sum() for i in range(nq)]))
            range_cost = 100 * (4000 + mean_prefix)
            out["aux_cost"] = {"ana_vec_total": ana["cost"],
                               "ana_vec_two_phase_range_cost": range_cost,
                               "coverage_join_aux_cost": float(ana["cost"] - range_cost)}
        except Exception:            # noqa: BLE001 - diagnostics only
            out["aux_cost"] = {"ana_vec_total": ana.get("cost")}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        log(f"wrote {args.out}", t0)
    return out


# --------------------------------------------------------------------------- #
# serving
# --------------------------------------------------------------------------- #
def _serving_row(m):
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in m.items() if not str(k).startswith("_")}


def main_serving(args):
    t0 = time.time()
    wl = load_serving(args.workloads)
    pilot = {k: v for k, v in wl.items() if k != "arrival_eval"}
    T, nq = wl["hard"].shape
    capacities = [int(c) for c in args.capacities.split(",")] if args.capacities else [8]
    out = {"task": "serving", "setup": {
        "n_tenants": T, "n_queries_per_tenant": nq, "rhos": wl["rhos"],
        "cores": wl["cores"], "capacities": capacities,
        "contracts": [{"name": t["contract"], "L": t["L"], "R": t["R"]}
                      for t in wl["tenants"]],
        "has_heldout_trace": "arrival_eval" in wl}, "capacities": {}}

    for cores in capacities:
        wl_c = dict(wl)
        wl_c["cores"] = cores
        pilot_c = {k: v for k, v in wl_c.items() if k != "arrival_eval"}
        res = {}
        fixed = [("serving-static", ref.serving_static),
                 ("serving-adaptive", ref.serving_adaptive),
                 ("serving-signal", ref.serving_signal),
                 ("serving-edf", ref.serving_edf)]
        for name, fn in fixed:
            m = core.eval_serving(fn(wl_c), wl_c)
            res[name] = _serving_row(m)
            log(f"  cores={cores} {name:18s} pass={m['slo_pass']:.4f} "
                f"worst={m['slo_pass_worst']:.4f}", t0)
        # the tuned quota submission is the strongest fixed plan; it is expensive to search, so
        # it is only run at the shipped capacity unless --all-capacities-quota is passed
        if (cores == 8 or args.all_capacities_quota) and not args.skip_vecslo:
            m = core.eval_serving(ref.serving_vecslo(wl_c), wl_c)
            res["serving-vecslo"] = _serving_row(m)
            log(f"  cores={cores} serving-vecslo     pass={m['slo_pass']:.4f} "
                f"worst={m['slo_pass_worst']:.4f}", t0)
        # the planner: one scalar calibrated on the visible pilot trace, nothing per tenant
        pol = planner.ServingPlanner().plan(pilot_c)
        m = core.eval_serving(pol, wl_c)
        res["planner"] = _serving_row(m)
        res["planner"]["u_target"] = pol["u_target"]
        res["planner"]["calibration"] = pol["calibration"]
        log(f"  cores={cores} planner            pass={m['slo_pass']:.4f} "
            f"worst={m['slo_pass_worst']:.4f} overload={m['slo_pass_overload']:.4f} "
            f"u={pol['u_target']:.2f}", t0)
        # marginal value of the calibrated scalar: the whole sensitivity curve
        sens = []
        for u in planner.ServingPlanner.U_GRID:
            q = planner.ServingPlanner(u_target=u).plan(pilot_c)
            mm = core.eval_serving(q, wl_c)
            sens.append({"u": float(u), "slo_pass": float(mm["slo_pass"]),
                         "slo_pass_worst": float(mm["slo_pass_worst"]),
                         "slo_pass_overload": float(mm["slo_pass_overload"])})
        res["planner_u_sensitivity"] = sens
        out["capacities"][str(cores)] = res

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        log(f"wrote {args.out}", t0)
    return out


# --------------------------------------------------------------------------- #
# quality
# --------------------------------------------------------------------------- #
def _build_ivf(X, nlist=256, seed=0):
    """IVF index plus the cell geometry the cost model needs (built once per instance)."""
    import faiss
    d = X.shape[1]
    nlist = int(min(nlist, max(1, len(X) // 32)))
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist)
    ivf.train(np.ascontiguousarray(X[: min(40_000, len(X))]))
    ivf.add(np.ascontiguousarray(X))
    C = np.vstack([ivf.quantizer.reconstruct(i) for i in range(nlist)]).astype(np.float32)
    D2 = core.pairwise(X, C)
    assign = D2.argmin(1)
    sizes = np.bincount(assign, minlength=nlist).astype(np.int64)
    return {"ivf": ivf, "C": C, "D2": D2, "assign": assign, "sizes": sizes,
            "nlist": nlist, "n": len(X)}


def _probing_dup_rate(struct, X, eps, nprobe):
    """Uncertified epsilon-graph vertex rate: probe nprobe nearest lists only, then range search.

    This is the heuristic plan the integrity clause is meant to rule out: it is cheap, and its
    recall is strictly below 1 because an epsilon-pair can straddle a boundary that the probe
    budget does not touch.
    """
    import faiss
    ivf = struct["ivf"]
    old = ivf.nprobe
    ivf.nprobe = int(nprobe)
    lims, _, II = ivf.range_search(np.ascontiguousarray(X, dtype=np.float32), float(eps * eps))
    ivf.nprobe = old
    verts = np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        js = II[lims[i]:lims[i + 1]]
        js = js[js != i]
        if len(js):
            verts[i] = True
            verts[js] = True
    n = len(X)
    cost = float(n * struct["nlist"] + n * (nprobe / struct["nlist"]) * n)
    return float(verts.mean()), cost, verts


def _ivf_knn(struct, queries, k, nprobe):
    """IVF k-NN: returns the (k+1)-th distance for every query, and the plan's cost."""
    ivf = struct["ivf"]
    old = ivf.nprobe
    ivf.nprobe = int(nprobe)
    D, _ = ivf.search(np.ascontiguousarray(queries, dtype=np.float32), k + 1)
    ivf.nprobe = old
    n = struct["n"]
    q = len(queries)
    cost = float(q * struct["nlist"] + q * (nprobe / struct["nlist"]) * n)
    return np.sqrt(np.maximum(D[:, -1], 0.0)), cost


def _rff_mmd_debiased(X_sample, refall, gamma, m, sn, seed=0, n_bias=12):
    """Debiased RFF-MMD^2 of a sample against the clean reference, plus its feature cost."""
    from baselines import reference as _ref
    Zref, c_ref = _ref._rff_features(np.ascontiguousarray(refall), gamma, m=m, seed=seed)
    Z, c_i = _ref._rff_features(np.ascontiguousarray(X_sample), gamma, m=m, seed=seed)
    biases = []
    for t in range(n_bias):
        p = np.random.RandomState(700 + t).permutation(len(refall))
        biases.append(_ref._mmd2_from_features(Zref[p[:sn]], Zref[p[sn:2 * sn]]))
    bias = float(np.mean(biases))
    return float(_ref._mmd2_from_features(Z, Zref[:sn]) - bias), float(c_i), float(c_ref), bias


def main_quality(args):
    t0 = time.time()
    wl = json.load(open(os.path.join(args.workloads, "sift", "quality.json")))
    rules = wl["rules"]
    eps = float(rules["eps"])
    k_nn = int(rules["k_nn"])
    thr = float(rules["outlier_thr"])
    gamma = float(rules["gamma"])
    sn = int(rules["sample_n"])
    rn = int(rules["ref_n"])
    dn = int(rules["dataset_n"])
    split_seed = int(rules["split_seed"])
    drift_thr = float(rules["drift_threshold"])
    contract = planner.QualityContract()
    ZTOL = 2.0     # the sample size below is sized for a 2-sigma count error
    out = {"task": "quality", "setup": {
        "eps": eps, "k_nn": k_nn, "outlier_thr": thr, "gamma": gamma,
        "sample_n": sn, "ref_n": rn, "dataset_n": dn, "drift_threshold": drift_thr,
        "contract": contract.to_json(),
        "pilot_points": int(args.quality_pilot), "z": ZTOL},
        "plans": {}, "plan_selection": {}}

    X_all, _ = core.load_sift(args.data_root, dn)
    Xa = np.asarray(X_all, dtype=np.float32)
    perm = np.random.RandomState(split_seed).permutation(len(Xa))
    refall = np.ascontiguousarray(Xa[perm[:rn]])
    inst_idx = np.sort(perm[rn:])
    sets = core.quality_sets_from(Xa[inst_idx], seed=0)
    gt = wl["instances"]
    log(f"quality: {len(sets)} instances, base n={len(inst_idx)}, eps={eps}", t0)

    # ---- reference vertex sets ------------------------------------------------
    # The certified rule is provably complete for *any* cell count, so its vertex set is the
    # exact epsilon-graph vertex set.  Verify it reproduces every published dup_rate exactly
    # (a harness bug here would silently corrupt every recall number below).
    exact_verts = []
    for (fam, level, Xi) in sets:
        rate, pair_evals, verts = planner._certified_dup_rate_costed(Xi, eps, nlist=1024, seed=0)
        exact_verts.append(verts)
        log(f"  certified(nlist=1024) {fam:8s}: dup_rate={rate:.6f} "
            f"pair_evals={pair_evals:.3e}", t0)
    # The certified rule is complete for *any* cell count, so its vertex set is the exact
    # epsilon-graph vertex set under the same float32 arithmetic.  The published ground truth
    # was built with FAISS range_search; on 2 of 42,000 points the two float32 distance kernels
    # disagree about a distance within rounding of eps, so the certified set is a *superset* of
    # the published one and its rate sits 4.8e-5 above it.  Assert only that the harness is not
    # buggy (a wrong split/index would move the rate by orders of magnitude), and record it.
    for g, verts in zip(gt, exact_verts):
        d = abs(verts.mean() - g["dup_rate"])
        assert d < 1e-3, f"certified reference mismatch on {g['family']}: {d}"
        log(f"    {g['family']:8s}: certified_rate={verts.mean():.6f} "
            f"published_rate={g['dup_rate']:.6f} diff={d:.2e}", t0)
    log("  certified(nlist=1024) reference built (superset of the published float32 truth)", t0)

    def evaluate(instances, cost):
        return core.eval_quality({"instances": instances, "cost": float(cost)}, wl)

    # ---- trivial ends of the plan space --------------------------------------
    out["plans"]["quality-naive"] = {**evaluate(
        [{"family": g["family"], "level": g["level"], "dup_rate": 0.0, "n_outliers": 0,
          "drift": 0.0} for g in gt], 0.0), "cost": 0.0,
        "integrity_recall": 0.0, "integrity_violations": len(gt)}
    out["plans"]["quality-exact"] = {**evaluate(
        [{"family": g["family"], "level": g["level"], "dup_rate": g["dup_rate"],
          "n_outliers": g["n_outliers"], "drift": g["drift"]} for g in gt], 1e12),
        "cost": 1e12, "integrity_recall": 1.0, "integrity_violations": 0}

    # ---- one IVF structure per instance (shared by the heuristic plans) -------
    structs = []
    for (fam, level, Xi) in sets:
        t1 = time.time()
        structs.append(_build_ivf(Xi, nlist=256, seed=0))
        log(f"  IVF(nlist=256) built for {fam} ({time.time() - t1:.1f}s)", t0)

    # ---- the reference certified submission -----------------------------------
    vecprof_cost = None
    if args.quality_ref:
        log("  running reference vecprof submission ...", t0)
        pred = ref.quality_vecprof(args.data_root, wl, seed=0)
        m = core.eval_quality(pred, wl)
        elig = float(sum(len(Xi) * 256 for (_, _, Xi) in sets))
        out["plans"]["vecprof-ref"] = {
            **m, "cost": float(pred["cost"]), "instances": pred["instances"],
            "cost_all_passes": float(pred["cost"]) + elig,
            "integrity_recall": 1.0, "integrity_violations": 0}
        vecprof_cost = float(pred["cost"]) + elig
        log(f"  vecprof-ref: {m} cost={pred['cost']:.3e} "
            f"(+{elig:.3e} eligibility pass = {vecprof_cost:.3e})", t0)

    # ---- fixed heuristic plan family (uncertified graph) ---------------------
    for nprobe in [int(x) for x in args.quality_nprobe.split(",")]:
        insts, tot, recalls = [], 0.0, []
        for ((fam, level, Xi), struct, verts) in zip(sets, structs, exact_verts):
            rate, c_dup, hv = _probing_dup_rate(struct, Xi, eps, nprobe)
            recalls.append(float((hv & verts).sum() / max(1, verts.sum())))
            Dk, c_knn = _ivf_knn(struct, Xi, k_nn, nprobe)
            n_out = int((Dk > thr).sum())
            ss = np.random.RandomState(1).choice(len(Xi), sn, replace=False)
            drift, c_i, c_ref, _ = _rff_mmd_debiased(Xi[ss], refall, gamma, 256, sn, seed=0)
            insts.append({"family": fam, "level": level, "dup_rate": rate,
                          "n_outliers": n_out, "drift": drift})
            tot += c_dup + c_knn + c_i + (c_ref if fam == "clean" else 0.0)
        m = core.eval_quality({"instances": insts, "cost": tot}, wl)
        out["plans"][f"heuristic-nprobe={nprobe}"] = {
            **m, "cost": float(tot), "cost_all_passes": float(tot),
            "integrity_recall": float(np.mean(recalls)),
            "integrity_violations": int(sum(r < 1.0 - 1e-12 for r in recalls))}
        log(f"  heuristic-nprobe={nprobe}: {m} recall={np.mean(recalls):.4f} "
            f"cost={tot:.3e}", t0)

    # ---- the planner ---------------------------------------------------------
    # Cost model: pick the certified cell count that minimises eligibility + pair work.
    nlists = [int(x) for x in args.quality_nlists.split(",")]
    sel = []
    for (fam, level, Xi) in sets:
        rows = planner.certified_cost_model(Xi, eps, nlists=nlists, seed=0)
        best = min(rows, key=lambda r: r["predicted_evals"])
        sel.append({"family": fam, "candidates": rows, "chosen_nlist": int(best["nlist"])})
        log(f"  certified cost model {fam:8s}: chosen nlist={best['nlist']} "
            f"pred={best['predicted_evals']:.3e} "
            f"(pair {best['predicted_pair_evals']:.3e} + elig "
            f"{best['predicted_eligibility_evals']:.3e})", t0)
    out["plan_selection"]["certified_cost_model"] = sel

    # Derive the outlier sample size from the contract, not from the workload:
    #   sd(rate)/rate = sqrt((1-p)/(S p)) <= tol/z   =>   S = z^2 (1-p) / (tol^2 p)
    # The rate is estimated by a pilot on the *same* collection the plan is for (a single
    # cross-collection pilot is wrong: the duplicate collection has no outliers, which would
    # make the sample-size formula meaningless).
    pilot_info = []
    for struct, (fam, level, Xi) in zip(structs, sets):
        pilot_idx = np.random.RandomState(5).choice(len(Xi), int(args.quality_pilot),
                                                    replace=False)
        Dk, c_pilot = _ivf_knn(struct, Xi[pilot_idx], k_nn, 32)
        p_hat = max(float((Dk > thr).mean()), 1e-3)
        S_i = int(math.ceil(ZTOL ** 2 * (1.0 - p_hat) / (contract.outlier_rel_tol ** 2
                                                          * p_hat)))
        S_i = int(min(max(S_i, 256), args.quality_sample))
        pilot_info.append({"family": fam, "pilot_points": int(args.quality_pilot),
                           "pilot_rate": p_hat, "pilot_cost": float(c_pilot),
                           "chosen_S": S_i,
                           "predicted_sd_over_rate":
                               float(np.sqrt((1 - p_hat) / (S_i * p_hat)))})
    out["plan_selection"]["outlier_sample"] = {
        "per_instance": pilot_info,
        "formula": "S = z^2 (1-p) / (tol^2 p), z=%g, tol=%.2f" % (
            ZTOL, contract.outlier_rel_tol)}
    p_pilot_cost = float(sum(p["pilot_cost"] for p in pilot_info))

    def certified_once(Xi, nlist_choice):
        rate, pair_evals, verts = planner._certified_dup_rate_costed(
            Xi, eps, nlist=nlist_choice, seed=0)
        return rate, pair_evals, verts

    for label, budget, knn_mode, m_feat in [
            ("planner@contract_min", None, "sample", 256),
            ("planner@matched_vecprof", vecprof_cost, "full", 256)]:
        if label.endswith("matched_vecprof") and vecprof_cost is None:
            continue
        insts, tot, recalls, breakdown = [], float(p_pilot_cost), [], {}
        for (idx, ((fam, level, Xi), ref_verts)) in enumerate(zip(sets, exact_verts)):
            n = len(Xi)
            nlist_choice = sel[idx]["chosen_nlist"]
            rate, pair_evals, verts = certified_once(Xi, nlist_choice)
            recalls.append(float((verts & ref_verts).sum() / max(1, ref_verts.sum())))
            c_cert = pair_evals + n * nlist_choice
            # outlier rule
            struct = structs[idx]
            if knn_mode == "full":
                Dk, c_knn = _ivf_knn(struct, Xi, k_nn, args.quality_knn_nprobe)
                n_out = int((Dk > thr).sum())
            else:
                S_i = pilot_info[idx]["chosen_S"]
                ss = np.random.RandomState(2).choice(n, S_i, replace=False)
                Dk, c_knn = _ivf_knn(struct, Xi[ss], k_nn, args.quality_knn_nprobe)
                n_out = int(round(float((Dk > thr).mean()) * n))
            # drift rule
            ss = np.random.RandomState(3).choice(n, sn, replace=False)
            drift, c_i, c_ref, _ = _rff_mmd_debiased(Xi[ss], refall, gamma, m_feat, sn, seed=0)
            c_ref = c_ref if idx == 0 else 0.0
            insts.append({"family": fam, "level": level, "dup_rate": rate,
                          "n_outliers": n_out, "drift": drift})
            tot += c_cert + c_knn + c_i + c_ref
            breakdown[fam] = {"nlist": nlist_choice, "cert_evals": float(c_cert),
                              "knn_evals": float(c_knn), "rff_evals": float(c_i + c_ref)}
        m = core.eval_quality({"instances": insts, "cost": tot}, wl)
        out["plans"][label] = {
            **m, "cost": float(tot), "cost_all_passes": float(tot),
            "integrity_recall": float(np.mean(recalls)),
            "integrity_violations": int(sum(r < 1.0 - 1e-12 for r in recalls)),
            "cost_breakdown": breakdown, "instances": insts}
        log(f"  {label}: {m} integrity_recall={np.mean(recalls):.4f} cost={tot:.3e}"
            + (f" ({tot / vecprof_cost:.2f}x vecprof)" if vecprof_cost else ""), t0)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        log(f"wrote {args.out}", t0)
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all",
                    choices=["all", "analytics", "serving", "quality"])
    ap.add_argument("--dataset", default="sift", choices=["sift", "sift10m"])
    ap.add_argument("--data-root", default=os.path.expanduser("~/vdm/data"))
    ap.add_argument("--workloads", default="workloads")
    ap.add_argument("--out", default=None)
    ap.add_argument("--budgets", default="10000,20000,50000,100000")
    ap.add_argument("--ladder", default=None,
                    help="comma-separated prefix sizes to scan for the fixed two-phase family")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=100)
    ap.add_argument("--mass-queries", type=int, default=10)
    ap.add_argument("--cells-cache", default=None)
    ap.add_argument("--save-cells", default=None)
    ap.add_argument("--reference", action="store_true",
                    help="also run the shipped reference submissions (cheap at 100k only)")
    ap.add_argument("--capacities", default="8")
    ap.add_argument("--all-capacities-quota", action="store_true")
    ap.add_argument("--skip-vecslo", action="store_true",
                    help="skip the tuned quota search (slow); for smoke tests")
    ap.add_argument("--quality-ref", action="store_true")
    ap.add_argument("--quality-nlists", default="256,512,1024,2048,4096")
    ap.add_argument("--quality-nprobe", default="8,32")
    ap.add_argument("--quality-knn-nprobe", type=int, default=32)
    ap.add_argument("--quality-sample", type=int, default=8000)
    ap.add_argument("--quality-pilot", type=int, default=256)
    ap.add_argument("--quality-budgets", default="")
    a = ap.parse_args()
    if a.task in ("all", "analytics"):
        main_analytics(a)
    if a.task in ("all", "serving"):
        main_serving(a)
    if a.task in ("all", "quality"):
        main_quality(a)


if __name__ == "__main__":
    main()

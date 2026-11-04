#!/usr/bin/env python3
"""Read out the committed planner_e2e quality/serving JSONs. Read-only, CPU-free.

This script does not run any experiment: it only re-formats the numbers already in
`experiments/data/planner_e2e_{quality,serving}.json`, which are the files cited by
`experiments/unified_planner_2026-11-05.md`.

    python scripts/readout_planner_e2e.py
"""
import argparse
import hashlib
import json
import os

PLANS_QUALITY = ["quality-exact", "vecprof-ref", "heuristic-nprobe=8",
                 "heuristic-nprobe=32", "planner@contract_min", "planner@matched_vecprof"]
PLANS_SERVING = ["serving-static", "serving-adaptive", "serving-signal", "serving-edf",
                 "serving-vecslo", "planner"]


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def readout_quality(path):
    q = json.load(open(path))
    plans = q["plans"]
    ref_cost = plans["vecprof-ref"]["cost_all_passes"]
    print(f"\n## quality  ({path}, sha256:{_sha(path)})")
    print("| plan | dup err | outlier err | drift err | detect | cert-set coverage "
          "(violations) | cost (evals) | x vecprof |")
    print("|---|---|---|---|---|---|---|---|")
    for name in PLANS_QUALITY:
        p = plans[name]
        cov = p.get("integrity_recall")
        cov = "n/a" if cov is None else f"{cov:.4f} ({p['integrity_violations']})"
        num = p.get("cost_all_passes", p["cost"])      # the x-axis is the all-passes cost
        print(f"| {name} | {p['dup_abs_err_mean']:.3e} | {p['outlier_abs_err_mean']:.1f} | "
              f"{p['drift_err_thr_units']:.3f} | {p['detect_acc']:.2f} | {cov} | "
              f"{p['cost']:.3e} | {num/ref_cost:.3f} |")
    # ---- contract table with per-seed spread (new) --------------------------
    seeds = q["setup"].get("seeds", 1)
    print(f"\ncontract compliance, mean ± sd over {seeds} seed(s) "
          f"(from each plan's 'seed_spread'):")
    print("| plan | contract rate | drift err (thr units) | cost |")
    print("|---|---|---|---|")
    for name in PLANS_QUALITY:
        p = plans[name]
        sp = p.get("seed_spread", {})

        def ms(key, fmt):
            if key in sp:
                return f"{sp[key]['mean']:.{fmt}f} ± {sp[key]['sd']:.{fmt}f}"
            return f"{p.get(key, float('nan')):.{fmt}f}"

        print(f"| {name} | {ms('contract_rate', 3)} | {ms('drift_err_thr_units', 4)} | "
              f"{ms('cost', 3)} |")

    print("\ncost breakdown (pilot + certified + knn + rff) vs recorded cost:")
    os_ = q["plan_selection"]["outlier_sample"]
    pilot = os_.get("pilot_cost")
    if pilot is None:      # newer schema: per-instance pilot records
        pilot = sum(p["pilot_cost"] for p in os_.get("per_instance", []))
    for name in ["planner@contract_min", "planner@matched_vecprof"]:
        cb = plans[name]["cost_breakdown"]
        s = pilot + sum(v["cert_evals"] + v["knn_evals"] + v["rff_evals"] for v in cb.values())
        rec = plans[name].get("cost_seed0", plans[name]["cost"])
        print(f"  {name}: sum={s:.0f} seed0_recorded={rec:.0f} match={s == rec}")
    print("\nplanner cost-model selection (predicted == measured certified evals):")
    for row in q["plan_selection"]["certified_cost_model"]:
        fam, nl = row["family"], row["chosen_nlist"]
        pred = [c for c in row["candidates"] if c["nlist"] == nl][0]["predicted_evals"]
        meas = plans["planner@contract_min"]["cost_breakdown"][fam]["cert_evals"]
        print(f"  {fam:8s} nlist={nl} predicted={pred:.0f} measured={meas:.0f} "
              f"match={pred == meas}")
    print("\nper-instance errors of the planner rows vs ground truth:")
    wl = json.load(open(os.path.join(os.path.dirname(path), "..", "..", "workloads",
                                     "sift", "quality.json")))
    thr = wl["rules"]["drift_threshold"]
    gt = wl["instances"]
    for name in ["planner@contract_min", "planner@matched_vecprof"]:
        for p, g in zip(plans[name]["instances"], gt):
            print(f"  {name:24s} {p['family']:8s} dup_err={abs(p['dup_rate']-g['dup_rate']):.3e} "
                  f"out={p['n_outliers']:4d}/{g['n_outliers']:3d} "
                  f"drift_err={abs(p['drift']-g['drift'])/thr:.3f}")


def readout_serving(path):
    s = json.load(open(path))
    st = s["setup"]
    print(f"\n## serving  ({path}, sha256:{_sha(path)})")
    print(f"tenants={st['n_tenants']} queries/tenant={st['n_queries_per_tenant']} "
          f"rhos={st['rhos']} capacities={st['capacities']} "
          f"held_out={st['has_heldout_trace']}")
    print("\n| cores | policy | mean | worst | overload | tenant worst | p95 | p99 | "
          "goodput | served |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for cap in [str(c) for c in st["capacities"]]:
        for name in PLANS_SERVING:
            v = s["capacities"][cap].get(name)
            if v is None:
                print(f"| {cap} | {name} | not run | | | | | | | |")
                continue
            print(f"| {cap} | {name} | {v['slo_pass']:.4f} | {v['slo_pass_worst']:.4f} | "
                  f"{v['slo_pass_overload']:.4f} | {v['tenant_pass_worst']:.4f} | "
                  f"{v['p95_latency']:.1f} | {v['p99_latency']:.1f} | "
                  f"{v['goodput_qps']:.0f} | {v['served_frac']:.4f} |")
    # ---- per-regime vector (7 regimes), the aggregate's blind spot ----------
    caps = [str(c) for c in st["capacities"]]
    for cap in caps:
        cd = s["capacities"][cap]
        names = [n for n in PLANS_SERVING if n in cd]
        print(f"\nper-regime slo_pass at cores={cap} (rhos = {st['rhos']}):")
        print("| policy | " + " | ".join(f"rho={r:g}" for r in st["rhos"]) + " |")
        print("|---" * (len(st["rhos"]) + 1) + "|")
        for name in names:
            pr = cd[name].get("per_regime")
            if not pr:
                print(f"| {name} | per-regime not recorded |")
                continue
            print(f"| {name} | " + " | ".join(f"{v:.3f}" for v in pr["slo_pass"]) + " |")
        print(f"\nper-regime worst-tenant pass / served fraction at cores={cap} "
              f"(planner vs the best fixed plan by mean):")
        fixed = [n for n in names if n.startswith("serving-")]
        best = max(fixed, key=lambda n: (cd[n]["slo_pass"], cd[n]["slo_pass_worst"]))
        print("| policy | " + " | ".join(f"rho={r:g}" for r in st["rhos"]) + " |")
        print("|---" * (len(st["rhos"]) + 1) + "|")
        for name in (best, "planner"):
            pr = cd[name].get("per_regime")
            if not pr or "worst_tenant_pass" not in pr:
                print(f"| {name} | per-regime not recorded |")
                continue
            cells = [f"{w:.3f} / {sf:.3f}" for w, sf in
                     zip(pr["worst_tenant_pass"], pr.get("served_frac", []))]
            print(f"| {name} | " + " | ".join(cells) + " |")

    print("\nplanner vs best fixed plan that was actually run, and the calibrated u:")
    for cap in [str(c) for c in st["capacities"]]:
        cd = s["capacities"][cap]
        fixed = [n for n in PLANS_SERVING if n.startswith("serving-") and n in cd]
        best = max(fixed, key=lambda n: (cd[n]["slo_pass"], cd[n]["slo_pass_worst"]))
        b, p = cd[best], cd["planner"]
        print(f"  cores={cap}: best fixed={best} | dmean={p['slo_pass']-b['slo_pass']:+.4f} "
              f"dworst={p['slo_pass_worst']-b['slo_pass_worst']:+.4f} "
              f"doverload={p['slo_pass_overload']-b['slo_pass_overload']:+.4f} | "
              f"u_chosen={cd['planner']['u_target']}")
        u_per = cd["planner"].get("u_target_per_regime")
        pr_per = cd["planner"].get("profile_per_regime")
        if u_per is not None:
            print(f"           per-regime actions (u): {[round(x,3) for x in u_per]}")
        if pr_per is not None:
            print(f"           per-regime actions (profile): {pr_per}")
        grid = {r["u"]: r for r in cd["planner_u_sensitivity"]}
        u_mean = max(grid, key=lambda u: grid[u]["slo_pass"])
        u_worst = max(grid, key=lambda u: grid[u]["slo_pass_worst"])
        print(f"           eval-grid argmax (single-u policies): mean u={u_mean} "
              f"({grid[u_mean]['slo_pass']:.4f}), worst u={u_worst} "
              f"({grid[u_worst]['slo_pass_worst']:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="experiments/data/planner_e2e_quality.json")
    ap.add_argument("--serving", default="experiments/data/planner_e2e_serving.json")
    a = ap.parse_args()
    readout_quality(a.quality)
    readout_serving(a.serving)


if __name__ == "__main__":
    main()

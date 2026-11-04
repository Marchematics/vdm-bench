"""Build the deterministic VDM-Bench workloads from raw datasets."""
import argparse
import json
import os

import numpy as np

from vdmbench import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="workloads")
    ap.add_argument("--dataset", default="sift", choices=["sift", "sift10m", "mnist", "scifact"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--quality-n", type=int, default=50_000)
    ap.add_argument("--only", default="all",
                    choices=["all", "analytics", "quality", "serving"],
                    help="regenerate just one task family (others are left untouched)")
    ap.add_argument("--serving-rhos", default="0.70,1.00,1.15,1.60,2.00,2.50,3.00",
                    help="comma-separated aggregate utilisation levels (1.0 = peak; "
                         "1.6-3.0 are overload)")
    ap.add_argument("--serving-skew", default="4,1,1,1,1,1,1,1",
                    help="comma-separated per-tenant arrival weights (hot tenant first)")
    a = ap.parse_args()
    out = os.path.join(a.out, a.dataset) if a.out == "workloads" else a.out
    os.makedirs(out, exist_ok=True)
    manifest = {"dataset": a.dataset}

    do_analytics = a.only in ("all", "analytics")
    do_quality = a.only in ("all", "quality") and a.dataset == "sift"
    do_serving = a.only in ("all", "serving") and a.dataset == "sift"

    if do_analytics:
        print("[analytics] ...", flush=True)
        wa = core.build_analytics(a.data_root, dataset=a.dataset, n=a.n)
        np.savez_compressed(os.path.join(out, "analytics.npz"),
                            queries=wa["queries"], radii=wa["radii"], truth=wa["truth"],
                            anchors=wa["anchors"], r_cov=wa["r_cov"], k=wa["k"],
                            exact_covered=wa["exact_covered"],
                            coverage_chosen=wa["coverage_chosen"],
                            X_join_n=wa["X_join_n"], r_join=wa["r_join"],
                            join_pairs=wa["join_pairs"], n=wa["n"])
        manifest["analytics_n"] = wa["n"]

    if do_quality:
        print("[quality] ...", flush=True)
        wq = core.build_quality(a.data_root, n=a.quality_n)
        with open(os.path.join(out, "quality.json"), "w") as f:
            json.dump(wq, f, indent=2)
        manifest["quality_instances"] = len(wq["instances"])

    if do_serving:
        print("[serving] ...", flush=True)
        skew = [float(x) for x in str(a.serving_skew).split(",")]
        rhos = [float(x) for x in str(a.serving_rhos).split(",")]
        ws = core.build_serving(rhos=rhos, skew=skew, eval_seed=1000 + 7)
        np.savez_compressed(os.path.join(out, "serving.npz"),
                            hard=ws["hard"], score=ws["score"],
                            deadline=ws["deadline"], recall_req=ws["recall_req"],
                            arrival=ws["arrival"], arrival_eval=ws["arrival_eval"])
        with open(os.path.join(out, "serving_meta.json"), "w") as f:
            json.dump({"tenants": ws["tenants"], "cores": ws["cores"],
                       "skew": ws["skew"], "rhos": ws["rhos"],
                       "rates_qps": ws["rates_qps"],
                       "mean_base_service_ms": ws["mean_base_service_ms"],
                       "held_out_trace": True,
                       "contracts": ws["contracts"],
                       "hard_frac": ws["hard_frac"],
                       "hard_deadline_scale": ws["hard_deadline_scale"],
                       "hard_effort_rule": "recall = min(recall(nprobe), recall(nprobe // 2))",
                       "overload_rho": ws["overload_rho"],
                       "policy_format": "efforts/priority/admission may be (T,nq) "
                                        "load-blind or (R,T,nq) load-indexed",
                       "surface": ws["surface"]}, f, indent=2)
        manifest["serving_tenants"] = len(ws["tenants"])
        manifest["serving_rhos"] = ws["rhos"]

    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("done ->", out)


if __name__ == "__main__":
    main()

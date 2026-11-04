"""Score a submission against one VDM-Bench task."""
import argparse
import json
import os

import numpy as np

from vdmbench import core


def load_wl(task, wl_dir):
    if task == "analytics":
        z = np.load(os.path.join(wl_dir, "analytics.npz"))
        return {k: z[k].item() if z[k].shape == () else z[k] for k in z.files}
    if task == "quality":
        wl = json.load(open(os.path.join(wl_dir, "quality.json")))
        wl["contract"] = core.QUALITY_CONTRACT   # published contract, enforced by eval_quality
        return wl
    if task == "serving":
        z = np.load(os.path.join(wl_dir, "serving.npz"))
        meta = json.load(open(os.path.join(wl_dir, "serving_meta.json")))
        wl = {"hard": z["hard"], "arrival": z["arrival"],
              "tenants": meta["tenants"], "cores": meta["cores"],
              "rhos": meta.get("rhos", [meta.get("rho")])}
        for k in ("score", "arrival_eval", "deadline", "recall_req"):
            if k in z.files:
                wl[k] = z[k]
        for k in ("skew", "rates_qps", "hard_deadline_scale"):
            if k in meta:
                wl[k] = meta[k]
        return wl
    raise SystemExit("unknown task")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["analytics", "quality", "serving"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--workloads", default="workloads")
    ap.add_argument("--dataset", default="sift", choices=["sift", "sift10m", "mnist", "scifact"])
    ap.add_argument("--predictions", required=True)
    a = ap.parse_args()
    wdir = os.path.join(a.workloads, a.dataset) if os.path.isdir(os.path.join(a.workloads, a.dataset)) else a.workloads
    wl = load_wl(a.task, wdir)
    pred = json.load(open(a.predictions))
    if a.task == "analytics":
        m = core.eval_analytics(pred, wl, a.data_root, dataset=a.dataset)
    elif a.task == "quality":
        m = core.eval_quality(pred, wl)
    else:
        m = core.eval_serving(pred, wl)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()

"""Diagnose the planner's rho=2.0 deficit against VecSLO on the shipped serving workload.

Answers: is the deficit (a) queue-state blindness of the planner's admission, or (b) the
policy space (per-tenant admission quotas + per-tenant hard-effort quota)?  Decomposes the two
policies at cores=8, rho=2.0 by swapping one lever at a time, and reports per-tenant
served/pass so the mechanism is visible.

Read-only: no files are written. CPU-only; run on the experiment host.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdmbench import core, planner  # noqa: E402
from baselines import reference as ref  # noqa: E402


def load(workloads):
    meta = json.load(open(os.path.join(workloads, "sift", "serving_meta.json")))
    z = np.load(os.path.join(workloads, "sift", "serving.npz"))
    wl = {"hard": z["hard"], "arrival": z["arrival"], "tenants": meta["tenants"],
          "cores": meta["cores"], "rhos": meta["rhos"]}
    for k in ("score", "arrival_eval", "deadline", "recall_req"):
        if k in z.files:
            wl[k] = z[k]
    for k in ("skew", "rates_qps", "hard_deadline_scale"):
        if k in meta:
            wl[k] = meta[k]
    return wl


def as3d(a, n_regimes):
    a = np.asarray(a)
    return np.repeat(a[None], n_regimes, 0) if a.ndim == 2 else a


def main():
    ap_args = sys.argv[1:]
    workloads = ap_args[0] if ap_args else "workloads"
    wl = load(workloads)
    cores = 8
    wl = dict(wl)
    wl["cores"] = cores
    r = int(np.flatnonzero(np.isclose([float(x) for x in wl["rhos"]], 2.0))[0])
    NR = len(wl["rhos"])
    vec = ref.serving_vecslo(wl)
    plan = planner.ServingPlanner().plan(wl)
    ev = as3d(vec["efforts"], NR); av = as3d(vec["admission"], NR)
    pv = as3d(vec["priority"], NR)
    ep = as3d(plan["efforts"], NR); ap = as3d(plan["admission"], NR)
    pp = as3d(plan["priority"], NR)

    def ev_at(name, eff, pri, adm):
        service, recall = core._serving_service_recall(wl, eff[r])
        lat = core._simulate(wl["arrival_eval"][r], service, cores, pri[r], adm[r])
        lat = np.where(adm[r], lat, 1e6)
        ok = (lat <= wl["deadline"]) & (recall >= wl["recall_req"])
        print(f"{name:26s} served={adm[r].mean():.3f} pass={ok.mean():.3f} "
              f"admitted-pass={ok.sum()/max(1, adm[r].sum()):.3f}")
        print(f"  tenant served={[round(float(x),3) for x in adm[r].mean(1)]}")
        print(f"  tenant pass  ={[round(float(x),3) for x in ok.mean(1)]}")

    print(f"rho=2.0 at cores={cores}")
    ev_at("vecslo", ev, pv, av)
    ev_at("planner (debt)", ep, pp, ap)
    ev_at("vec-eff + plan-adm", ev, pp, ap)
    ev_at("plan-eff + vec-adm", ep, pv, av)
    print("\nreading: if 'plan-eff + vec-adm' recovers most of the gap, the lever is the "
          "admission policy space; if 'vec-eff + plan-adm' does, it is effort.")


if __name__ == "__main__":
    main()

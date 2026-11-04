"""Run the contract-aware vecprof-ref submission and print the drift instrumentation.

Prints, per instance: the estimated MMD^2, the CI half-width (raw and in threshold units),
the declared tolerance, the defined sample size, the final subsample size, the drift cost and
the total per-instance cost; then the eval_quality contract result (rate, met flags, failures).
"""
import argparse
import json

from baselines import reference as ref
from vdmbench import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--workloads", default="workloads")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    wl = json.load(open(a.workloads + "/sift/quality.json"))
    rules = wl["rules"]
    thr = float(rules["drift_threshold"])
    pred = ref.quality_vecprof(a.data_root, wl, verbose=True)
    m = core.eval_quality(pred, wl)
    print("\n=== per-instance drift instrumentation ===")
    print(f"{'family':8s} {'drift':>12s} {'drift/thr':>9s} {'CI thr':>7s} {'tol':>4s} "
          f"{'n':>5s} {'k':>5s} {'drift_cost':>11s} {'inst_cost':>11s}")
    for inst in pred["instances"]:
        print(f"{inst['family']:8s} {inst['drift']:12.4e} {inst['drift']/thr:9.3f} "
              f"{inst['drift_ci_thr_units']:7.3f} {inst['drift_tol_thr_units']:4.1f} "
              f"{inst['sample_n']:5d} {inst['subsample_n']:5d} "
              f"{inst['drift_cost']:11.3e} {inst['cost']:11.3e}")
    print(f"\ntotal cost = {pred['cost']:.4e}")
    print("\n=== eval_quality ===")
    print(json.dumps({k: v for k, v in m.items() if not k.startswith("_")}, indent=1))
    print("contract_met   =", m["_contract_met"])
    print("contract_failures =", json.dumps(m["_contract_failures"], indent=1))
    if a.out:
        json.dump({"prediction": pred, "metrics": m}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

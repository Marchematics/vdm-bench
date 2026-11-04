"""Verify the quality workload's ground truth is reproducible, and measure its own sampling noise.

Why: the drift diagnostic suggested the hole instance's ground truth (+1.097 threshold units) is
far above the value the exact estimator returns on *other* 4000-samples of the same collection
(-0.46 thr). Before concluding the shipped estimator is noisy, check that the diagnostic rebuilds
the workload exactly (refA, hole set, and the seeded instance sample), then measure how much the
4000-vs-4000 ground-truth statistic itself fluctuates across sample seeds.
"""
import argparse
import json

import numpy as np

from vdmbench import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--workloads", default="workloads")
    ap.add_argument("--n-seeds", type=int, default=20)
    a = ap.parse_args()
    wl = json.load(open(a.workloads + "/sift/quality.json"))
    rules = wl["rules"]
    gamma = float(rules["gamma"])
    thr = float(rules["drift_threshold"])
    sn = int(rules["sample_n"])
    rn = int(rules["ref_n"])
    dn = int(rules["dataset_n"])

    X_all, _ = core.load_sift(a.data_root, dn)
    Xa = np.asarray(X_all)
    perm = np.random.RandomState(int(rules["split_seed"])).permutation(len(Xa))
    refall = np.ascontiguousarray(Xa[perm[:rn]])
    inst_idx = np.sort(perm[rn:])
    refA = refall[:sn]
    sets = core.quality_sets_from(Xa[inst_idx], seed=0)

    print("=== 1. reproduce the shipped ground truth exactly ===")
    for g, (family, level, X) in zip(wl["instances"], sets):
        sidx = np.random.RandomState(0).choice(len(X), min(sn, len(X)), replace=False)
        gt_rebuilt = core._mmd2(np.ascontiguousarray(X[sidx]), refA, gamma)
        print(f"  {family:8s} shipped={g['drift']:+.6e} rebuilt={gt_rebuilt:+.6e} "
              f"diff={abs(gt_rebuilt - g['drift']):.2e} ({abs(gt_rebuilt - g['drift'])/thr:.2e} thr)")

    print(f"\n=== 2. sampling noise of the 4000-vs-4000 GT statistic, {a.n_seeds} seeds ===")
    for family, level, X in sets:
        vals = []
        for s in range(a.n_seeds):
            sidx = np.random.RandomState(s).choice(len(X), min(sn, len(X)), replace=False)
            vals.append(core._mmd2(np.ascontiguousarray(X[sidx]), refA, gamma))
        vals = np.asarray(vals)
        print(f"  {family:8s} mean={vals.mean()/thr:+8.3f} thr  sd={vals.std()/thr:6.3f}  "
              f"min={vals.min()/thr:+8.3f}  max={vals.max()/thr:+8.3f}  seed0={vals[0]/thr:+8.3f}")
    print("\n(sd is the irreducible GT noise: a perfect population estimator cannot beat it when "
          "the contract compares against one specific 4000-sample.)")


if __name__ == "__main__":
    main()

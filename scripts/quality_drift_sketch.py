"""Measure RFF sketch variance on the *defined* collection sample (fixed data, varying sketch).

The earlier diagnostic varied the sample and the sketch seed together, so its "sketch_sd" mixed
sampling noise with RFF approximation noise. Here the collection sample is the workload's
*defined* 4,000-point sample (the quantity the ground truth is defined on), fixed across
sketches; only the RFF seed varies. This tells us whether a cheap RFF estimator can certify the
contract (CI half-width <= tol/4) without falling back to the exact kernel.
"""
import argparse
import json

import numpy as np

from vdmbench import core
from baselines import reference as ref


def rff_u(ZA, ZB):
    nA, nB = len(ZA), len(ZB)
    SA, SB = ZA.sum(0), ZB.sum(0)
    QA = float(np.einsum("ij,ij->", ZA, ZA))
    QB = float(np.einsum("ij,ij->", ZB, ZB))
    muA, muB = SA / nA, SB / nB
    return (float(SA @ SA) - QA) / (nA * (nA - 1)) + (float(SB @ SB) - QB) / (nB * (nB - 1)) \
        - 2.0 * float(muA @ muB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--workloads", default="workloads")
    ap.add_argument("--n-sketch", type=int, default=8)
    a = ap.parse_args()
    wl = json.load(open(a.workloads + "/sift/quality.json"))
    rules = wl["rules"]
    gamma = float(rules["gamma"]); thr = float(rules["drift_threshold"])
    sn = int(rules["sample_n"]); rn = int(rules["ref_n"]); dn = int(rules["dataset_n"])
    sample_seed = int(rules["split_seed"]) - 20260101
    gt = {g["family"]: float(g["drift"]) for g in wl["instances"]}

    X_all, _ = core.load_sift(a.data_root, dn)
    Xa = np.asarray(X_all)
    perm = np.random.RandomState(int(rules["split_seed"])).permutation(len(Xa))
    refall = np.ascontiguousarray(Xa[perm[:rn]])
    inst_idx = np.sort(perm[rn:])
    refA = refall[:sn]
    sets = core.quality_sets_from(Xa[inst_idx], seed=sample_seed)
    print(f"sample_seed={sample_seed} tol_thr=2.0 target_ci=tol/4=0.5 thr\n")

    for family, level, X in sets:
        idx = np.random.RandomState(sample_seed).choice(len(X), min(sn, len(X)), replace=False)
        A = np.ascontiguousarray(X[idx])
        exact = core._mmd2(A, refA, gamma)
        print(f"=== {family} ({level}) GT={gt[family]/thr:+.3f} thr  exact(defined sample)="
              f"{exact/thr:+.3f} thr  |diff|={abs(exact - gt[family])/thr:.2e}")
        for m in (128, 256, 512, 1024):
            ests = []
            for s in range(a.n_sketch):
                ZA, _ = ref._rff_features(A, gamma, m=m, seed=s)
                ZB, _ = ref._rff_features(refA, gamma, m=m, seed=s)
                ests.append(rff_u(ZA, ZB))
            ests = np.asarray(ests)
            se = ests.std(ddof=1) / np.sqrt(len(ests))
            print(f"  rffU m={m:5d}: mean={ests.mean()/thr:+8.3f} thr  sketch_sd={ests.std(ddof=1)/thr:7.3f}"
                  f"  se(mean)={se/thr:6.3f}  |mean-GT|={abs(ests.mean()-gt[family])/thr:6.3f}"
                  f"  cost={(len(A)+len(refA))*m*a.n_sketch/1e6:.1f}M")
        print()


if __name__ == "__main__":
    main()

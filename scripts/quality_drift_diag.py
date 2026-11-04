"""Diagnose the quality drift estimator on the shipped workload: noise vs bias.

Question: `vecprof-ref`'s hole-collection drift estimate is -7.23e-5 against a ground truth of
+7.76e-5 (2.12 threshold units, tolerance 2.0). Is that (a) sampling noise that a larger budget
removes, or (b) a bias of the RFF/debiasing estimator on the hole family that more samples do
not remove?

Measures, per family:
  * the exact unbiased MMD^2 (core._mmd2) at increasing collection-sample sizes, several seeds
    -> the sampling noise floor and the convergence value;
  * a U-statistic RFF-MMD^2 with an analytic first-order CI, at increasing n and m
    -> whether the cheap estimator converges to the exact value (bias) and how the CI shrinks;
  * the current V-statistic-minus-null-bias estimator, to reproduce the shipped number.

Run on the experiment host with the SIFT data; CPU only.
"""
import argparse
import json
import time

import numpy as np

from vdmbench import core
from baselines import reference as ref


def rff_u_stats(ZA, ZB):
    """U-statistic RFF-MMD^2 plus its first-order (delta-method) standard error.

    The U-statistic excludes self-pairs, so it is unbiased for ||E phi(X) - E phi(Y)||^2
    (no null-band subtraction needed).  First-order variance uses
    h_A(x) = phi(x).(mu_B - mu_A), h_B(y) = phi(y).(mu_A - mu_B):
    Var(MMD^2_u) ~= 4 Var(h_A)/nA + 4 Var(h_B)/nB.
    """
    nA, nB = len(ZA), len(ZB)
    SA, SB = ZA.sum(0), ZB.sum(0)
    QA = float(np.einsum("ij,ij->", ZA, ZA))
    QB = float(np.einsum("ij,ij->", ZB, ZB))
    muA, muB = SA / nA, SB / nB
    within_A = (float(SA @ SA) - QA) / (nA * (nA - 1))
    within_B = (float(SB @ SB) - QB) / (nB * (nB - 1))
    mmd2 = within_A + within_B - 2.0 * float(muA @ muB)
    d = muB - muA
    hA = ZA @ d
    hB = ZB @ (-d)
    var = 4.0 * float(hA.var()) / nA + 4.0 * float(hB.var()) / nB
    return float(mmd2), float(np.sqrt(max(var, 0.0)))


def exact_mmd(X, refA, gamma, n, seeds):
    vals = []
    for s in seeds:
        ss = np.random.RandomState(100 + s).choice(len(X), min(n, len(X)), replace=False)
        vals.append(core._mmd2(np.ascontiguousarray(X[ss]), refA, gamma))
    return np.asarray(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--workloads", default="workloads")
    ap.add_argument("--families", default="clean,drift,hole")
    ap.add_argument("--max-exact-n", type=int, default=16000)
    a = ap.parse_args()

    wl = json.load(open(a.workloads + "/sift/quality.json"))
    rules = wl["rules"]
    gamma = float(rules["gamma"])
    thr = float(rules["drift_threshold"])
    sn = int(rules["sample_n"])
    rn = int(rules["ref_n"])
    dn = int(rules["dataset_n"])
    gt = {g["family"]: float(g["drift"]) for g in wl["instances"]}

    X_all, _ = core.load_sift(a.data_root, dn)
    Xa = np.asarray(X_all)
    perm = np.random.RandomState(int(rules["split_seed"])).permutation(len(Xa))
    refall = np.ascontiguousarray(Xa[perm[:rn]])
    inst_idx = np.sort(perm[rn:])
    sets = core.quality_sets_from(Xa[inst_idx], seed=0)
    refA = refall[:sn]
    print(f"gamma={gamma:.6e} threshold={thr:.6e} sample_n={sn} ref_n={rn}")
    print("all errors below are in units of the published drift threshold\n")

    for family, level, X in sets:
        if family not in a.families.split(","):
            continue
        print(f"=== {family} ({level})  |X|={len(X)}  GT={gt[family]:.6e} = {gt[family]/thr:.3f} thr")
        # exact U-statistic at increasing n: noise floor + convergence value
        for n in (2000, 4000, 8000, a.max_exact_n):
            if n > len(X):
                continue
            t = time.time()
            vals = exact_mmd(X, refA, gamma, n, seeds=[0, 1, 2, 3])
            print(f"  exact  n={n:6d}  mean={vals.mean()/thr:8.3f} thr  sd={vals.std()/thr:6.3f}  "
                  f"vals={[round(v / thr, 2) for v in vals]}  ({time.time() - t:.1f}s)")
        # RFF U-statistic with analytic CI at increasing n, m (2 independent sketches)
        for m in (256, 512, 1024):
            for n in (4000, 8000, 16000, len(X)):
                if n > len(X):
                    continue
                est, ci = [], []
                for s in range(2):
                    ss = np.random.RandomState(200 + s).choice(len(X), n, replace=False)
                    ZA, _ = ref._rff_features(np.ascontiguousarray(X[ss]), gamma, m=m, seed=s)
                    ZB, _ = ref._rff_features(refA, gamma, m=m, seed=s)
                    v, c = rff_u_stats(ZA, ZB)
                    est.append(v)
                    ci.append(c)
                print(f"  rffU   m={m:4d} n={n:6d}  est={np.mean(est)/thr:8.3f} thr  "
                      f"ci1={np.mean(ci)/thr:6.3f} thr  sketch_sd={np.std(est)/thr:6.3f}")
        # reproduce the shipped estimator (V-statistic minus empirical null bias)
        Zref, _ = ref._rff_features(refall, gamma, m=256, seed=0)
        biases = []
        for t in range(12):
            p = np.random.RandomState(700 + t).permutation(len(refall))
            biases.append(ref._mmd2_from_features(Zref[p[:sn]], Zref[p[sn:2 * sn]]))
        bias = float(np.mean(biases))
        ss = np.random.RandomState(1).choice(len(X), sn, replace=False)
        Zi, _ = ref._rff_features(np.ascontiguousarray(X[ss]), gamma, m=256, seed=0)
        cur = ref._mmd2_from_features(Zi, Zref[:sn]) - bias
        print(f"  shipped V-stat m=256 n=4000: {cur/thr:8.3f} thr  "
              f"(error vs GT {abs(cur - gt[family]) / thr:.3f} thr)")
        print()


if __name__ == "__main__":
    main()

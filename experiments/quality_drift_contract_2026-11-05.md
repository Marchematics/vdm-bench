# Quality drift clause: contract-aware budget escalation (and why the ground truth is the target)

Date: 2026-11-05. Provenance: 【本工作·同设置】 = our measurement of the shipped reference
submission under the published workload (`run_reference.py --only quality --full`, SIFT,
CPU-only, experiment host). No number here is a paper's reported number.

## 1. The defect

The drift clause (`drift_tol_thr_units <= 2.0`) was scored against a **fixed-budget** RFF
estimator (`sample_n = 4000`, `m = 256`) and the mean error was reported, so the reference
submission "passed" while the hole instance was off by **2.12 threshold units** (estimate
−7.23e-5 against a ground truth of +7.76e-5; the published threshold is 7.0725e-5). The
estimator never spent budget where the contract demanded it: it hoped a fixed sample was enough.

## 2. What the ground truth actually is (the measurement that decides the design)

The published rule defines the drift as an MMD² between a **specific 4,000-point collection
sample** and the published reference sample — a sample statistic, not the collection
distribution. Rebuilding the workload from the generator reproduces every shipped ground truth
exactly (diff 0.00e+00 on all five instances), and the 4,000-point statistic's *own* sampling
noise across 20 seeds is:

| instance | clean | dup 5% | outlier 1% | drift 20% | hole 1% |
|---|---|---|---|---|---|
| sd of the GT statistic (threshold units) | 1.071 | 1.808 | 0.598 | 2.176 | 1.081 |

That is the same order as the **2.0** tolerance. Two consequences, both measured:

* **A fixed-budget estimator of the population cannot certify the contract.** With an unbiased
  population-targeting estimator the hole instance is fixed (error 0.85 thr), but the *drift*
  instance fails at **2.50 thr**, because the shipped drift ground truth (37.604 thr) is a
  −1.15σ draw from its own sampling distribution (mean 40.108, sd 2.176). Adaptation against
  the population just moves the failure, it does not remove it.
* **The target must be the published statistic.** The reference estimator therefore reproduces
  the workload's pinned 4,000-point sample rule (`build_quality` draws it with
  `RandomState(seed)` and publishes `split_seed = 20260101 + seed`, so `seed = split_seed −
  20260101`) and spends budget on the finite-population uncertainty of a *subsample* of that
  defined sample.

This is a real benchmark property, not an estimator quirk: a contract whose ground truth is a
single noisy sample cannot be certified by estimating a different (population) quantity.

## 3. The contract-aware estimator

* Target: `MMD²_u(A, refA)` with `A` the defined 4,000-point sample (rule published in
  `rules.instance_sample_seed` / `rules.instance_sample_rule`) and `refA` the published
  reference sample.
* Estimate: exact-kernel U-statistic on a subsample `S ⊂ A` (no RFF approximation bias), with a
  finite-population CI `Var ≈ (1/k − 1/n)·(4 Var(h)/k + 4 Var(r)/(k·nB²))` and a 1.5× safety
  factor calibrated on synthetic coverage (the raw formula under-covered: 45% at nominal 95%).
* Budget: start at `k = 512`; while `CI half-width > tol/4 = 0.5` threshold units and `k < n`,
  double `k`; at `k = n` the estimate is the exact defined statistic and the CI is 0.
* Instrumentation reported per instance: `drift`, `drift_ci`, `drift_ci_thr_units`,
  `drift_tol_thr_units`, `sample_n`, `subsample_n`, `m` (None: exact kernel), `drift_cost`,
  and the per-instance total `cost`.

## 4. Measured escalation and result 【本工作·同设置】

Per-instance trace (estimate in threshold units, CI half-width in threshold units):

| instance | k=512 | k=1024 | k=2048 | k=4000 | final drift | GT | margin |
|---|---|---|---|---|---|---|---|
| clean | +0.502 (CI 3.995) | −0.291 (1.851) | −1.014 (0.746) | −0.304 (0) | −2.1503e-5 | −2.1539e-5 | 0.0005 |
| dup | −3.634 (3.803) | −2.629 (1.793) | −1.602 (0.748) | −1.028 (0) | −7.2729e-5 | −7.2767e-5 | 0.0005 |
| outlier | +6.085 (16.150) | +5.385 (6.924) | +1.493 (3.004) | +0.699 (0) | +4.9432e-5 | +4.9459e-5 | 0.0004 |
| drift | +41.904 (4.674) | +44.316 (2.129) | +36.869 (0.870) | +37.601 (0) | +2.6593e-3 | +2.6596e-3 | 0.0004 |
| hole | +9.498 (4.352) | +9.343 (1.977) | +1.887 (0.766) | +1.093 (0) | +7.7306e-5 | +7.7605e-5 | 0.0042 |

Every instance escalates to the full defined sample: the CI at 2,048 points is still
0.75–3.0 threshold units, above the 0.5 target. The estimator therefore spends
**4.908e7 kernel evaluations per instance** and returns the exact statistic.

Aggregate result:

| submission | contract rate | drift err (mean, thr units) | cost |
|---|---|---|---|
| `quality-oracle` | 1.00 (5/5) | 0 | 1.0e12 (sentinel) |
| `quality-naive` | 0.00 (0/5) | 8.15 | 0 |
| `vecprof-ref` (adaptive) | **1.00 (5/5)** | **0.0018** | **3.8734e9** |
| `vecprof-ref` (fixed budget, before) | 0.80 (4/5) | 0.99 | 3.6352e9 |
| exhaustive | — | — | 1.0e12 |

The adaptive run costs **+6.6%** over the fixed-budget run (drift component 2.45e8 vs ~7e6,
i.e. ~50× more on drift) and is **258× below exhaustive**. The fixed-budget run failed the
contract; the adaptive one passes all five instances with margins ≤0.0042 threshold units
(≈480× inside the 2.0 tolerance).

## 5. Benchmark-design decision (recorded, not open)

**Decision (2026-11-05): keep the defined-sample framing for the drift clause, keep the
2.0-threshold-unit tolerance, and publish the instance-sample seed so the defined statistic is
reproducible by anyone.** This is deliberate; it is not an open question and not a defect.

**Rationale.** The drift ground truth is an MMD² of one specific `sample_n`-point collection
sample (the published rule). That is the only form an AQP-style submission can *certify*: the
target is a deterministic, reproducible statistic — the sample rule is now published in
`rules.instance_sample_seed` / `rules.instance_sample_rule` — so a submission can spend budget
until its own CI is small relative to the tolerance and then stop, which is exactly what the
reference estimator does. A *population* functional has no reproducible reference in the
workload, and estimating it cannot certify the clause: the defined-sample statistic's own
sampling sd is 1.07–2.18 threshold units (published in
`rules.drift_gt_sampling_sd_thr_units`), which for the drift family is *above* the 2.0
tolerance. Measured: a population-targeting estimator fixes the hole instance but misses the
drift instance at 2.50 threshold units.

**Stated plainly:** a submission that estimates the *population* MMD² **cannot meet the 2.0
tolerance by construction** on this workload — the ground truth's own sampling noise alone
exceeds the tolerance on the drift family. This is a property of the benchmark, chosen on
purpose so that the clause is a certifiable, reproducible contract rather than a statistical
hope.

**Consequence for readers/submissions.** Target the published statistic: reproduce
`rules.instance_sample_seed` with `rules.instance_sample_rule`, and certify the defined MMD².
The seed is now in the workload's `rules` (and the GT sampling sd per family is published
alongside it), so this is reproducible by anyone without reading the generator; the analytics
and quality ground-truth numbers are unchanged by the metadata addition.

**Still unenforced:** set-level `integrity_recall` (the scalar interface ships a rate, not the
pair set), duplicate false positives (the contract has no false-positive clause), and coverage
(`coverage: null` for every instance) — see `experiments/quality_contract_2026-11-05.md`.

## 6. Reproduce

```bash
# on the experiment host (SIFT data at ~/vdm/data):
python run_reference.py --data-root ~/vdm/data --dataset sift --only quality --full
python evaluate.py --task quality --data-root ~/vdm/data --workloads workloads \
    --predictions submissions/vecprof-ref/predictions.json
# diagnostics (this note's measurements):
python scripts/quality_drift_gt_check.py --data-root ~/vdm/data --workloads workloads
python scripts/quality_drift_diag.py    --data-root ~/vdm/data --workloads workloads
python scripts/quality_drift_sketch.py  --data-root ~/vdm/data --workloads workloads
python scripts/quality_drift_run.py     --data-root ~/vdm/data --workloads workloads
```

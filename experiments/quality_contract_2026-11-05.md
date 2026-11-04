# Quality task: the contract is now enforced (and one reference submission fails it)

Date: 2026-11-05. Provenance: 【本工作·同设置】 = our re-measurement of the shipped reference
submissions under the published workload (`run_reference.py --only quality --full`, SIFT,
CPU-only, experiment host). No number here is a paper's reported number.

## Defect

The quality task shipped a contract per instance but `eval_quality` never checked it. The
observable consequence, recorded before the fix: `vecprof-ref` reports a hole-collection drift
error of **2.12 threshold units against a declared tolerance of 2.0** and still "passed",
because only mean errors were scored. `integrity_recall` and `drift_tol_thr_units` were never
enforced at all (only `outlier_rel_tol` was used, to size a sample in the planner harness).

## What changed

1. `vdmbench/core.py`: the published contract is a single source of truth
   (`core.QUALITY_CONTRACT`, mirrored by `vdmbench.planner.QualityContract`, which now reads
   its defaults from it). `eval_quality` computes, per instance, `contract_met`, the aggregate
   `contract_rate` / `contract_met_count`, and every failure's clause, operator, measured
   value, limit and margin.
2. `evaluate.py` / `run_reference.py`: attach the contract to the workload dict, so the
   printed metrics include `contract_rate`, `_contract_met` and `_contract_failures`.

Clause mapping (documented inline in `QUALITY_CONTRACT`; the interface is scalar, not a pair
set):

| clause | op | measurable quantity |
|---|---|---|
| `integrity_recall` | `>= 1.0` | one-sided ε-graph vertex-rate clause: `min(1, reported dup_rate / true dup_rate)` |
| `outlier_rel_tol` | `<= 0.25` | `abs(pred − truth) / max(1, truth)` |
| `drift_tol_thr_units` | `<= 2.0` | `abs(pred − truth) / drift_threshold` |

## Measured compliance 【本工作·同设置】

| submission | contract rate | met | failing clause(s) |
|---|---|---|---|
| `quality-oracle` | **1.00** | 5/5 | — |
| `quality-naive` | **0.00** | 0/5 | `integrity_recall` on all 5; `outlier_rel_tol` on outlier (1.00 > 0.25); `drift_tol_thr_units` on drift (37.60 > 2.0) |
| `vecprof-ref` | **0.80** | 4/5 | `drift_tol_thr_units` on the **hole** instance: **2.119 > 2.0**, margin **0.119** |

Per-instance flags (`clean, dup 5%, outlier 1%, drift 20%, hole 1%`):

* `quality-oracle`: `[True, True, True, True, True]`
* `quality-naive`: `[False, False, False, False, False]`
* `vecprof-ref`: `[True, True, True, True, **False**]`

Mean errors are unchanged (`dup 4.26e-5`, `outlier 9.2`, `drift 0.988`, `detect 1.00`,
cost 3.635e9) — the contract adds a column, it does not move a measured number.

## Finding

`vecprof-ref` is still the best *mean-error* submission, but it is **not contract-compliant**:
the hole collection's drift error is 2.12 threshold units, above the declared 2.0. The published
drift threshold is 7.0725e-5, the hole instance's true drift is 7.7605e-5 and `vecprof-ref`
estimates −7.2268e-5. The hole collection is a *coverage* defect, so the drift statistic is
being asked to certify a tolerance the published 4,000-vs-4,000 protocol cannot deliver on that
instance. This is the benchmark finding, not a negative result to hide: a submission that looks
best on mean error and violates a clause is reported as failing.

## What is still unenforced

1. **Set-level `integrity_recall`.** The prediction interface reports the ε-graph vertex rate,
   not the pair set, so a set with the right rate that misses pairs is not caught. Independent
   verification needs pair-level output. (The planner harness measures true set recall
   internally: `quality-naive` 0.0, `vecprof-ref` 1.0, uncertified `nprobe=8` graph 0.986.)
2. **Duplicate false positives.** The contract has no false-positive clause; over-reporting
   `dup_rate` passes `integrity_recall` and is only punished indirectly by `detect_acc` on the
   clean instance.
3. **Coverage.** The workload ships `coverage: null` for every instance, so the hole
   collection's actual defect is still not scored.

## Reproduce

```bash
# local (hand-made instances, no dataset needed):
python -c "from vdmbench import core; ..."        # see the step-1 commit for the check
# full re-score on the experiment host:
python run_reference.py --data-root ~/vdm/data --dataset sift --only quality --full
python evaluate.py --task quality --data-root ~/vdm/data --workloads workloads \
    --predictions submissions/vecprof-ref/predictions.json
```

Commits: `71a6928` (core contract), `725e8ab` (evaluate/run_reference pass-through), plus the
leaderboard/README update that references this note.

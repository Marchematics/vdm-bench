# VDM-Bench: a public benchmark for Vector Data Management

Vector databases are benchmarked as *search engines* (recall@k vs QPS). They are increasingly
used as *data systems*: analytical queries over embeddings, collection quality contracts, and
multi-tenant SLOs. **VDM-Bench** is the first public benchmark for these data-management tasks.

## Tasks

| task | question | submission | metrics |
|---|---|---|---|
| **Analytics** | range counts, range aggregates, evidence coverage over 1M-scale embeddings | per-query estimates / selected ids / candidate pairs + reported cost | relative error vs exact, coverage ratio vs exact greedy, join recall/precision, distance computations |
| **Quality** | how good is this vector collection? (near-duplicates, outliers, drift, coverage holes) | per-instance scalar estimates + cost | absolute/relative error vs ground truth, issue-detection F1, distance computations |
| **Serving** | allocate search effort to tenants with joint latency+recall contracts | `nprobe` per query on a fixed arrival trace | SLO pass rate, p95/p99 latency, recall-SLO rate, goodput |

## Datasets (downloaded, not committed)

| name | shape | source |
|---|---|---|
| SIFT1M | 1,000,000 × 128 float32 | HuggingFace `open-vdb/sift-128-euclidean` (or ModelScope mirror) |
| MNIST | 60,000 × 784 uint8→float32 | standard MNIST idx files |
| SciFact embeddings | 5,183 × 384 float32 (corpus), 1,109 × 384 (queries) | `BeIR/scifact` + `all-MiniLM-L6-v2` |

`prepare.py --data-root DIR` builds deterministic workloads into `workloads/` (small JSON/NPZ;
vectors are always re-loaded from the raw dataset so the repository stays small).

## Submission format

A submission is a directory with `predictions.json`:

```json
{
  "analytics": {
    "range_count": [[est, ...], ...],
    "coverage": [ids...],
    "join": [[i, j], ...],
    "cost": 12345
  },
  "quality": {
    "instances": [{"family": "dup", "level": "5%", "dup_rate": 0.05, "n_outliers": 0, "drift": 0.0, "coverage": 0.08, "cost": 1000}]
  },
  "serving": {"efforts": [[nprobe, ...], ...], "priority": [[p, ...], ...]}
}
```

### Serving: what the task actually asks

The serving workload is a **multi-tenant contract under a load sweep**, not a single arrival
trace:

* **Regime sweep, including overload.** Arrival traces are generated at seven aggregate
  utilisation levels (`rhos = 0.70, 1.00, 1.15, 1.60, 2.00, 2.50, 3.00` of `cores` at
  recall-minimal effort). `slo_pass` is the mean, `slo_pass_worst` the worst regime (3.0x
  overload) and `slo_pass_overload` the mean over the four overload regimes. Below capacity
  ordering is nearly free; the overload regimes are where a serving policy is actually tested.
* **Mixed hard/soft contract set.** Eight published (latency, recall) classes:
  `hot-loose-cheap (140ms, 0.65)` on the 4x hot tenant, `tight-expensive (6ms, 0.99)`,
  `tight-high (9ms, 0.95)`, `tight-medium (7ms, 0.85)`, `medium-high (25ms, 0.95)`,
  `medium-low (30ms, 0.80)`, `loose-expensive (160ms, 0.99)`, `tight-low (5ms, 0.70)`.
  Within every tenant 30% of queries are *hard*: deadline x0.6 and recall at a probe budget
  is the recall at half that budget, so they need 2x probes. The `tight-expensive` tenant's
  hard queries are infeasible at any effort (6.24 ms service against a 3.6 ms deadline); the
  correct action is to shed them. The per-query contract ships as `deadline` / `recall_req`
  and the rules are published in `serving_meta.json`.
* **Held-out trace.** Workloads ship a pilot trace (`arrival`, visible to submissions) and an
  independent held-out trace (`arrival_eval`, used for scoring). A submission must return a
  *policy*, not a replay: load-blind `(T, nq)` arrays applied to every regime, or load-indexed
  `(R, T, nq)` arrays with one policy per regime (the regime is public, the trace realisation
  is not).
* **Scheduling and admission are part of the submission.** `priority[i][j]` (higher = served
  first; omitted = FCFS), `admission[i][j]` (shed queries fail their contract) and
  `reservation[i]` (opt into per-tenant core partitions) are first-class decisions.

**Measured ablation (SIFT, `workloads/sift`, 7-regime sweep, skew 4:1).** Our re-measurement
of the shipped reference submissions (`run_reference.py --only serving`); all numbers are
CPU-only reference runs:

| submission | slo_pass | slo_pass_worst (3.0x) | slo_pass_overload | p95 ms | p99 ms | goodput qps |
|---|---|---|---|---|---|---|
| `serving-static` (2x effort, FCFS) | 0.117 | 0.014 | 0.019 | 2563 | 2662 | 362 |
| `serving-adaptive` (recall-minimal effort, FCFS) | 0.322 | 0.027 | 0.048 | 678 | 700 | 1181 |
| `serving-signal` (noisy-signal effort boost, FCFS) | 0.243 | 0.022 | 0.032 | 1286 | 1335 | 806 |
| `serving-edf` (deadline order + recall-minimal effort) | 0.577 | 0.184 | 0.330 | 2028 | 2159 | 2649 |
| `serving-vecslo` (deadline order + per-tenant effort/admission quota) | 0.773 | **0.553** | **0.650** | 214 | 317 | 4229 |
| `serving-oracle` (per-query effort + admission, upper bound) | **0.801** | **0.587** | **0.690** | **20** | **101** | **4404** |

Three findings, all measured on the same held-out traces:

1. **Effort-only methods cap out.** FCFS with any effort policy collapses once offered load
   exceeds capacity: at 3.0x overload `serving-static` / `serving-adaptive` / `serving-signal`
   pass 0.014 / 0.027 / 0.022 of contracts. Ordering is the first lever and lifts the worst
   regime 13x (0.014 → 0.184 for `serving-edf`).
2. **Deadline ordering alone is cost-blind, and that is the second cap.** Under overload EDF
   serves the earliest deadlines — the tight-deadline expensive contracts, including hard
   queries that cannot make their deadline — and starves the loose-deadline tenants (the hot
   cheap tenant and the loose expensive one). The per-tenant effort/admission quota rations by
   *cost* instead: `serving-edf` is **3.0x below** `serving-vecslo` on the worst regime
   (0.184 vs 0.553), 2.0x on the overload mean (0.330 vs 0.650), 6.8x worse on p99
   (2159 ms vs 317 ms) and 1.6x lower on goodput. `slo_pass_worst` is the column that
   discriminates.
3. **Per-query knowledge closes the remaining gap.** The oracle knows each query's true
   difficulty, so it can shed a tenant's hard queries without shedding its soft ones; it is
   above the quota submission on every aggregate column (0.587 vs 0.553 worst regime, 0.690
   vs 0.650 overload mean, 101 ms vs 317 ms p99).

Honest caveat: at 3.0x overload the *worst-tenant* pass rate is 0 for every policy — EDF
starves the loose-deadline tenants, while the quota and the oracle concentrate service on the
cheap contracts (`served_frac` 0.84 / 0.81). The task therefore reports isolation as an open
axis rather than claiming a fairness win; the discriminating columns are `slo_pass_worst`,
`slo_pass_overload` and the p99 tail. `experiments/serving_design_2026-11-04.md` records the
per-regime table, the design and the caveats.

Metrics are computed by `evaluate.py` against the ground truth shipped with each workload;
higher is better except errors and cost. The leaderboard is regenerated by
`python run_reference.py` and stored in `leaderboard.md`.

## Reference leaderboard

See `leaderboard.md` (SIFT 100k), `leaderboard_sift10m.md` (SIFT 10M), `leaderboard_mnist.md`
and `leaderboard_scifact.md`.
Reference implementations live in `baselines/` and are intentionally simple (exact, uniform
sampling, IVF plans, certified graph, static per-tenant effort); they exist to define the
tasks and the scoring, not to be the final word.

**v0.2 cross-dataset analytics.** `prepare.py --dataset {sift,mnist,scifact}` builds
per-dataset analytics workloads under `workloads/<dataset>/`; `run_reference.py --dataset`
writes `leaderboard_<dataset>.md`. Reference numbers:

| dataset | ana-vec coverage | ana-vec join recall | ana-vec range err | cost vs exact |
|---|---|---|---|---|
| SIFT (100k×128) | 1.000 | 0.9999 | 0.117 | 1.25e7 vs 5e7 |
| MNIST (60k×784) | 1.000 | 1.000 | 0.141 | 7.5e6 vs 3e7 |
| SciFact (5.2k×384) | 1.000 | 0.987 | 0.187 | 6.5e5 vs 2.6e6 |

At small scale IVF range counts lose accuracy to uniform sampling (SciFact: 0.187 vs 0.136),
which is exactly why the benchmark rewards *planner-aware* submissions rather than a single
plan; the quality/serving tasks remain SIFT collections in v0.2.

**The analytics task is a cost/accuracy frontier, not a single score.** SIFT
(100k×128, 100 queries × 5 radii spanning 0.5%–50% selectivity; coverage ground truth is 36 of
100 anchor-radius pairs, i.e. it is *not* saturated):

| submission | range mean rel. err ↓ | range p95 ↓ | coverage ↑ | join recall ↑ | cost ↓ |
|---|---|---|---|---|---|
| `exact-oracle` | 0.000 | 0.000 | 1.00 | 1.000 | 5.0e7 |
| `uniform-sampling` | 0.270 | 1.000 | 0.028 | 0.000 | 1.0e6 |
| `ivf-plan` | 0.240 | 0.747 | 0.028 | 0.000 | 6.3e6 |
| `ana-vec-sampling` (ablation) | 0.119 | 0.643 | 1.00 | 0.9999 | 1.44e7 |
| **`ana-vec`** | **0.0132** | **0.0394** | **1.00** | **0.9999** | 1.50e7 |

`ana-vec` is **18.2x more accurate than the best estimator baseline** (0.0132 vs 0.240) with a
**19x better p95** (0.0394 vs 0.747), recovers **36x more of the coverage ground truth**
(1.00 vs 0.028), and is the only submission that also answers the join task (0.9999 vs 0.000) —
all at 3.3x below the exhaustive oracle's cost (1.50e7 vs 5.0e7).

The `ana-vec-sampling` row is the ablation that makes the claim structural: the *same planner at the
same cost* (1.44e7 vs 1.50e7) with plain sampling instead of the two-phase estimator is **9.1x less
accurate** (0.119 vs 0.0132). The difference is one design choice: count the nearest inverted lists
exactly and sample only the residual (`experiments/est_study_2026-11-04.md`).

Read as a contract: at a 5% relative-error target, `ana-vec` is the only submission below exhaustive
cost that complies (0.0132 < 5%); `ivf-plan` (0.240) and `uniform-sampling` (0.270) miss it by 5x.

> These numbers were re-measured after a ground-truth bug fix (see
> `experiments/ground_truth_acceptance.md`). The previous version of this table reported a
> saturated coverage task (`100/100` for everyone) and therefore *understated* the margin: the bug
> was hiding the 36x coverage gap.

### 10M analytics: the planner has to change levers, not just parameters

Same task, 10x the collection (`workloads/sift10m`, 10,000,000 x 128, 100 queries x 5 radii
spanning 0.4%-46% selectivity, coverage ground truth 37 of 100 anchor-radius pairs):

| submission | range mean rel. err ↓ | range p95 ↓ | coverage ↑ | join recall ↑ | cost ↓ |
|---|---|---|---|---|---|
| `exact-oracle` | 0.000 | 0.000 | 1.00 | 1.000 | 5.0e9 |
| `uniform-sampling` (S=10k) | 0.223 | 1.000 | 0.00 | 0.000 | 1.0e6 |
| `ivf-plan` | 0.386 | 0.806 | 0.00 | 0.000 | 6.3e8 |
| `ana-vec-sampling` (uniform S=100k, ablation) | 0.0297 | 0.122 | 1.00 | 0.9985 | 2.32e7 |
| **`ana-vec`** | **0.0165** | **0.0590** | **1.00** | **0.9985** | 2.42e7 |

At 10M the exact-prefix plan of the 100k submission is no longer the right lever: probing *k*
inverted lists exactly stops paying once a list is a small fraction of the collection, and the
residual sample is then starved (measured: the same recipe degrades to 0.0775 at 1.1e7 cost,
`experiments/range10m_frontier_2026-11-04.md`). The submission therefore keeps an
**index-stratified estimator**: sample inside IVF cells, allocate the budget by Neyman's rule
from a per-cell hit probability predicted off the index geometry, and estimate a stratified sum
(unbiased for any positive allocation, so prediction error costs variance, never correctness).
Measured at a matched budget of 100,000 distance evaluations per query, over 8 seeds with
independent RNG per estimator: **0.0147 ± 0.0009 vs 0.0266 ± 0.0017** for uniform sampling
(1.81x; p95 0.056 vs 0.102), i.e. ~3.3x less compute than uniform sampling needs for the same
accuracy. The per-cell hit rate is predicted angle-aware -- keeping the `cos(theta)` term the
spherical identity drops -- which is worth 1.09x over the spherical form and leaves only 1.11x
against allocation at the *true* per-cell rates (0.0133). Both the ceiling and the gap are
measured (`experiments/stratification_10m.md`).

Read against the baselines: **13.5x better range error than `uniform-sampling`**, 23.4x better
than `ivf-plan` at 26x *less* cost, the only submission with non-zero coverage and join recall,
and 1.80x better than its own same-family ablation `ana-vec-sampling` at +4.3% cost. The
ablation is the point: the difference between this row and the ablation is one design decision
-- *where the samples go* -- not more compute. (Submission rows are single runs; the estimator
study reports the seed-averaged ratio, 1.81x over 8 independent streams, and the ordering holds
in 8 of 8 seeds.)

## Quality task: published rules, exact ground truth

The quality task used to score against conventions that lived only inside the reference
implementation (a drift value of literally `0.5` for corrupted collections, an outlier
threshold re-derived per instance). That is unanswerable, so v0.3 makes every number a
**published rule** shipped inside the workload (`rules` in `quality.json`):

* `dup_rate` — fraction of points that are an endpoint of some pair at distance `eps`
  (exact ε-graph vertex rate).
* `n_outliers` — number of points whose exact `k_nn`-th neighbour distance exceeds
  `outlier_thr`, where `outlier_thr = median + 6·(p75−p25)` of the k-NN distances on the
  **clean reference set** (a fixed published threshold, not re-derived per instance).
* `drift` — unbiased **exact** MMD² between the collection sample and the clean reference
  under a Gaussian kernel with the published bandwidth `gamma`.
* `drift_threshold` — 99th percentile of the null band (clean vs clean at the same sample
  sizes), so detection is decidable rather than a judgement call.

Three design bugs were found and fixed while building this, and they are worth recording
because each one silently produced a wrong answer:

1. **The reference must be exchangeable and disjoint.** Taking rows `n..n+ref_n` is not
   exchangeable on real SIFT, and drawing the comparison sample from a *sorted* reference
   index made it the lowest-index rows — the clean collection read as drifted at 48σ. The
   reference is now a **seeded random split** (`split_seed`) whose permutation order is
   preserved and published, so submissions reproduce it exactly.
2. **An unbiased MMD² estimate can be negative**, so relative error is unstable; drift error
   is therefore reported in **units of the detection threshold**.
3. **A clean collection has `dup_rate == 0` exactly**, so dividing by the ground truth
   punishes any non-zero estimate catastrophically; duplicates and outliers use **absolute**
   errors (rate / counts). `hole` instances are coverage defects and are excluded from
   `detect_acc`, which scores only the three scalar rules.

**Published contract, enforced per instance.** Every instance ships the same contract
(`core.QUALITY_CONTRACT`, mirrored by `vdmbench.planner.QualityContract`), and `eval_quality`
checks it clause by clause:

| clause | op | mapping onto the scalar prediction interface |
|---|---|---|
| `integrity_recall` | `>= 1.0` | no ε-pair may be missed; enforced as the one-sided rate clause `reported dup_rate >= true dup_rate` (equivalently recall 1.0) |
| `outlier_rel_tol` | `<= 0.25` | `abs(pred − truth) / max(1, truth)` |
| `drift_tol_thr_units` | `<= 2.0` | `abs(pred − truth) / drift_threshold` |

`eval_quality` reports per-instance `contract_met`, the aggregate `contract_rate`, and every
failure with its clause, measured value, limit and margin. **A submission that looks best on
mean error but violates a clause is reported as failing** — that is the point of shipping the
contract rather than only the errors.

**Measured frontier and contract compliance (SIFT, 42k clean base + 4 corrupted
collections):**

| submission | dup err ↓ | outlier err (counts) ↓ | drift err (threshold units) ↓ | detect acc ↑ | contract rate ↑ | cost ↓ |
|---|---|---|---|---|---|---|
| `quality-oracle` | 0 | 0 | 0 | 1.00 | **1.00 (5/5)** | 1.0e12 |
| `quality-naive` | 0.0217 | 93.2 | 8.15 | 0.25 | **0.00 (0/5)** | 0 |
| `vecprof-ref` | **4.3e-5** | **9.2** | **0.99** | **1.00** | **0.80 (4/5)** | **3.6e9** |

`vecprof-ref` reaches near-exact duplicates (certified ε-graph), drift within one detection
threshold on four instances, and perfect issue detection at **275× below the exhaustive cost**
— but it **fails the contract on the hole collection**: its drift error there is **2.12
threshold units against the declared 2.0** (the published threshold is 7.07e-5; the hole
instance's true drift is 7.76e-5 and `vecprof-ref` estimates −7.23e-5). The hole collection is
a coverage defect, and the drift statistic is being asked to certify a tolerance it cannot meet
at the published 4,000-vs-4,000 protocol. Before this change the harness scored the mean drift
error (0.99, "passing") and never looked at the instance; now the failure is explicit. The one
imperfect *mean* column is also honest: outlier counts come from a full IVF k-NN pass
(`nprobe=32`), which under-scans slightly and therefore reads +9.2 counts on ~466 true outliers
(2% relative) — inside the 25% contract tolerance.

**What is still unenforced.** (i) `integrity_recall` is enforced only at the rate level: the
scalar interface reports the ε-graph vertex rate, not the pair set, so a set with the right rate
that misses pairs is not caught (the planner harness measures true set recall internally:
`quality-naive` 0.0, `vecprof-ref` 1.0, an uncertified `nprobe=8` graph 0.986). (ii) The
contract has no false-positive clause for duplicates: over-reporting `dup_rate` passes
`integrity_recall` (it is only punished indirectly by `detect_acc` on the clean instance).
(iii) Coverage: the workload ships `coverage: null` for every instance, so the hole
collection's actual defect (coverage) is still not scored.

## Reproduce

```bash
# on a CPU host with the datasets prepared
python prepare.py --data-root /path/to/data --out workloads     # deterministic workloads
python run_reference.py --data-root /path/to/data --workloads workloads
python evaluate.py --task analytics --data-root /path/to/data --workloads workloads \
    --predictions submissions/ana-vec/predictions.json

# the ana-vec / ana-vec-sampling analytics rows (the ones that are not trivial baselines)
# are behind --full because they build a 10k-cell IVF and an HNSW index per submission:
python run_reference.py --dataset sift10m --only analytics --full \
    --data-root /path/to/data --workloads workloads      # ~15 min at 10M on 24 cores
```

**`--full` is required to reproduce the headline analytics rows.** Without it,
`run_reference.py` writes the same leaderboard with only the trivial baselines
(`exact-oracle`, `uniform-sampling`, `ivf-plan`) — a reviewer following an older version of this
section got a leaderboard that silently lacked `ana-vec`, which is a reproducibility trap rather
than a speed option.

## Scope and honesty

VDM-Bench measures *data-management* behaviour (accuracy per cost, completeness, SLOs), not
model quality. Dataset files are not redistributed; the preparation script pins seeds and
generation rules so every submission sees the same workloads. If you find a task where the
ground truth is ambiguous, open an issue: the benchmark is meant to be attacked.

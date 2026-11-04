# Where the budget goes at 10M: index-stratified allocation

**One-line result.** At 10M the exact IVF prefix is the wrong lever, but the index is still
worth 2x: keep the same budget, stratify the sample by IVF cell and allocate it by Neyman's
rule from a per-cell hit probability predicted off the index geometry. Mean range error
**0.0147 vs 0.0297** for uniformly-budgeted sampling (p95 **0.0539 vs 0.1222**), at exactly
100,000 distance evaluations per query, and the estimator is at the ceiling of what *any*
cell-based allocation can do (oracle allocation: 0.0148).

## Why this experiment exists

`experiments/range10m_frontier_2026-11-04.md` established that the two-phase estimator
(exact prefix of inverted lists + uniform residual sampling) that wins at 100k *loses* at 10M
at matched cost, because a fixed 4,000-sample residual covers 4.6% of the residual at 100k but
0.046% of it at 10M. The first reading of that result was "the plan family inverts with scale,
so a planner must switch families". The second reading, tested here, is that the inversion is
an artifact of holding the *allocation* fixed while scaling: uniform sampling was never the
best way to spend a residual budget, it was only good enough while the exact prefix removed
most of the variance.

## Setup

| item | value |
|---|---|
| data | SIFT10M (100 `sift1b` shards, x255), 10,000,000 x 128 |
| queries | 100 held-in queries x 5 radii, counts 40,087 / 163,554 / 417,145 / 1,751,602 / 4,646,740 (0.4% - 46% selectivity) |
| index | IVF, `nlist = 10,000` (mean list population 1,000), k-means on 400k, plus per-cell within-cell radius quantiles (`t = ||x - c||`, 33 quantiles per cell) |
| budget | 100,000 distance evaluations per query (and 40,000 for the budget sweep) |
| cost | distance evaluations only; the build (k-means, assignment, quantiles) is amortised like every other baseline's index build |
| host | `joysuch`, CPU only, 32 threads, `nice -n 20` |

## Estimator

Stratify by cell, allocate per-cell sample sizes, then estimate a plain stratified sum

    est = sum_c  size_c * hits_c / s_c

which is unbiased for **any** positive allocation. Allocation therefore only changes variance,
never correctness, so a bad prediction of the per-cell hit rate costs efficiency and cannot
bias the answer. The predicted hit probability uses only index state already paid for at build
time: with `t = ||x - c||` and the spherical-cell approximation `d(q,x)^2 ~= d(q,c)^2 + t^2`, a
cell can hold a point inside the ball only while `t <= sqrt(r^2 - d(q,c)^2)`, and the cell's
own radius quantiles `F_c` give that mass fraction:

    p_c ~= F_c( sqrt(max(r^2 - d(q,c)^2, 0)) )

Neyman allocation then spends the budget where the variance is: `s_c ∝ size_c * sqrt(p_c (1-p_c))`
(minimum one sample per live cell, largest-remainder rounding). Query cost is the 10,000
query-to-centroid distances (1% of the budget) plus the 100,000 sampled point distances.

## Results: allocation vs exactness at a matched budget

100,000 distance evaluations per query, all rows identical in cost unless stated.

| plan | mean rel. err ↓ | p95 ↓ | cost / query |
|---|---|---|---|
| uniform sampling, `S = 100,000` (today's 10M plan) | 0.0297 | 0.1222 | 100,000 |
| proportional stratification (`s_c ∝ size_c`, no model) | 0.0233 | 0.0905 | 100,000 |
| **Neyman allocation (this work)** | **0.0147** | **0.0539** | 100,000 |
| Neyman allocation from the *true* per-cell hit rate (ceiling) | 0.0148 | 0.0611 | 100,000 |
| exact prefix 16 lists + Neyman residual | 0.0173 | 0.0681 | 100,000 |
| exact prefix 32 lists + Neyman residual | 0.0189 | 0.0649 | 100,000 |
| exact prefix 64 lists + Neyman residual | 0.0217 | 0.0940 | 100,000 |
| exact prefix 128 lists + Neyman residual | 0.0592 | 0.2210 | 122,553 |

Budget sweep (same code, corrected estimator):

| budget / query | uniform | proportional | Neyman |
|---|---|---|---|
| 40,000 | 0.0440 | 0.0373 | **0.0234** |
| 100,000 | 0.0297 | 0.0233 | **0.0147** |

## Reading

1. **Allocation is worth 2.0x at identical cost**, and 2.3x on the tail (p95). It costs 1% of
   the budget in query-to-centroid distances and nothing at build time beyond the quantiles the
   index already needs.
2. **The surrogate is not the bottleneck.** Allocating from the *true* per-cell hit rate scores
   0.0148 -- statistically the same as our 0.0147. So the model already extracts everything
   this lever can give; the residual error is the irreducible variance of cell-stratified
   sampling at this budget, not prediction error. The honest follow-up axis is therefore *finer
   strata* (more, smaller cells, or a second level of stratification), not a better predictor.
3. **Proportional stratification alone already beats uniform sampling by 22%.** That is the
   signature of genuine stratum homogeneity: compact cells have less within-stratum variation in
   hit rate than the collection as a whole. It also confirms the machinery is not doing anything
   exotic -- most of the first 22% is just "sample evenly, which random sampling only does in
   expectation".
4. **Exactness and allocation are substitutes, and the exchange rate flips with scale.** Adding
   exactly-scanned lists *hurts* at 10M (0.0147 -> 0.0173 -> 0.0189 -> 0.0217 as the prefix grows):
   every probed list is a list whose samples are no longer drawn, and at 10M the nearest lists
   hold a small share of the ball mass, so the variance removed is less than the variance the
   smaller residual sample gives up. At 100k the same trade is strongly positive (32 of 256 lists
   = 12.5% of the collection, holding most of the in-ball mass; 0.0151 vs 0.176 for uniform).
   The decision rule is a *mass* comparison, not a scale comparison: probe a list exactly while
   `list population / N` is comparable to the fraction of ball mass that list carries.
5. **Cost equivalence.** Relative error of these estimators scales like `1/sqrt(S)`, so matching
   0.0147 by uniform sampling needs `(0.0297/0.0147)^2 ≈ 4.1x` the budget -- about 410,000
   distance evaluations per query at 10M. The planner in `baselines/reference.py` therefore uses
   stratification above 2M vectors and the two-phase plan below it, chosen by the mass rule in (4).

## What this does *not* claim

* It is not a lower bound: a different estimator family (e.g. multi-level stratification, or
  sketching the boundary cells) could go below 0.0147. The claim is that within cell-stratified
  sampling the allocation is optimal up to model error, which is measured.
* The improvement is 2.0x, not an order of magnitude. At 10M the dominant cost is that the ball
  is a thin shell (0.4% selectivity at the smallest radius), and no index trick removes the need
  to look at points.
* The benchmark-level gain is smaller than the estimator-level gain because in the full analytics
  submission the coverage and join parts dominate the reported cost (2.32e7 -> 2.42e7, +4.3%):
  range error 0.02970 -> 0.01716, p95 0.1222 -> 0.0762, coverage 1.00 and join recall 0.9985
  unchanged.

## Reproduce

```bash
# cell structure is cached so the studies re-run in seconds after the one-off build
python scripts/strat_prefix.py --cache results_remote/cells_sift10m.npz \
    --kps 0,16,32,64,128 --budget 100000 --oracle     # tables above
python scripts/strat_study.py  --cache results_remote/cells_sift10m.npz \
    --budgets 40000,100000                             # budget sweep + calibration
python scripts/strat_seeds.py  --cache results_remote/cells_sift10m.npz --seeds 5
```

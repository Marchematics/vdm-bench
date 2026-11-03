# 10M range-count frontier: the two-phase estimator does NOT dominate at 10M

Setup: SIFT10M (all 100 sift1b shards, x255 rescaled), 100 queries x 5 radii
(counts 40,087 / 163,554 / 417,145 / 1,751,602 / 4,646,740), exact ground truth.
IVF nlist = 10,000 (list population 1,000). Cost = distance evaluations, CPU-only, `joysuch`.

| estimator | mean rel. err ↓ | p95 ↓ | cost ↓ |
|---|---|---|---|
| uniform S=10,000 | 0.0803 | 0.337 | 1.0e6 |
| uniform S=40,000 | 0.0423 | 0.175 | 4.0e6 |
| uniform S=100,000 | **0.0289** | **0.121** | **1.0e7** |
| two-phase frac=0.0125 (KP=125) | 0.0775 | 0.289 | 1.14e7 |
| two-phase frac=0.05 (KP=500) | 0.0453 | 0.135 | 4.39e7 |
| two-phase frac=0.125 (KP=1250) | 0.0245 | 0.0705 | 1.09e8 |
| two-phase frac=0.25 (KP=2500) | 0.0111 | 0.0359 | 2.19e8 |

## Reading

1. **At matched cost, plain uniform sampling wins at every point at 10M.** Extrapolating the
   measured uniform curve (error ∝ 1/sqrt(S)): at 1.09e8 the expected uniform error is ~0.0088
   against the two-phase 0.0245; at 2.19e8 ~0.0062 against 0.0111. So the two-phase estimator is
   dominated at this scale.
2. **The reason is the sample budget, not the probe count.** At 100k the 4,000-sample residual
   estimate covers 4.6% of the residual (87.5k points); at 10M the same 4,000 samples cover
   0.046% of the residual (8.75M points). The prefix is exact either way, but the *residual*
   estimate degrades 100x. Scaling the probe fraction does not fix this.
3. **This is not a bug, it is the plan-flip the paper predicts.** 100k: two-phase 0.0143 vs the
   best classical plan 0.240 (16.8x). 10M: IVF-plan 0.3864 vs uniform sampling 0.0289-0.0803 —
   the ranking of plan families *inverts* with scale. That inversion is the justification for a
   cost-based planner rather than a single tuned estimator.
4. Consequence: `ana-vec` selects the plan by n and selectivity, exactly as the planner does.
   At 10M it uses the sampling plan (cost 1.0e7, err 0.0289) and is **13.4x more accurate than
   the IVF plan** (0.3864), which is what an index-first submission would ship.

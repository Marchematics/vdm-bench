# One planner, three task families: does a cost model beat per-task tuning?

**Status: partial.** Analytics is measured and reported below. Quality and serving were run and
their raw selections are committed (`experiments/data/planner_e2e_{quality,serving}.json`) but are
**not yet read out** — do not cite them.

## Setup
One planner (`vdmbench/planner.py`) over the same plan primitives the reference submissions use:
exact IVF prefix with `k` lists, Neyman allocation over index cells, uniform sampling, certified
probe rule, EDF ordering + per-tenant quotas. It sees only a per-query cost budget and a contract,
and returns a plan. Scored at matched budget against every fixed plan, 5 seeds, builder threads
pinned (`VDMBENCH_BUILD_THREADS=8`).

## Analytics, 100k SIFT (nlist=256, 100 queries x 5 radii)
| budget/q | uniform | best fixed plan | planner |
|---|---|---|---|
| 4,000 | 0.1885 | 0.0758 (two-phase k=4) | **0.0720** |
| 50,000 | 0.0447 | 0.00110 (two-phase k=64) | picks k per budget |

## Analytics, 10M SIFT (nlist=10,000, cell cache reused)
| budget/q | uniform | fixed plan | planner |
|---|---|---|---|
| 40,000 | 0.0449 | 0.0294 (k=0, i.e. no exact prefix; allocation only) | 0.0298 |
| 100,000 | 0.0274 | 0.0151 (k=0) | picks k=0 |

**What this establishes:** the lever inversion is *selected*, not hard-coded — at 100k the planner
buys exactness (it beats the best fixed prefix size at the tight budget), and at 10M it stops
probing lists entirely and spends the budget on allocated samples. It does not lose to the best
fixed plan at either scale.

**What it does not establish (yet):** (i) that the planner wins on quality and serving, (ii) that
its prediction of *which* lever wins matches the outcome in every regime, (iii) anything at 100M+.
Those are the next three steps, in that order.

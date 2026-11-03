# Acceptance checks for a regenerated analytics workload

Three rounds of silent breakage taught the same lesson twice: a workload can look "generated"
and still be measuring nothing. These are the checks to run before any leaderboard number is
believed. They are cheap and each one caught a real bug.

| check | pass condition | bug it catches |
|---|---|---|
| `uniq(coverage_chosen) == k` | 30 distinct indices | greedy summing over anchors instead of points (picked the same point k times) |
| `exact_covered < len(anchors)` | e.g. 36/100, not 100/100 | `r_cov` so loose that one point covers everything (task saturated) |
| `exact_covered > 0` | not 0 / len | radius so tight nothing is coverable, or a wrong scale on the vectors |
| leaderboard `coverage_ratio` in [0, ~2] | never `29.0` | ground truth and evaluation disagreeing |
| `len(join_pairs) > 0` and stable across refactors | 25,439 at 100k | an "optimisation" that silently changes the ground truth |
| `truth` counts monotone in radius | strictly increasing | radius/scale mistakes |

## Root causes found

1. **Greedy over the wrong axis.** `rem.sum(axis=0)` sums per *anchor*; the argmax was then used as
   a *point* index. The greedy only ever examined the first `len(anchors)` rows of X. At 10M this
   reported `exact_covered = 0`; at 100k it reported 100/100 by accident of the first 100 rows.
2. **`r_cov` derived from the head of the file.** `1-NN` estimated on the first 5,000 rows grows
   with n (0.05% of 10M), inflating `r_cov` to 566 and making the task trivially satisfiable.
   Fixed to an exact 1-NN over the whole collection, and the multiplier reduced from 1.5x to 1.0x.
3. **Join ground truth as an O(1e4 x 1e4) python loop** — 41.7 s of a 42 s build, independent of n.
   Replaced with an exact FAISS radius search (200k build: 42.4 s -> 7.0 s).

## Deployment check (this one bit hardest)

Files were uploaded with `tar` over ssh and the uploads **failed silently for several rounds**: the
remote `vdmbench/core.py` was a stale copy while the docs claimed the fixes were in. Every sync now
goes through `base64` + an explicit `md5sum` comparison on both sides, and `pkill` patterns are
written as `[f]ull_regen` so they cannot match the shell that issues them.

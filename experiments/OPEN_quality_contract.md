# Open task: enforce the quality task's contract

The quality task ships a contract per instance (`integrity_recall`, `outlier_rel_tol`,
`drift_tol_thr_units`) but `eval_quality` never checks it. Measured consequence: `vecprof-ref`
reports a hole-collection drift error of **2.12 against a declared tolerance of 2.0** and still
passes, and planner plans are ranked on cost alone with no notion of contract satisfaction.

Recipe (kept here because two subagent attempts died before landing it):

1. `vdmbench/core.py`, quality section: compute per instance `contract_met` (recall clauses `>=`,
   error clauses `<=`, each mapping documented inline), aggregate `contract_rate`, and for each
   failure the clause name and margin.
2. `evaluate.py`: pass the contract through and print `contract_met` / `contract_rate` next to the
   existing metrics.
3. Re-score `quality-oracle` / `quality-naive` / `vecprof-ref` on `ssh joysuch`
   (`~/vdm/data`, repo mirror `~/vdm/vdm-bench`, `nice -n 20`, `OMP_NUM_THREADS=8`, base64+md5
   sync both sides).
4. Update the quality section of `leaderboard_sift.md` and README. **The analytics rows must stay
   byte-identical** (ana-vec 0.013201, ana-vec-sampling 0.119493) — they are final.
5. A submission that looks best on mean error but violates a clause must be reported as failing;
   that is a benchmark finding, not something to hide.

What must not change: metric definitions, the workloads, or any existing analytics/quality number
that was measured under the current definitions.

## Status: implemented (2026-11-05)

Steps 1-4 are landed. `eval_quality` enforces the contract (`71a6928`), `evaluate.py` /
`run_reference.py` pass it through (`725e8ab`), the SIFT quality section is populated in
`leaderboard_sift.md` / `leaderboard.md`, and the measured compliance table plus what is still
unenforced is in `experiments/quality_contract_2026-11-05.md` and the README quality section.
Measured on the experiment host: `quality-oracle` 1.00 (5/5), `quality-naive` 0.00 (0/5),
`vecprof-ref` **0.80 (4/5)** — the hole-collection drift clause fails at **2.119 > 2.0**.

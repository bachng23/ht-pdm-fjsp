# Coordination benchmark v1 — local verification

Verified on macOS, 2026-10-05, before full-run handoff. No full run or remote
transfer was launched. Implementation is on the isolated branch
`codex/maintenance-coordination-benchmark-v1`; unrelated main checkout changes
were preserved.

* Regression suite: 475 passed, 190.04 seconds, one existing SB3/rich tqdm
  experimental warning. Subsequently added an explicit positive-overdue
  team-return test and reran the complete new benchmark test module: 36 passed.
* Ruff check/format and Git whitespace gate pass.
* Actual CLI smoke used `uv run --frozen --no-sync ... python -m
  ht_pdm_fjsp.maintenance_coordination_benchmark --profile smoke --device cpu`.
* Smoke directory: `artifacts/maintenance_coordination_benchmark_v1_smoke_20261005T140714Z`.
  Manifest COMPLETED, all 21 engineering audits true. Five saved/reloaded models,
  240 physical transitions,60 training episodes,40 optimizer steps,
  120 learned test,5 learned development,75 reference,24 message-off episodes,
  eight exact rows. All scientific statuses ENGINEERING_ONLY.
* Independent read-only audit `scripts/audit_maintenance_coordination_benchmark.py`
  PASS: 216 complete traced episodes,945 ticks,525 proposal ticks. Frozen source
  hashes, checkpoint actions/tokens/proposals, independent urgency arbitration,
  separate scalar costs, per-machine starts/wait/unavailability and development
  reference selection replay agree. Training physical panels match across arms.
* The added training test forces two already-overdue failed machines to STOP;
  each H4 episode has base cost48,waiting cost96,total144, and verifies the actual
  per-agent trainer's Monte Carlo/probability reconciliation on positive penalties.
* New tests also cover mixed N/K padding and STOP remapping, no actor peer-state
  leakage, hidden occupant-status removal, IPPO local vs MAPPO global value
  observations, learned peer effects/message-off, legal collisions and idle
  capacity diagnostics, partial matching vs exhaustive enumeration, isolated
  lookahead vs physical one-machine DP, uncertainty/SLA/nominal gates,
  seed/overwrite/dirty-full guards, and end-to-end checkpoint/trace replay.

Full protocol remains untouched by smoke performance: 50 models,6m transitions,
20,000 learned test +1,000 learned development +1,260 references +4,000 message-off
and53 exact rows. No baseline, price, seed, stopping rule or hypothesis was changed
from smoke scores. Four confirmatory contrasts use98.75% paired Student-t intervals.
The best reference is selected only from the separate development panel. Local
information is a synthetic sensitivity, explicitly approved by the user; no
unconditional MARL necessity claim is made. Parameter counts are equal among the
four proposal models; unused branches mean active capacity can differ. The shared
fixed execution arbiter and central_ar factorization are disclosed in the plan.

Run command uses a detached checkout pinned to the pushed revision and a separate
virtual environment. Outputs use the canonical repository artifact root and a
fresh UTC timestamp. Keep the terminal connected; no resume contract is offered.

Final CLI smoke and replay repeated after clarifying only exact-oracle artifact
filenames in the plan; all counts/gates unchanged. Handoff shell syntax verified
with `bash -n`. Rsync include rules tested in local dry-run: nested model and log
selected, unrelated experiment excluded; no remote connection/transfer made.

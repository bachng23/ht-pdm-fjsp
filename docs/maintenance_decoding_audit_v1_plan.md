# Frozen-checkpoint serial decoding audit v1

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-08
- Verification Status: UNVERIFIED
- Version Label: maintenance_decoding_audit_v1

## Hypothesis and intervention

Evaluate the selected, immutable solver-comparison v1 checkpoints. No training,
checkpoint reselection, stochastic execution, critic use for serial decisions,
new information features, reward changes or changed physical dispatch semantics.
Two primary contrasts: exact joint-MAP minus legacy serial greedy, separately
for cooperative_local_ppo and cooperative_full_ppo on specialized N4/K2/H12.
Hypothesis: greedy conditional decoding explains a practically relevant part of
the cost gap. The hypothesis may fail: maximizing policy probability is not
minimizing expected cost. MAP is a centralized diagnostic, not distributed execution
or a quality oracle.

Joint-MAP exhaustively scores every feasible matching using the SAME rotating
machine order, conditional reservation masks and frozen actor. Sum conditional
log probabilities; maximize, ties within 1e-6 choose first matching in the existing
catalogue. This finite precision convention is recorded; selection regret must
not exceed 1e-6. Require complete probability mass within 1e-5 of one. Greedy
execution must call the unmodified legacy actor. The MAP likelihood evaluator
uses a single encoded state shared across candidates, no critic and no lookahead.

## Panel, comparisons and stopping rule

Full: 10 original training replicas 160000..160009, all four conditions and
H12/H24, same 32 labeled test profiles 163000..163031. These profiles were already
opened; the new sealed shock panel supports a prospective decoder contrast
conditional on these familiar profiles, NOT unseen-profile generalization.
Common-state diagnostic uses only original development profile 161000 and new
shock 170000: 12 states/condition, collected with the original selected rule.
Sealed closed-loop shocks 171000..171019. No development outcome changes decoder,
checkpoint, thresholds, budget or panel. Smoke uses full TRAIN profile 165020
via the inherited smoke resolver, shocks 173000..173001, H4/H6, the first source
replica only. Smoke never instantiates full test profiles or full test shocks.
Separate protocol namespace; registry checks historical and source seed overlap.

Learned controllers: central matching greedy, central fixed greedy, cooperative
local greedy/MAP, cooperative full greedy/MAP. Both central references use their
unchanged legacy decoder. Controls: original development-selected rule per
condition, and joint rollout with original continuation/scenario count. Independent
RL is omitted because neither primary contrast requires it. Keep source model
weights and source checkpoint-selection metadata unchanged and hash all used files.

Full counts: 40 loaded checkpoints, 60 learned controller/replica pairs + 2
controls, 317440 closed-loop episodes, 5713920 physical decision intervals.
Common-state diagnostics: 20 serial models * 48 states = 960 decoder audits;
62 controller/replica pairs * 48 states * 3 timed repeats = 8928 latency rows,
preceded by one untimed warm pass/controller. Fixed grid, no early stopping or
resume. A crash preserves partial outputs and FAILED manifest. Full requires
Linux and a clean Git checkout. Default CPU, one torch thread.

## Metrics and statistics

Primary priced cost difference MAP-greedy on specialized H12; base cost and
cost components secondary. Practical gate: priced reduction >=2%, at least
8/10 replica wins, and simultaneous two-sided Bonferroni 97.5% paired-t CI upper
bound <0 (two primary contrasts). Guard: episode waiting violation increase
<=2 percentage points, and terminal-pending requests/episode increase <=0.05
relative to SAME frozen greedy checkpoint arm, primary condition. Guard thresholds
are research margins, not plant SLA requirements. Nominal/H24/skill-mask and
comparisons to central/rule/planner are exploratory with 95% CIs. No smoke CIs
or scientific PASS. Pair on model seed, profile, shock, condition and horizon;
average profiles/shocks inside training replica; never treat episode/tick rows as
independent training replicas. Report all seeds and both primary results, including
failure. CI is conditional on selected checkpoint population and fixed labeled
profiles/shocks; training is not rerun. No equivalence claim from a CI crossing zero.

Secondary: request served/censored/observed overdue, terminal zero-wait censored
arrivals, served-wait distribution (clearly conditional); per-episode max wait,
episode violation, failures, maintenance/downtime/failure cost. Request service
start ends waiting, not repair completion. Censoring prevents starvation-free claims.
Full feasibility and cost/request reconciliation on every episode.

On common development states, log complete candidate actions/log probabilities,
normalization, greedy action/rank/probability, MAP action/probability,
matching disagreement and MAP log-probability gain. For common-state latency:
same inputs, no critic for serial, central retains legacy diagnostic critic;
per-controller clear planner caches, one warm pass + three repeats, validate each
assignment. P50/P95 and mean; no inference of network/decentralized scalability.
MAP has different information aggregation/arbitration requirements, so do not
reuse the old logical communication model or claim communication savings.

## Provenance, engineering gate and outputs

Input --source-run must be a COMPLETED original solver-comparison v1 run, matching
profile/algorithm/feature/environment contract; full requires the original full
commit 4b1cffb0cc1e2413e27f0e2c3a7730f8d3e3d7a8. Verify manifest-listed selection,
config, reference-selection, snapshot and checkpoint hashes before evaluation.
Full additionally verifies frozen source files against current inherited modules.
Reject missing/changed checkpoints, inconsistent payload seed/steps/algorithm,
unsafe paths or different settings before creating a run or consuming compute.

Tests: contrived greedy-MAP disagreement, independently score teacher-forced
prefixes, probability normalization with occupied workers/skill masks, WAIT,
rotating order, local critic not accessed, lexicographic catalogue tie handling,
checkpoint identity, profile/seed sealing, duplicated/incomplete evaluation grid,
paired statistics, interrupted-run persistence and end-to-end smoke schema.
Mac smoke also uses one real full-run selected replica to validate compatibility.

Outputs in fresh UTC timestamp directory: manifest/config/protocol/seed registry
snapshots, source_provenance.json, evaluation_configs.json, episodes.partial.csv,
episodes.csv, request_metrics.csv, decoding_audit.csv, latency_states.json,
latency.csv, paired_replica_metrics.csv, summary.json, decoding_summary.json,
latency_summary.json and request_summary.json. No copied models or training
logs. Compact request and aggregate episode logging; complete candidate
distributions only on development states, no massive decision CSV. Manifest counts
and SHA-256 all output artifacts, including source snapshots. Raw input is read-only.

Follow docs/experiment_workflow.md: tests and Mac smoke -> commit and push -> exact
human-run lab compound command -> human-run rsync -avhP --partial. No SSH or full
lab run by assistant. Training count is zero; multi-seed evaluation uses tqdm.

## Entry command and source availability

Module entry: `uv run python -m ht_pdm_fjsp.maintenance_decoding_audit --profile full
--device cpu --source-run artifacts/maintenance_solver_comparison_v1_full_cpu_20261007T151955Z
--output-dir artifacts/<new_UTC_timestamp>`. The source directory must remain
available on the lab. Preflight verifies all 40 required selected checkpoints and
metadata before any evaluation. No downloaded model or network service is required.

Full output estimate: approximately 0.3-0.5 GB including the duplicate partial/final
episode CSV; no full decisions.csv or model copies. Wall-time is not a stopping
criterion and cannot be reliably extrapolated from H4/H6 Mac smoke. Leave the lab
terminal open; fixed-grid tqdm reports actual progress. The unchanged UV lockfile
provides all dependencies; no added package is required.

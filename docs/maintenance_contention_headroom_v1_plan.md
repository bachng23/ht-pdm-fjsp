# Maintenance contention/headroom v1 — locked protocol

Locked before implementation/results, 2026-10-06 (Asia/Taipei). This experiment
measures opportunity before training more RL. It can find little headroom.
Use the repository experiment workflow; full execution on the Ubuntu lab is
handed to the user after verified commit/push. No SSH or transfer by assistant.

## Intervention and questions

Same WaitingEnv physical dynamics: restoration on completion, shared pending
pool, simultaneous feasible matching, PM1/CM2/failure15/downtime6, overdue
price12 per failed-wait tick beyond B4. Same age threshold3, failure probability
.6, per-machine service duration sampled1..3, restoration sampled.5/.75/1.
Primary N4,H12; secondary N4,H24. Only capacity changes: low K4, middle K2,
high K1. Workers have identical profiles for each machine. Thus own best
service options and costs are unchanged by treatment, unlike scaling duration.
Names low/middle/high refer to assigned capacity; report observed utilization,
urgent excess and queueing instead of assuming monotonic realized contention.
No claim about heterogeneous skill allocation; the study isolates service
priority and shared capacity. No real distributed information constraint added.

Two quantities differ: capacity scarcity cost (joint optimum minus independent
capacity-relaxed optimum), and avoidable decision cost (policy minus joint
optimum). Higher unavoidable cost alone does not imply room for RL/MARL.

## Controllers

Four references: existing risk_skill_rule, deadline_matching, isolated H<=6
lookahead_matching; plus independent_dp_matching. The latter optimizes each
machine exactly to the actual horizon under its failure law/service/restoration
and overdue price, assuming workers freely available in the future. It matches
positive marginal start-now versus defer-now bids under today's real capacity.
Each machine's independent value is also a valid capacity-relaxed lower bound.

joint_rollout enumerates every feasible machine subset, with 32 independent
model-based future hazard scenarios and the development-selected reference
as continuation. It rolls to the REAL episode end, not a truncated forecast.
All root actions share scenarios. Future simulated shocks come from a separate
planner seed role and never from the environment's realized future events.
This is an achievable model-based planner, not an exact N4 optimum. No gain
from it cannot establish a ceiling; its sample selection may be imperfect.
Save root candidate scores, selected/baseline root scores and Monte Carlo SE.

Equivalent worker assignments are canonicalized (sorted selected machines to
sorted idle workers). For identical workers, permutations have identical cost
and future physical behavior modulo worker IDs. Enumerating machine subsets
therefore covers the exact joint action space modulo this symmetry, including
STOP. Test the symmetry and vectorized simulator against scalar WaitingEnv.

References are selected separately per capacity, only on development PRICED
cost over ten cohorts x20 environment seeds on H12; deterministic name ties.
Freeze selection before any test/exact policy results. Save every reference,
including independent DP, not only the chosen one. All test controllers see
current state/model parameters; none sees actual future shocks.

## Exact panel

N3,H6, K3/K2/K1 paired across capacity using the first three machines of each
cohort. Uniform over all27 healthy initial age vectors (0..2). Full joint
stochastic DP of the common PRICED objective with expected hazard branches.
Cache states after equivalent worker permutation normalization. Per-job cap
250000 joint states, fixed before smoke. A cap produces CAPPED with partial
artifacts and unknown exact headroom, never an approximate optimum or exclusion
silently called complete. Every cohort/capacity job must be attempted.

Compare all four references to the exact joint optimum and independent lower
bound. Save expected policy cost, base/priced objective scope, regret, priority/
allocation decomposition (allocation zero here), early versus late regret and
Bellman residual. Exact state values/actions exported as gzipped JSONL for
reproducibility. Early is first half of H; late is second half, to expose finite
horizon effects. Low K>=N provides a negative control: exact joint optimum
must equal the independent relaxation and independent DP matching must be
optimal. No inference that N3/H6 establishes the N4/H12 or H24 ceiling.

## Metrics and confirmatory hypotheses

Primary: equal-weight mean cohort cost on50 paired evaluation environment seeds
at N4,H12; independent unit is ten randomized workload cohorts. PRICED objective
is base cost+12*overdue ticks. H_room_low/middle/high compare joint_rollout with
the development-selected best reference for each capacity. Each passes iff
mean cost reduction>=5%, wins>=8/10 cohorts, upper bound<0 for the paired
99% two-sided Student-t(df9) CI (critical3.2498355415921254; conservative
Bonferroni for three contrasts), and base-cost mean<=110% of reference.
Smoke scientific statuses ENGINEERING_ONLY. Complete test grids required.

Report base/failure/maintenance/downtime costs, waits/p95/violations/terminal
pending, utilization, failed backlog, urgent excess beyond maximum feasible
matching, feasible subset count, choice opportunity frequency, planning score
spread/MC SE, and runtime. Scarcity/decision gaps on the exact panel are
secondary; per-capacity trends and H24 contrasts are exploratory. State caps
are counted and block ceiling conclusions for affected jobs. SLA B4 is soft;
capacity can make it physically infeasible, so report violations explicitly
rather than pretending the diagnostic guarantees a hard bound.

## Seeds, budget and stopping

Full cohorts134000..134009, development135000..135019, test136000..136049.
Smoke cohort137000, development137020, test137010..137012. No training seeds,
no RL transitions, optimizer or teacher labels; this is a model-based diagnostic.
Cohort configuration role separate from paired environment role; within a
cohort/environment seed all capacities, methods and horizon prefixes share
initial ages/hazard grids. Development/test shocks disjoint; configs shared
by design (known-model study), not unseen-workload generalization. Historical
registry and reserved201..300,601..700 remain closed.

Full: 15000 test episodes (5 controllers x3 capacities x10 cohorts x50 test
seeds x2 horizons),2400 reference development episodes;30 exact jobs, up to
120 exact policy rows if none capped. Smoke:90 test,12 development,3 exact jobs
and up to12 exact rows. Smoke H4 primary/H6 long, exact H3,4 scenario forecasts;
full 32 scenarios. Exact state cap same. Fixed panels, final outputs only;
no adaptive budgets, checkpoint selection, early performance stopping or resume.

## Artifacts and engineering gates

Fresh UTC directory, refuse nonempty output, full requires clean Git.
manifest.json (protocol, commit, runtime, expected/actual counts, status),
source_snapshot (code+plan+seed registry), benchmark_config/resolved_config,
evaluation_configs.json, controller_checkpoints/*.json with reload checks,
reference_selection.json, development_episodes.csv, episodes.partial/episodes.csv,
decisions.csv (state, physical action, planner diagnostics), machine_metrics.csv,
exact_jobs.csv, exact_metrics.csv, oracle_tables/*.jsonl.gz, paired_cohort_metrics.csv,
summary.json, budget.json (training/optimizer/teacher rows zero). Controller
JSON replaces neural weights; training files not applicable. Partial results
and FAILED/CAPPED status preserved. Training progress not applicable; evaluation,
cohort/exact searches show tqdm. Memory is bounded by fixed caches/state cap.

Tests + local Mac CLI smoke before commit/push. Gates: scalar/vector transitions,
worker symmetry, independent value lower bound, exact Bellman/decomposition,
matched streams, deterministic checkpoint reload/scenario replay, feasible
assignments, cost/machine reconciliation, all panels and zero training budget.
Provide exactly the lab full-run command and local partial-preserving rsync.
A positive planner gap supports optimization headroom, not MARL necessity; a
small exact gap supports limited headroom only on the exact panel. RL learning
and operational distributed constraints remain separate questions.

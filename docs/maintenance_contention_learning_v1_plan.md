# Maintenance contention learning v1 — frozen protocol

Locked 2026-10-06 before implementation or any new experiment results.
Engineering amendment before scientific runs: deterministic near-tie tolerance
1e-6, verified on synthetic logits; no budget/metric/selection changes.
Parent diagnostic: maintenance_contention_headroom_v1, commit89ccb60. New
sealed cohorts/shocks; prior 134–137 panels remain historical. Question: can
centralized PPO learn economically useful shared-capacity maintenance decisions
on the identical-worker problem where a joint rollout planner showed headroom?
This experiment tests RL learnability. It does not compare or establish MARL.

## Environment and policy

Unmodified WaitingEnv, identical technician columns, duration1..3 and
restoration.5/.75/1 per machine; failure threshold3/probability.6;
PM1/CM2/failure15/downtime6, B4 and overdue price12. Completion restoration,
shared pending demand and simultaneous feasible matching remain unchanged.
N4,H12 training: deterministic 80% K2 /20% K4 mixture by episode index (every
fifth episode K4). Primary test N4,K2,H12; K4/H12 nominal guard. Secondary K1
and H24 (OOD horizon; K1 not trained). Known simulator model with freshly drawn
training configuration seeds; every reserved development/test seed is excluded.
The finite parameter generator can produce repeated physical profiles across
independent seeds; this is not a guarantee of unseen parameter combinations.

One learner: central_subset_ppo. Each action is the complete chosen machine
subset including STOP, with a feasibility mask for ongoing services and idle
worker capacity. Sorted chosen machines match sorted idle workers; worker
identity has no economic information in this identical-worker environment.
The entity encoder receives current age/failure/wait/remaining, own duration/
restoration/service flag/imminent hazard; global features include remaining
horizon, capacity, free workers, model costs/failure law and overdue parameters.
All agents/machines' current information is available. No future shocks, teacher
labels, planner actions or previous trained checkpoints are used for PPO.
Deterministic evaluation breaks logits within1e-6 of the maximum by canonical
lexicographic subset order, avoiding unstable ties between symmetric machines.
Shared machine/global MLPs with width64 and pooled all/selected/unselected machine
representations produce joint subset logits plus one value. Actor learns
non-additive subset scores; it is a centralized policy, not separate agents.

## Training, checkpoints, selection

10 training seeds; 491520 physical steps/model, 40960 complete H12 episodes,
320 rollouts of1536 steps, vector inference groups32 envs, minibatch192,
4 PPO epochs (10240 optimizer steps/model). Discount1, complete-episode Monte
Carlo returns, reward=-priced interval cost/20, normalized advantage, Adam3e-4,
clip.2, entropy.01, value weight.5, gradient norm.5, one CPU torch thread.
No adaptive epochs/budget, training early stop, test-feedback tuning or resume.
Torch action/minibatch RNG and config/environment RNG have separate seed roles.
Every training episode saves config seed/hash and shocks seed; no held-out
configuration/hazard seed may enter training. Count budgets exactly.

Save initial model and checkpoints at80/160/240/320 rollouts. Evaluate all five
on fixed development K2 and K4/H12, 10 cohorts x20 shocks each. Initial model is
an engineering/learning-curve diagnostic and is not selectable. Among four
trained checkpoints choose minimum K2 priced cost that meets dev nominal base
cost<=110% selected K4 reference and K2 waiting-violation rate<=selected K2
reference+2 percentage points. Earlier step breaks exact ties. If no feasible
checkpoint, choose final and mark DEVELOPMENT_INFEASIBLE; no silent omission.
Freeze each selection before opening any new test result. All train seeds stay
in the test analysis including development-infeasible ones.

Four deterministic references (risk_skill_rule, deadline_matching,
lookahead_matching, independent_dp_matching) use existing diagnostic definitions.
Select best per capacity on development H12 priced cost only, name tie.
Joint rollout (32 forecast scenarios, full remaining horizon, independent
planner RNG, common forecasts across root actions) uses that selected reference
as continuation. Save every reference, not only selected. Planner is achievable,
not an N4 optimum; its runtime/model knowledge are reported separately from PPO.
No new exact panel: the previous exact diagnostic already established headroom.

## New seed splits and fixed workload

Full training seeds138000..138009. Development cohort139000..139009;
development environment140000..140019. Sealed test cohorts141000..141009;
test environments142000..142049. Training configuration/environment seeds are
counter-derived SHA256 roles from training seed; explicitly check against every
held-out/historical numeric seed before use. Cohort role remains
contention_configuration; environment initial/failures unchanged. All test
controllers share cohort/hazard streams and H12/H24 prefixes. No checkpoint
selection on test; new seed draws independent of training/development, without an unseen-profile claim.

Smoke train143000, dev cohort143020/shock143030; test cohort143010,
shocks143040..143042. Smoke H4 train/primary, H6 secondary. Two rollouts128
steps each (32 complete H4 episodes/rollout), group8, minibatch32, epochs2,
checkpoints1/2 plus initial; width16,4 planner scenarios. It validates engineering
only, does not tune hyperparameters or establish scientific performance.

Full counts: 10models,4915200 training steps,409600 training episodes,
102400 optimizer steps;22400 development episodes (2400 reference +20000
learner including initialization);45000 test episodes (30000 selected learned
models +15000 deterministic controls/planner);810000 test decision intervals,
180000 test machine rows. Smoke:1model,256 steps,64 episodes,16 optimizer
steps;18 dev episodes (12 refs+6 learner);108 test episodes,540 decision
intervals,432 machine rows. Training progress and all multi-seed eval use tqdm.

## Primary hypothesis, uncertainty and stopping

H_learn, one confirmatory contrast: selected PPO vs development-selected K2
reference on N4,H12. Unit: 10 independent training seeds, each mean across the
same10 new cohorts x50 paired test shocks. Reference/planner evaluated once per
panel, not fabricated as independent training replicates. Training-seed CI is
conditional on this finite held-out workload/shock panel. Cohort-level results
are exploratory and separately reported; no pseudo-replication of5000 episodes.

PASS iff mean priced-cost reduction>=5%, wins>=8/10 training seeds, upper95%
two-sided paired Student-t(df9, critical2.2621571628540993) CI<0, base-cost
reduction>=3%, K4/H12 nominal base-cost mean<=110% reference, and K2/H12 mean
waiting-violation rate<=reference+2 percentage points. Report every gate and
all10 outcomes. Primary all seeds included; no exclusion for bad dev selection.
Smoke status ENGINEERING_ONLY. No pass interpretation from budget-limited runs.

Report base/failure/maintenance/downtime/overdue cost, failure counts, waits,
violation/p95/max/terminal backlog, utilization/urgent excess, feasibility,
planning/inference runtime. Gap capture=(reference−learner)/(reference−planner)
on priced and base costs, as exploratory learnability diagnostic. If planner
fails to beat reference on new panel, gap capture is undefined; no clamp and no
ceiling claim. Values may be negative or>1. H24/K1 contrasts, cohort consistency,
initial-to-checkpoint dev curves and seeds are exploratory. Need operational
information/communication constraints plus matched centralized comparisons
before investigating a MARL necessity claim.

## Artifacts and engineering gates

Fresh UTC output; refuse nonempty directory; full requires clean Git and CPU.
Manifest RUNNING/FAILED/COMPLETED, source snapshot plus plan/seed registry/lock,
benchmark/resolved/evaluation configs, seed_audit, budget, training_episodes.csv,
training_progress.csv, training_coverage.json, model checkpoints (initial and
4fixed saves +selected model with settings/state_dict/contract), development
and selection JSON/CSV, episodes.partial.csv/episodes.csv, decisions.csv,
machine_metrics.csv, paired_seed_metrics.csv, cohort_metrics.csv, summary.json.
Checkpoints reload exactly (parameters, outputs, masked probabilities/actions).
No silent resume; FAILED preserves partial journals/checkpoints.

Tests and local macOS CLI smoke before commit/push. Verify mask covers all
physical actions modulo worker symmetry; sampling/logprob/value reevaluation;
Monte Carlo reward/return signs and complete episode boundaries; optimizer and
physical budgets; actual weight updates; gradient finite; current-state feature
contract (no future access); same test streams across controllers; fresh seed
splits/no test-config training; deterministic reload and selected checkpoint
replay; cost/machine reconciliation and full grids; output overwrite refusal.
Push verified code, then stop tool execution and hand off exact lab command and
partial-preserving rsync. Never SSH/start/transfer full run on user's behalf.

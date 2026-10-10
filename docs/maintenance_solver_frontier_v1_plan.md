# Maintenance solver frontier v1 — locked protocol

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: authorized implementation, Mac smoke, human lab handoff
- Origin Date: 2026-10-10 Asia/Taipei
- Verification Status: UNVERIFIED scientific result
- Version Label: maintenance_solver_frontier_v1
- Parent commit: a210ad67679b5d9e19f9da409c458e28c09f679f

## Question and scope

Does frozen centralized RL offer a useful quality/compute tradeoff against rules
and a vectorized rollout planner on the CURRENT simulated maintenance benchmark?
Any solver may win. The user confirms no deployment specification exists. Do
not fabricate a deadline, call simulated configurations operational requirements,
claim workload/scale generalization, or claim every planner is covered.

The falsifiable screening hypothesis is centralized RL cost non-inferiority
within 2% AND at least twofold lower mean latency on BOTH common states and
closed-loop trajectories, with relative waiting/pending guards. The user
explicitly chose the 2% cost margin on 2026-10-10, before new test data. Also
report a quality-advantage screen using at least 2% lower cost. Screens are
research-direction evidence, not joint confirmatory tests or deployment SLAs.

## Frozen task and solvers

Keep existing physics/prices/waiting-limit4/waiting-price12, current-state full
information, legal matching, terminal semantics. N4, condition-specific K1/K2,
four existing nominal/specialized/skill_mask/homogeneous conditions, H12/H24.
No new training. No MARL arm, scaling, workload manipulation or fine-tuning.

- Rule: choose among four existing references per condition on development H12
  mean objective; lexical ties. Report selected rule at H12 AND H24.
- Planner: ALL three 8/32/128 simulated forecast-scenario budgets; each chooses
  continuation separately among the same four rules, per condition, development
  H12 only. Never select the winning budget using test, never assume more
  scenarios guarantee better quality. Preserve all feasible root matching
  candidates and full remaining-horizon lookahead; no future environment shocks.
- RL: all10 final central_matching_ppo/covered checkpoints, train seeds
  180000..180009, from coverage run maintenance_training_coverage_v1_full_cpu_
  20261009T044236Z, clean commit53d180a5fc18c53e850eff7278d753fbe295d647.
  No checkpoint/replica selection or weight updates. Plot/report family mean
  and every replica; the mean is not a deployable ensemble policy.

Source preflight requires COMPLETED, expected counts, clean exact source commit,
relevant file hashes/sizes, final-only selection and model payload identity,
feature/environment contracts, finite tensors and exact491520steps/320updates.
Copy immutable checkpoints into new output and verify hashes. Source training
wall time is recorded per model. Historical training CPU seconds and matched
hardware evidence are unavailable: do not manufacture compute amortization.

Planner implementation evaluates all roots/scenarios in NumPy batches, inherited
scalar waiting physics, with cached expected-value continuation. New scenario-
major independent forecast stream makes smaller budgets exact scenario prefixes
at every future time. Shared forecasts across candidates reduce MC noise.
Changing budget still changes decisions/visited states. Validate vectorized
scores and chosen action against independent scalar rollout on identical events.
This is an optimized rollout baseline with heuristic continuation, not an
optimality certificate. Any decision-affecting engineering change after test
opening requires a new protocol; no post-test planner tuning.

## Seeds, development and sealed tests

No new training seeds. Intentional frozen historical model seeds above.
Familiar profiles: dev161000..161015, test163000..163031, already opened in prior
studies. Fresh dev shocks204000..204019, sealed test205000..205019.
Timing order206000, bootstrap207000..207007, reserved separately from historical
panels and checked by registry. Planner forecasts use SHA-derived independent
role solver_frontier_v1_forecast:<time>, never read environment event arrays.

Development: every rule and every budget/continuation, all16profiles×20shocks×
4conditions at H12, 20,480episodes total. Selection frozen before opening test.
No RL development selection: models were frozen before this experiment.

Test:14controllers (one selected rule, three planners, ten RL replicas) ×
4conditions×H12/H24×32profiles×20shocks =71,680episodes;
1,290,240physical ticks,286,720machine rows. Each controller uses same exogenous
shock panel; trajectories differ. Fixed complete grid, no early stopping,
seed exclusion, repeated runs to obtain a preferred winner or post-test tuning.

Smoke: frozen seed180000 only, engineering physical profiles165020/165010 both
map to FULL TRAIN pool; dev209020/test209040..209041. Horizons4/6, budgets2/4/8,
timing209060/bootstrap209061.64development episodes,80test episodes,400test
physical ticks,320machine rows. No fresh full test shocks/profile configurations
are opened in tests or Mac smoke. Smoke has no training and is not scientific
performance evidence.

## Timing and resource contract

CPU only; explicit OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1,
Torch1thread. Same hardware/runtime for every measured solver in this run.

Common-state timing:16devprofiles×4conditions×(12+24)ticks =2,304shared states
from selected-rule development trajectories using devshock204000. Each of14
controllers runs serially in a fresh spawned process; randomized controller order.
One cache-cleared cold inference per cell, clear caches again, one warm pass,
16measured passes with randomized state order and shared states. Total516,096
warm rows and112cold rows. Per controller/cell H12 has3,072 samples, H24 has6,144.
State repetitions are correlated, not independent effective sample size. p99 is
descriptive, with counts and tail counts, not a precise tail guarantee. Smoke
40states, two repeats/controller,400warm rows and40cold rows.

Timed select begins from raw current config/state, includes native frontend,
forecasting/neural computation, matching/decode and feasibility validation. RL
retains the existing select's critic/log-probability diagnostics, explicitly
reported. This is not a claim of maximally optimized actor-only serving.
Model load and worker preparation time reported separately; interpreter startup
not included. Cold inference one observation per cell: no cold-start percentile
claim. Shared cached continuation optimizations are available to rule/planner;
clear per worker then warm identically. All budgets have identical cache protocol.

Closed-loop test: record EACH decision latency and p50/p95/p99/max by cell and
replica on actual solver trajectories, randomized controller/episode order.
Excluded from decision latency: simulator transition and scientific journaling;
included in phase wall/CPU time. Raw input is a complete simulated current state;
no transport or telemetry acquisition latency is measured. Peak RSS from isolated
common-state workers includes Python/Torch runtime; closed-loop parent lifetime
RSS is explicitly labeled and must not be treated as isolated solver memory.
Record checkpoint bytes, selection compute and historical training wall time.

## Statistics and directional screens

For each8condition/horizon cells compare RL family against each4controls;
all32cost contrasts reported, no post-test chosen primary cell or planner budget.
Primary estimand: RL mean cost / reference mean cost, averaged over balanced
training replicas, labeled profiles and shocks. Paired crossed empirical
percentile bootstrap: resample training replicas, profiles and shocks separately;
reuse environment weights for both solvers and exact bootstrap draws across
controls within each cell.20,000draws, fixed resampling seed. Nominal Bonferroni
cost CI confidence1-.05/32=.9984375. These are approximate bootstrap intervals
with finite tails, not certified finite-sample familywise coverage. Profiles may
be permutation-equivalent; inference concerns labeled benchmark distributions,
not physically novel workload classes or independent episode samples.

Practical cost noninferiority screen: cost-ratio upper CI≤1.02. Cost advantage
screen: upper CI≤.98. Lower compute screen: mean decision latency ratio≤.5 in
BOTH common-state and closed-loop panels. Guards on the same cell: waiting
violation difference≤+.02 (2percentage points), terminal pending difference≤+.05
requests/episode. Guards/latency are descriptive, without familywise inferential
certification. Report every gate, raw estimates and uncertainty; absence of a
passed screen is not proof of inferiority or equivalence. Joint screen status
must not be labeled statistically confirmed superiority or operational adequacy.

Always show 0/1/2/5% cost-margin sensitivity, labeled secondary, without changing
2% main margin. Cost/compute assessed together; no single-metric winner. Plot
all budgets; points/empirical frontier are descriptive, with all raw data
available. Do not hide waiting regressions with pooled means. Report served/
censored requests, exact waiting-cost decomposition, overdue ticks, observed
waits and pending. Pending at horizon is not permanent starvation; censored
waits are lower bounds. Lower failure counts alone do not prove better service.

## Artifacts, stopping and engineering gate

Timestamped new output, reject nonempty dirs/no resume, immutable prior runs.
Interrupted/error run gets FAILED manifest, preserves partial CSV/log/checkpoints.
No automatic lab launch/SSH/rsync. Full Linux clean checkout only, human handoff.
tqdm development, latency workers and multi-seed evaluation.

Manifest: commit/status/hardware/runtime/config/expected+actualcounts/hash inventory.
Files: source_snapshot, resolved_config, seed/profile audits, checkpoint provenance,
frozen model copies, development_episodes, development/reference/planner selection,
latency_states, per-worker warm/cold CSV and replica summary/resource, aggregate
latency_summary, closed_loop_latency and summary, resource_usage, episodes.partial/
episodes, machine/request metrics, request_wait_decomposition, feasibility_audit,
paired_seed_metrics, summary, quality_compute PNG/PDF.

Mac gate: planner forecast prefixes/scalar score parity/legal matching; sealed
panels/complete-grid rejection; provenance and corrupted/nonfinal checkpoints;
bootstrap pairing/resampling/2% classification; worker isolation and timing counts;
manifest immutability/failure handling; independent replay of all smoke physical
transitions and SHA/size inventory. Then commit/push, exact lab command and human
rsync handoff. Future scaling requires a separate design and policy able to run
at unseen N/K without fine-tuning; this experiment authorizes no such claim.

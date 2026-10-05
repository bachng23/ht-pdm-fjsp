# Maintenance allocation context × overdue waiting, v1

Locked before implementation/smoke/full outcomes. This is a new diagnostic
protocol following priority/allocation v2, not an amendment of its failed H1.

## Questions and factors

Does giving the allocation head each technician's peer-machine context improve
base economic cost? Can an explicit cost of excessive corrective waiting reduce
neglected failures within an acceptable economic tradeoff?

Four PPO arms: `local_cost`, `context_cost`, `local_wait`, `context_wait`.
All learn machine/STOP and allocation, use the same initialization and number
of parameters. The allocation head receives selected edge, common context,
prefix, and updated technician embedding. Local arms replace only the last
embedding by zeros. Context arms use the per-technician embedding aggregated
across compatible machines. This is static at the physical step; reservations
still update masks and prefix, not the graph embeddings.

Physics, failure streams, start costs and completion restoration remain v2.
Base cost = maintenance + failure + unavailable cost (1 PM, 2 CM, 15 failure,
6 per unavailable machine tick). Waiting limit B=4 ticks. A failed, unserviced
machine incurs an overdue tick exactly when its next pending wait exceeds B.
Wait arms train on base cost + 12 × overdue ticks; cost arms train on base cost.
The bound and price are synthetic sensitivity assumptions, not a calibrated
factory SLA. This soft cost does not guarantee a hard waiting constraint.
All arms observe B and their training price. Evaluation also records a common
service-adjusted score = base cost + 12 × overdue ticks for EVERY arm.

All deterministic neural decisions use lowest feasible index within 1e-5
absolute normalized-logit distance of the maximum. Sampled PPO distributions
remain unchanged. The same deployment rule applies to all four arms. This
reduces near-tie sensitivity; it is not a cross-platform bitwise guarantee.
The fixed risk/skill heuristic is evaluated once and reused descriptively.

## Hypotheses and gates

H_context: `context_cost` improves over `local_cost` on the equal-weight mean
of n3_nominal, n3_pressure, n4_nominal, n4_pressure. Primary metric is BASE cost.
Pass requires at least 5% mean reduction, wins in >=8/10 training seeds,
97.5% two-sided paired-t CI upper <0, and nominal mean cost increase <=10%.

H_wait: `context_wait` versus `context_cost` reduces equal-weight mean overdue
waiting ticks per machine on n3_pressure and n4_pressure by >=50%, wins >=8/10,
and 97.5% paired-t CI upper <0. Additionally each of these two families must
have pooled p95 maximum pending wait <=4, primary BASE cost increase <=10%,
and nominal BASE cost increase <=10%. A zero control overdue mean makes the
relative-benefit hypothesis inapplicable/fail rather than claiming a benefit.

Both intervals use df9 critical value 2.685010847 (Bonferroni two hypotheses).
Independent unit is TRAINING seed, conditional on the fixed evaluation panel.
Machine/tick/episode rows are not treated as independent training replicates.
Report both gates separately; do not collapse into a claim of MARL necessity.

Other context/wait main effects, interaction, family effects, common adjusted
score, p95 waits, terminal failed machines with zero service starts, failed
deferral and allocation-regret decomposition are descriptive. All raw metrics
and CI values are retained even when a gate fails. No algorithm selection or
hyperparameter adaptation based on these results.

## Populations and budgets

Full: train 122000..122009, development 123000..123019, evaluation
124000..124049. Smoke: train 125000, development 125020, evaluation
125010..125012. Historical registry includes previous declared/manifest seed
panels through priority/allocation v2. Sealed 201..300 and 601..700 remain shut.
Role-separated SHA256 config, initial, hazard, action and minibatch streams.
Same training config/initial/shock panel across arms; only price differs.

Full uses the v2 randomized N=2/3/4, K=1/2, H=12 training population and PPO
settings unchanged: hidden64, 120000 physical transitions per arm/seed,
rollout480, minibatch120, four epochs, gamma=lambda=1, LR3e-4, clip.2,
entropy.01, value.5, gradclip.5, reward/20. Final checkpoint only; no resume,
early stop, teacher labels, test training, tuning or best-seed selection.
40 models, 4.8 million transitions, 400000 episodes, 10000 rollouts and
160000 optimizer updates. CPU single thread.

Smoke: hidden16, 48 transitions/model, H=4, rollout24, minibatch12, two epochs;
4 models, 192 transitions, 48 training episodes, 32 optimizer updates.
These tiny runs test engineering, not the scientific hypotheses.

Eight test families: the seven v2 families plus n4_pressure_long (same n4
pressure config, H=24 full/H=8 smoke). Long horizon is OUT OF DISTRIBUTION,
secondary only; it probes terminal truncation and cannot establish steady-state
performance. No added terminal repair credit/penalty.
Full: 16000 learned test episodes, 800 development, 420 heuristic episodes
(400 test +20 development), 41 exact rows. Smoke: 96 test, 4 development,
25 heuristic (24+1), 5 exact. Exact N2 DP uses the ORIGINAL base objective,
unrestricted matching, all nine healthy initial ages equally weighted, H=5
full/H=3 smoke. Each policy still sees its own waiting price. This floor and
regret decomposition describe base cost only, never a waiting-constrained
optimum or an N3/N4 ceiling. No exact labels used by PPO.

## Artifacts and stopping rule

A fresh timestamped directory, nonempty output refusal, clean Git full gate,
source snapshot/hash (including reused modules), runtime/revision, plan and
historical registry. `manifest.json`, `seed_audit.json`, `benchmark_config.json`,
`resolved_config.json`, `evaluation_configs.json`, `parameter_counts.csv`,
`training_episodes.csv`, `training_progress.csv`, `training_coverage.json`,
`episodes.partial.csv`, `episodes.csv`, `development_episodes.csv`,
`reference_episodes.csv`, `decisions.csv`, `reference_decisions.csv`,
`machine_metrics.csv`, `reference_machine_metrics.csv`, `coordination.csv`,
`paired_seed_metrics.csv`, `summary.json`, `exact_small_metrics.csv`,
`small_oracle_actions.csv`, `small_oracle_audit.json`, per-arm/seed model.pt.

Episode objective is the arm's TRAINING objective. `base_cost`, `waiting_cost`,
`overdue_waiting_ticks`, `service_adjusted_cost`, `overdue_ticks_per_machine`
make cross-arm comparisons explicit. Tick traces contain beginning state,
action sequence, economic components and overdue count. Machine rows record
maximum wait, terminal failure/wait, service starts and deferral exposures.

Abort on invalid assignment, budget/schema/key discrepancy, return or cost
reconciliation error, nonfinite loss, checkpoint mismatch, exact residual or
seed violation. Preserve FAILED manifest and partial outputs; no silent retry.
Smoke must finish all counts/audits with save/load policy equality, source hash,
PPO probability/MC reconciliation, equal capacity, identical physical training
populations, and decision-trace replay. Tests must include penalty boundary,
zero-price physics equivalence, technician-head symmetry/context information,
sampling/logprob replay, deterministic tie behavior, and statistical gates.

## Interpretation

A context gain identifies value of this representation in this PPO budget,
not proof that MARL is required. A waiting improvement at excessive economic
cost is a tradeoff, not success. A waiting failure does not show the underlying
constraint is infeasible. The local control has matched allocated parameters,
but its zero-input branch has fewer effective features. The original v2 model
is not a matched control because decoder and dimensions now differ.

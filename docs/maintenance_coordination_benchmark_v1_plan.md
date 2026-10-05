# Maintenance coordination benchmark v1 — locked protocol

Locked before implementation/smoke, 2026-10-05. This study can reject the value
of learning or coordination. Local execution is a synthetic sensitivity approved
by the user; no measured operational communication constraint exists yet.

## Questions and arms

Same completion-based WaitingEnv physics, economic costs PM=1, CM=2,
failure=15, downtime=6; failed machines wait in a common pool, technicians
passive, restoration only on completion. All learners train on base cost plus
12 per overdue failed-wait tick, waiting limit B=4. This is a soft price, not
a guaranteed constraint. Economic/base cost remains a separate reported metric.

Five fresh trained arms, same encoder width and transition/optimizer budget:
* central_ar: full-state autoregressive ContextActorCritic PPO scheduler.
* central_proposal: full-information machine actors with simultaneous proposals.
* ippo_local: shared local actor, individual local critic, team reward (IPPO).
* mappo_local: identical local actors/actions, centralized shared critic (CTDE).
* mappo_message: same centralized critic plus one round of learned 4-scalar
  messages per compatible machine-technician edge, mean peer aggregation.

IPPO is also MARL. These are PPO adaptations with complete-episode Monte Carlo
returns, not claims to replicate every published MAPPO/IPPO implementation.
Method motivation: https://arxiv.org/abs/2103.01955 and
https://arxiv.org/abs/2011.09533 .

Proposal models have identical parameter count and initialization for each seed;
unused full/message branches are retained. Active capacity differs and is stated.
Each machine simultaneously chooses STOP or an idle compatible technician.
A fixed public arbiter awards a contested worker to the highest reported local
urgency (failed status, wait, age, imminent hazard), then smallest machine ID.
No second proposal or reassignment. This shared execution arbiter is disclosed.
All four proposal arms use it; central_ar uses its original sequential reservation.

Local actor sees own six machine features and public ID, its own edge durations
and restoration, public technician idle/remaining time (occupant failure bit
removed), counts, remaining horizon and common cost/hazard parameters. It never
sees other machine states or compatibility. Central critic can see all states.
Only message/full actor branches consume peer embeddings. Learned messages also
reveal participation/compatibility through their presence; no privacy claim.
Message-off evaluation of the same checkpoint is an OOD diagnostic, not a
separately trained equal-budget control. Record scalar message counts, not
unmeasured distributed latency, energy, privacy, or dollar communication cost.

## Strong references and selection

risk_skill_rule; deadline_matching (optimal partial matching of deadline/risk/
service bids); lookahead_matching (optimal partial matching of isolated-machine
expected marginal gains). Matching uses bitmask dynamic programming with stable
lexicographic ties, optional STOP, no negative-gain dispatch.
Lookahead computes exact single-machine adaptive wait/start costs up to six
remaining ticks, including completion restoration, failure probabilities, PM/CM,
downtime and overdue price, then compares starting each idle worker now against
waiting now. Future worker availability is assumed unconstrained; it ignores
future competition and is NOT a joint MPC/oracle. It sees no future random shocks.
These heuristics can communicate local bids and are legitimate distributed
references. They receive no policy training transitions or test tuning.

Select reference using only development episodes: lowest mean base cost among
references whose empirical machine-record p95 wait <=4. If none eligible, record
infeasibility and do not pass learned-versus-reference hypotheses. Save all
reference test results, but never reselect on test. All methods share price/B.

## Confirmatory hypotheses

Four paired contrasts; independent unit is training seed (10), each seed's score
is the equal-weight mean of four primary family means over the same 50 eval seeds.
H_learning: central_ar vs development-selected reference, base cost.
H_multiagent: mappo_message vs selected reference, base cost.
H_critic: mappo_local vs ippo_local, service_adjusted_cost.
H_message: mappo_message vs mappo_local, service_adjusted_cost.
Each must reduce its primary cost by >=5%, win >=8/10 train seeds, and have
upper bound <0 on a paired difference two-sided 98.75% Student-t interval
(Bonferroni four contrasts, t(df=9)=3.110934823179475). For each learner, pooled
and each-training-seed pressure-family p95 waiting must be <=4; base cost and
nominal base cost must not exceed 110% of control. Report each condition.
Reference eligibility is additional for the first two hypotheses. Smoke labels
are ENGINEERING_ONLY, never PASS scientific claims. Incomplete full panels abort.
No composite 'MARL necessary' claim: even positive results support conditional
benefits under the declared information/action contracts. A real distributed
execution requirement and its costs are still needed for application necessity.

## Panels, budgets and stopping

Full train seeds 128000..128009, development 129000..129019, test 130000..130049.
Smoke train 131000, development 131020, test 131010..131012. SHA256 role-separated
config/environment/action/minibatch streams; every arm uses identical physical
training populations per seed and eligibility-independent hazard grids.
Historical declared/readable seed registry and reserved 201..300,601..700 stay
closed. No old checkpoint, exact labels or test configurations enter training.

Full per model 120000 physical transitions, rollout480, epochs4, minibatch120,
hidden64, Adam3e-4, clip.2, entropy.01, value.5, gradclip.5, reward scale20,
gamma=lambda1, 4000 optimizer steps. Five arms ×10 seeds: 50 models,
6,000,000 transitions, 500,000 H12 episodes, 200,000 optimizer steps.
IPPO/MAPPO ratios clipped per-agent; policy/value/entropy averaged per physical
row over actual agents (1/N), with identically weighted advantage normalization.
Busy agents take forced STOP; these transitions still train the critic.
Central_ar clips its autoregressive joint sequence ratio; this factorization
is an explicit difference, controlled by central_proposal descriptively.
Final checkpoint only, no early stopping, selection or resume. Smoke uses
48 steps, H4, rollout24, epochs2, minibatch12, hidden16: 240 transitions,
60 episodes, 40 optimizer steps over five models.

Primary N3/N4 nominal and pressure. Secondary small N2, N5 pressure,
N5/K3 sparse, N4 pressure H24. Eight test families; same development N3/K2.
Full 20,000 learned test, 1,000 learned development, 1,260 reference episodes,
4,000 message-off secondary episodes; smoke 120 test,5 development,75 references,
24 message-off. Record per-machine and step-level traces for test/reference/
message-off; only development episode/machine rows needed for reference selection.
Exact N2/K2 economic DP evaluated on all healthy age initial states for 50 models
and three references (53 rows; smoke8). Base-cost diagnostic only; not an SLA
constrained optimum, and no ceiling conclusion on N3/N4. States capped25,000.

## Secondary metrics and artifacts

Report all component costs, failures, PM/CM starts, failed/unavailable ticks,
empirical machine wait p95/max and violation fraction, terminal failed never
started, collision contenders/rejections, avoidable requested-but-unassigned
capacity from maximum feasible matching, proposals and accepted grants,
message scalar payload. Critic-only, message-only, global-proposal vs local,
central_ar vs proposal factorization and message-off are separately interpretable.

Fresh timestamped output, refuse nonempty directories; full requires clean git.
manifest.json has protocol, revision, runtime, settings, source hashes, expected/
actual counts, RUNNING/COMPLETED/FAILED; immutable source_snapshot including
plan+seed registry; benchmark/resolved/evaluation_configs.json; training_episodes/
progress.csv, per-model checkpoint model.pt, parameter_counts.csv,
training_coverage.json, reference_selection.json, episodes.partial/episodes.csv,
development_episodes.csv, reference_episodes.csv, message_off_episodes.csv,
machine_metrics.csv, decisions.csv, exact_small_metrics.csv,
small_oracle_actions.csv and small_oracle_audit.json,
paired_seed_metrics.csv, summary.json. Partial artifacts survive interruption.

Engineering gates: tests, local Mac smoke, legal matching, raw-proposal replay,
team-return and per-agent sampled-probability replay, matching initialization,
identical physical training population, finite gradients, checkpoint reload,
complete panel/budget, economic and machine-level reconciliation, seed audit.
Training/evaluation use tqdm. Follow docs/experiment_workflow.md: verified code
commit/push, then only hand off exact compound run and partial-preserving rsync;
no SSH or remote run/transfer on the user's behalf. CPU-only implementation.

Artifact filename clarification before handoff: the existing exact DP exports
action-value CSV plus audit JSON; filenames above corrected to match those
exports. Hypotheses, observations, seeds, budgets, metrics and gates unchanged.

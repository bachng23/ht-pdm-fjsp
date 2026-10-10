# Maintenance opportunity cost v1 — locked protocol

Material Passport: academic-research-suite / experiment-agent / plan;
2026-10-10; UNVERIFIED until execution; protocol v1.

## Question and hypotheses

Does technician assignment have material value beyond timing/priority, and
does a simple scarcity-aware rule remove that headroom? This is a simulator
diagnostic, not a new RL algorithm or a claim of real-world calibration.

H-mechanism: with A failed, T0 faster on A, T1 slower on A, and only T0
eligible for B, the expected cost difference between assigning A to T0 and
T1 depends on B's age and can change sign. Both diagnostic root actions
serve exactly A. The continuation is the same unrestricted exact optimum.
A sign change is conditional evidence; not its prevalence in normal episodes.

H-assignment (primary): in skill_mask N4/K2/H12, flexible joint rollout
has lower expected priced cost than fixed-allocation joint rollout.
H-headroom (primary): flexible joint rollout has lower priced cost than
the strongest fast rule selected exclusively on development.

## Physics and information

Inherit WaitingEnv pending_matching_overdue_waiting_v1 unchanged from
verified solver-comparison commit 4b1cffb. Non-FIFO partial matching; zero
duration encodes ineligibility; nonpreemptive service; no degradation while
servicing; restoration on completion. CM waiting ends at service START;
waiting_limit=4 is a soft price, not a completion deadline.
Costs: PM=1, CM=2, unavailable=6, failure=15, overdue waiting=12/tick.
Finite undiscounted cost with zero terminal value; report terminal pending,
failed and still-servicing counts. This objective can reward deferral near
the boundary; no claim of fairness or infinite-horizon performance.

Full episode profiles expand the historical generator: base durations
2..5, failure_age 3..5, p in {0.25,0.45,0.6}; max_age=8. Conditions:
homogeneous (both workers interchangeable, restoration .75), specialized
(preferred worker base duration/restoration 1, other base+1/.75), skill_mask
(specialized with two machines restricted to the common worker). Duration,
hazard and skill variability are public. Report eligibility edge count,
sum of best per-machine service rates, and number of single-option machines.
Cross-condition differences include effective-capacity changes and are not
a pure causal effect of topology. Every machine and technician has an edge.

Configuration cohorts are paired across conditions. Full development and
test configurations are disjoint EVEN after relabeling machines/technicians
within each condition. They are not claimed disjoint from every historical
physical system. New seed ranges are checked against inherited registries.

## Controllers and development selection

Rules: risk_matching, deadline_matching, adaptive_isolated_matching, and
scarcity penalties lambda in {0,1,3,6}. Scarcity rule starts with isolated
dynamic marginal benefits; subtract lambda * duration(m,j) * sum over other
nonservicing machines of [risk weight / number of eligible FREE workers],
when j is a free eligible option for them. Risk weight is 1 for failed
machines, otherwise p when age+1 reaches failure threshold, else 0.
Select best scarcity lambda and best overall fast rule separately per
condition by mean development priced cost; ties by frozen candidate order.

Joint planners: 128 independent forecast scenarios per physical decision,
common forecast seed indexed by cohort/shock/time (no controller identity),
full remaining episode lookahead; common adaptive_isolated_matching
continuation. Fixed planner permits one assignment for each served machine
subset: minimum sum duration/restoration, ties lexicographic. Flexible
planner permits every feasible partial matching. Inherited planner evaluates
each candidate using the SAME forecast shock tape and continuation.
Fixed restriction applies to the root at every real decision; forecast
continuation is common and unrestricted. Therefore this is not a comparison
of globally optimal fixed vs flexible policies. Forecasts are independent
of actual future failure tape. Approximate planners are quality references,
not certified optima. No tuning of forecast budget on test, no latency win
claim from the tiny enumeration implementation.

Report five controllers: isolated, selected scarcity rule, selected best
fast rule, fixed rollout, flexible rollout (duplicates retained as explicit
roles if development chooses the same rule).

## Splits, budgets and stopping

No training, optimizer, checkpoint or training seeds (NOT APPLICABLE).
Full: 32 development cohorts x 8 shocks; 64 sealed test cohorts x 32 shocks;
3 conditions, H12, five test roles = 30,720 test episodes / 368,640 decisions.
Development evaluates seven rule candidates = 5,376 episodes.
Seeds: development configurations 9,100,000..9,100,031;
development shocks 9,200,000..9,200,007;
test configurations 9,300,000..9,300,063;
test shocks 9,400,000..9,400,031.
Smoke uses disjoint 9,500,000/9,600,000 ranges, 2 development cohorts,
1 engineering evaluation cohort, 2 shocks, H4, 8 forecast scenarios.
Smoke never simulates or reads full test outcomes. Structural split audits
may inspect test configuration metadata only, not run its simulator.

Exact diagnostic: N2/K2, A failed, A ages {3,5}, B ages 0..5,
service times ((2,3),(3,0)), restoration ((1,1),(1,1)), threshold=4,
p in {.25,.6}; horizons {6,8}, 48 full states. Smoke: A age3, B age{0,3},
p=.6, horizon4, two states. Zero terminal value, same original costs.
Exact enumeration integrates only active independent Bernoulli hazards.
One solver cache per (config,horizon), hard cap 250,000 unique value states;
cap exhaustion fails the run and preserves artifacts; never silently fall
back to Monte Carlo. No parameter search to force a sign reversal.

Run all locked episodes. No performance-dependent early stopping, extra
seeds, best-seed filtering, resume, or automatic crash retry. A crash is
FAILED with partial output. A new attempt needs a new directory. All loops
show tqdm. CPU, one BLAS/OpenMP thread, no torch inference or training.

## Analysis and research decision

Positive paired cost differences favor flexible rollout. Average paired
shock differences inside each profile, then bootstrap independent profiles
(20,000 resamples with a fixed analysis seed). Primary family has TWO
contrasts at skill_mask: use 97.5% two-sided percentile CIs (Bonferroni).
Report paired means, relative improvement (ratio of panel means), profile
win fraction, all per-profile differences. Other conditions and contrasts
use 95% descriptive CIs and are EXPLORATORY. Uncertainty is conditional on
the fixed shock panel; episodes are not training replicas. No p-values.

Primary learning-investment screen: lower CI for relative improvement over
fixed planner AND best fast rule exceeds 1%. This 1% is a research budget
screen, NOT a calibrated application requirement. Mechanism sign reversal
must also be observed in the locked grid. Service guard: mean flexible
overdue-wait ticks must not exceed best-rule mean by .25/episode and mean
terminal pending must not exceed it by .1/episode; guards are descriptive,
not safety certifications or extra significance claims. If CI overlaps the
margin, report INCONCLUSIVE rather than equivalence. If robust benefit is
only timing/priority, simplify the proposed learning method.

Secondary: base/maintenance/failure/unavailability/waiting cost, failures,
overdue-request counts, wait mean/p95 on served requests with served and
censored counts, observed waiting for ALL requests, terminal backlog,
latency p50/p95/p99 (feature/scoring/matching included), candidate counts,
forecast sample counts. Latency is closed-loop and hardware dependent.
Exact diagnostic saves both conditional assignment Q values, their
difference, regret against unrestricted exact optimum, globally optimal
root action, and best action under fixed candidates; never call approximate
rollout reference gaps optimality gaps.

## Engineering gates and artifacts

Test exact DP against brute force, scalar/batched transitions, common
candidate scores, masks, fixed candidates, relabeling invariance of exact
VALUE (tie-selected actions may differ), split/seed separation, replay,
request censoring, statistical pairing, immutability and failed manifests.
Run repository tests and macOS smoke before commit/push. Smoke must complete
all rows; finite/reconciled costs; valid actions; replay PASS; actual counts
match; progress visible; full test unopened; no model reload requirement.

artifacts/<new UTC run id>/ contains plan.md, resolved_config.json,
seed_registry.json, manifest.json (commit/dirty/runtime/hardware/status/
hashes/audit/counts), source_snapshot/, development_episodes.csv,
development_selection.json, episodes.partial.csv, episodes.csv,
decisions.csv (state/action/next state/components), requests.csv,
latency.csv, action_values.csv, coordination.csv, profile_differences.csv,
summary.json. Console log is beside the directory. Journals flush each row;
final CSVs/manifest are hashed. A nonempty directory is rejected.

Follow docs/experiment_workflow.md: tests -> Mac smoke -> commit/push ->
user executes the exact lab command -> user executes rsync -> analysis on
downloaded completed artifacts. Assistant never SSHes or runs full/rsync.

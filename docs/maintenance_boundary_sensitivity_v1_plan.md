# Maintenance boundary sensitivity v1 — locked protocol

## Material Passport

Origin: academic-research-suite / experiment-agent / plan; 2026-10-10;
UNVERIFIED until execution; protocol v1. Follow-up to opportunity-cost v1,
whose test is now open. No outcome from the new sealed panel has been viewed.

## Question

Is the planner's low H12 cost accompanied by deferred corrective maintenance
because of zero terminal value? Does cost headroom over strong rules persist
away from the episode boundary, and is flexible assignment useful there?

Preserve WaitingEnv physics and every original coefficient: N4/K2, PM=1,
CM=2, unavailable=6, failure=15, waiting_limit=4, waiting_price=12.
Partial matching and nonpreemptive completion remain unchanged. Report CM
waiting ending at START separately from total failed machines, including CM
already in service. No SLA/deadline constraint is introduced.

## Factorial design and terminal treatment

H in {12,24,48}; conditions homogeneous, specialized, skill_mask. Four
planners: fixed_zero, flexible_zero, fixed_tail, flexible_tail; one best_rule
selected on development for each condition/H. No RL/training/checkpoints.

Zero planner scores expected priced costs until the evaluation cutoff H.
Tail planner scores the SAME pre-cutoff costs/forecast continuation, then
adds expected cost of L=12 real model ticks under frozen deadline_matching.
This is a policy-evaluation terminal value, not an arbitrary backlog penalty
or an exact infinite-horizon value. It includes ongoing service, subsequent
degradation, new failures, PM/CM and overdue costs on ALL machines. The
continuation after H is shared by every controller and fixed before test.

Within each candidate forecast before H, continuation is the inherited
adaptive_isolated_matching with horizon H-t-i, IDENTICAL for zero and tail;
after H it switches to deadline_matching for L ticks. Thus adding terminal
cost does not silently change pre-cutoff continuation. 128 scenarios per
decision; full remaining H lookahead (no search-horizon cap). Fixed vs
flexible root restrictions are inherited unchanged. Common forecast tapes
are indexed by cohort/shock/physical time and exclude controller/H identity;
pre-cutoff portions shared whenever forecast lengths differ. Forecasts are
independent of actual future shocks. All candidates are feasible matchings.

For actual evaluation, every trajectory continues for L=12 ticks after H
with the SAME deadline rule. Dynamics and in-progress jobs continue without
reset. Actual initial/failure tapes share physical machine/time prefixes
across all controllers and H. Report C_H, observed continuation C_tail,
C_H+C_tail, cutoff backlog and post-tail backlog separately. The observed
tail is one held-out sample of terminal cost, not a fresh optimized policy.
Zero and tail optimize different finite objectives; compare the controllers
using common reported objectives rather than calling cost columns equal.

## Profiles, splits, rules and budgets

Reuse v1 profile-generation distribution (base durations 2..5; thresholds
3..5; p=.25/.45/.6). Different conditions still change effective capacity;
no pure causal topology claim. Exclude ALL 96 prior development/test physical
cohorts within each condition, invariant to machine/worker relabeling and
ignoring horizon. Their signatures and provenance are committed in
configs/maintenance_boundary_sensitivity_v1_profile_exclusions.json, so the lab
run has no dependency on old artifact directories. The separate
seed_registry.json config records ONLY this experiment's raw seeds, so inherited
audits do not misinterpret prior-seed exclusions as newly reused seeds.
New dev/test profiles
also must be physically disjoint ignoring H. No unique-instance claim at
arbitrary sizes; N/K remains fixed.

Full: 16 development cohorts x 4 shocks, 64 new test cohorts x 8 shocks.
Development evaluates the seven inherited rule candidates, including
scarcity lambda {0,1,3,6}, at every H/condition: 4,032 episodes. Best rule
selection metric: extended physical cost at H12/H24; interior cost/machine
tick at H48, using ticks [12,36). Ties use frozen candidate order. Tail
continuation remains deadline_matching, regardless of which rule wins.
Do not change the selected rule on test. All training seeds are N/A.

Development config seeds 10,100,000..10,100,015; shocks
10,200,000..10,200,003. Test config seeds 10,300,000..10,300,063;
shocks 10,400,000..10,400,007. Analysis seed 10,700,000.
Smoke uses disjoint 10,500,000/10,600,000 ranges, 2 dev cohorts/1 evaluation
cohort, 2 shocks, H={4,8}, L=2 and 8 forecast scenarios. Smoke outcomes are
engineering only. Full test configuration metadata may be checked for
splits, but its outcomes never run locally.

Full test: 23,040 episodes; 645,120 pre-cutoff decisions + 276,480 tail
decisions = 921,600 physical decisions. Development: 161,280 physical ticks.
All budgets fixed, no performance-dependent stopping or extra seeds, no
automatic retry/resume. A failed/interrupted run retains partial artifacts
and FAILED manifest. New attempts use new UTC directories. CPU/thread=1,
tqdm for development and paired multi-seed evaluation; no training stage.

## Primary hypotheses, uncertainty and research screens

Primary condition is skill_mask. Average paired shock differences within
each of 64 profiles; bootstrap profiles 20,000 times with frozen seeds.
Three primary contrasts use two-sided 98.333333% percentile CIs (Bonferroni
family of three); uncertainty conditional on the fixed shock panel.

P1: H12 fixed_zero cutoff pending minus fixed_tail cutoff pending. Hypothesis:
adding modeled terminal continuation reduces deferred CM. Material research
screen: lower CI > .1 machine/episode. Also report total failed, in-progress
CM and observed extended cost to avoid crediting starts as completed repairs.

P2: H48 best_rule minus fixed_tail INTERIOR priced cost per machine-tick,
using ticks [12,36). Hypothesis: headroom survives away from the boundary.
Material screen: lower CI of relative improvement >1%. Report C_H, extended
cost and all component costs separately; an interior window is not stationary
or infinite-horizon performance and may reflect earlier decisions.

P3: H48 fixed_tail minus flexible_tail on the same interior cost. Hypothesis:
flexible assignment adds material value under the new boundary treatment.
Material screen: lower relative CI >1%. If upper CI <1%, label below research
margin; if CI spans margin, label inconclusive; zero overlap is not equivalence.

Other H/condition/contrast metrics get descriptive 95% CIs and exploratory
labels. These margins are research investment thresholds, not factory-calibrated
requirements. Describe total cost and backlog together. A timing-learning
follow-up needs P2 plus descriptive service guard: fixed_tail mean INTERIOR
pending and total failed must not exceed best_rule by .1 machine, and interior
overdue ticks per physical tick must not exceed best_rule by .25/12. P1/P3
diagnose mechanism and architecture direction; they are not required to make
P2 true. Report any guard failure and avoid cost-only deployment claims.

## Secondary metrics and artifact schema

For each episode report all priced/economic cost components before/after H,
interior components, pending/total failed/servicing/CM in service at H and
H+L, time profiles, requests served and censored at BOTH cutoffs, observed
wait and overdue counts. Carryover requests pending at H are followed through
the tail with request IDs preserved, recording whether service starts and
remaining censoring; service start is not repair completion. Record shared
physical/forecast seed provenance and feasibility/scalar-batch audits.
Latency is online closed-loop with separate tail-rule timings and counts;
tail planner performs more simulation work, so no equal-compute claim.

Each new artifact directory: plan.md, resolved_config.json, seed_registry.json,
manifest.json (commit/dirty/device/runtime/hardware/status/hash/count/audit),
source_snapshot/, development_episodes.csv, development_selection.json,
episodes.partial.csv, episodes.csv, decisions.csv (phase/state/action/next/
physical metrics), requests.csv (cutoff/extended scopes), carryover.csv,
time_profiles.csv, latency.csv, coordination.csv, profile_differences.csv,
summary.json. Log beside directory. No raw-run overwrite.

Tests must check zero planner equality to inherited implementation, prefix
forecast sharing, exact finite deterministic tail scores, additive base/tail
cost reconciliation, ongoing-job preservation, old/new split exclusion,
cutoff vs extended censoring, bootstrap units and three-contrast correction,
full budget counts, failed/interrupted manifests and nonempty-output rejection.
Run repository tests and CLI macOS smoke; verify counts/hashes/audits and
full_test_outcomes_opened=false before commit/push. Then hand off exact lab
compound command and Mac rsync command per docs/experiment_workflow.md;
assistant does not SSH, launch full, or transfer results.

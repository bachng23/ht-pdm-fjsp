# Maintenance solver comparison v1 — frozen scientific protocol

ARS experiment-agent; user-authorized implementation, 2026-10-07 Asia/Taipei.
Parent physics/source: 693c348. Status: protocol locked before smoke/full results.
Smoke is engineering only. This study permits planner or centralized RL to win.

## Problem and scope

Finite-horizon maintenance dispatch, N4, passive heterogeneous technicians;
select preventive/corrective starts and a feasible partial matching. Reuse
WaitingEnv without changing service completion, occupancy, hazards, PM/CM costs,
failure price15, unavailability price6, wait bound4 and overdue price12.
Soft waiting price is not a hard SLA or a plant-calibrated economic valuation.
STOP is legal; pending demands persist. This is not a full FJSP/makespan study.
All methods see known current model parameters, never actual future shocks.

## Controllers and information

Controls: risk_matching, deadline_matching, lookahead_matching,
adaptive_isolated_matching, joint_rollout and fixed_allocation_rollout.
Reference selection is development-only, minimum priced cost among four rules.
Isolated DP values plus exact matching remain a CENTRALIZED nonlearning baseline.
Rollouts use32 common forecast scenarios and the selected rule continuation;
they are achievable model-based comparators, not optimality bounds.

Five learned arms use identical initial tensors/hidden64 and physical budgets:
central_matching_ppo, central_fixed_allocation_ppo, independent_local_ppo,
cooperative_local_ppo, cooperative_full_ppo. Central arms inherit the full-state
categorical matching/restricted-allocation heads. Serial machine arms visit all
four machines in public rotating order, condition on reservations, and choose
WAIT or a feasible idle worker; all starts execute simultaneously at one physical
tick. Reservation masks cannot erase a request or advance physical time.
Serial masks fix the prior simultaneous-worker symmetry defect without adding
graph/mixer/attention modules. Likelihood uses the sampled serial action sequence,
including teacher-forced reservation prefixes when reevaluated for PPO.

Local actors see own state/edges, public worker status, static costs/hazard,
horizon and public order/reservations, but no other machine ages/failures/waits
or private edge profiles. Independent learning uses per-machine cost returns,
local critic and per-machine PPO ratios; cooperative local uses team return,
central critic and joint-sequence PPO ratio. This contrast tests that package,
not a pure isolated causal effect of critic alone. Full cooperative actor sees
full current state, matching centralized information. It tests policy
factorization, not a demonstrated requirement for decentralized deployment.
Local policies still require resource telemetry and serial reservation messages;
they are not communication-free. Equal total parameters are not equal active
capacity; report actor/critic contracts rather than claiming identical capacity.
Independent actors share parameters; own-cost gradients are pooled across
machines, without team-return or other-agent action derivatives. Cooperative
entropy bonus is the sum of conditional entropies along sampled prefixes, not
an exact enumeration of joint-policy entropy. Serial evaluation invokes actor
only; critic is training/development-value machinery and is not required for
local action execution. Central heads retain inherited critic diagnostic work,
which is included in their measured latency. This asymmetric overhead is stated
explicitly; do not claim an intrinsic actor-only algorithm speed advantage.

## Profile and seed separation

Enumerate128 labeled specialized profiles (16 duration vectors,2 common workers,
4 exceptional machines). Split by fixed seeded permutation into80 train,16 dev,
32 test profiles. Report all32 test profiles with paired shocks. This tests new
labeled physical combinations within a finite generator, not plant or arbitrary
permutation-class generalization. Homogeneous projections may coincide across
splits; record that explicitly. No full test rollout/checkpoint evaluation on Mac.
Conditions: specializedK2 primary, homogeneousK2, nominal homogeneousK4 and
skill_maskK2 sensitivity; horizons12 primary/24 sensitivity. Train80% specialized,
20% nominal, H12. No unseen-size/scale or measured network deployment claim.

Full training seeds160000..160009; configuration and shock RNG counters use
solver_comparison_v1 role namespaces. Dev cohort seeds161000..161015, shock
seeds162000..162009; test cohort seeds163000..163031, shock seeds164000..164019.
Smoke train165000; dev165020; test165010; dev shock165030, test165040..165042.
Profile permutation seed166000; smoke uses three internally disjoint IDs from
the full training partition and never opens the full sealed test panel. Registry includes historical declared
panels and previous heterogeneous full/smoke panels; reject numeric overlaps.

## Budget and selection

Each full model491520 physical steps,40960 complete H12 episodes,320 rollouts1536,
32 vector envs, minibatch192,4 epochs =10240 optimizer steps. PPO Adam3e-4,
clip.2, entropy.01,value.5,gradient clip.5,gamma1,reward scale20,one CPU thread.
Checkpoint at80/160/240/320, initialization diagnostic only. Select lowest dev
specialized cost subject to nominal base<=110% selected nominal rule and waiting
violation<=selected specialized rule+2pp; retain final if none feasible and mark
DEVELOPMENT_INFEASIBLE. No seed dropping, no early stop/resume, no teacher labels.
Freeze every selection before test. Fixed budget stops regardless of outcome;
do not retune or add architectures to rescue failed gates after seeing test.
Smoke hidden16,H4/H6,256steps/model,rollout128,B8,minibatch32,2epochs/checkpoints1/2,
4 forecast scenarios; five models. All smoke verdicts ENGINEERING_ONLY.

## Hypotheses and inference

Four confirmatory contrasts on specializedK2/H12, Bonferroni family4,
two-sided98.75% paired t intervals. Economic/guard thresholds are fixed:
H_coordination: joint_rollout vs adaptive_isolated_matching (joint lookahead vs
isolated valuation, both with centralized matching; not coordination vs none), priced reduction>=3%,
base reduction>=2%, >=80% of32 test profile means win, upper CI<0.
H_learning: central_fixed_allocation_ppo vs dev-selected rule, priced>=5%,
base>=3%, >=8/10 replica means win, upper CI<0.
H_cooperative: cooperative_local_ppo vs independent_local_ppo, priced>=2%,
>=8/10 replica means win, upper CI<0.
H_decomposition: cooperative_full_ppo vs central_matching_ppo, priced>=2%,
>=8/10 replica means win, upper CI<0.
Every contrast additionally requires left nominal base<=110% selected rule
and left specialized wait-violation<=selected rule+2pp.
Coordination CI uses32 profile means conditional on common shock panel; learned
CIs use10 paired replica means conditional on common32-profile/shock panel.
Do not treat all episodes as independent training replicas. Also report paired
profile summaries and uncorrected95% descriptive CIs for economic/assignment/
information contrasts. H24/other conditions are descriptive, never new primary.
FAILED gate is not equivalence or proof an algorithm family cannot succeed.

## Metrics and artifacts

Primary priced cost and base cost; components, failures, service occupancy,
per-machine waiting, request-level wait-to-start with terminal right censoring,
overdue fraction, terminal pending, utilization and contention. Report request
waits separately from time-sampled waits. No finite-horizon absence-of-starvation
guarantee. Serial reservation diagnostics and physical matching must reconcile.
Record end-to-end decision latency (feature extraction through decode), p50/p95,
episode totals, training/development walltime, model parameters. Logical wire
payload counts include telemetry, dispatch and serial reservation rounds, with
fixed documented numeric serialization. These are modeled byte counts, not
measured network latency/energy. Do not turn serial in-process calls into claims
about parallel deployment speed. Compare quality/compute descriptively; do not
declare economic optimality or amortization without deployment request volume.
Online timings are observed mixed-cache, in fixed controller/replica order.
Additionally benchmark every selected controller on the same48 development
states (first dev cohort/shock, four conditions, H12); clear isolated/gain caches
before each controller, do one untimed full-state-set warm-up, then three timed
passes. Save per-state timings and cache info. Smoke uses16 states. This is a
steady-state common-state CPU harness benchmark, not a network measurement;
local-policy frontend still constructs shared tensors/masks, included in timing.
Failure at final boundary H creates a zero-duration right-censored request;
service started but unfinished is an observed wait-to-start event. Never report
served-only p95 as p95 of all demands.

New timestamped directory, refuse nonempty output. RUNNING/COMPLETED/FAILED
manifest with source/config/protocol/lock hashes and Git/runtime provenance;
training progress/episode CSVs, initial/fixed/selected checkpoints, dev/reference
selection, profile registry, seed/budget audits, test episode/decision/machine/
request CSVs, contrasts and latency/communication summaries. Partial CSV flushed;
FAILED preserves artifacts. tqdm during training and multiseed evaluation.
Unit/integration tests plus local Mac smoke before commit/push. Then human lab
run and rsync handoff only, as docs/experiment_workflow.md. Never SSH/launch/rsync.

## Engineering clarifications before full handoff

Added during implementation/smoke: explicit own-return parameter-sharing
semantics, no critic execution in serial inference, common-state timing/cache
policy, per-phase walltime and final-boundary request censoring. These clarify
the executable/measurement contract; no full outcomes were opened, and no
hypothesis, scientific budget, seed panel, threshold or checkpoint rule changed.
Initial smoke artifacts are retained separately from the final verified smoke.

Full counts:50 models,24,576,000 training steps,2,048,000 training episodes,
512,000 optimizer steps,82,560 development episodes,286,720 test episodes,
5,160,960 test decision rows and1,146,880 machine rows, plus request/latency logs.
This is larger than the prior heterogeneous run; reserve multiple GB for raw
artifacts. No precise runtime estimate is inferred from tiny engineering smoke.
CLI: `python -m ht_pdm_fjsp.maintenance_solver_comparison --profile smoke|full
--device cpu --output-dir <new-directory>`. Full is Linux-only and requires a
clean committed checkout. No existing checkpoint or downloaded dataset needed.

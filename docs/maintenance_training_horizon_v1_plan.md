# Maintenance training horizon v1 — locked protocol

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: authorized implementation + local smoke
- Origin Date: 2026-10-10 Asia/Taipei
- Verification Status: UNVERIFIED scientific result
- Version Label: maintenance_training_horizon_v1
- Parent commit: 53d180a5fc18c53e850eff7278d753fbe295d647

## Question, hypothesis and scope

Does changing the training episode horizon from12 to24, at identical physical
transition and optimizer budgets, reduce skill-mask H24 cooperative cost and
its gap to centralized RL? This is a fresh six-cell controlled experiment:
central_matching_ppo, cooperative_local_ppo, cooperative_full_ppo x h12/h24.
The falsifiable hypotheses are >=2% within-cooperative cost reduction and >=25%
closure of each cooperative-minus-central gap on skill-mask H24. Any solver can
win. Coverage gains from the preceding study do not authorize a MARL necessity
claim or a new architecture.

Both regimes use covered schedule repeated every five complete episodes:
[specialized, skill_mask, specialized, skill_mask, nominal] =40/40/20%.
No homogeneous training is added. Keep WaitingEnv prices, waiting limit4, service
physics, features/normalization code, architecture hidden64, greedy dispatch,
PPO hyperparameters and final-checkpoint selection unchanged. Longer horizon
changes remaining-time inputs, visited states, return length/variance and reset
frequency together. This intervention cannot isolate horizon extrapolation,
credit assignment, number of resets or critic quality individually.

## Budget and matching

Every model:491520 physical transitions,320rollouts x1536,32vector envs,
4epochs/rollout, minibatch192,10240 optimizer steps. Complete episodes only;
gamma1 and unmodified Monte Carlo return, no truncated bootstrap or extra reward.
H12:40960episodes/model =16384specialized+16384mask+8192nominal.
H24:20480episodes/model =8192specialized+8192mask+4096nominal.
All60models:29491200physical ticks,1843200training episodes,614400optimizer steps.
Equal ticks/updates do not ensure equal wall time, information, active policy
capacity or learning difficulty. Report measured training time per cell.

Fresh initial tensors identical across six cells within each replica seed.
Training configurations/environment seeds derived from same inherited episode
index roles. The first20480episode seed identities are shared between H12/H24;
all algorithms within each regime share the entire sequence. H12's remaining
20480episodes are extra resets, not extra physical steps. Same configuration
excluding horizon and exogenous event prefix for paired episode seed; actions,
RNG consumption mapping and visited states need not match between horizons.
Store full-sequence and shared-prefix hashes and check within/between regimes
respectively. Do not claim identical full training episodes across horizons.

Save initial and checkpoints at updates80/160/240/320, all provenance checked;
use final491520tick checkpoint only, no dev selection, early stopping or retries
with seed removal. Full360checkpoint files incl.final copies.

## Seed panels and transfer labels

Train replica/initial seeds190000..190009, development shocks194000..194009,
sealed fresh test shocks195000..195019. Reserve separately from solver, decoding
and coverage registries, including their full and smoke declarations. Training
role-derived seeds must not overlap any historical/reserved panel.
Use familiar original labeled development161000..161015 and test163000..163031
profiles from80/16/32partition. These profiles are ALREADY OPEN from preceding
studies; only fresh shock seeds are sealed. No unseen profile equivalence-class
claim. Both train regimes use the original80profile train pool.

Engineering smoke replica199000, dev profile165020/shock199020, test profile165010
and shocks199040..199041. Smoke h12/h24 LABELS mean shorter/longer training4/8
respectively; evaluation4/8,hidden16,256ticks/model,128rollout,8envs,epochs2,
minibatch32,checkpoints1/2. There are64/32episodes/model, not full scientific
horizons12/24. All smoke physical profiles belong to full TRAIN pool; never
open full shock trajectories or full test profile configurations during smoke.

## Evaluation, inference and guards

Evaluate all six cells plus selected_rule and joint_rollout on4conditions x
H12/H24 x32familiar profiles x20fresh shocks. Same shock panel across algorithms,
regimes and horizons; event prefixes share seeds, policy trajectories may differ.
Rules selected per condition by minimum development H12 objective among four
inherited controls; name tie-break. Rule is also planner continuation; planner
32forecast scenarios, never access realized future shocks. No optimality bound.
Final learned dev evaluation H12 diagnostic only, no weight reselection.
Latency on SAME selected-rule development trajectories for BOTH H12/H24,
4*(12+24)=144states, one warm pass plus three measured passes/controller.
One CPU torch thread. Logical payload bytes/rounds inherited, no actual network.

Four PRIMARY tests on skill_mask H24. Unit10paired TRAIN REPLICA means,
conditional on fixed familiar-profile/fresh-shock panel. Two-sided98.75% paired-t
CI (Bonferroni familywise.05/4), report every gate and every seed; no episode
pseudo-replication. Tests1/2 = objective(h24)-objective(h12) for local/full.
PASS requires mean priced reduction>=2%, >=8/10strictly negative deltas, CIupper<0.
Tests3/4 = [(coop_h24-central_h24)-(coop_h12-central_h12)] local/full.
PASS requires baseline mean gap>0, mean closes>=25%, >=8/10negative deltas and
CIupper<0. Baseline gap<=0 =>NOT_APPLICABLE; report residual gap even on PASS.
Gap closure alone can reflect central deterioration: interpret with within-policy
and central effects, never claim MARL gain from this contrast alone.

ALL FOUR tests additionally require these descriptive safeguards for the same
cooperative policy (h24 training relative to h12 training):
- skill-mask H24 waiting violation <=baseline+0.02 and pending <=baseline+0.05;
- specialized H12 AND nominal H12 objective <=1.05*baseline;
- homogeneous H24 objective <=1.05*baseline, waiting violation <=baseline+0.02,
  pending <=baseline+0.05. This new guard is locked BEFORE full run, motivated by
  prior coverage study's homogeneous regression, not added to change old verdicts.
Guards mean no material FURTHER regression relative to h12 training, not an
absolute SLA or demonstration that homogeneous is safe for deployment.

All other conditions/horizons, central horizon effect, residual gaps, latency,
request censoring and component costs exploratory95%CIs. n10, degenerate deltas,
outliers and unchecked normality limit generalization. Any sign-flip/bootstrap
supplement is labeled posthoc and must not replace preregistered gates.
Do not move primary to H12 or change guards after data are seen.

Mechanisms: waiting price and overdue ticks; served/censored requests and exact
waiting-cost decomposition, terminal pending; base/failure/down/maintenance
costs, failures, utilization. Keep finite-horizon censoring visible: served mean
alone is insufficient, pending at horizon may be a recent arrival, lower failure
counts can come from unrepaired machines. Cost decomposition is descriptive,
not proof of a particular architecture/critic/credit-assignment cause.

## Stopping and artifacts

Full fixed budget above. Linux clean committed checkout, human-only lab full
run; no SSH, tmux or assistant transfer. tqdm training and multi-seed eval.
New UTC timestamp directory; reject nonempty output, no resume, preserve partial
files and FAILED manifest on errors. Full317440test episodes,5713920evaluation
intervals,1269760machine rows,40960development episodes,26784latency rows.
No full decisions.csv; smoke has all128test episodes/768tick traces for replay.
Smoke training1536ticks,288episodes,96optimizer steps,24checkpoint files,
40development episodes,512machine rows,1152latency rows.

Artifact schema inherited coverage runner:manifest(commit/runtime/settings,
expected/actual counts,status,hash/size inventory); benchmark/resolved_config,
seed/profile audits, source_snapshot including protocol/registries/code;
training_episodes/progress.csv with training_regime AND actual training_horizon;
training_coverage.json with full/prefix sequence digests,mixture,budgets,timing;
checkpoint_selection(final-only), reference_selection,development_episodes;
episodes.partial/episodes.csv,machine_metrics,request_metrics with regime and
training_horizon (-1 references),waiting decomposition;latency states/CSV/summary;
paired_seed_metrics,summary,regime/algorithm/seed checkpoints and model.pt.
Identity labels h12/h24 are full training regimes, actual horizon field is
mandatory to disambiguate smoke4/8. Every evaluated episode cost/overdue/request
count must reconcile. Final checkpoint payload includes actual training horizon,
feature/environment contracts and final physical budget.

## Mac verification gate and limits

Regression tests plus focused tests for exact32env complete-return boundaries,
12/24training horizon override, equal tick/optimizer budgets, common seed prefix,
matching environment shock prefixes/configurations excluding horizon, checkpoint
protocol rejection and reload, difference-in-differences and all guards, complete
grid rejection, immutable output, FAILED manifest, and Mac full-run prohibition.
Smoke process0,complete counts/progress,finite and changed weights,all artifact
hashes/request costs,selected final-only checkpoints; independently replay smoke
physical transitions. Smoke is engineering evidence, not scientific result.

Entry command:ht-pdm-fjsp-maintenance-training-horizon --profile smoke|full
--device cpu --output-dir artifacts/<new-timestamp>. Full pending after Git push.

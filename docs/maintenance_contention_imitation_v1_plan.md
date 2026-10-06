# Maintenance contention imitation v1 — frozen protocol

## Material Passport

Schema: ARS 9. Material: code experiment protocol. Mode: run preparation.
Origin: user-authorized follow-up to learning v1 (82707db). Status: PLANNED.
Locked 2026-10-06 before implementation or new scientific results.

## Question and hypothesis

Can the same centralized subset policy learn useful coordination when supplied
planner supervision, and does that initialization help subsequent PPO?
Primary H: imitation-initialized PPO reduces K2/H12 priced cost by at least 2%
relative to fresh direct PPO, wins >=8/10 paired replicas, upper two-sided 95%
paired training-seed t CI <0, while retaining nominal and waiting guards.
One confirmatory arm contrast only. BC vs direct, hybrid vs BC, teacher agreement,
root-score regret, learning curves and planner gap capture are exploratory.
Absolute economic success is separately reported using the previous locked
>=5% priced / >=3% base gain over the development-selected reference, >=8/10
wins, upper95% CI<0, nominal base<=110%, K2 waiting violation<=ref+2pp.
Do not declare this experiment a MARL test or infer a unique cause of failure.

## Environment and arms

WaitingEnv, feature contract, width64 SubsetActorCritic and full masked subset
action space are unchanged from learning v1. Identical workers, N4, H12,
80% K2 /20% K4 training mixture; K2 primary, K4 nominal. K1 and H24 are
untrained sensitivities. Same newly drawn panels and initialization seed per
replica. Fresh training only: no prior result/checkpoint imports.

- direct_ppo: unchanged PPO,491520 physical steps,320 rollouts1536,32 envs,
  minibatch192,4 epochs,10240 optimizer steps; checkpoints80/160/240/320.
- planner_imitation: offline actor-only supervised learning,80 epochs over
  12288 labeled training states, minibatch192,5120 optimizer steps. Shared
  encoders/actor train; critic head is untrained. Adam3e-4,gradient clip.5.
  No value target or teacher-score features. Save20/40/60/80 epochs.
- imitation_ppo: starts from the development-selected imitation checkpoint
  (including encoder and unchanged critic head), then the exact same PPO budget
  and configuration/environment/action RNG roles as direct. PPO optimizer is
  fresh. BC initialization is diagnostic, not selectable as a PPO checkpoint.

Imitation data is generated once per replica and reused by the hybrid, not
counted twice. All arms have identical architecture but NOT equal total compute
or information: hybrid has additional supervision and optimization. Report
teacher generation steps, forecast root scoring, optimizer steps and timings
separately. This tests supervision/initialization value, not compute efficiency.

## Teacher data and diagnostics

1024 H12 training trajectories and128 separate imitation-validation trajectories
per replica, same capacity mixture. At every current state, label all feasible
root subsets with joint_rollout using32 independent simulated forecasts and
full remaining horizon; continuation is the development-selected reference
for that capacity. Actual behavior follows teacher80%, uniform feasible action20%
using a separate RNG, to add off-expert states. Forecasts never read the real
future event stream. Counter-derived configuration/environment/planner/behavior
seeds are checked against all declared historical/held-out panels. Store full
config/hash, compact current state, candidate scores and SEs, planner seed,
accepted targets (within1e-6 of minimum), executed action, next state, cost.

Supervised loss is -log(total policy probability of accepted teacher subsets).
Forced STOP-only states remain in dataset and counts. Validate probabilities,
feasibility and physical reconciliation. No critic training in BC. Diagnostic
train/validation teacher agreement and nonnegative teacher root-score regret (all rows and separately states with >1 feasible subset)
are recorded at initial and every fixed BC checkpoint; validation is never used
for checkpoint selection or training. First32 validation rows/replica receive
an independent128-forecast teacher replay, only to describe teacher noise; these
extra labels never train the network. No DAgger or adaptive dataset/budget.
Policy inputs remain exclusively current features, never teacher scores.

## Selection and fresh seeds

Full train144000..144009; development cohorts145000..145009/shocks146000..146019;
sealed test cohorts147000..147009/shocks148000..148049.
Smoke train149000,dev cohort149020/shock149030,test cohort149010/shocks149040..149042.
Registry inherits prior frozen history plus all learning-v1 full/smoke seeds.
Training/teacher roles derive numeric seeds from SHA256. Fail on overlaps.
Finite generator may repeat profiles: independence of seed draws is not proof
of unseen physical configurations.

References: existing four rules; choose minimum development H12 priced mean
per capacity. Initial+four fixed checkpoints per arm evaluated on K2/K4,
10 cohorts x20 shocks. Initial is not selectable. Select minimum K2 cost among
nominal<=110% ref and waiting<=ref+2pp checkpoints; earlier counter breaks ties.
If none feasible use final, mark DEVELOPMENT_INFEASIBLE and retain all seeds.
Hybrid starts from selected BC; all30 selected models frozen before test.
Teacher diagnostics use separate training-derived validation, not sealed test.
Test every selected model and four references+32-forecast planner on10 cohorts
x50 shocks,3 capacities,2 horizons. Controls run once, not10 fake replicas.

## Budget, uncertainty and stopping

Fixed budgets, no early stop, test feedback, retries/resume, or adaptive epochs.
30 selected models;9830400 PPO physical steps,819200 PPO episodes,
256000 actual optimizer steps (BC51200 +PPO204800). Teacher data138240 physical
steps/11520 trajectories,122880 training labels+15360 validation labels;
320 extra independent noise probes (not additional physical steps).
Development62400 episodes (2400 refs +60000 learned initial/fixed checkpoints).
Test105000 episodes,1890000 intervals,420000 machine rows.

Primary unit: paired10 replica means on same500 K2/H12 test episodes. Pairing
covers initialization, BC data seed and shared direct/hybrid PPO RNG roles;
trajectories can diverge.95% t(df9,2.2621571628540993) CI conditional on this fixed
workload panel. No episode pseudo-replication. Report cohort heterogeneity,
all seeds, base/failure/maintenance/downtime/overdue costs, waits/violations,
utilization/backlog, runtime and all gates. Primary PASS requires above2%/8wins/CI,
nominal<=110%reference and waiting<=reference+2pp. It does not require the
separate absolute economic gate; report both, never conflate them.

Smoke: width16,H4/H6,4 forecasts,2PPO rollouts128 (256steps,16 optimizer steps
per PPO arm),8 vector envs,minibatch32,epochs2;BC16 H4 training+4 validation
trajectories,4 BC epochs/minibatch32 (8 optimizer steps),checkpoints1/2/3/4.
Noise probes first2 validation rows with16forecasts. Three models;512 PPOsteps,
128 PPOepisodes,40 optimizersteps;80 teachersteps,64training/16validationlabels,
2probes;34 dev episodes;144 test episodes,720 intervals,576 machine rows.
ENGINEERING_ONLY, no scientific pass/fail or sealed full-panel use.

## Artifacts, tests and handoff

Fresh timestamped output, no nonempty-directory overwrite; clean Git for full,
CPU-only one torch thread. Manifest RUNNING/FAILED/COMPLETED with hashes and
source/protocol/lock snapshot. Seed audit, budget, resolved configs, reference
selection, teacher_rows.csv (train/validation), teacher_noise.csv, teacher_data.pt,
teacher coverage, BC progress/diagnostics, PPO progress/episodes, per-arm model
checkpoints, checkpoint selection including BC parent hash for hybrid, episodes
partial/final, decision/machine logs, per-arm seed/cohort metrics, primary contrast,
summary. FAILED keeps partial outputs and checkpoints.

Meaningful tests: supervision tied/masked loss and gradients, teacher seed/future
isolation, dataset and counters, no validation gradients, same initial weights,
exact BC->hybrid provenance, selected model roundtrip and full smoke action/physics
replay, complete grids, paired contrast gates, failure preservation and overwrite
refusal. Run existing tests + local Mac CLI smoke before commit/push.
Then stop tools and provide exactly two main commands: pinned detached lab
worktree full run and rsync -avhP --partial. Do not SSH/run/transfer remotely.

## Interpretation limits

Good BC but poor direct PPO supports an optimization/supervision explanation;
it does not prove credit assignment is the sole cause. Poor BC can reflect finite
coverage, teacher noise, approximation or optimization, not necessarily an
inadequate encoder. Hybrid benefit includes extra information/compute. Planner
is an achievable model-based reference, not an N4 optimum. Neither success nor
failure establishes MARL necessity.

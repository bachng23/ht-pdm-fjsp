# Heterogeneous maintenance MARL v1 — frozen protocol

## Material Passport

ARS Schema9; code experiment protocol; user-authorized heterogeneous-worker
follow-up. Locked2026-10-06 before implementation or scientific runs.
Status IMPLEMENTED. Full scientific run not executed locally. Parent imitation commit de85e82. WaitingEnv physics unchanged.

## Questions and two confirmatory contrasts

H_assignment: centralized learned full matching reduces priced cost >=3% and
base cost >=2% versus centralized learned machine selection with fixed allocation
on heterogeneous N4/K2/H12. H_MARL: full-information machine-agent cooperative
PPO reduces priced cost >=2% versus centralized full-matching PPO on that same
panel. Both require >=8/10 paired replica wins, upper97.5% two-sided paired t CI
below0, nominal homogeneousK4/H12 base <=110% selected rule and heterogeneousK2
waiting violation <=selected rule+2pp. Family two contrasts: Bonferroni .05/2,
Student t(df9) critical2.685010846816525. Do not weaken gates after results.
Economic benefit vs rule also reported with >=5% priced/>=3% base,8wins,95% CI
(df9 critical2.2621571628540993), nominal/wait guards, as descriptive secondary.
A MARL PASS supports this tested policy decomposition, never mathematical
necessity of multiple agents or real communication constraints.

## Paired environment treatment

N4; same failure age3/probability.6,PM1/CM2/failure15/downtime6,overdue limit4/
price12. Restoration only at service completion; waiting/service occupy resources
and incur downtime. New cohort latent draw: duration base2 or3 per machine;
three machines favor one randomly chosen specialist, the fourth favors the other.
Homogeneous columns each have duration base,restoration.75. Specialized columns
have favored duration base-1/restoration1; other duration base+1/restoration.5.
All edges compatible in this primary treatment. Row arithmetic mean duration
and restoration remain identical, N/K/costs/shock streams held fixed. This
preserves these marginal parameters, NOT optimal/effective throughput.

Conditions: homogeneousK2, specializedK2(primary), homogeneousK4(nominal),
skill_maskK2(secondary). Skill-mask deletes fallback edges of two machines that
favor the common specialist. Every machine/worker remains compatible with at
least one counterpart. This additionally changes effective feasible capacity;
report separately from the all-compatible heterogeneity effect.

Train deterministic80% specializedK2/20% homogeneousK4, every5th episode nominal;
H12. Test every condition H12/H24. HomogeneousK2 diagnoses treatment, skill_mask
and H24 are untrained sensitivities. Finite generator has only128 specialized
parameter profiles; fresh seed panels do not ensure unseen physical profiles.

## Three learned arms and execution

All share same width64 machine/technician/edge/global/pair encoders, actor scorer
and centralized critic, identical parameter counts and initial parameter values
per replica. Inputs: full current machine ages/failure/wait/service state,
current worker status, known edge duration/restoration/compatibility and model
cost/failure law/remaining horizon. No future shocks or planner teacher labels.
Public rotating machine arbitration rank is part of every arm's input contract.

- central_matching_ppo: one categorical over every feasible partial matching,
  STOP included. Internal N4/K4 maximum catalogue209; K2 has21 physical matchings.
- central_fixed_allocation_ppo: same scorer/encoder, feasible candidates restricted
  to one deterministic assignment per chosen machine subset. Allocation minimizes
  sum(duration/restoration), lex tie. Machine subset/STOP is learned; allocation
  stays fixed. Masked likelihood is computed over this restricted action set.
- machine_cooperative_ppo: four machine agents, shared actor parameters, each
  simultaneously proposes WAIT or a currently feasible free technician. All agents
  receive the same public full-state broadcast and their own machine context;
  this matches information available to centralized arms. Rotating public rank
  resolves conflicting nominations for each worker; losers remain waiting.
  Execution uses this shared coordinator, explicitly not communication-free.
  Each machine is a decision-maker; workers remain passive resources.

Cooperative PPO stores proposals BEFORE arbitration, log probability is sum of
agent log probabilities, entropy sum of agent entropies; common critic/return.
Many proposals may map to one executed matching: never evaluate PPO likelihood
on the resolved matching instead. Every feasible matching can be represented by
non-conflicting proposals. Rejected proposals are diagnostics, no artificial
penalty. Joint-ratio PPO is specified explicitly, not called canonical MAPPO.
Information is deliberately matched; no assumed observation limitation is added
in this experiment. With full information centralized control can in principle
represent any MARL joint policy, so conflict alone cannot prove MARL necessity.

## Training, development and seed roles

10 matched replicas per arm;491520 physical steps/model,40960 H12 episodes,
320 rollouts1536,32 vector envs,minibatch192,4epochs,10240 optimizer steps/model.
Adam3e-4,clip.2,entropy.01,value.5,grad norm.5,gamma1,complete-episode MC returns,
reward=-priced cost/20,normalized advantage,one CPU torch thread. Same config/
actual-shock RNG roles across arms; policy action RNG draw dimensions differ.
Fixed checkpoints80/160/240/320 plus initialization. No early stop, resume,
teacher data, old checkpoint imports or test-feedback hyperparameter changes.

Full train150000..150009,dev cohorts151000..151009/shocks152000..152019,
sealed test cohorts153000..153009/shocks154000..154049.
Smoke train155000,dev cohort155020/shock155030,test cohort155010/shocks155040..155042.
Inherited registry includes all previous declared imitation-v1 full/smoke seeds.
Counter-derived configuration/environment/action/minibatch roles use new
heterogeneous_learning prefixes and exclude reserved/historical numeric seeds.

Select best of four rule controllers independently per condition on development
H12 priced cost: risk_matching,deadline_matching,lookahead_matching(H<=6),
adaptive_isolated_matching(full remaining H, exact one-machine DP allowing
switching between compatible workers in future, ignoring competing machines).
References perform exact maximum-weight feasible matching, not FIFO or greedy
only. Store all rule results. Evaluate every learned checkpoint on primary and
nominal development (10cohorts x20 shocks each). Initialization excluded;
select minimum primary priced cost meeting nominal/wait guards, earlier counter
tie. If none feasible use final,mark DEVELOPMENT_INFEASIBLE and retain replica.
All30 selected models frozen before any sealed test rollout.

## Diagnostic planners and metrics

Joint rollout scores every feasible full matching using32 independent forecast
scenarios and selected rule continuation, full remaining horizon, common root
forecasts. Fixed-allocation rollout uses same forecasts/continuation but one
static assignment per machine subset. Both preserve worker identity, service
owner, duration and completion restoration during forecasts. These are feasible
model-based comparators, not optima. Homogeneous full/fixed planner decisions can
differ in worker labels but must have identical costs under common shocks;
this is an engineering invariant. No exact N4 optimality claim.

Primary cost/base differences/CI/wins and all guards; secondary breakdown of
maintenance/failure/downtime/overdue, failures, wait violations/max/p95, terminal
pending, utilization, contention (unmatched urgent demand on the graph of equally preferred free
workers, plus requests whose preferred workers are all busy), nomination collisions/rejections, specialist load, runtime. Report
heterogeneous vs homogeneous assignment-planner headroom, but causal mechanism
cannot be inferred merely from raw cost increase or conflict count.

## Counts, smoke and artifact contract

Full30 models,14745600 training steps,1228800 train episodes,307200 optimizer
steps;63200 development episodes (3200 references+60000 checkpoints),144000
test episodes (120000 learned+24000controls),2592000 physical test intervals,
576000 machine rows. Six controls/four conditions/two horizons share500 test
shock/cohort combinations. Replica CI conditional on this finite panel.

Smoke width16,H4/H6,4forecast scenarios,256steps/model,2rollouts128,B8,
minibatch32,2epochs/checkpoints1/2. Three models,768train steps,192episodes,
48optimizer steps;34dev episodes (16rules+18checkpoints),216test episodes,
1080test intervals,864machine rows. All smoke verdicts ENGINEERING_ONLY.

New UTC output, refuse nonempty directory, full requires clean commit, CPU only.
Manifest status/hash/source/protocol/lock snapshot; configs/seed audit/budget;
training episodes and progress tagged arm/config hashes; initial/fixed/selected
checkpoints with algorithm/feature/environment contracts; development, reference
and checkpoint selection; episodes.partial/final, decisions, machine metrics,
proposal/arbitration diagnostics, paired seed/cohort metrics and summary.
FAILED preserves partial artifacts. Training and multi-seed eval use tqdm.

Test matrix: all masks cover physical matching catalogue incl STOP/incompatible/
busy edges; arbitration preserves feasible execution and every matching is
representable; likelihood uses raw proposals; finite PPO gradients; same encoder/
parameter initialization and information exposure; no future-event access;
scalar/batched heterogeneous forecast physics including mixed service owners;
adaptive DP near-terminal values and homogeneous equivalence; paired seeds,
exact optimizer/physical counts, complete-episode returns, changed parameters,
checkpoint reload/provenance/dev-only selection, all smoke action/cost replay,
full grids/CI gates, failed manifests and overwrite refusal. Local Mac smoke
and tests before commit/push. Then stop tools and provide exactly lab run and
rsync -avhP --partial commands. Do not SSH/launch/transfer remotely.

Pre-full engineering correction: smoke count arithmetic is 9 controllers
(6 controls +3 learned) x4 conditions x2 horizons x3 shocks =216 episodes,
1080 intervals and864 machine rows. This corrects the initial 288/1440/1152
arithmetic only; scientific budgets, gates and sealed full panels are unchanged.
Tie handling for preferred-worker contention includes all equally good workers,
so interchangeable homogeneous technicians do not acquire artificial conflicts
from a lexicographic favorite.

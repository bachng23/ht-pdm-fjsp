# Parallel-Machine Maintenance Baseline Scan Plan

## Status and scope

- Branch: `codex/parallel-maintenance-baseline-scan`
- Experiment type: simulator validation plus development-set algorithm screen.
- This experiment replaces job-shop routing and precedence with production-rate
  accounting on six heterogeneous parallel machines. It is not a final test-set
  comparison and makes no claim that a new MARL method is needed.

## Research questions and hypotheses

The primary question is whether the new maintenance-and-technician simulator is
internally valid, reproducible, learnable, and non-trivial relative to simple
dispatching rules.

- **H1 (learnability):** at least one of masked centralized PPO, IQL, or QMIX
  lowers mean development objective by at least 5% relative to the best fixed
  heuristic after the locked training budget.
- **H2 (coordination signal):** QMIX has lower proposal-conflict incidence than
  IQL on the shared development panel.
- **H3 (non-triviality):** no fixed heuristic is uniformly best on objective,
  downtime, rework, and technician workload imbalance.

Failure to pass H1 means the next step is simulator/reward/calibration analysis,
not a more elaborate MARL algorithm. Failure to pass H2 argues against a
constraint-aware QMIX extension without further diagnostic evidence.

## Environment contract

- Six parallel heterogeneous machines, two technicians, and three failure types.
- One synchronous one-hour decision interval; horizon 168 hours.
- A working machine produces at its configured rate and accumulates effective
  age according to load. Conditional Weibull failure shocks are sampled over the
  age increment.
- Each machine agent proposes no request or one technician. The request is PM
  while working and CM while failed. A deterministic resolver prioritizes CM,
  longer waiting time, higher conditional failure risk, then machine id.
- A technician serves at most one machine and a machine receives at most one
  technician. Losing proposals stay pending; episodes never restart on conflict.
- Technician-specific skill and experience change log-normal service duration
  and repair success. Failed repair produces rework. Free technicians may be
  absent for a decision interval. Successful work increases type-specific
  experience.
- The dense shared reward is the negative incremental scalar objective. The
  undiscounted episode return must equal the negative final objective.

## Conditions

Fixed non-learning baselines:

1. `random_feasible`
2. `fifo`
3. `fibt` (fastest expected technician)
4. `risk_first`
5. `balanced_greedy`

Learned baselines:

6. `masked_ppo` with a centralized discrete catalog of conflict-free joint
   proposals
7. `iql` with one utility network per machine
8. `qmix` with a shared utility network and monotonic state-conditioned mixer

All conditions use the same transition kernel, objective weights, resolver, and
evaluation shocks. This first scan intentionally excludes MAPPO, recurrent
agents, graph mixers, and constraint-aware mixers.

## Metrics

Primary endpoint:

- mean development objective, lower is better.

Secondary and mechanism endpoints:

- production, downtime, failure count, PM count, CM count;
- total ticket waiting, rework count, early-PM cost;
- proposal conflicts and conflict-step incidence;
- per-technician utilization and workload imbalance;
- episode return/objective identity;
- invalid execution, duplicate machine assignment, and duplicate technician
  assignment counts;
- wall-clock training and deterministic inference time;
- mean and dispersion across independent training seeds.

## Seeds and stopping rule

Smoke profile:

- training seeds: `77100` for each learned condition;
- development seeds: `63190:63193` (Python stop-exclusive syntax);
- 800 joint environment steps per learned condition;
- intended only as an engineering gate.

Full profile:

- independent training seeds: `77000:77005`;
- development seeds: `63100:63150`;
- fixed 100,000 joint environment steps per learned condition;
- checkpoint at 20%, 40%, 60%, 80%, and 100% of budget;
- evaluate the final-budget checkpoint only; intermediate checkpoints are
  retained for later learning-curve diagnostics and do not alter stopping.

Sealed final-test seeds are `63200:63300`. They are recorded in the protocol but
must not be evaluated by this runner.

## Smoke and full gates

The smoke gate passes only when:

1. all unit and integration tests pass;
2. every condition completes every requested development episode;
3. all learned models save and reload;
4. return equals negative objective within `1e-6` in every episode;
5. invalid, duplicate-machine, and duplicate-technician executions are zero;
6. same seed and action sequence reproduce the same trace and metrics;
7. training and multi-seed evaluation display `tqdm` progress;
8. the sealed test panel remains closed.

The full gate applies checks 2--8, requires five complete training replicates per
learned condition, and reports H1--H3 without changing the locked budget or seed
panels.

## Artifact schema

Each invocation writes to a new timestamped directory and refuses a non-empty
output directory:

```text
artifacts/<timestamped-run-id>/
├── parallel_maintenance_manifest.json
├── resolved_config.json
├── episodes.partial.csv
├── episodes.csv
├── decisions.csv
├── coordination.csv
├── training_progress.csv
├── summary.json
├── masked_ppo/<train-seed>/model.zip
├── iql/<train-seed>/model.pt
└── qmix/<train-seed>/model.pt
```

The manifest records profile, Git revision, runtime versions, exact seed panels,
training settings, output inventory, status, and audit checks. Partial episode
results are refreshed after each completed evaluation block.

## Interpretation boundary

This is a development screening experiment. It may identify learnability,
coordination, or calibration problems, but it cannot establish final algorithm
superiority. The sealed test panel can be opened only in a separately planned,
replicated confirmation experiment after the simulator and candidate algorithm
are frozen.

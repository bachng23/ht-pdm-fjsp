# Committed-Conflict RL Baseline Scan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-24
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1
- Parent mechanism: `clean-conflict-attribution-v1`
- Sealed status: seeds `63200:63300` remain closed

## Research question

Can standard independent, value-decomposition, and centralized value-learning
baselines learn the coordination pressure created by committed technician
requests, without changing the validated simulator transition kernel or reward?

The environment is fixed to `commit1_window0_sub0`: rejected maintenance
requests stop the requesting machine until service, while maintenance windows
and strong technician substitution remain disabled. This is a baseline scan,
not an algorithm-improvement claim.

## Frozen RL interface

- Six machine agents act simultaneously once per simulator hour.
- Each local action is `noop`, `technician_0`, or `technician_1`.
- Masks remove only physically unavailable actions. They do not remove
  duplicate technician proposals, so conflict remains learnable.
- Local observations expose the machine state/request/risk and observable
  technician availability, expected duration, skill, and success probability.
- The centralized state contains the complete local-action tensor and masks.
- All learned agents receive the same unshaped team reward, exactly the negative
  incremental simulator objective.
- The wrapper must be transition- and reward-identical to
  `ConflictConsequenceEnv` for equal seeds and joint actions.

## Fixed baselines

Learned baselines receive equal interaction budgets and checkpoint schedules:

1. `iql`: separate per-machine Q networks trained on the common team reward;
2. `vdn`: shared per-action utility network with additive joint value;
3. `qmix`: the same shared utility network with a monotonic state-conditioned
   mixer;
4. `centralized_dqn`: one global network over all `3^6 = 729` joint actions.

Fixed non-learning references are `reactive_fibt`, `independent_preventive`,
and `coordinated_preventive`.

## Hypotheses and decision logic

- **H1 (centralized learnability):** at the final checkpoint, centralized DQN
  has lower mean objective than IQL across matched training/evaluation seeds and
  wins on at least two of three training seeds.
- **H2 (cooperative factorization):** at least one of VDN or QMIX has lower mean
  objective than IQL and wins on at least two of three training seeds.
- **H3 (nonlinear mixing):** QMIX has lower final mean objective than VDN. This
  is secondary and does not gate simulator usability.

Interpretation is fixed before results:

| Pattern | Diagnostic interpretation |
|---|---|
| Centralized and factorized methods beat IQL | standard cooperative RL is sufficient |
| Centralized beats IQL, VDN/QMIX do not | representation/factorization bottleneck |
| No learned method approaches heuristic references | optimization/budget/interface bottleneck |
| All learned methods perform similarly well | no new RL algorithm is justified |

No algorithm is promoted from smoke results. No new mixer is implemented in
this experiment.

## Metrics

Primary endpoint: final-checkpoint objective, first averaged over the 50 common
evaluation seeds within each training seed, then compared across the three
matched training seeds.

Secondary metrics:

- objective at 25%, 50%, and 100% of the training budget;
- production, downtime, failures, preventive/corrective work, waiting, rework;
- proposal conflicts, conflict-step incidence, committed waiting;
- win rate against IQL on common evaluation seeds;
- distance to independent and coordinated preventive heuristic references;
- training loss, Q magnitude, gradient norm, and elapsed time.

Performance hypotheses are descriptive with only three training seeds. The
full-run audit gate checks experiment integrity, not whether a hypothesis wins.

## Seeds, budget, and stopping rule

### Smoke

- Training seeds: `91900`.
- Evaluation seeds: `64390:64393`.
- Budget: 1,200 joint steps per learned baseline.
- Parallel environments: 4.
- Checkpoints: 300, 600, and 1,200 steps.

### Full development run

- Training seeds: `91000`, `92000`, `93000`.
- Evaluation seeds: `64200:64250` (50 common seeds).
- Budget: 200,000 joint steps per baseline and training seed.
- Parallel environments: 8.
- Checkpoints: 50,000, 100,000, and 200,000 steps.
- Replay capacity: 50,000 transitions; learning starts at 5,000; batch 256;
  one gradient step every 8 collected transitions; target update every 2,000
  steps; Adam learning rate `3e-4`; discount `0.99`; epsilon `1.0 -> 0.05`
  over 60% of the budget.

There is no early stopping, best-checkpoint selection, seed extension, or
result-dependent hyperparameter change. The final checkpoint is primary.

## Smoke and full gates

Smoke passes only when:

- wrapper parity, mask, reward, checkpoint save/load, and conflict-reachability
  tests pass;
- every learned condition trains to 1,200 steps;
- all 36 learned checkpoint evaluations and 9 heuristic episodes complete;
- all reward, feasibility, duplicate-assignment, count, and sealed-seed audits
  pass;
- training and multi-seed evaluation display `tqdm` progress.

The full run is scientifically valid when all 12 training cells, 1,800 learned
checkpoint evaluations, and 150 heuristic episodes complete and all audits
pass. Performance hypotheses never change manifest completion status.

## Artifact schema

Each invocation refuses a non-empty output directory and writes:

```text
artifacts/<timestamped-run-id>/
├── committed_rl_manifest.json
├── source_config.json
├── resolved_config.json
├── heuristic_episodes.csv
├── learned_episodes.partial.csv
├── learned_episodes.csv
├── decisions.partial.csv
├── decisions.csv
├── coordination.partial.csv
├── coordination.csv
├── summary.json
└── <algorithm>/train_seed_<seed>/
    ├── settings.json
    ├── training_progress.csv
    ├── training_episodes.csv
    ├── model.pt
    └── checkpoints/model_<steps>_steps.pt
```

The manifest records the exact commit, resolved device, seed panels, budgets,
completed cells, runtimes, output inventory, sealed status, and audit gate.
Artifacts are development evidence; the future sealed panel is not evaluated.

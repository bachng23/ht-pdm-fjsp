# QMIX baseline fidelity experiment

## Research question and hypotheses

This diagnostic asks whether the feed-forward utility network used by the
current machine-agent QMIX baseline is responsible for unstable learning and
repeated coordination conflicts on the locked Brandimarte MK01-derived
`two_specialists_x2_0` environment.

- H1: recurrent QMIX has a lower mean development objective than feed-forward
  QMIX at 300,000 joint environment steps.
- H2: recurrent QMIX reduces production conflicts and rejected proposals per
  1,000 joint steps.
- H3: recurrent QMIX reduces between-training-seed dispersion and repeated
  per-agent rejection streaks.
- H4: if both conditions retain a zero-preventive policy, missing memory and
  identity are not sufficient explanations for the preventive-maintenance
  shortcut.

This is a baseline-fidelity diagnostic, not a TWC-QMIX performance claim.

## Locked conditions

Both conditions use one agent per machine, identical candidate features,
action masks, safe-noop semantics, reward, deterministic conflict resolver,
global state, monotonic QMIX mixer, optimizer, discount, epsilon schedule and
environment configuration.

1. `ff_qmix`: the existing shared candidate-scoring MLP and transition replay.
2. `recurrent_qmix`: a shared candidate encoder and GRU utility network,
   learned machine identity embedding, previous proposed-action features,
   previous accepted/wait/rejected outcome, episode-sequence replay, and a
   Double-Q target. The monotonic mixer is unchanged.

The recurrent condition deliberately bundles the canonical recurrent utility
features. If it passes the gate, a later component ablation will separate GRU,
identity, previous action/outcome, sequence replay and Double Q.

## Budget, seeds and stopping rule

- Fixed full budget: 300,000 joint environment steps per condition and train
  seed.
- Checkpoints: 100,000, 200,000 and 300,000 steps.
- Full training seeds: `76000,77000,78000`.
- Full development evaluation seeds: `61700:61750`, paired across all cells.
- Smoke training seed: `76100` with 600 steps per condition.
- Smoke evaluation seeds: `61990:61993`.
- Sealed future test seeds: `62000:62100`; this experiment must not open them.

Every cell runs once to its fixed budget. Stop only on an exception, non-finite
loss, incomplete episode, failed reward identity, invalid execution, duplicate
operation/technician execution, missing checkpoint, or malformed artifact.
There is no performance-based early stopping, retry or hyperparameter tuning.

## Endpoints and decision rule

The primary endpoint is the recurrent-minus-feed-forward paired mean objective
at 300,000 steps, summarized within each training seed over the common
development seeds. Secondary endpoints are makespan, total cost, failures,
preventive/corrective counts, maintenance waiting, production and technician
conflicts per 1,000 joint steps, rejected proposals per 1,000 joint steps,
proposal acceptance ratio, maximum repeated per-agent rejection streak, TD
loss, Q statistics and gradient norm.

Recurrent QMIX is promoted as the strong thesis baseline if its grand mean
objective is no worse than feed-forward QMIX and at least two of three
training-seed objective deltas are nonpositive. Conflict reductions are
mechanism evidence and do not override an objective regression. Regardless of
the result, all seeds and uncertainty are reported; this development gate is
not a confirmatory significance claim.

Before training, the runner evaluates the existing masked-SPT and
health-threshold policies on the same scaled configuration and development
panel. This reward sanity audit is descriptive and cannot select or alter the
locked reward.

## Artifact schema and smoke gate

Each new timestamped run contains:

- `qmix_fidelity_manifest.json`;
- source and scaled benchmark configuration snapshots;
- `reward_sanity_episodes.csv`;
- final and partial episode, decision and coordination CSVs;
- `qmix_fidelity_summary.json`;
- one directory per condition and training seed containing settings,
  `training_progress.csv`, `training_episodes.csv`, three checkpoints and the
  final model.

Training and evaluation use `tqdm`. The smoke gate requires both conditions to
train, save and reload; every checkpoint/evaluation cell to exist; reward and
objective identity to hold; all feasibility/coordination audits to pass; the
manifest to finish with `COMPLETED`; and the sealed panel to remain closed.

## Interpretation boundary

The experiment tests a bundled baseline correction. A favorable result does
not identify which recurrent component caused the change. A null result does
not prove that memory is irrelevant outside this environment or budget. The
deterministic resolver may still conceal proposal-level coordination failures,
and development seeds must not be presented as held-out confirmation.

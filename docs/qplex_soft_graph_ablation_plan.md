# QPLEX soft-graph ablation

## Material Passport

- Artifact: locked executable code-experiment plan
- Environment: Brandimarte MK01-derived `two_specialists_x2_0`
- Status: protocol locked before training results
- Primary unit of replication: training seed

## Research question and hypotheses

The environment-attribution diagnostic showed that the all-feasible resource
graph inflates coordination scope, whereas the policy-intent top-2 graph is
mostly local but is too imprecise for hard connected-component partitions.
This experiment asks whether the same graph is useful as soft centralized
training context in a full-team QPLEX mixer without changing decentralized
execution or the environment dynamics.

Four conditions are locked:

1. `qmix`: the existing feed-forward QMIX baseline;
2. `qplex_null_graph`: QPLEX with the graph input fixed to zero;
3. `qplex_all_feasible`: QPLEX conditioned on conflicts among all feasible
   production and maintenance actions;
4. `qplex_policy_intent_top2`: QPLEX conditioned on conflicts among each
   agent's two highest-valued feasible actions.

The three QPLEX conditions have the same architecture and parameter count.
Only the graph tensor differs. Graphs influence positive QPLEX advantage
weights; they never mask actions, split connected components, alter rewards,
or change the transition/resolver logic. Graph construction is performed from
the current pre-resolution observation and current Q ranking, with no future
resolver label.

- H1 (mixer effect): `qplex_null_graph` is no worse than `qmix` at 300,000
  joint environment steps.
- H2 (primary graph effect): `qplex_policy_intent_top2` improves the mean
  objective over `qplex_null_graph` by at least 3% and does so for at least
  four of five training seeds.
- H3 (intent-filter effect): `qplex_policy_intent_top2` has a lower grand mean
  objective than `qplex_all_feasible`.
- H4 (safety): the intent graph does not increase the failure mean by more
  than 5% relative to the null graph.

The feed-forward agent utility network, replay, action masks, safe-noop
semantics, reward, resolver, optimizer, discount, epsilon schedule and
environment configuration remain fixed. Recurrence is excluded because the
preceding recurrent-QMIX fidelity experiment failed its locked 300k promotion
gate; adding recurrence here would confound mixer and graph effects.

## Budget, seeds and stopping rule

- Fixed full budget: 300,000 joint environment steps per condition and train
  seed.
- Checkpoints: 100,000, 200,000 and 300,000 steps.
- Full training seeds: `76000,77000,78000,79000,80000`.
- Full development evaluation seeds: `61700:61750`, paired across every cell.
- Smoke training seed: `76100`, 1,200 steps per condition.
- Smoke evaluation seeds: `61990:61993`.
- Sealed future test seeds: `62000:62100`; this experiment must not open them.

Every cell runs once to its fixed budget. Stop only on an exception,
non-finite loss, incomplete episode, failed reward identity, invalid execution,
duplicate operation/technician execution, malformed graph, missing checkpoint,
or malformed artifact. There is no performance-based early stopping, retry,
checkpoint selection or hyperparameter tuning.

## Endpoints and locked decisions

The primary endpoint is the mean objective at 300,000 steps, first averaged
over the paired development episodes within each training seed and then across
training seeds. The training seed, not the evaluation episode, is the primary
replicate. Report all per-seed paired deltas and their sample dispersion.

Secondary endpoints are makespan, total cost, failures, preventive/corrective
maintenance, maintenance waiting, production and technician conflicts per
1,000 joint steps, rejected proposals per 1,000 joint steps, proposal
acceptance, maximum repeated rejection streak, TD loss, Q statistics and
gradient norm. Mechanism diagnostics include graph density, actual conflict
incidence and QPLEX advantage-weight summaries; they cannot override an
objective regression.

Promote `qplex_policy_intent_top2` only if all of the following hold:

1. grand-mean objective is at least 3% lower than `qplex_null_graph`;
2. its objective is lower in at least four of five training seeds;
3. grand-mean objective is lower than `qplex_all_feasible`;
4. grand-mean failures are no more than 5% above `qplex_null_graph`;
5. all reward, feasibility, graph and coordination audits pass.

H1 is reported separately: QPLEX-null is mixer-noninferior only if its grand
mean objective is no higher than QMIX and at least three of five seed deltas
are nonpositive. The 300k checkpoint is primary; earlier checkpoints describe
learning curves only.

## Artifact schema and gates

Each timestamped run contains:

- `qplex_soft_graph_manifest.json`, including Git revision, runtime, device,
  seeds, thresholds, model settings, completion state and audit results;
- source `benchmark_config.json` and resolved `scaled_config.json`;
- `reward_sanity_episodes.csv`;
- final and partial episode, decision, coordination and graph-mixer CSVs;
- `qplex_soft_graph_summary.json`;
- one directory per condition and training seed containing settings,
  `training_progress.csv`, `training_episodes.csv`, checkpoints and final model.

Training and multi-seed evaluation use `tqdm`. Smoke passes only when every
condition trains, all checkpoints save and reload, all expected evaluation
cells finish, schemas are complete, graph tensors are symmetric and
zero-diagonal, reward/coordination audits pass, progress is shown, and the
sealed panel remains closed. Full completion uses the same integrity gate;
the scientific promotion decision is reported separately.

## Interpretation boundary

This is a development experiment on one environment configuration. A positive
result estimates the incremental value of policy-conditioned graph context
under the locked learner and budget; it does not show that hard QSCAN
partitions are valid or that the graph is causally optimal. A null result does
not rule out other graph encoders, recurrent agents, budgets or environments.
The development seeds must not be described as held-out confirmation, and the
sealed test panel may be opened only after the method is frozen in a separate
confirmatory protocol.

## Implementation references

The mixer follows the duplex-dueling construction in Wang et al., “QPLEX:
Duplex Dueling Multi-Agent Q-Learning,” ICLR 2021
(`https://arxiv.org/abs/2008.01062`) and the authors' DMAQ reference
implementation (`https://github.com/wjh720/QPLEX`). The resource graph is an
experiment-specific extra input to the positive advantage-weight network; it
is not part of the original QPLEX claim.

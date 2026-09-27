# Experiment 2A — RA-QMIX stress-generalization checkpoint sweep

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Origin Date: 2026-09-28
- Verification Status: IMPLEMENTED / LOCAL CPU SMOKE VERIFIED — full run pending
- Version Label: `ra_qmix_stress_checkpoint_sweep_v1`

## 1. Motivation and research question

The completed leave-one-out development run showed an exploratory pattern: every
algorithm improved on the nominal environment between 20k and 50k training
episodes while becoming worse on all three stress scenarios. The largest
component contrasts under `combined_pressure` were associated with the queue
mixer and counterfactual-consistency loss.

This follow-up was specified after observing that run, so it is a new
prospective development experiment rather than a reinterpretation of the old
primary endpoint.

Research question: when training continues from 20k to 50k nominal episodes,
do the queue mixer or counterfactual loss increase degradation under
`combined_pressure`, and at what checkpoint does stress performance turn?

## 2. Locked hypotheses and estimands

For algorithm `a`, training seed `s`, and scenario `c`, define

`change(a,s,c) = mean_cost(a,s,c,50k) - mean_cost(a,s,c,20k)`.

Positive change is degradation because lower objective cost is better. Define
the two co-primary interaction estimands:

- `I_queue(s) = change(Full,s,combined_pressure) - change(NoQueue,s,combined_pressure)`
- `I_cf(s) = change(Full,s,combined_pressure) - change(NoCF,s,combined_pressure)`

H-queue and H-CF predict positive interactions. A directional replication is
recorded separately for each contrast when its mean interaction is positive
and at least four of five training-seed interactions are positive. This is a
predeclared descriptive rule, not a p-value or proof of causality. Null,
negative, or unstable results remain valid outcomes.

Secondary confirmatory endpoints apply the same 20k→50k estimand to
`slow_service`, `early_failure`, and nominal. Checkpoints 5k, 10k, 30k, and 40k
are exploratory and localize trajectory shape or a possible turning point.
Standard QMIX is a reference, not a component-isolating control.

## 3. Conditions and controls

Algorithms:

| ID | Purpose |
|---|---|
| `qmix` | Standard-QMIX reference |
| `tqmix_no_queue` | Isolate queue-mixer contribution conditional on edge + CF |
| `tqmix_no_cf` | Isolate CF-loss contribution conditional on edge + queue |
| `tqmix` | Full RA-QMIX |

All conditions use the same nominal training environment, action masks,
observations, replay/training settings, exploration schedule, and evaluation
policy as the completed leave-one-out experiment. The four evaluation scenarios
are unchanged: `in_distribution`, `early_failure`, `slow_service`, and
`combined_pressure`. Recovery remains at maintenance start.

`tqmix_no_edge` is excluded because the experiment is focused on the two
components implicated in late stress degradation. The edge/capacity question
belongs to the separate parameter-matched capacity-control experiment.

## 4. Seeds, budgets, and stopping rule

| Item | Full CPU run | macOS smoke |
|---|---|---|
| Train seeds | 83100–83104 | 83090 |
| Development evaluation seeds | 83500–83599 | 83490–83492 |
| Checkpoints | 5k, 10k, 20k, 30k, 40k, 50k | 2, 4, 6, 8, 10, 12 |
| Algorithms | 4 | 4 |
| Scenarios | 4 | 4 |
| Sealed seeds | 201–300, never evaluated | never evaluated |

A repository-wide textual seed audit performed before implementation found no
prior protocol/code declaration of the full-run train or evaluation ranges.
They are also disjoint from the prior leave-one-out train seeds 11–15 and eval
seeds 101–200. The manifest records all panels explicitly.

Training continues to the fixed 50k budget. There is no performance-based
early stopping and no post-hoc checkpoint selection. A run stops only for an
error, non-finite loss, or failed engineering audit. Interrupted runs remain
`FAILED`; reruns use a new timestamped artifact directory.

Full scale: 20 training trajectories, 1,000,000 training episodes, 120 saved
checkpoints, and 48,000 evaluation episodes. Smoke scale: 4 trajectories, 48
training episodes, 24 checkpoints, and 288 evaluation episodes.

## 5. Metrics and analysis

Primary metric is mean `combined_pressure` objective over 100 matched evaluation
seeds, first within each training seed and checkpoint. Report all five
training-seed values, mean, SD, minimum, maximum, and positive-seed count for
each co-primary interaction.

Secondary/mechanism metrics retain the previous artifact definitions: failure,
downtime, maintenance, queue-waiting, collision and other cost components;
failure events; downtime steps; queue length/waiting/censoring; service counts;
request diagnostics; technician utilization; and TD/CF/total losses.

Analysis order:

1. Verify completion, unique keys, checkpoint counts, finite losses, cost
   reconciliation, resource semantics, seed disjointness, and sealed status.
2. Average matched evaluation episodes within each training seed.
3. Compute 20k→50k changes and the two co-primary interactions.
4. Apply the locked directional-replication rule without p-values.
5. Inspect dense checkpoint curves and mechanisms as exploratory evidence.

Evaluation episodes do not replace independent training replications. The
effective replication count for algorithm-level uncertainty is five training
seeds. Component removal can affect optimization and representation, so an
interaction is evidence conditional on this architecture, not a universal
causal claim.

## 6. Artifact and engineering contract

Each run writes to a new timestamped directory and contains:

```text
manifest.json
benchmark_config.json
resolved_config.json
parameter_counts.csv
training_progress.csv
training_episodes.csv
episodes.partial.csv
episodes.csv
coordination.csv
budget_summary.csv
checkpoint_changes.csv
component_interactions.csv
primary_results.csv
summary.json
<algorithm>/train_seed_<seed>/checkpoints/budget_<n>/model.pt
```

Training and evaluation expose `tqdm` progress. The runner refuses a non-empty
output directory, blocks any overlap with sealed seeds, records Git/runtime/
device metadata, streams progress/partial rows, and writes `FAILED` metadata on
exceptions.

Smoke passes only if the process exits zero; all 288 evaluation rows, 24
checkpoints, and four trajectories are present; every trajectory performs an
optimizer update; losses are finite; checkpoint save/load works; costs
reconcile within `1e-6`; resource audits pass; and the sealed panel stays closed.

The full run is CPU-first on the Ubuntu lab. Per repository workflow, Codex
implements and smokes locally, commits and pushes, and then hands the user one
compound lab command plus the canonical `rsync` command. Codex does not SSH to
the lab or launch the full run.

Local verification on 2026-09-28 completed all 4 smoke trajectories, 24
checkpoints, and 288 evaluation rows. All schema, finite-loss, cost,
resource-semantics, unique-key, and sealed-panel audits passed; the repository
test suite passed 255 tests. These smoke results are engineering evidence only.

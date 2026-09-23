# MARL budget-screening protocol

## Material Passport

- Artifact: executable code-experiment plan
- Environment: `HTPdMFJSP-v0`, Brandimarte MK01-derived configuration
- Algorithms: cooperative IQL and QMIX
- Primary endpoint: mean development objective by checkpoint (lower is better)
- Status: protocol amendment locked before the first full budget screen

## Protocol amendment — 2026-09-23

The completed engineering smoke runs showed that strict same-step three-way
events are too sparse to choose a training budget: all three smoke policies had
zero events. Before any full budget-screen result was observed, the screen was
therefore narrowed to the thesis baselines, IQL and QMIX, and the plateau rule
was changed from three-way incidence to makespan. This is a prospective
engineering amendment, not a response to full-run performance.

Strict three-way incidence remains a secondary diagnostic. Constraint exposure
is reported separately for machine contention, technician contention,
precedence blocking, maintenance waiting, pairwise intersections, and the strict
three-way intersection so sparse signals cannot hide denser constraint pressure.

## Purpose and hypothesis

This is an engineering screen, not a confirmatory algorithm comparison. It keeps
the locked Brandimarte MK01-derived `two_specialists_x2_0` environment with six
machine agents. The hypothesis is that batched collection from eight independent
environments removes enough Python/inference overhead that IQL and QMIX can be
screened cheaply, and that their useful learning may already plateau by 200k or
300k joint environment transitions.

## Design

- Algorithms: cooperative IQL and QMIX. MAPPO is outside this screening scope.
- One development training seed per algorithm: `76000`.
- Checkpoints: 100k, 200k, 300k, 400k, and 500k transitions.
- Paired development evaluation panel: seeds `61900:61950` (50 episodes).
- Collection: eight independent environments. Both algorithms use one batched
  action inference call per vector step and retain one gradient update per four
  stored transitions.
- The future test panel `62000:62100` remains sealed.

## Outcomes and stopping rule

The primary endpoint is mean objective at each checkpoint. Makespan is the
co-primary budget guard and failures are the safety guard. Secondary endpoints
are maintenance wait, production and technician conflicts per 1,000 joint
steps, precedence-blocked operations per joint step, constraint-step incidence,
pairwise intersections, strict three-way steps, and strict three-way episode
incidence. Training always stops at 500k; there is no data-dependent early stop.

For each algorithm, select the earliest of 200k or 300k that simultaneously:

1. has mean objective within 5% of the best mean objective observed from
   100k--500k;
2. has mean makespan within 5% of the best mean makespan from 100k--500k; and
3. has mean failures no more than 0.10 above the best checkpoint.

The proposed budget for a later replicated run is the maximum selected budget
across both algorithms. If either algorithm fails the rule, do not claim a
200k/300k plateau; inspect the curves before spending on replicated training.

## Artifact contract

Each timestamped run contains the raw and scaled configs, a manifest with exact
software/settings hashes, five checkpoints per algorithm, training progress and
episode logs, checkpoint-level episode/decision/coordination CSV files, and a
JSON summary containing the locked selection rule and gate result. Partial CSVs
are refreshed after every evaluated checkpoint. Coordination audits require zero
invalid, duplicate-operation, and duplicate-technician executions.

The `smoke` profile uses 400 transitions, four environments, and three evaluation
episodes only to validate execution and artifact shape. Its output must not be
interpreted as a budget recommendation.

## Success gates and interpretation limits

Smoke passes only if both algorithms train, all ten checkpoint-evaluation cells
finish, every checkpoint saves and reloads, schemas are complete, coordination
audits pass, progress bars are visible, and the sealed test panel stays closed.
The full engineering gate additionally requires both algorithms to support a
200k or 300k budget under the locked rule. A failed selection is an informative
screening outcome and triggers curve inspection rather than an altered rule.

Constraint metrics are descriptive exposure measures. Step-level co-occurrence
is not a counterfactual estimate that a conflict caused objective or makespan.
The training seed, development panel, and stopping rule may not be changed after
viewing the full screen; any follow-up requires a labeled protocol amendment.

## Canonical execution and artifact schema

Each run uses a new UTC timestamped directory and writes config snapshots, a
manifest, five checkpoints and training logs per algorithm, partial and final
episode/decision/coordination CSVs, and a summary JSON with the locked selection
rule and normalized constraint metrics. The full run uses CPU on the Ubuntu lab
unless an explicit later amendment changes the device.

The exact command and remote handoff follow `docs/experiment_workflow.md`.

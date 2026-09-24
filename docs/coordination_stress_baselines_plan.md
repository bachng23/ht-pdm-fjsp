# Coordination-Stress Baseline Experiment

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-24
- Verification Status: UNVERIFIED (protocol; results not yet run)
- Version Label: code_plan_v1
- Parent: `committed_conflict_rl_baseline_plan.md`
- Sealed panel: `63200:63300` remains closed

## Research question and hypotheses

Can standard factorized value learners reach a coordinated heuristic reference
when technician coordination pressure is materially larger than in the first
committed-conflict RL panel? The reference is a fixed policy, not an optimum.

- H1 (screen): at least one predeclared cell has an independent-minus-coordinated
  paired objective gap of at least 5% of the absolute coordinated objective,
  a 95% paired interval above zero, independent worse on at least 70% of seeds,
  independent conflict-step incidence of at least 2.5%, zero coordinated
  conflicts, and random feasible worse than coordinated on mean objective.
  A window-enabled cell also requires at least one missed-window or overdue
  event in the screen; otherwise that factor has not become active.
- H2 (factorization): at least one of VDN or QMIX reaches within 5% of the
  coordinated reference, or outperforms it, on the fresh RL evaluation panel.
- H3 (diagnostic bottleneck): centralized DQN reaches within 5% of that
  reference while both VDN and QMIX are more than 5% worse. This is a
  diagnostic pattern only; optimization, budget, and seed stability must also
  be reviewed before attributing a cause to the mixer or credit assignment.

The screen selection is exploratory. H2/H3 are descriptive development-panel
results. No algorithm-superiority or causal representation claim follows from
this experiment.

## Conditions and fixed policies

All conditions share six machines, two technicians, a 168-hour horizon, the
selected parent cost/maintenance calibration, the same simulator kernel, and
the same unshaped shared reward. Commitment is enabled in every condition.
Only one extra factor changes relative to the parent:

1. `commit1_window0_sub0` (parent anchor)
2. `commit1_window1_sub0` (maintenance window)
3. `commit1_window0_sub1` (strong technician substitution)

The heuristic screen runs `random_feasible`, `independent_preventive`, and
`coordinated_preventive`. Random feasible samples machine order and then a
masked, unreserved `noop` or available technician for each machine, removing
technicians after assignment. The two greedy policies use the same priority,
candidate set, and fastest-technician preference; coordinated greedy reserves
each technician before visiting the next machine.

The selected cell trains `iql`, `vdn`, `qmix`, and `centralized_dqn`, with the
same RL interface and hyperparameters as the parent experiment. The wrapper
must match the raw kernel on all three cells for equal seed and joint actions.

## Panels, selection, and stopping

Full screen seeds: `64500:64600` (100 common seeds, no training). Full RL
evaluation seeds: `64600:64700` (100 fresh common seeds). Training seeds:
`94000:94010` (10 independent seeds per learner). These do not overlap the
parent or sealed panels. Smoke uses training seed `94900`, screen seeds
`64900:64903`, and RL evaluation seeds `64910:64912` solely for engineering.

The screen selects the qualifying cell with the largest relative paired
independent-minus-coordinated gap, breaking ties by cell id. If none qualifies,
the full run finishes with `status=COMPLETED`, `outcome=SCREEN_ONLY`; it does not train or silently relax
the rule. Smoke trains the anchor cell regardless of screen performance, and
is never scientific evidence.

Every learned method receives 500,000 joint environment steps, eight parallel
environments, and checkpoints at 100,000, 300,000, and 500,000 steps. The final
checkpoint is primary; earlier checkpoints diagnose learning dynamics only.
Replay capacity is 50,000; learning starts at 5,000; batch size 256; one
gradient step per eight collected transitions; target update every 2,000
steps; Adam learning rate `3e-4`; discount `0.99`; epsilon decreases from 1.0
to 0.05 over 60% of the budget. No adaptive extension, early stopping,
best-checkpoint selection, or result-dependent tuning. Smoke uses 40 steps,
one training seed, and two checkpoints at 20 and 40 steps.

## Metrics and analysis

Primary endpoint: mean final-checkpoint objective. Average each learner over
the same 100 RL evaluation seeds within each training seed; report the 10
training-seed means and paired method differences across matched training
seeds. For comparison with fixed heuristics, subtract their mean on the same
evaluation seeds. A 95% t interval across training-seed differences and the
number of seed wins describe uncertainty; evaluation episodes from one model
are not counted as independent training replicates.

The screen reports per-cell paired greedy gap, interval, relative gap,
independent-worse proportion, conflict incidence, random objective, and audit.
Secondary RL metrics: production, downtime, failures, preventive/corrective
jobs, waiting, conflicts, committed waiting, rework, missed windows, overdue
steps, return/objective identity, invalid executions, duplicate assignments,
and training time. Report learning
curves at all checkpoints. Inspect whether centralized performance is stable
and whether VDN/QMIX have plateaued before interpreting H3.

## Gates and artifacts

Smoke passes when wrapper parity, save/load, complete episode counts,
progress display, reward identity, feasibility, schema, and sealed-panel
closure pass. Full integrity requires complete screen results, a valid
selection or explicit `SCREEN_ONLY`, complete training/evaluation if selected,
all checkpoints and model files, zero invalid or duplicate assignments, reward
identity within `1e-6`, and no sealed seeds evaluated. Performance hypotheses
do not alter integrity status.

Each run refuses a nonempty output directory and writes a new UTC timestamped
artifact directory with `manifest.json`, source, screen-cell, and selected
resolved configs,
`screen_episodes.csv`, `screen_summary.json`, `heuristic_episodes.csv`,
`learned_episodes.partial.csv`, `learned_episodes.csv`, `coordination.csv`,
`decisions.csv` (fixed trace subset), `summary.json`, and per-algorithm,
per-seed settings, progress, episodes, checkpoints, and final models. The
manifest records revision, runtime, device, seeds, selection, completion,
output inventory, and audits. Full run execution and result transfer follow
`docs/experiment_workflow.md`; no sealed test evaluation occurs here.

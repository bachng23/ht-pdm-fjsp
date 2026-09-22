# MARL failure-risk guard diagnostic plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan/run
- Origin Date: 2026-09-22
- Verification Status: UNVERIFIED
- Version Label: marl_risk_guard_diagnostic_v1

## Motivation and locked hypothesis

The fresh combined-policy replication showed that broadcast context reduced
coordination rejection by 55% and tardiness by 7.4%, but also reduced preventive
maintenance by 9.6% and increased mean failures by 0.030 per episode. Episodes
with one additional failure had an average objective regression of roughly 36.

This diagnostic asks whether that tail risk is specifically caused by choosing
high-risk production when preventive maintenance is currently feasible.

The locked hypothesis is that a deterministic failure-risk guard applied to the
completed combined checkpoints will reduce failures without increasing mean
objective relative to the same unguarded checkpoints. The guard is a diagnostic
intervention, not yet a proposed final learning algorithm.

## Locked intervention

For each machine agent and decision epoch:

1. retain the environment's native validity mask;
2. identify feasible production actions whose conditional failure probability
   is at least 0.18;
3. only when the same machine has a feasible preventive-maintenance action,
   mask those risky production actions before deterministic policy inference;
4. never manufacture a wait, maintenance, or otherwise invalid action.

The threshold `0.18` is the benchmark's existing
`preventive_probability_threshold`; it is not tuned on this diagnostic panel.

## Conditions, metrics, seeds, and stopping

- `independent_actor_mappo_global_critic`: unguarded reference.
- `independent_actor_mappo_broadcast_context`: unguarded combined policy.
- `independent_actor_mappo_broadcast_context_risk_guard`: the same combined
  checkpoints and weights with only the locked action-mask intervention.
- Source models: all ten training seeds 15000--24000 from the completed fresh
  replication run.
- Full diagnostic seeds: 56000--56199, paired across all conditions and models.
- Smoke diagnostic seeds: 56900--56904 using source seeds 15000 and 16000.
- Reserved final test: 50000--50099 remains unopened.
- No training, checkpoint selection, tuning, adaptive retry, or early stopping.
- Replication unit: independent training seed. Environment seeds are paired
  repeated measurements.

Co-primary endpoints are guard-minus-unguarded-combined objective and failure
deltas. Secondary endpoints are makespan, tardiness, total cost, maintenance,
coordination conflicts, rejection rate, guard opportunities, masked actions,
changed agent decisions, and selected production risk.

## Locked mechanism gate

The intervention supports a risk-aware learning follow-up only if all checks
pass across the ten source training seeds:

1. one-sided 95% upper confidence bound for objective delta is at most zero;
2. one-sided 95% upper confidence bound for failure delta is below zero;
3. at least 7/10 objective deltas and 7/10 failure deltas are nonpositive;
4. the guard changes at least one selected agent action in every training seed;
5. all coordination audits pass.

The independent reference is reported on the same fresh panel but is not part
of the mechanism gate. Passing the gate does not authorize the held-out final
test; it motivates training a risk-aware policy rather than deploying the hard
guard directly.

## Artifact schema and integrity gates

Each timestamped run contains:

- `marl_risk_guard_manifest.json`
- `benchmark_config.json`
- `marl_risk_guard_episodes.csv` and `.partial.csv`
- `marl_risk_guard_decisions.csv` and `.partial.csv`
- `marl_risk_guard_coordination.csv` and `.partial.csv`
- `marl_risk_guard_summary.json`
- the terminal log produced by `tee`

Full completion requires 3 conditions x 10 training seeds x 200 diagnostic
seeds = 6,000 unique episodes, reward/objective identity, zero invalid or
duplicate executions, complete decision traces, matching config checksum, and
an unopened final panel. Multi-model and multi-seed evaluation uses `tqdm`.

## Local smoke command

```bash
RUN_ID="marl_risk_guard_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
uv run ht-pdm-fjsp-marl-risk-guard \
  --profile smoke \
  --device cpu \
  --source-run lab_results/marl_combined_replication_20260922T065843Z \
  --output-dir "artifacts/$RUN_ID"
```

The final handoff supplies one foreground Ubuntu command to fetch the committed
branch, verify the source artifact, run the full diagnostic, and tee a
timestamped log. A separate Mac command retrieves all artifacts from
`bachng@100.111.83.52` with `rsync -avhP --partial`.

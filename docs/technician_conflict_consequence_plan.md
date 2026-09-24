# Technician-Conflict Consequence Diagnostic Plan

## Scope and parent evidence

- Branch: `codex/technician-conflict-consequence-diagnostic`.
- Type: simulator mechanism experiment, not an RL algorithm comparison.
- Parent calibration selected `f36_pm0p35_risk0p005`: failure cost 36, PM
  duration factor 0.35, and PM risk threshold 0.005.

The parent experiment produced 2.88% independent conflict steps but only a
`+5.01` independent-minus-coordinated objective delta whose paired confidence
interval included zero. This experiment tests whether operational consequences,
rather than a large explicit conflict penalty, make technician collisions
materially costly.

## Hypotheses

- **H1 (commitment):** shutting a machine down when it requests maintenance
  increases the paired independent-minus-coordinated objective gap relative to
  non-committed requests.
- **H2 (window):** a missed two-hour PM window with elevated subsequent hazard
  increases that gap relative to requests without a window.
- **H3 (substitution):** a strong specialist/substitute service-time and success
  gap increases that gap relative to the base skill matrix.
- **H4 (material conflict loss):** at least one cell has 3%--10% independent
  conflict-step incidence, independent objective at least 5% worse than
  coordinated preventive dispatch with a paired 95% confidence interval above
  zero, and preventive dispatch still at least 5% better than reactive FIBT.

Failure of H4 means the simulator still lacks consequential coordination
pressure and does not justify a constraint-aware QMIX experiment.

## Locked 2 x 2 x 2 factorial

All cells use the selected parent configuration. Three binary factors vary:

| Factor | Low | High |
|---|---|---|
| Request commitment | rejected PM request keeps producing | every request shuts the machine down until service begins |
| PM window | no deadline | two-hour window; missing it multiplies conditional failure hazard by 3 until maintenance |
| Substitution severity | calibrated base skill/duration matrix | non-specialist duration is at least 2.5 times specialist duration and non-specialist base skill is capped at 0.25 |

Requests persist after first submission. A rejected committed request remains in
`waiting-for-technician`; an uncommitted request keeps producing but stays in the
queue. Failed machines always remain down. Conflict penalty remains at the base
value 1.5; material loss should arise from production, waiting, missed-window,
failure, duration, and rework consequences.

## Fixed policies

1. `reactive_fibt`: repairs failed machines only, reserving technicians centrally.
2. `coordinated_preventive`: PM/CM priority plus fastest available unique
   technician; no proposal collision; retained as a simple comparator.
3. `independent_preventive`: every eligible machine independently chooses its
   fastest available technician; the deterministic resolver selects one winner.
4. `coordinated_matching`: primary coordinated comparator; enumerates feasible
   one-step matchings and allows PM to wait rather than use a substitute whose
   expected duration exceeds 1.75 times the request's specialist duration.

All policies use the same transition kernel and keyed shocks.

## Metrics and attribution

Primary endpoint per cell:

\[
\Delta J_s = J_{independent,s}-J_{coordinated\ matching,s}.
\]

Report its mean, paired 95% t interval, median, and independent loss rate over
common seeds. Secondary metrics are production, downtime, failures, PM, CM,
waiting, rework, missed windows, overdue steps, proposal conflicts, conflict
steps, technician utilization, and workload imbalance.

Factor attribution uses paired difference-in-differences. For commitment:

\[
I_C = \Delta J(C=1)-\Delta J(C=0)
\]

within the same seed, window, and substitution setting. Window and substitution
interactions are defined analogously. Report mean and paired 95% intervals.

## Seeds and stopping rule

- Smoke: all eight cells, seeds `63590:63593`.
- Full development panel: all eight cells, seeds `63500:63550`.
- Sealed confirmation seeds remain `63200:63300` and must stay closed.
- Each policy completes one 168-step episode per cell and seed. No adaptive
  stopping or result-dependent factor expansion is allowed.

## Promotion rule

A cell qualifies only if:

1. independent conflict incidence is in `[0.03, 0.10]`;
2. mean independent-minus-coordinated objective gap is at least 5% of the
   absolute coordinated objective;
3. the paired 95% lower confidence bound for that gap is above zero;
4. independent is worse on at least 70% of seeds;
5. coordinated matching improves at least 5% over reactive FIBT;
6. coordinated policies have zero proposal conflicts;
7. all reward, feasibility, completion, and sealed-panel audits pass.

Among qualifying cells choose the smallest number of enabled high factors, then
the larger lower confidence bound, then lexicographic cell id.

## Artifacts

Every invocation refuses a non-empty output directory and writes:

```text
artifacts/<timestamped-run-id>/
├── conflict_consequence_manifest.json
├── resolved_base_config.json
├── cells.json
├── episodes.partial.csv
├── episodes.csv
├── decisions.csv
└── summary.json
```

Multi-cell/multi-seed evaluation displays `tqdm`. This diagnostic performs no RL
training. A promoted cell must next undergo a separately planned baseline RL
replication before any new mixer is implemented.

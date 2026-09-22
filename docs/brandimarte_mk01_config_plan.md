# Brandimarte MK01 HT-PdM-FJSP configuration plan

## Material Passport

- Artifact: executable configuration plan
- Production source: Brandimarte MK01, mirrored from SchedulingLab/fjsp-instances
- Production source path: `configs/source/brandimarte_mk01.fjs`
- Extended instance label: `ht_pdm_fjsp_brandimarte_mk01_v1`
- Status: implementation and local smoke validation
- Scope: standard FJSP production layer with a synthetic degradation and maintenance overlay

## Locked question and hypothesis

The configuration task asks whether the simulator can preserve the standard
MK01 routing and processing-time data while adding a six-machine, two-technician
HT-PdM layer that remains feasible and produces non-degenerate stochastic
maintenance behavior.

The acceptance hypothesis is that both bundled heuristic policies complete all
local smoke episodes without an audit failure, while the production-first policy
encounters at least one failure somewhere in the smoke panel and the
health-threshold policy executes at least one preventive action. These are
configuration checks, not algorithm-performance claims.

## Frozen production data

- 10 jobs, 6 machines, and 55 precedence-ordered operations.
- Machine eligibility and processing times are copied exactly from MK01.
- All load factors are fixed to `1.0`; load factors are not part of MK01.
- Every job receives schema-compatible due date `40.0`, equal to the published
  production-only optimum. Tardiness cost is zero, so due dates do not affect
  the objective in this configuration.
- The maintenance-free validation target is the published MK01 optimum
  makespan of `40`; it is a parser/model validation reference, not the optimum
  of the stochastic extended problem.

## Synthetic maintenance overlay

- One machine agent per machine, for six agents total.
- Technician T1 is eligible for M1-M4; T2 is eligible for M3-M6.
- M3 and M4 provide overlapping technician coverage; M1-M2 and M5-M6 create
  skill bottlenecks.
- Preventive restoration is `0.8` and corrective restoration is `1.0` for all
  eligible technician-machine pairs. Service durations remain heterogeneous.
- Machine Weibull parameters are synthetic and fixed before algorithm training.

## Metrics and seed split

- Structural metrics: 10 jobs, 6 machines, 55 operations, exact alternatives
  and processing times relative to the source `.fjs` file.
- Runtime metrics: completion, feasibility audit, makespan, failures,
  preventive/corrective counts, downtime, and maintenance waiting time.
- Local smoke seeds: `61000:61005`.
- Configuration diagnostic seeds, if later needed: `61100:61200`.
- These seed ranges are development-only and must not be reused as a final
  algorithm test panel.

## Stopping rule

Stop configuration work after the structural tests pass, all ten smoke policy
episodes complete and pass feasibility audits, at least one production-first
episode contains a failure, and at least one health-threshold episode contains
preventive maintenance. If the stochastic checks fail, change only the
synthetic overlay, document the change, and rerun the same development seeds.

## Artifact schema

Each smoke run uses a new UTC timestamped directory:

`artifacts/brandimarte_mk01_config_smoke_<UTC timestamp>/`

The existing benchmark runner writes `config_snapshot.json`, `metadata.json`,
`episodes.csv`, `summary.json`, and `traces.json`. Prior runs are never
overwritten silently.

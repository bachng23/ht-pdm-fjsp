# Literature-derived technician-conflict factorial screening

## Material Passport

- Origin skill: ARS experiment-agent
- Artifact: preregistered executable simulation experiment plan
- Instance: `ht_pdm_fjsp_brandimarte_mk01_v1`
- Architecture: six synchronous machine agents and two shared technicians
- Execution target: CPU-only Ubuntu lab at `bachng@100.111.83.52`
- Scope: development mechanism screening; not an RL algorithm comparison

## Research question and locked hypotheses

This experiment asks which literature-derived simulator mechanisms make shared-
technician contention observable, non-pathological, and avoidable through
coordination while preserving operation precedence and resource feasibility.

The primary hypothesis is that no single pressure control is sufficient, but
interactions among persistent requests, preventive-maintenance time windows,
overlapping skills, technician unavailability, and longer heterogeneous service
times create measurable coordination headroom.  Coordination headroom means
that an independent greedy assignment produces conflicts that a conflict-aware
matching policy can avoid without changing production eligibility or precedence.

The following directional mechanism hypotheses are locked before the full run:

1. request persistence increases observed request and contention incidence;
2. maintenance windows increase temporal overlap relative to threshold-only PM;
3. skill overlap increases avoidable conflicts and gives the matcher alternatives;
4. technician unavailability and longer service times increase capacity pressure;
5. at least one interaction containing time windows and skill overlap produces
   more avoidable contention than either factor alone.

## Simulator contract

The experiment uses the frozen Brandimarte MK01 production routes, processing
times, Weibull deterioration, costs, and precedence ordering.  At each event
epoch every idle machine produces at most one proposal.  Production proposals
select the local shortest-processing-time ready operation.  Maintenance has
priority over production when a corrective request or eligible preventive
request is active.  All proposals are logged before a rotating-priority
resolver accepts a conflict-free subset.

Two fixed technician-assignment policies are evaluated:

- `independent_greedy`: each machine independently claims its fastest qualified
  technician;
- `conflict_aware_matcher`: a deterministic maximum-cardinality, minimum-duration
  matching assigns distinct currently serviceable technicians.

The matcher is a mechanism reference, not an algorithm-performance baseline.
It receives the same requests and technician state as the greedy policy.

Operation precedence, operation-machine compatibility, one task per machine,
and one maintenance task per technician are hard constraints.  A proposal
conflict is observable before resolution; no conflict may survive into the
executed schedule.

## Full factorial conditions

Five binary factors produce all 32 combinations, including the baseline,
single-factor interventions, all pairs, triples, quadruples, and the full model.

| Code | Off | On |
|---|---|---|
| `Q` request persistence | busy/off-shift technician requests are masked | requests persist and may claim a busy/off-shift qualified technician |
| `W` PM timing | failure-probability threshold 0.18 | two deterministic overlapping release-deadline windows per machine |
| `O` skill overlap | frozen two-specialist eligibility | both heterogeneous technicians cover all six machines |
| `A` availability | technicians always on shift | staggered deterministic unavailable intervals |
| `D` service pressure | frozen service duration | preventive and corrective durations multiplied by 2.0 |

Window releases, deadlines, and unavailable-interval boundaries are simulator
events.  Maintenance remains non-preemptive after it starts.  The factor values
and every resolved condition configuration are written to the run artifacts.

## Metrics

Primary selection metrics under `independent_greedy`:

- fraction of episodes with at least one technician proposal conflict;
- technician-conflict units per maintenance proposal;
- avoidable conflict fraction;
- conflict reduction under the paired conflict-aware matcher.

Mechanism metrics:

- contention-opportunity epochs and units;
- optimal serviceable matching cardinality;
- avoidable and capacity-unavoidable wait units;
- busy/off-shift suppressed requests and persistent queued requests;
- maintenance wait, PM-window deadline misses, technician utilization;
- production duplicate claims and proposal rejection rate.

Outcome and audit metrics include makespan, objective, failures, preventive and
corrective maintenance, invalid executions, duplicate operation executions,
duplicate technician executions, task overlap, and precedence completeness.
The independent experimental unit is a stochastic environment seed; every
condition-policy comparison is paired by seed.

## Seed split

- macOS smoke: `61700:61703` (3 seeds per condition and policy; 192 episodes);
- full development screen: `61800:61900` (100 seeds; 6,400 episodes);
- sealed confirmation panel: `62100:62300` (200 seeds), not opened here.

This screen has no training seeds because no learned policy is fitted.  The full
development screen may nominate conditions for a later confirmatory environment
experiment and subsequent MARL training; it cannot open the sealed panel.

## Locked selection rule and stopping rule

Stop after exactly one full 100-seed factorial panel.  Do not change factors,
windows, gates, or seeds after inspecting its output.  A condition qualifies as
coordination-bearing only if all of the following hold for independent greedy:

1. technician conflicts occur in 10% to 50% of episodes;
2. conflict units are 1% to 10% of maintenance proposals;
3. at least 50% of conflict units are avoidable at their decision epoch;
4. the matcher reduces conflict units by at least 30%;
5. rejected proposals are at most 20% of all non-wait proposals;
6. both policies complete every episode with zero invalid or duplicate execution;
7. the executed schedules pass machine, technician, and precedence audits.

Among qualifying conditions, select the largest paired mean conflict reduction;
break ties by lower greedy objective, then fewer enabled factors, then condition
name.  If no condition qualifies, report `selected_condition = null`; do not tune
and rerun this panel.  Smoke success requires completion, schema checks, progress
display, and feasibility audits, not passage of the scientific selection gate.

Main-factor effects and pairwise difference-in-differences are descriptive.
Higher-order cells are retained in raw artifacts but are not promoted to causal
claims.  This diagnostic identifies simulator mechanisms associated with
learnable contention under the two fixed policies; it does not establish that a
particular MARL algorithm will learn the coordination strategy.

## Artifact schema

Every invocation rejects a non-empty output directory and writes:

- `benchmark_config.json`;
- `resolved_conditions.json`;
- `technician_factorial_manifest.json`;
- `technician_factorial_episodes.partial.csv`;
- `technician_factorial_episodes.csv`;
- `technician_factorial_coordination.partial.csv`;
- `technician_factorial_coordination.csv`;
- `technician_factorial_effects.csv`;
- `technician_factorial_summary.json`.

The manifest records the Git commit, runtime, factor definitions, condition and
policy order, seeds, sealed panel, status, output files, and audit gate.  `tqdm`
shows progress over every condition-policy-seed episode.


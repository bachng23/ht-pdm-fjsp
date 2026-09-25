# Passive-technician simulator v2 validation plan

## Material Passport

- Origin skill: experiment-agent
- Origin mode: plan
- Origin date: 2026-09-25
- Verification status: UNVERIFIED
- Version label: code_plan_v1

## Research question and hypotheses

This engineering experiment asks whether the passive shared-technician model can
represent maintenance timing and technician selection without violating machine
or technician capacity.

- **H1 (physical validity):** every evaluated trajectory has zero duplicate
  machine assignments, duplicate technician assignments, queue duplication,
  ineligible service starts, and invalid state transitions.
- **H2 (temporal validity):** a machine remains unavailable for its full service
  duration, is restored only when service completes, and neither ages nor fails
  while in service.
- **H3 (reproducibility):** repeating a policy with the same episode seed gives
  an identical state/event trace, and failure shocks keyed by episode seed,
  machine, and time are independent of policy execution order.
- **H4 (non-degeneracy):** the development panel contains both failures and
  maintenance jobs, and at least one fixed policy produces positive queue wait.

Failure of H1--H3 blocks learning experiments. Failure of H4 triggers a new,
separately documented calibration rather than an adaptive change to this run.

## Locked environment contract

- One learning entity per machine; technicians are passive resources.
- Machine modes are `OPERATING`, `QUEUED_PM`, `QUEUED_CM`, `IN_SERVICE_PM`,
  `IN_SERVICE_CM`, and `FAILED`.
- An operating machine may defer or request preventive maintenance. A failed
  machine may defer or request corrective maintenance. Queued and in-service
  machines have only the wait action.
- Eligibility and service duration are fixed for every machine-technician pair.
- A queued preventive-maintenance machine keeps operating and aging. If it
  fails, the queued request becomes corrective without losing its FIFO place.
- A machine in service produces nothing, does not age, and cannot fail or
  receive another technician.
- Restoration occurs at service completion.
- Requests are FIFO across decision epochs. Simultaneous requests use rotating
  machine priority `(machine_id - time) mod machine_count` to avoid permanent
  lowest-index preference.
- Failure shocks are deterministic functions of episode seed, machine, and time.
- The final transition includes a locked residual-age and unfinished-work cost.

## Conditions

The validation runner compares four fixed policies on the same shocks:

1. `defer_all`;
2. `reactive_fastest`;
3. `threshold_fastest`;
4. `workload_aware`.

An exact finite-horizon dynamic program is evaluated only on a two-machine,
one-technician, four-step oracle cell using the same transition function.

## Metrics

Primary endpoints:

- total physical-invariant violations;
- reward/objective identity error;
- deterministic replay mismatches.

Secondary endpoints:

- objective and terminal cost;
- failures, PM starts, and CM starts;
- operating, downtime, service, and queue-wait machine-steps;
- proposal-contention units and invalid actions;
- per-technician busy time and service starts;
- exact expected cost on the oracle cell.

## Seeds and stopping rule

- Smoke seeds: `9100:9103`.
- Pilot/development seeds: `9100:9130`.
- Full validation seeds: `9100:9200`.
- Sealed evaluation seeds: `9200:9300`, recorded but never evaluated here.
- Every policy runs exactly once on every selected seed for the fixed 24-step
  horizon. There is no adaptive stopping or result-dependent parameter change.

## Smoke and full gates

The gate passes only if:

1. all unit and integration tests pass;
2. all requested policy-seed episodes complete;
3. invariant violations, identity errors, and replay mismatches are zero;
4. the manual service-duration, queue-conversion, capacity, terminal-cost, and
   common-random-number traces match their expected values;
5. the exact oracle returns a finite nonnegative value;
6. progress is displayed with `tqdm`;
7. the sealed panel remains closed.

H4 is reported separately and is not allowed to override H1--H3.

## Artifact schema

Each invocation refuses a non-empty output directory and writes:

```text
artifacts/<timestamped-run-id>/
├── manifest.json
├── resolved_config.json
├── episodes.partial.csv
├── episodes.csv
├── decisions.csv
├── coordination.csv
└── summary.json
```

The manifest records the Git revision, runtime, profile, seed panels, simulator
and objective versions, output inventory, and audit status. Decision rows record
pre-state, raw and accepted actions, service starts/completions, failures,
step-cost components, post-state, and a trace digest.

## Interpretation boundary

This experiment validates simulator mechanics and screens fixed policies. It
does not establish MARL superiority, select a learning architecture, or provide
confirmatory performance evidence. All v1 results remain historical artifacts
and are not compared numerically with v2 results.

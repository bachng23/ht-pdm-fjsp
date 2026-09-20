# Minimal HT-PdM-FJSP benchmark plan

## Material Passport

- Artifact: code experiment plan
- Status: implemented MVP specification
- Scope: synthetic simulator validation, not plant validation
- Primary unit: one policy × one stochastic seed
- Reproducibility target: exact replay for identical config, policy, and seed

## Objective

Build the smallest complete environment in which production routing changes
machine degradation, degradation changes failure and maintenance decisions, and
maintenance requests compete for heterogeneous technicians.

## Frozen MVP contract

1. All jobs are available at time zero and contain precedence-ordered operations.
2. Each operation has one or more eligible machines, processing times, and load
   factors.
3. Machine effective age increases by
   `alpha_m * processing_time * load_factor`.
4. Conditional failure probability is derived from a two-parameter Weibull
   cumulative hazard over the operation's effective-age interval.
5. Failed operations restart after corrective maintenance.
6. Preventive maintenance is permitted only while a machine is idle; corrective
   maintenance has priority.
7. Each maintenance action requires one eligible technician. Duration and
   restoration effectiveness are technician- and machine-dependent.
   The bundled instance has one cross-trained technician and one M1-only
   technician, so simultaneous requests can create a genuine skill bottleneck.
8. The primary production objective is makespan. The environment also reports
   tardiness, failures, maintenance counts, downtime, utilization, and total
   weighted cost.

## Baselines

- `production_first_spt`: shortest-processing-time routing, no preventive
  maintenance.
- `health_threshold_spt`: the same production rule plus preventive maintenance
  when the conditional failure probability of the next selected operation
  exceeds a fixed threshold; technicians are assigned by minimum service time.

## Validation gates

- Weibull probability is bounded and monotone in age/exposure.
- Same seed produces byte-equivalent metrics and trace records.
- Failure shocks use a deterministic key `(seed, job, operation, machine,
  attempt)`, providing common random numbers across policies without depending
  on event ordering.
- Every completed operation respects precedence and machine eligibility.
- No machine or technician has overlapping tasks.
- Every maintenance assignment respects the skill matrix.
- Maintenance reduces effective age according to the selected technician.
- Both baselines complete the bundled instance across a multi-seed smoke panel.
- CLI output can be regenerated exactly from the config and seed specification.

## Explicit limitations

- Synthetic parameters; no claim of industrial calibration.
- Linear effective-age accumulation and two-parameter Weibull hazard.
- No noisy health observation, job arrivals, setup time, spare parts, technician
  shifts, travel, energy, or learned policy in the MVP.
- Threshold policy observes true effective age; it is an oracle-health baseline.
  A noisy belief interface is the next extension.

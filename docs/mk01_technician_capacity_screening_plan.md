# MK01 queue-aware technician-capacity screening plan

## Decision and scope

This is a development-set mechanism experiment, not an algorithm-performance
claim.  It selects a technician-capacity configuration in which maintenance
coordination is observable before any further MARL training is attempted.  The
Brandimarte MK01 job shop, degradation model, SPT routing rule, and preventive
threshold `0.18` stay fixed.

## Hypotheses

- H1: the current two-specialist, unit-duration configuration is under-binding.
- H2: longer service times and/or one shared technician increase queue incidence
  and maintenance waiting time under the queue-aware simulator.
- H3: at least one condition produces moderate binding: maintenance waits in
  10--40% of seeds without mean technician utilization exceeding 85%.

## Factorial panel

- Technician topology: the current two overlapping specialists, or one shared
  technician eligible for all six machines.
- Service-duration multiplier: `1.0`, `1.5`, or `2.0`, applied to preventive and
  corrective durations only.
- The shared technician uses the fastest original eligible duration and its
  restoration value for each machine and maintenance kind before multiplication.
- Policy: queue-aware health-threshold SPT with threshold `0.18`.

The six conditions use identical seeds.  Smoke seeds are `61430:61435`; the
full development panel is `61500:61700`.  Seeds `62000:62100` remain sealed for
a later confirmatory comparison.

## Endpoints and selection rule

Primary endpoint: fraction of episodes with positive maintenance queue waiting
time.  Secondary endpoints: preventive/corrective wait incidence and time,
technician utilization, total cost, makespan, maintenance counts, failures, and
processed events.  Paired 95% t intervals compare each condition with the
current two-technician, `1.0x` reference.

A condition passes the moderate-binding gate when queue incidence is in
`[0.10, 0.40]`, mean utilization is positive and no greater than `0.85`, mean
queue waiting is positive, and all simulations pass the existing feasibility
audit.  Among passing conditions, select queue incidence closest to `0.25`;
ties prefer two technicians and then the lower duration multiplier.  If none
passes, report no selection rather than relaxing the gate after observing data.

The mechanism check separately reports whether queue incidence and mean wait
are non-decreasing with duration multiplier within each topology.  It is
diagnostic and is not required by the selection gate.

## Stopping rule and artifacts

Run the six-condition panel once for all specified seeds.  Stop early only on a
simulation/audit exception or `max_events`; partial episode CSV is checkpointed
every ten episodes.  Each invocation must use a fresh timestamped directory and
writes:

- `technician_capacity_manifest.json`
- `condition_configs.json`
- `technician_capacity_episodes.csv` and `.partial.csv`
- `technician_capacity_summary.json`
- the exact benchmark input as `benchmark_config.json`

The full run is accepted only with the manifest status `COMPLETED`, the expected
1,200 episode rows, and no feasibility failure.  The full run is performed on
the rented Ubuntu server; the local Mac run is smoke-only.

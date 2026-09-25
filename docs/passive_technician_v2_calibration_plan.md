# Passive-technician v2.1 calibration plan

## Material Passport

- Origin skill: experiment-agent
- Origin mode: plan
- Origin date: 2026-09-25
- Verification status: UNVERIFIED
- Version label: code_plan_v1

## Purpose and hypotheses

This exploratory calibration selects a development scenario for the passive
shared-technician simulator before any MARL training. It asks whether fixed
machine ages, technician eligibility, and machine-technician service times can
create meaningful maintenance-timing and technician-selection pressure without
making the system permanently saturated.

- **H1 (mechanical validity):** every trajectory has zero assignment, queue,
  eligibility, state-transition, reward-identity, and replay violations.
- **H2 (selection activation):** at least one candidate makes
  `threshold_fastest` and `workload_aware` produce different technician choices
  and different state traces on the development seeds.
- **H3 (controlled congestion):** at least one candidate produces positive
  queueing and contention while keeping both technician utilization and queue
  length inside the locked non-saturation bounds below.
- **H4 (distributed risk):** at least one candidate produces failures on two or
  more machines without one machine accounting for more than 70% of failures.

Failure of H1 blocks all further experiments. Failure of H2--H4 means that no
candidate is selected and a separately documented calibration amendment is
required.

## Locked factorial

All candidates use two passive technicians, failure age 6, maximum age 10,
failure probability 0.30, horizon 36, and the v2 objective coefficients.

The full `2 x 2 x 3` factorial contains:

1. machine count: 4 or 6;
2. initial ages:
   - `synchronized`: every machine starts at age 0;
   - `staggered`: machine `i` starts at `floor(i * 6 / machine_count)`;
3. fixed skill structure:
   - `alternating_specialists`: all machines are eligible for both technicians;
     even machines have service times `(2, 5)` and odd machines `(5, 2)`;
   - `shared_bottleneck`: all machines are eligible for both technicians and
     prefer technician 0, cycling through `(2, 4)`, `(2, 5)`, and `(3, 4)`;
   - `mixed_eligibility`: machine types cycle through
     `((2,5),(true,false))`, `((2,3),(true,true))`,
     `((3,2),(true,true))`, and `((5,2),(false,true))`.

Technician eligibility and service times stay fixed for the whole episode.
Technicians have no observations, policies, learning state, or actions.

## Fixed policies

Every candidate is evaluated with common keyed failure shocks under:

1. `reactive_fastest`;
2. `threshold_fastest`;
3. `workload_aware`.

Both preventive policies use the same age threshold. Their only intended
difference is technician selection: `threshold_fastest` minimizes service
duration, while `workload_aware` minimizes queued workload plus service time.

## Metrics

Mechanical endpoints:

- invariant violations;
- reward/objective identity error;
- deterministic replay mismatches;
- episode and decision-row completeness.

Calibration endpoints, computed per candidate:

- paired objective difference between preventive policies;
- fraction of seeds with different trace digests;
- fraction of matched request opportunities with different technician actions;
- fraction of workload-aware requests assigned to a strictly slower technician;
- queue-positive step fraction and maximum queue length;
- proposal contention per episode;
- utilization of each technician and utilization imbalance;
- failures per machine and maximum machine failure share;
- downtime, waiting, PM/CM starts, terminal cost, and objective.

## Candidate qualification and selection rule

A candidate qualifies only if the full development run satisfies all of:

1. zero invariant violations and replay mismatches, with identity error at most
   `1e-9`;
2. preventive-policy traces differ on at least 50% of seeds;
3. matched maintenance requests choose different technicians at least 5% of
   the time;
4. workload-aware uses a strictly slower eligible technician for at least 2%
   of its maintenance requests;
5. queue-positive fraction is between 1% and 35% of machine decision steps;
6. observed maximum queue length is between 1 and 3;
7. each technician utilization is between 10% and 90%;
8. failures occur on at least two machines and maximum failure share is at most
   70%;
9. at least 50% of seeds have a nonzero paired objective difference between the
   two preventive policies.

Qualifying candidates receive the locked score

```text
0.35 * trace-divergent-seed fraction
+ 0.30 * matched-request action-divergence fraction
+ 0.20 * nonzero paired-objective fraction
+ 0.15 * (1 - technician-utilization imbalance).
```

The candidate with the highest score is selected. Ties are broken by lower
mean objective across the two preventive policies, then lexicographic candidate
ID. The rule does not require workload-aware to outperform fastest assignment.

The smoke profile checks mechanics and that at least one candidate activates a
technician-choice difference. Qualification and selection are authoritative
only for the full development profile.

## Seeds and stopping rule

- Smoke seeds: `9100:9103`.
- Pilot seeds: `9100:9130`.
- Full development seeds: `9100:9200`.
- Sealed evaluation seeds: `9200:9300`, recorded but never evaluated.

Every policy-candidate-seed combination runs exactly once for 36 steps, plus
one deterministic replay. There is no adaptive stopping, candidate deletion,
or parameter modification after results are observed.

## Artifact schema

Each invocation refuses a non-empty output directory and writes:

```text
artifacts/<timestamped-run-id>/
├── manifest.json
├── resolved_candidates.json
├── episodes.partial.csv
├── episodes.csv
├── decisions.csv
├── coordination.csv
├── candidate_summary.csv
├── candidate_summary.json
├── pairwise.csv
└── selected_candidate.json
```

The manifest records Git provenance, profile, seed panels, stopping rule,
versions, completion status, and output inventory. The selected-candidate file
must explicitly contain `null` when no candidate qualifies.

## Interpretation boundary

This is exploratory environment calibration on development seeds. It may select
a scenario for a later preregistered learning experiment. It does not evaluate
MARL, establish algorithm superiority, or authorize use of the sealed seed
panel.

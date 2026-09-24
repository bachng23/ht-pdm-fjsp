# Skill-aware technician-cluster mixture calibration

## Material Passport

- Origin skill: ARS experiment-agent
- Artifact: preregistered executable simulation experiment plan
- Parent calibration: `technician_window_calibration_full_cpu_20260923T201919Z`
- Instance: `ht_pdm_fjsp_brandimarte_mk01_v1`
- Execution target: CPU-only Ubuntu lab at `bachng@100.111.83.52`
- Scope: development environment calibration; not an RL comparison

## Research question and locked hypothesis

The parent calibration found a discontinuity between the natural schedule (10%
technician-conflict incidence on its development seeds) and deterministic skill-
aware clustering (91% incidence for `group3_gap0p5`).  This experiment asks
whether an episode-level mixture of those two frozen schedules can produce a
main environment with 25% to 30% conflict incidence while retaining avoidable
coordination headroom and limiting downstream schedule distortion.

The locked hypothesis is that at least one mixture probability from 0.10 through
0.40 produces 25% to 30% conflict incidence, a 2% to 6% maintenance-proposal
conflict rate, at least 70% avoidable conflicts, and no more than 5% degradation
in greedy mean objective or makespan relative to the natural `p=0` reference.

## Simulator and conditions

Every condition fixes `Q1W1O1A1D0`: persistent requests, preventive-maintenance
windows, full heterogeneous two-technician skill overlap, availability calendars,
and no duration multiplier.  Production routes, precedence, processing times,
deterioration, technician durations, window width, and all other simulator
settings remain frozen.

For each environment seed, a stable SHA-256-derived uniform value `u` in `[0,1)`
is computed independently of the simulator random streams.  A condition with
mixture probability `p` uses the clustered schedule when `u < p`; otherwise it
uses the natural parent schedule.  The clustered schedule is exactly the prior
`group3_gap0p5` condition: `M1-M3` release at 12 and 36, while `M4-M6` release at
12.5 and 36.5.  The natural schedule is the original parent release mapping.

The same seed, `u`, and realized schedule are shared by independent greedy and
the conflict-aware matcher.  Reusing `u` across probabilities makes schedule
assignments nested and supports paired comparisons.

The locked grid is:

- references: `p=0.00` and `p=1.00`, not selection-eligible;
- candidates: `p={0.10,0.15,0.20,0.25,0.30,0.35,0.40}`.

## Metrics

Primary metric under independent greedy:

- fraction of episodes with at least one technician proposal conflict.

Coordination and anti-pathology metrics:

- technician-conflict units per maintenance proposal;
- aggregate avoidable conflict fraction;
- paired conflict reduction under the matcher;
- capacity-unavoidable wait units per maintenance request onset;
- busy/off-shift claims per maintenance proposal;
- maintenance wait and PM-window deadline misses;
- paired and aggregate objective and makespan relative to `p=0`;
- production conflicts, failures, PM and CM counts;
- realized clustered-schedule fraction for each condition.

All feasibility, invalid-execution, and duplicate-execution audits remain hard
requirements.

## Seed split and stopping rule

- macOS smoke: `62800:62803` (3 seeds per condition-policy cell; 54 episodes);
- full development: `62900:63100` (200 paired seeds; 3,600 episodes);
- sealed confirmation: `63200:63500` (300 seeds), not opened here.

Stop after one full development panel.  Do not change probabilities, thresholds,
selection gates, or seeds after inspecting it.  There is no training and no
checkpoint in this simulator calibration.

## Locked selection rule

A selection-eligible probability qualifies only if:

1. greedy technician-conflict incidence is between 25% and 30%, inclusive;
2. aggregate conflict rate is between 2% and 6% of maintenance proposals;
3. aggregate avoidable conflict fraction is at least 70%;
4. the matcher reduces mean technician conflicts by at least 50%;
5. greedy capacity-unavoidable wait units per request onset are at most 0.25;
6. greedy busy/off-shift claims are at most 35% of maintenance proposals;
7. greedy mean objective and makespan are each at most 5% worse than `p=0`;
8. every episode and both policies pass all feasibility and execution audits.

Among qualifying candidates, select incidence closest to 27.5%; break ties by
higher avoidable fraction, lower objective degradation, lower probability, then
condition name.  If none qualifies, write `selected_condition = null`.  The two
references cannot be selected.

Smoke success requires completion, schema validation, progress display, and
audits, not passage of the scientific gate.

## Artifact schema

Each invocation rejects a non-empty output directory and writes:

- `benchmark_config.json`;
- `resolved_mixture_conditions.json`;
- `technician_mixture_manifest.json`;
- `technician_mixture_episodes.partial.csv` and final `.csv`;
- `technician_mixture_coordination.partial.csv` and final `.csv`;
- `technician_mixture_summary.json`.

Each episode row records the probability, stable uniform value, and realized
schedule.  The manifest records the Git commit, seed panels, schedules, mixture
derivation, policy and condition order, completion status, selected condition,
output files, and global audit gate.  `tqdm` covers all multi-seed episodes.

## Interpretation boundary

This experiment calibrates a stochastic environment distribution over two fixed
release schedules.  It may nominate one probability for a separately registered
confirmation panel and later MARL training.  It does not show that any learned
algorithm can exploit the coordination opportunity, and results do not
generalize to schedules or mixture mechanisms outside this grid.

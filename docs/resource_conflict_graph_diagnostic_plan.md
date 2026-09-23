# Resource-conflict graph diagnostic

## Material Passport

- Artifact: executable code-experiment plan
- Environment: Brandimarte MK01-derived `two_specialists_x2_0`
- Source models: completed IQL/QMIX budget-screen checkpoints
- Primary condition: QMIX at 300,000 joint environment steps
- Status: protocol locked before diagnostic results

## Research question and hypotheses

This diagnostic asks whether feasible machine actions induce small, dynamic
resource-conflict subteams often enough to justify a QSCAN-inspired extension
of recurrent QPLEX.

At each dispatch epoch, the graph contains all six machine agents. Two agents
are adjacent when at least one pair of their currently feasible non-wait
actions either targets the same production operation `(job_id,
operation_index)` or requires the same technician. Components include isolated
agents.

- H1: under the 300k QMIX checkpoint, the epoch-weighted mean largest connected
  component size is at most 3.5 agents.
- H2: at least 50% of QMIX epochs contain a connected component of size 2 or 3.
- H3: at most 10% of QMIX epochs form one connected component containing all six
  agents.

The architecture diagnostic supports dynamic subteams only if H1--H3 all hold.
IQL is evaluated on the same seeds as a trajectory-distribution robustness
check but does not control the gate.

## Locked models, seeds, and stopping rule

- Source run: the completed MARL budget screen for IQL and QMIX.
- Checkpoint: 300,000 steps for both conditions.
- Training seed inherited from the source run: `76000`.
- Full development evaluation seeds: `61900:61950`, paired across conditions.
- Smoke evaluation seeds: `61990:61993`.
- Sealed future test seeds: `62000:62100`; they must not be opened.
- Device: CPU by default; this experiment performs evaluation only and does not
  train or alter a model.

Every condition/seed cell runs once to episode completion. Stop only on a
missing or incompatible source artifact, malformed graph partition, invalid or
duplicate execution, failed reward/objective identity, incomplete episode,
sealed-seed overlap, or other exception. There is no data-dependent stopping,
retry, or threshold tuning.

## Metrics and estimands

The three primary epoch-weighted statistics for each condition are:

1. mean largest connected-component size per epoch;
2. fraction of epochs containing at least one component of size 2 or 3;
3. fraction of epochs whose graph is one component containing all six agents.

Secondary graph diagnostics are mean component size including singletons, mean
non-singleton component size, component count, isolated-agent count, graph
density, and production/technician edge counts. Episode-level versions and
normal-approximation 95% intervals across evaluation seeds are reported so a
few long episodes cannot be mistaken for cross-seed stability. Objective,
makespan, failures, maintenance counts, resolver conflicts, and coordination
audits verify that the replayed evaluation contract is intact; they do not
select the graph thresholds.

## Artifact schema and gates

Each new timestamped run contains:

- `resource_graph_manifest.json`, including source hashes, source commit,
  current commit, device, seeds, checkpoint, protocol thresholds and status;
- source `benchmark_config.json` and `scaled_config.json` snapshots;
- `resource_graph_epochs.partial.csv` and final `.csv`, with component sizes
  serialized as JSON and edge/resource diagnostics for every epoch;
- `resource_graph_episodes.partial.csv` and final `.csv`;
- `resource_graph_summary.json` with pooled epoch estimands, episode-level
  uncertainty, the QMIX architecture decision, and audit gate.

Smoke passes when both policies load, all six smoke episodes finish, progress
is displayed with `tqdm`, graph partitions cover exactly six unique agents,
all coordination/reward audits pass, schemas are complete, and the sealed panel
remains closed. Full completion additionally requires 100 episodes and at least
one logged epoch per episode. The architecture decision is reported regardless
of pass/fail; a negative result favors recurrent QPLEX without dynamic
subteams.

## Interpretation boundary

Graph topology is policy- and trajectory-distribution dependent. Associations
between component structure and resolver conflicts are descriptive and do not
show that a graph mixer will improve objective or reduce conflicts. Passing the
gate justifies implementing and testing a dynamic-subteam treatment; it is not
evidence that QSCAN or QPLEX is superior. Development seeds may not be described
as held-out confirmation.

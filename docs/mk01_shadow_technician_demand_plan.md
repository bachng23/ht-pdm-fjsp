# MK01 shadow technician-demand diagnostic

## Material Passport

- Origin skill: ARS experiment-agent
- Origin mode: plan/run
- Artifact: executable simulation diagnostic plan
- Instance: `ht_pdm_fjsp_brandimarte_mk01_v1`
- Architecture: six machine agents, two shared technicians, safe-noop mask
- Status: implementation and local smoke validation
- Scope: development mechanism diagnostic, not an algorithm comparison

## Question and locked hypothesis

The safe-noop diagnostic showed that valid technician conflicts are sparse, but
the simulator masks preventive and corrective actions whenever every qualified
technician is busy. This diagnostic asks whether that availability mask hides a
non-degenerate three-way interaction among production choice, maintenance need,
and shared-technician capacity.

The locked hypothesis is that threshold 0.18 creates latent preventive requests
while all qualified technicians are busy, and that reconstructing those requests
without changing the trajectory exposes non-degenerate technician contention.

## Shadow reconstruction

The executed policy and simulator remain unchanged. Before each real joint
action, every machine is evaluated for a shadow maintenance request:

1. a waiting-corrective machine requests corrective maintenance;
2. otherwise, the local SPT production candidate requests preventive maintenance
   when its failure probability reaches the condition threshold;
3. the request selects the fastest currently idle qualified technician, or the
   fastest qualified technician overall when all are busy;
4. a request is marked mask-suppressed only when all qualified technicians are
   busy.

Request onsets, busy-suppression onsets, and technician-contention spell onsets
are deduplicated across consecutive decision snapshots. Shadow requests never
change actions, events, rewards, or simulator state.

## Conditions

All conditions use safe-noop joint SPT and paired stochastic seeds:

1. `shadow_reactive_safe_noop`: corrective requests only;
2. `shadow_threshold_018_safe_noop`: intended PdM threshold;
3. `shadow_threshold_005_safe_noop`: descriptive aggressive-PdM condition.

## Metrics

Primary mechanism metrics:

- episodes with threshold-preventive busy-suppression onset;
- episodes with shadow technician contention;
- distinct shadow request and busy-suppression onsets;
- distinct contention-spell onsets and their rate per request onset.

Secondary metrics include request snapshots, preferred-technician assignment,
visible technician conflicts, production conflicts, objective, makespan,
failures, preventive/corrective maintenance, and rejection rate. The independent
unit is one seed; condition contrasts are paired and use two-sided 95% t
intervals as descriptive estimates.

## Seed split and stopping rule

- Local macOS smoke: `61420:61425`, five seeds per condition.
- Full development panel: `61500:61700`, 200 already-opened seeds per condition.
- Future confirmation panel: `62000:62100`, kept unopened.

Stop after one full panel. Do not tune thresholds or gates after inspecting the
full results. A failed gate leads to resource/action-space redesign on
development seeds, not to opening the future panel.

## Locked gate

All seven checks must pass:

1. all 600 episodes complete;
2. all feasibility and shadow-accounting audits pass;
3. executed safe-noop policies have zero forced preventive proposals;
4. threshold 0.18 has preventive busy-suppression onset in at least 10% of seeds;
5. threshold 0.18 has shadow contention in at least 10% of seeds;
6. its busy-suppression-onset rate is at least 1% of request onsets;
7. its contention-spell-onset rate is at least 1% of request onsets.

## Artifact schema

Every invocation rejects a non-empty output directory and writes:

- `benchmark_config.json`;
- `shadow_demand_manifest.json`;
- final and incremental `.partial.csv` episode, decision, and coordination tables;
- `shadow_demand_summary.json`.

Progress is displayed with `tqdm` over every condition-seed episode.

## Rented Ubuntu server command

```bash
ssh -p 20002 root@194.93.49.15 'export PATH="/root/.local/bin:$PATH" && cd /root/ht-pdm-fjsp && git fetch origin && (git switch codex/mk01-shadow-technician-demand || git switch --track origin/codex/mk01-shadow-technician-demand) && git pull --ff-only && (command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh) && export PATH="/root/.local/bin:$PATH" && uv sync --frozen && set -o pipefail && RUN_ID="mk01_shadow_demand_$(date -u +%Y%m%dT%H%M%SZ)" && mkdir -p "artifacts/$RUN_ID" && uv run ht-pdm-fjsp-shadow-demand --profile full --output-dir "artifacts/$RUN_ID" 2>&1 | tee "artifacts/$RUN_ID.log"'
```

## Pull all artifacts back to the Mac

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" && rsync -avhP --partial -e "ssh -p 20002" root@194.93.49.15:/root/ht-pdm-fjsp/artifacts/ "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```

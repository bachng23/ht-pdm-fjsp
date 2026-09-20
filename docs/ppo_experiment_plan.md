# MaskablePPO experiment plan

## Material Passport

- Artifact: executable code-experiment plan
- Environment: `HTPdMFJSP-v0`
- Algorithm: `sb3-contrib` MaskablePPO with `MultiInputPolicy`
- Primary endpoint: mean test objective (lower is better)
- Secondary endpoints: makespan, tardiness, failures, maintenance, return
- Status: implementation and local smoke validation

## Design

- Training begins from seed 10000; unseeded episode resets draw deterministic
  new seeds from Gymnasium's seeded RNG stream.
- Validation seeds: 20000–20019.
- Held-out test seeds: 30000–30099.
- The full test compares PPO with masked SPT, health-threshold, joint-risk,
  reactive CP-SAT, and scenario rolling-horizon policies.
- Every evaluation episode must terminate, pass the existing feasibility audit,
  and satisfy `episode_return = -objective`.
- Smoke gate: 512 steps and five validation/test seeds on local macOS.
- Full default: 500,000 steps, four vector environments, and 100 test seeds on
  Ubuntu. This is an engineering baseline, not a preregistered final budget.

## Ubuntu lab run

```bash
git clone https://github.com/bachng23/ht-pdm-fjsp.git
cd ht-pdm-fjsp
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv sync --extra dev
RUN_ID="ppo_full_$(date -u +%Y%m%dT%H%M%SZ)"
uv run ht-pdm-fjsp-ppo \
  --profile full \
  --device auto \
  --output-dir "artifacts/$RUN_ID"
```

For a detachable terminal, start `tmux new -s ht-pdm-fjsp`, run the command,
then detach with `Ctrl-b d`. Reattach with `tmux attach -t ht-pdm-fjsp` to view
the live `tqdm` bar. Four periodic recovery checkpoints plus the final model
are written during training.

## Pull all experiment results to the local Mac

Run this on the Mac. The trailing slashes are intentional.

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results"
rsync -avhP --partial \
  bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```

If the repository is cloned elsewhere on the lab machine, replace the remote
path before running `rsync`.

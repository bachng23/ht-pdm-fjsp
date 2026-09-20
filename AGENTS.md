# Experiment workflow

For every code experiment in this repository:

1. Plan the hypothesis, metrics, seed splits, stopping rule, and artifact schema.
2. Implement and run tests plus a small local macOS smoke experiment first.
3. Commit and push the verified code to GitHub.
4. Run the full experiment on the Ubuntu lab host at `bachng@100.111.83.52`.
5. Show progress with `tqdm` during training and multi-seed evaluation.
6. Store each run in a new timestamped artifact directory; never overwrite a
   prior full run silently.
7. Pull the complete artifact directory back to the local Mac with `rsync`,
   preserving partial transfers and displaying progress.

The canonical commands are documented in `docs/ppo_experiment_plan.md`.

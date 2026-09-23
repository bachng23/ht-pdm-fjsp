# Experiment workflow

For every code experiment in this repository:

1. Plan the hypothesis, metrics, seed splits, stopping rule, and artifact schema.
2. Implement and run tests plus a small local macOS smoke experiment first.
3. Commit and push the verified code to GitHub.
4. After pushing, show the exact compound command for the user to run in the
   Ubuntu lab terminal at `bachng@100.111.83.52`; do not SSH to the lab or
   launch the full run on the user's behalf.
5. Show progress with `tqdm` during training and multi-seed evaluation.
6. Store each run in a new timestamped artifact directory; never overwrite a
   prior full run silently.
7. Show the exact `rsync` command for the user to run on the local Mac after
   the full run finishes; do not start the transfer on the user's behalf. The
   command must preserve partial transfers and display progress.

The canonical commands and handoff contract are documented in
`docs/experiment_workflow.md`.

# Experiment Log: `00-explorer-pipeline/connect`

* **Target Repo:** `OncoLens-explorer` (branch `exp/00-explorer-pipeline/connect`, base `main` at `a9215a1`)
* **Opened:** `2026-11-11 20:38`

> The experiment's own working memory. Append-only, one entry per round, newest last. Write here everything someone continuing **this** experiment needs: settings, dead ends, debugging, partial numbers. A failed round is an entry too; say whether the failure was technical or a real result.
> What a **different** experiment would need to know does not stay here: it goes to the lab (`experiment-lab-loop` → What-to-tell-the-lab).

## Round 1 · 2026-11-11 · smoke (local)
* **Tried:** Wrote the whole pipeline (`oncolens/`: config, data, model, train, runlog, hub, runner, watch) and `configs/connect.toml`. Commit: the first commit on this branch.
* **Run plan:** Question: does one run go through every part on the session's own machine (fetch 2,000 patients, train, log to the run folder and W&B, save to the Hub, read back)? Watch: the printed evaluations and the run folder. Stop early: any exception. Settings: `train.steps=40 train.eval_every=10 train.checkpoint_every=20`, CPU, through `runner --where local`.
* **Saw:** Finished in a few seconds (about 16 steps/s). Cohort: 2,000 patients, split by patient hash. Plumbing-label numbers at step 40: val AUROC 0.80, val loss 0.53 (not a result; see `oncolens/data.py`). `hub_roundtrip_ok: true`: the weights downloaded from the Hub equal the trained ones. `watch` read the run from the folder, and from W&B after one fix. Run files: `runs/01-smoke/`.
* **Technical failures fixed in this round (not results):** `wandb.Run.path` is already a string in the installed version, so joining it produced a broken path and the W&B source of `watch` failed; the run now records `entity/project/id` itself. Hub progress bars flooded the run output; disabled in `hub.py`.
* **Decided:** The local path works. Next: the same run on a Colab VM, long enough (3,000 steps) to be read while it is running.
* **Told the lab:** Nothing yet.

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

## Round 2 · 2026-11-11 · colab
* **Tried:** The same run on a Colab CPU VM through `runner --where colab`. Three runs and two runner fixes (commits `5969c1f`, then the one-connection-per-poll commit after it).
* **Run plan:** Question: does the whole path work from a rented VM, and can the run be read from this session while it is still running, from the mirrored folder and from W&B? Watch: the runner's own timeline, `oncolens.watch` on both sources mid-run. Stop early: the run failing, or no state coming back for a minute. Settings: `configs/connect.toml` with `train.steps` 3,000 / 40,000 / 60,000, CPU.
* **Saw:**
  * First attempt died before training: a technical failure. The VM accepts one ssh connection at a time and frees it slowly (the CLI answers HTTP 429, exit 255); the long silent `uv sync` was the command that lost. Fixed in the runner: install and train are one background job on the VM, a refused connection is retried, and each poll is one connection that brings the process state and the files together.
  * `02-colab` (3,000 steps): finished, `hub_roundtrip_ok: true`, run folder mirrored back. It ran at about 135 steps/s, so it was over in 20 seconds, before anyone could read it mid-run.
  * `03-colab-long` (40,000 steps, about 300 steps/s): the runner mirrored 12 intermediate states about 10 seconds apart.
  * `04-live-read` (60,000 steps): read mid-run in the same command from both sources. Folder mirror at step 855 with a 1.4 s old heartbeat; W&B a few seconds later, at step 6,413 when the folder was at 8,327. So the folder mirror is about one poll behind and W&B about ten seconds behind.
  * The plumbing model overfits: val loss rises from 0.46 (step 500) to about 2.0 while train loss goes to 0.01 and val AUROC stays near 0.79. This says something about a 2,000-patient toy, not about OncoLens. But nothing in `watch` said so while it happened: the summary shows last values, not that the two curves had parted. Candidate for the observability experiment.
  * Run files: `runs/02-colab/` in full; `runs/03-colab-long/` and `runs/04-live-read/` without `metrics.jsonl` (5-8 MB each at one row per step; the full series are on W&B under the same run names).
* **Decided:** The stopping point is reached: a Colab run finished, was read while running from both sources, its model reads back from the Hub identical, its run folder is on the branch. Closing.
* **Told the lab:** final conclusion: closing. Claim `pipeline-connected-explorer-colab-wandb-hub`; system page written (Experiment Language, quirks: one ssh connection at a time, a short run is over before it can be watched, one row per step makes large files).

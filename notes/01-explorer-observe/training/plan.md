---
exp_name: 01-explorer-observe/training
title: "See inside a training run while it runs"
date: 2026-11-11
target_repo: OncoLens-explorer
base_branch: exp/00-explorer-pipeline/connect
exp_branch: exp/01-explorer-observe/training
linked_gaps: [G3]
inspired_by: [SPARK-01]
data_source: "OncoLens-data@master:wiki-llm/datasets/msk-chord.md"  # same cohort and plumbing label as experiment 00
---
# Plan: `01-explorer-observe/training` — See inside a training run while it runs

> This page is the **current** best description of how to realise the idea. Rewrite it in place after every round that teaches something. The history is in `log.md`, next to this file.

## 1. The Idea
* **Idea:** Put measurements inside the training loop (what each layer is doing, how the patients are being learned, what the predictions look like) and detectors over the resulting stream that state, in a plain sentence and at the step it becomes true, what is happening to the run. Then whoever watches, a person on W&B or an agent on the mirrored folder, knows what the run is doing while it is doing it, and can stop a run that has already answered its question or cannot.
* **Why it is worth trying:** `G3`, from the user: "I want to know what happened at live in the train process"; and the lab's purpose says a run is analyzed continuously while it runs. Experiment 00 showed the hole: its runs overfitted in plain sight (validation loss fourfold up) and `watch` printed last values and said nothing.
* **The test of the idea:** plant faults and see whether the tools name them while the run is going, without being told what was planted: (a) the default config, which is known to overfit; (b) a learning rate far too high; (c) a learning rate far too low. Supported if each fault is named by an event while the run is still running, the event's step is close to when the fault became visible in the curves, and a healthy stretch of a run raises nothing false.
* **Crossed with what the lab knows:** The lab saw (claim `pipeline-connected-explorer-colab-wandb-hub`) that a run is readable mid-run from the mirror and from W&B, with about ten seconds of delay, and (system page quirks) that a short run is over before it can be watched and that one row per step makes large files. So: runs here last minutes, and metrics rows become sparse. Nothing from outside sources is in the cortex yet. The measures themselves (update-to-weight ratio, dead ReLU units, forgetting events, calibration error) are standard practice, used here from general knowledge, not from a source the lab stored.
* **Alternatives considered:**
  * A notebook dashboard read by a person (`notebook-observability` skill). Dropped as the first step: the reader that must not miss things here is an agent on a wide request, and it reads text and JSON. A notebook on top of the same functions is a small later step.
  * W&B alerts and panels only. Dropped: they live outside the repo, an agent cannot read a panel, and a detector there cannot be re-run on an old run.
  * Saving per-patient trajectories to the run folder. Dropped for now: patient-level values would leave the process. Aggregates only; worth revisiting when the task is real (noted in Open Questions).

## 2. What We Know and How We Will Do It
* **Experiment Language** (from `cortex/systems/OncoLens-explorer.md`):
  * **Extension point:** "a view of a run" (`oncolens/watch.py`). That is not enough for this idea: a view can only show what the loop logged, and the loop logs losses only. So this experiment **adds to the shared set**, and says so here.
  * **Shared components used as they are:** config loading, data, model, `training_step`, Hub, runner (one name added to its mirror list).
  * **What stays fixed:** data, model, label, optimizer, seed. With `track.probe_every = 0` and `track.log_every = 1` (the defaults) a run does what it did in experiment 00.
  * **Added to or changed in the shared set:**
    * `oncolens/probes.py` (new): in-loop measurements, on with `track.probe_every`. Aggregates only.
    * `oncolens/detect.py` (new, and a new **extension point**: a detector is a function with `@detector`): pure functions over the logged rows, used inside the loop and from outside on any run, old runs included.
    * `oncolens/runlog.py`: `events.jsonl` and `RunLog.event`; rows kept in memory for the detectors.
    * `oncolens/train.py`: the loop calls probes and detectors; `track.log_every` averages train scalars into sparse rows (fix for the large-file quirk).
    * `oncolens/watch.py`: sparklines, the layer table, events, `analyze` and `--report` (writes `analysis.md` to the run folder).
    * `configs/observe.toml`: a run made to be watched.

Commands (from the lab root):
```bash
uv run scripts/lab.py run --exp 01 -- 'uv run -m oncolens.runner --where colab --run 02-overfit --config configs/observe.toml'
uv run scripts/lab.py run --exp 01 -- 'uv run -m oncolens.runner --where colab --run 03-too-hot --config configs/observe.toml train.lr=3.0'
uv run scripts/lab.py run --exp 01 -- 'uv run -m oncolens.watch notes/01-explorer-observe/training/runs/02-overfit --follow'
uv run scripts/lab.py run --exp 01 -- 'uv run -m oncolens.watch notes/01-explorer-observe/training/runs/02-overfit --report'
```

## 3. Open Questions
* Do the detectors name each planted fault mid-run, and how late?
* Do they stay quiet on the healthy part of a run?
* What do the probes cost in speed?
* Later, not here: per-patient trajectories (which patients are learned late, or never) need a rule for where patient-level values may be kept.

## 4. Stopping Point
Enough: the three planted runs on Colab, each read mid-run, each fault named by an event before the run ended, no false event on the healthy stretch, and the cost of the probes known. Not worth more: more than about four rounds of tuning detector thresholds on this toy, since thresholds fitted to a 2,000-patient MLP say little about the models that come later; then close with what holds and list what does not.

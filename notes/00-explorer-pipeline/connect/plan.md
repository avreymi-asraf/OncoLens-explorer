---
exp_name: 00-explorer-pipeline/connect
title: "Connect the pipeline end to end"
date: 2026-11-11
target_repo: OncoLens-explorer
base_branch: main
exp_branch: exp/00-explorer-pipeline/connect
linked_gaps: [G2]
inspired_by: []
data_source: "OncoLens-data@master:wiki-llm/datasets/msk-chord.md"  # public cBioPortal study msk_chord_2024, first 2,000 patients, fetched at run time
---
# Plan: `00-explorer-pipeline/connect` — Connect the pipeline end to end

> This page is the **current** best description of how to realise the idea. Rewrite it in place after every round that teaches something. The history is in `log.md`, next to this file.

## 1. The Idea
* **Idea:** Build the smallest pipeline that goes all the way: a small public cohort in, one short training run on a rented machine (Colab), the run readable while it runs, the model saved to the Hugging Face Hub, the run folder back on the branch. The claim is narrow: *the pipeline is connected*. Its numbers are not a baseline.
* **Why it is worth trying:** `G2`: the target repo is empty and has no Experiment Language. Every later experiment needs these parts, and the lab's purpose says a run must be analyzed while it runs, with its data reachable in real time. This experiment writes the shared components and the system page section from what it builds.
* **Crossed with what the lab knows:** The cortex is empty (first experiment). From `OncoLens-data` (read, not tested here): MSK-CHORD is the closest public analogue of the Hadassah data and is CC BY-NC-ND, so rows are fetched at run time, cached in a git-ignored folder and never committed or re-published. The prediction target is undefined (`G1`), so the label here is a *plumbing label* and is named as such in the code.
* **Alternatives considered:**
  * Synthea synthetic records (the data repo's "job A: code and CI fixtures"). Zero license questions, but it needs a Java generator and a FHIR conversion step before the first row exists. Dropped for now; worth it when a fixture is needed that can be committed.
  * Run only in the session's own machine. Dropped: the point is the connection to the machine heavy runs will use.
  * A real task with a first evaluation. Not possible before `G1` is decided; written as a Spark at close.

## 2. What We Know and How We Will Do It
The Experiment Language does not exist yet: creating it is this experiment's idea. What it builds (all in `oncolens/`):

| Component | File | What it does |
|---|---|---|
| Config | `config.py`, `configs/*.toml` | One `RunConfig` per run; a variant is `section.key=value` |
| Data | `data.py` | Fetch and cache the cohort, features, split by patient (hash of the patient id), aggregates |
| Model (extension point) | `model.py` | `MODELS` registry, chosen with `model.name` |
| Run entry (extension point: `training_step`) | `train.py` | One config in, one run folder and one model on the Hub out |
| Run folder and live stream | `runlog.py` | `config.json`, `metrics.jsonl`, `status.json`, `summary.json`, mirrored to W&B as they happen |
| Hub | `hub.py` | Checkpoints to a private Hub repo under `<exp-name>/<run>/`, and a read-back check |
| Runner | `runner.py` | `--where local|colab`; a Colab run ships committed code, runs in the background, its run folder is mirrored back every few seconds, the VM is always released |
| Observation | `watch.py` | Read a run while it runs, from its folder or from W&B: progress, speed, each scalar's latest value and movement, warnings |

* **Experiment Language** (from `cortex/systems/OncoLens-explorer.md`): "Not written yet"; this experiment writes it.
  * **Extension point:** none exists; `model.py` (`MODELS`) and `train.py` (`training_step`) are created as the first two.
  * **Shared components used as they are:** none exist.
  * **What stays fixed:** nothing yet.
  * **Added to or changed in the shared set:** all of the table above.

Commands (from the lab root):
```bash
uv run scripts/lab.py run --exp 00 -- 'uv run -m oncolens.runner --where local --run 01-smoke --config configs/connect.toml train.steps=40'
uv run scripts/lab.py run --exp 00 -- 'uv run -m oncolens.runner --where colab --run 02-colab --config configs/connect.toml train.steps=3000'
uv run scripts/lab.py run --exp 00 -- 'uv run -m oncolens.watch notes/00-explorer-pipeline/connect/runs/02-colab --follow'
```

## 3. Open Questions
None left for this experiment. Both were answered in round 2 (see `log.md`): the whole path works from a Colab VM, and a run is readable mid-run from the mirrored folder (about one poll behind) and from W&B (about ten seconds behind).

## 4. Stopping Point
Reached in round 2. Was: one run on Colab that finished, was read while running from both sources, left a model on the Hub that reads back identical, and left its run folder on the branch. Improving the model's numbers was out of scope.

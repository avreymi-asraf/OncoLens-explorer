"""The run folder and the live stream: what a run leaves behind, written while it runs.

Every run writes the same files, so any run can be watched and any two runs compared:

    <run-dir>/config.json     the RunConfig, the experiment and run names, the code commit
    <run-dir>/metrics.jsonl   one JSON object per logged event, appended and flushed as it happens
    <run-dir>/events.jsonl    one JSON object per event the detectors raised (oncolens/detect.py), as it was raised
    <run-dir>/status.json     state (running | finished | failed), step, heartbeat, links; rewritten atomically
    <run-dir>/summary.json    final numbers, written once at the end

The same events go to Weights & Biases as they happen, so a run on another machine can be watched from anywhere.
Row keys: `step`, `time` (unix seconds), then `<group>/<name>` scalars, e.g. `train/loss`, `val/auroc`.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

NOTES_DIR = "notes"


def current_exp(default: str = "adhoc") -> str:
    """The experiment name, from the lab's environment (`lab.py run` sets LAB_ROOT and EXP_DIR)."""
    lab_root, exp_dir = os.environ.get("LAB_ROOT"), os.environ.get("EXP_DIR")
    if lab_root and exp_dir:
        return Path(exp_dir).relative_to(Path(lab_root) / "experiments").as_posix()
    return os.environ.get("ONCOLENS_EXP", default)


def run_dir_for(exp: str, run: str) -> Path:
    """Where a run's small files live on the experiment branch: notes/<exp-name>/runs/<nn>-<short-name>/."""
    return Path(NOTES_DIR) / exp / "runs" / run


def write_json(path: Path, obj: dict) -> None:
    """Atomic, so a watcher never reads half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def code_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or os.environ.get("ONCOLENS_COMMIT", "unknown")


class RunLog:
    """Open at the start of a run, call `log` for every event, `finish` or `fail` at the end."""

    def __init__(self, out_dir: Path, config: dict, exp: str, run: str, total_steps: int, use_wandb: bool = True):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.exp, self.run, self.total_steps = exp, run, total_steps
        self.info: dict = {"exp": exp, "run": run, "commit": code_commit(), "started": time.time()}
        write_json(self.dir / "config.json", {**self.info, "config": config})
        self.rows: list[dict] = []  # everything logged so far, for the detectors
        self._metrics = (self.dir / "metrics.jsonl").open("a", encoding="utf-8")
        self._events = (self.dir / "events.jsonl").open("a", encoding="utf-8")
        self._wandb = None
        if use_wandb:
            self._wandb = _init_wandb(config, exp, run, self.info["commit"])
            self.info["wandb_url"] = self._wandb.url
            self.info["wandb_path"] = f"{self._wandb.entity}/{self._wandb.project}/{self._wandb.id}"
        self.status("running", step=0)

    def note(self, **info) -> None:
        """Facts about the run that are not a time series (data aggregates, links). Kept in status.json and W&B config."""
        self.info.update(info)
        if self._wandb:
            self._wandb.config.update(info, allow_val_change=True)

    def log(self, step: int, **scalars: float) -> None:
        row = {"step": step, "time": round(time.time(), 3), **scalars}
        self.rows.append(row)
        self._metrics.write(json.dumps(row) + "\n")
        self._metrics.flush()
        if self._wandb:
            self._wandb.log(scalars, step=step)

    def event(self, event: dict) -> None:
        """An event raised by a detector: to events.jsonl, to the process output, and to W&B (as `events/<kind>` = its step)."""
        self._events.write(json.dumps({"time": round(time.time(), 3), **event}) + "\n")
        self._events.flush()
        print(f"[event {event['severity']}] step {event['step']}: {event['message']}", flush=True)
        if self._wandb:
            self._wandb.log({f"events/{event['kind']}": event["step"]}, step=max(event["step"], self.rows[-1]["step"] if self.rows else 0))
            self._wandb.summary[f"event/{event['kind']}/{event['subject']}"] = event["message"]

    def status(self, state: str, step: int, **extra) -> None:
        write_json(self.dir / "status.json", {"state": state, "step": step, "total_steps": self.total_steps,
                                              "heartbeat": time.time(), **self.info, **extra})

    def finish(self, step: int, summary: dict) -> None:
        write_json(self.dir / "summary.json", {**self.info, **summary})
        self.status("finished", step)
        self._close(summary, exit_code=0)

    def fail(self, step: int, error: str) -> None:
        self.status("failed", step, error=error)
        self._close({}, exit_code=1)

    def _close(self, summary: dict, exit_code: int) -> None:
        self._metrics.close()
        self._events.close()
        if self._wandb:
            self._wandb.summary.update(summary)
            self._wandb.finish(exit_code=exit_code)


def _init_wandb(config: dict, exp: str, run: str, commit: str):
    import wandb

    key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_TOKEN")
    if not key:
        raise SystemExit("missing secret: WANDB_TOKEN (set it, or run with track.wandb=false). This is a setup failure, not a result.")
    os.environ["WANDB_API_KEY"] = key
    os.environ.setdefault("WANDB_SILENT", "true")
    return wandb.init(project=os.environ.get("WANDB_PROJECT", "oncolens-explorer"), name=f"{exp}/{run}", group=exp,
                      config={**config, "exp": exp, "run": run, "commit": commit}, dir=os.environ.get("WANDB_DIR"))

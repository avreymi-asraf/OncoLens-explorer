"""Observation: read a run while it runs, from its run folder or from Weights & Biases.

    uv run -m oncolens.watch notes/<exp-name>/runs/<run>            # one look
    uv run -m oncolens.watch notes/<exp-name>/runs/<run> --follow   # keep looking until the run ends
    uv run -m oncolens.watch wandb:<entity>/<project>/<run-id>      # the same, from anywhere, without the folder

The functions are the tools; the command line only calls them. In a notebook or an analysis script:

    run = read_run("notes/<exp-name>/runs/<run>")
    steps, loss = series(run.rows, "train/loss")

A new view of a run (a metric, a check, a plot) is added here, for every experiment, not inside one experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

STALE_AFTER_SECONDS = 120  # a running run whose heartbeat is older than this is probably dead


@dataclass
class Run:
    source: str
    status: dict      # state, step, total_steps, heartbeat, links (see oncolens/runlog.py)
    rows: list[dict]  # metrics.jsonl rows: step, time, <group>/<name> scalars


def read_run(source: str | Path) -> Run:
    """A run folder path, or `wandb:<entity>/<project>/<run-id>`."""
    source = str(source)
    return _read_wandb(source.removeprefix("wandb:")) if source.startswith("wandb:") else _read_folder(Path(source))


def _read_folder(folder: Path) -> Run:
    status_path, metrics_path = folder / "status.json", folder / "metrics.jsonl"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"state": "not started"}
    rows = []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # the last line can be half written while the run is going
                continue
    return Run(str(folder), status, rows)


def _read_wandb(path: str) -> Run:
    import wandb

    os.environ.setdefault("WANDB_API_KEY", os.environ.get("WANDB_TOKEN", ""))
    run = wandb.Api().run(path)
    rows = [{"step": r.pop("_step"), "time": r.pop("_timestamp", None), **{k: v for k, v in r.items() if not k.startswith("_")}}
            for r in run.scan_history()]
    state = {"running": "running", "finished": "finished"}.get(run.state, "failed")
    status = {"state": state, "step": rows[-1]["step"] if rows else 0, "total_steps": run.config.get("train", {}).get("steps"),
              "heartbeat": rows[-1]["time"] if rows else None, "exp": run.config.get("exp"), "run": run.config.get("run"),
              "commit": run.config.get("commit"), "wandb_url": run.url}
    return Run(f"wandb:{path}", status, rows)


def series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    """(steps, values) of one scalar, skipping rows that do not have it."""
    points = [(r["step"], r[key]) for r in rows if r.get(key) is not None]
    return [s for s, _ in points], [v for _, v in points]


def keys(rows: list[dict]) -> list[str]:
    return sorted({k for r in rows for k in r} - {"step", "time"})


def summarize(run: Run, window: int = 20) -> dict:
    """Where the run stands: progress, speed, and for each scalar its latest value and how it moved."""
    status, rows = run.status, run.rows
    out: dict = {"source": run.source, "state": status.get("state"), "step": status.get("step", 0),
                 "total_steps": status.get("total_steps"), "exp": status.get("exp"), "run": status.get("run"),
                 "links": {k: status[k] for k in ("wandb_url", "hub_url") if status.get(k)}, "scalars": {}}
    if status.get("heartbeat"):
        out["heartbeat_age_seconds"] = round(time.time() - status["heartbeat"], 1)
    timed = [r for r in rows if r.get("time")]
    if len(timed) > 1 and timed[-1]["time"] > timed[0]["time"]:
        out["steps_per_second"] = round((timed[-1]["step"] - timed[0]["step"]) / (timed[-1]["time"] - timed[0]["time"]), 2)
    for key in keys(rows):
        steps, values = series(rows, key)
        recent, earlier = values[-window:], values[-2 * window:-window]
        entry = {"last": values[-1], "at_step": steps[-1], "n": len(values), "recent_mean": sum(recent) / len(recent)}
        if earlier:
            entry["change"] = entry["recent_mean"] - sum(earlier) / len(earlier)  # recent window minus the one before
        out["scalars"][key] = entry
    out["warnings"] = checks(run)
    return out


def checks(run: Run) -> list[str]:
    """Signs that a run needs attention now. Each is one plain sentence."""
    found = []
    status = run.status
    if status.get("state") == "failed":
        found.append(f"The run failed: {status.get('error', 'no error recorded')}")
    if status.get("state") == "running" and status.get("heartbeat") and time.time() - status["heartbeat"] > STALE_AFTER_SECONDS:
        found.append(f"No heartbeat for {time.time() - status['heartbeat']:.0f}s while the state says running: the process may be dead.")
    for key in keys(run.rows):
        _, values = series(run.rows, key)
        if any(isinstance(v, float) and not math.isfinite(v) for v in values):
            found.append(f"{key} has non-finite values.")
    return found


def render(summary: dict) -> str:
    total = summary.get("total_steps")
    head = f"{summary.get('exp')}/{summary.get('run')}  [{summary['state']}]  step {summary['step']}" + (f"/{total}" if total else "")
    if "steps_per_second" in summary:
        head += f"  {summary['steps_per_second']} steps/s"
    if "heartbeat_age_seconds" in summary and summary["state"] == "running":
        head += f"  heartbeat {summary['heartbeat_age_seconds']}s ago"
    lines = [head]
    for key, entry in summary["scalars"].items():
        change = f"  change {entry['change']:+.4f}" if "change" in entry else ""
        lines.append(f"  {key:<24} last {entry['last']:.4f} @ {entry['at_step']:<6} recent mean {entry['recent_mean']:.4f}{change}")
    lines += [f"  {name}: {url}" for name, url in summary["links"].items()]
    lines += [f"  WARNING: {w}" for w in summary["warnings"]]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Look at a run while it runs.")
    parser.add_argument("source", help="Run folder, or wandb:<entity>/<project>/<run-id>")
    parser.add_argument("--follow", action="store_true", help="Keep printing until the run finishes or fails")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between looks with --follow")
    parser.add_argument("--json", action="store_true", help="Print the summary as one JSON object")
    args = parser.parse_args()
    while True:
        summary = summarize(read_run(args.source))
        print(json.dumps(summary) if args.json else render(summary), flush=True)
        if not args.follow or summary["state"] in ("finished", "failed"):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

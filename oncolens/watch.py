"""Observation: read a run while it runs, from its run folder or from Weights & Biases.

    uv run -m oncolens.watch notes/<exp-name>/runs/<run>            # one look
    uv run -m oncolens.watch notes/<exp-name>/runs/<run> --follow   # keep looking until the run ends
    uv run -m oncolens.watch wandb:<entity>/<project>/<run-id>      # the same, from anywhere, without the folder
    uv run -m oncolens.watch <source> --report                      # write analysis.md: the run's story so far

What a look shows: progress and speed, the curves as sparklines, what each layer is doing (gradient, size of its
updates, dead units), how the patients are being learned, and the events the detectors raise (oncolens/detect.py).

The functions are the tools; the command line only calls them. In a notebook or an analysis script:

    run = read_run("notes/<exp-name>/runs/<run>")
    steps, loss = series(run.rows, "train/loss")
    layers(run.rows)        # one row per layer: latest gradient norm, update size, dead units
    events(run)             # what the detectors say about this run, recomputed from its rows
    analyze(run)            # the story of the run as a dict: best step, when curves parted, who moved when

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

from oncolens.detect import detect, keys_under, series

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
    out["layers"] = layers(rows)
    out["rows"] = rows  # for render(); dropped from --json
    return out


def events(run: Run) -> list[dict]:
    """What the detectors say about this run, recomputed from its rows (works on runs older than a detector)."""
    return detect(run.rows)


def checks(run: Run) -> list[str]:
    """Signs that a run needs attention now. Each is one plain sentence."""
    found = []
    status = run.status
    if status.get("state") == "failed":
        found.append(f"The run failed: {status.get('error', 'no error recorded')}")
    if status.get("state") == "running" and status.get("heartbeat") and time.time() - status["heartbeat"] > STALE_AFTER_SECONDS:
        found.append(f"No heartbeat for {time.time() - status['heartbeat']:.0f}s while the state says running: the process may be dead.")
    return found + [f"[{e['severity']}] {e['message']}" for e in events(run)]


def layers(rows: list[dict]) -> list[dict]:
    """One dict per layer with the latest probe values: grad, weight, update, and dead (when the layer feeds a ReLU)."""
    names = [k.split("/", 1)[1] for k in keys_under(rows, "update/")]
    dead_after = {k.split("/")[1]: series(rows, k)[1][-1] for k in keys_under(rows, "act/") if k.endswith("/dead")}
    table = []
    for name in names:
        entry = {"layer": name}
        for kind in ("grad", "weight", "update"):
            values = series(rows, f"{kind}/{name}")[1]
            if values:
                entry[kind] = values[-1]
        update_steps, updates = series(rows, f"update/{name}")
        entry["update_peak_step"] = update_steps[max(range(len(updates)), key=updates.__getitem__)]
        if name in dead_after:
            entry["dead"] = dead_after[name]
        table.append(entry)
    return table


def sparkline(values: list[float], width: int = 40) -> str:
    """The shape of a series in one line of text (averaged into `width` buckets, scaled min to max)."""
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not values:
        return ""
    size = max(1, math.ceil(len(values) / width))
    buckets = [sum(values[i:i + size]) / len(values[i:i + size]) for i in range(0, len(values), size)]
    low, high = min(buckets), max(buckets)
    return "".join("▁▂▃▄▅▆▇█"[int((b - low) / (high - low) * 7) if high > low else 0] for b in buckets)


def analyze(run: Run) -> dict:
    """The story of the run so far, as facts: where validation was best, when the curves parted, which layer
    moved when, how the patients were learned, and every event."""
    rows = run.rows
    out: dict = {"state": run.status.get("state"), "step": run.status.get("step"), "events": events(run)}
    steps, val = series(rows, "val/loss")
    _, train = series(rows, "train_eval/loss")
    if val:
        best = min(range(len(val)), key=val.__getitem__)
        out["best_val"] = {"step": steps[best], "val_loss": val[best], "val_auroc": series(rows, "val/auroc")[1][best]}
        out["now"] = {"step": steps[-1], "val_loss": val[-1], "train_loss": train[-1] if train else None,
                      "val_auroc": series(rows, "val/auroc")[1][-1]}
        out["steps_after_best"] = steps[-1] - steps[best]
    table = layers(rows)
    if table:
        out["layers"] = table
    for key in ("patients/train_right", "patients/val_right", "patients/train_forgotten", "pred/ece", "gap/loss"):
        values = series(rows, key)[1]
        if values:
            out[key] = {"first": values[0], "last": values[-1]}
    return out


def report(run: Run) -> str:
    """analyze() as a short Markdown page for the run folder."""
    a, status = analyze(run), run.status
    lines = [f"# Run `{status.get('exp')}/{status.get('run')}`: what happened", "",
             f"State **{a['state']}** at step {a['step']} of {status.get('total_steps')}. Code commit `{status.get('commit')}`.", ""]
    if "best_val" in a:
        b, n = a["best_val"], a["now"]
        lines += ["## Curves", f"* Validation loss was best at step {b['step']}: {b['val_loss']:.4f} (AUROC {b['val_auroc']:.3f}).",
                  f"* At step {n['step']}: validation loss {n['val_loss']:.4f}, training loss {n['train_loss']:.4f}, AUROC {n['val_auroc']:.3f}. "
                  f"{a['steps_after_best']} steps were run after the best point.", ""]
    if "layers" in a:
        lines += ["## Layers (latest probe)", "| Layer | Gradient norm | Weight norm | Update / weight | Update peaked at step | Dead units |", "|---|---|---|---|---|---|"]
        lines += [f"| {l['layer']} | {l.get('grad', float('nan')):.3g} | {l.get('weight', float('nan')):.3g} | {l.get('update', float('nan')):.2e} | "
                  f"{l['update_peak_step']} | {format(l['dead'], '.0%') if 'dead' in l else 'n/a'} |" for l in a["layers"]]
        lines.append("")
    named = {"patients/train_right": "Training patients on the right side of 0.5", "patients/val_right": "Validation patients on the right side of 0.5",
             "patients/train_forgotten": "Training patients that were right and became wrong (count)", "pred/ece": "Calibration error on validation",
             "gap/loss": "Validation loss minus training loss"}
    rows = [f"| {label} | {a[key]['first']:.3f} | {a[key]['last']:.3f} |" for key, label in named.items() if key in a]
    if rows:
        lines += ["## Patients and predictions", "| | First evaluation | Latest |", "|---|---|---|", *rows, ""]
    lines += ["## Events", *([f"* step {e['step']} [{e['severity']}] {e['message']}" for e in a["events"]] or ["* None."]), ""]
    return "\n".join(lines)


def render(summary: dict) -> str:
    total = summary.get("total_steps")
    head = f"{summary.get('exp')}/{summary.get('run')}  [{summary['state']}]  step {summary['step']}" + (f"/{total}" if total else "")
    if "steps_per_second" in summary:
        head += f"  {summary['steps_per_second']} steps/s"
    if "heartbeat_age_seconds" in summary and summary["state"] == "running":
        head += f"  heartbeat {summary['heartbeat_age_seconds']}s ago"
    lines = [head]
    rows = summary.get("rows", [])
    for key, entry in summary["scalars"].items():
        if key.split("/")[0] in ("grad", "weight", "update", "act", "events"):
            continue  # per-layer values are in the layer table below
        change = f"  change {entry['change']:+.4f}" if "change" in entry else ""
        lines.append(f"  {key:<26} {sparkline(series(rows, key)[1]):<40} last {entry['last']:.4f} @ {entry['at_step']:<6}{change}")
    if summary.get("layers"):
        lines.append("  layer        grad norm   update/weight   dead units")
        lines += [f"  {l['layer']:<12} {l.get('grad', float('nan')):<11.3g} {l.get('update', float('nan')):<15.2e} "
                  f"{format(l['dead'], '.0%') if 'dead' in l else 'n/a'}" for l in summary["layers"]]
    lines += [f"  {name}: {url}" for name, url in summary["links"].items()]
    lines += [f"  ! {w}" for w in summary["warnings"]]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Look at a run while it runs.")
    parser.add_argument("source", help="Run folder, or wandb:<entity>/<project>/<run-id>")
    parser.add_argument("--follow", action="store_true", help="Keep printing until the run finishes or fails")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between looks with --follow")
    parser.add_argument("--json", action="store_true", help="Print the summary as one JSON object")
    parser.add_argument("--report", action="store_true", help="Write analysis.md (the run's story so far) and print it")
    args = parser.parse_args()
    while True:
        run = read_run(args.source)
        summary = summarize(run)
        if args.report:
            text = report(run)
            if not args.source.startswith("wandb:"):
                (Path(args.source) / "analysis.md").write_text(text + "\n", encoding="utf-8")
            print(text, flush=True)
        elif args.json:
            print(json.dumps({**{k: v for k, v in summary.items() if k != "rows"}, "analysis": analyze(run)}), flush=True)
        else:
            print(render(summary), flush=True)
        if not args.follow or summary["state"] in ("finished", "failed"):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

"""Detectors: turn the stream of a run's scalars into events, each one a plain sentence about what is happening.

A detector is a pure function over the rows logged so far (`metrics.jsonl` rows) that returns events. The same
functions run inside the training loop (so an event is recorded at the step it became true) and on any finished or
running run from outside (`oncolens.watch`), including runs made before a detector existed.

    events = detect(rows)                       # every event the rows support
    Monitor().update(rows) -> new events        # only the ones not reported before

An event: {"step", "kind", "subject", "severity" ("info" | "warning" | "critical"), "message"}.
`kind` + `subject` identify it; it is reported once per run.

EXTENSION POINT: a new detector is a function decorated with `@detector`. Thresholds are constants beside it.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Callable

Event = dict
DETECTORS: list[Callable[[list[dict]], list[Event]]] = []


def detector(fn):
    DETECTORS.append(fn)
    return fn


def series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    """(steps, values) of one scalar, skipping rows that do not have it."""
    points = [(r["step"], r[key]) for r in rows if r.get(key) is not None]
    return [s for s, _ in points], [v for _, v in points]


def keys_under(rows: list[dict], prefix: str) -> list[str]:
    return sorted({k for r in rows for k in r if k.startswith(prefix)})


def _event(step: int, kind: str, subject: str, severity: str, message: str) -> Event:
    return {"step": step, "kind": kind, "subject": subject, "severity": severity, "message": message}


@detector
def non_finite(rows: list[dict]) -> list[Event]:
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                return [_event(row["step"], "non_finite", key, "critical", f"{key} became {value} at step {row['step']}: the run is producing no usable numbers from here.")]
    return []


OVERFIT_RISE = 0.10      # validation loss this far (relative) above its best ...
OVERFIT_PATIENCE = 3     # ... for this many evaluations in a row, while training loss kept falling


@detector
def overfitting(rows: list[dict]) -> list[Event]:
    steps, val = series(rows, "val/loss")
    _, train = series(rows, "train_eval/loss")
    if len(val) < OVERFIT_PATIENCE + 1 or len(train) != len(val):
        return []
    for i in range(OVERFIT_PATIENCE, len(val)):
        best = min(range(i), key=lambda j: val[j])
        recent = range(i - OVERFIT_PATIENCE + 1, i + 1)
        if best < recent[0] and all(val[j] > val[best] * (1 + OVERFIT_RISE) for j in recent) and train[i] < train[best]:
            return [_event(steps[i], "overfitting", "val/loss", "warning",
                           f"Validation loss was best at step {steps[best]} ({val[best]:.4f}) and is {val[i]:.4f} at step {steps[i]}, "
                           f"while training loss went from {train[best]:.4f} to {train[i]:.4f}: the model is fitting the training patients only.")]
    return []


STALL_EVALS = 6          # after this many evaluations ...
STALL_IMPROVEMENT = 0.01  # ... training loss has improved by less than this (relative)


@detector
def not_learning(rows: list[dict]) -> list[Event]:
    steps, train = series(rows, "train_eval/loss")
    if len(train) < STALL_EVALS:
        return []
    i = STALL_EVALS - 1
    if all(math.isfinite(v) for v in train[: i + 1]) and train[i] > train[0] * (1 - STALL_IMPROVEMENT):
        moving = [v[-1] for k in keys_under(rows, "update/") if (v := series([r for r in rows if r["step"] <= steps[i]], k)[1])]
        detail = f" Layers move by about {median(moving):.1e} of their size per step." if moving else ""
        return [_event(steps[i], "not_learning", "train_eval/loss", "warning",
                       f"Training loss is {train[i]:.4f} at step {steps[i]}, from {train[0]:.4f} at step {steps[0]}: nothing is being learned.{detail}")]
    return []


BLOWUP_FACTOR = 10.0     # training loss this many times its first value


@detector
def blowing_up(rows: list[dict]) -> list[Event]:
    steps, loss = series(rows, "train/loss")
    for step, value in zip(steps[1:], loss[1:]):
        if math.isfinite(value) and value > loss[0] * BLOWUP_FACTOR:
            return [_event(step, "blowing_up", "train/loss", "critical",
                           f"Training loss is {value:.3g} at step {step}, {value / loss[0]:.0f} times its first value ({loss[0]:.3g}): the updates are too large.")]
    return []


DEAD_FRACTION = 0.5      # more than this share of a layer's units never fire on validation


@detector
def dead_units(rows: list[dict]) -> list[Event]:
    found = []
    for key in keys_under(rows, "act/"):
        if not key.endswith("/dead"):
            continue
        for step, value in zip(*series(rows, key)):
            if value > DEAD_FRACTION:
                layer = key.split("/")[1]
                found.append(_event(step, "dead_units", layer, "warning",
                                    f"{value:.0%} of the units after {layer} are zero for every validation patient at step {step}: that part of the model has stopped contributing."))
                break
    return found


UPDATE_TOO_LARGE = 0.1   # a layer changing by more than this share of its own size in one step
UPDATE_TOO_SMALL = 1e-6  # a layer changing by less than this, after the first steps


@detector
def update_size(rows: list[dict]) -> list[Event]:
    found = []
    for key in keys_under(rows, "update/"):
        layer = key.split("/", 1)[1]
        steps, values = series(rows, key)
        for step, value in zip(steps, values):
            if value > UPDATE_TOO_LARGE:
                found.append(_event(step, "update_too_large", layer, "warning",
                                    f"Layer {layer} changed by {value:.0%} of its own size in one step (step {step}): steps this large overwrite what was learned."))
                break
        if len(values) >= 5 and all(v < UPDATE_TOO_SMALL for v in values[-5:]):
            found.append(_event(steps[-5], "update_too_small", layer, "warning",
                                f"Layer {layer} has moved by less than {UPDATE_TOO_SMALL:.0e} of its size per step since step {steps[-5]}: it is effectively frozen."))
    return found


def detect(rows: list[dict]) -> list[Event]:
    """Every event the rows support, in step order."""
    return sorted((e for fn in DETECTORS for e in fn(rows)), key=lambda e: e["step"])


class Monitor:
    """Remembers what was already reported, so each event comes out once."""

    def __init__(self):
        self._seen: set[tuple[str, str]] = set()

    def update(self, rows: list[dict]) -> list[Event]:
        new = [e for e in detect(rows) if (e["kind"], e["subject"]) not in self._seen]
        self._seen.update((e["kind"], e["subject"]) for e in new)
        return new

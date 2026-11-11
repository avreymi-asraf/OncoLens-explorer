"""The run entry point: one RunConfig in, one run folder and one model on the Hub out.

    uv run -m oncolens.train --config configs/connect.toml --run 01-smoke [section.key=value ...]

EXTENSION POINT: `training_step` is where a new objective or update rule goes in. The loop around it (data,
logging, evaluation, checkpoints) is shared and stays as it is.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

from oncolens.config import RunConfig, load_config
from oncolens.data import describe, load_cohort
from oncolens.detect import Monitor
from oncolens.hub import Hub
from oncolens.model import build_model
from oncolens.probes import Probes
from oncolens.runlog import RunLog, current_exp, run_dir_for


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC: the probability that a positive is scored above a negative, a tie counting as half.
    nan when one class is absent. A model that gives every patient the same score gets 0.5."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    both = np.concatenate([pos, neg])
    _, inverse, counts = np.unique(both, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]  # average rank within a group of ties
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def training_step(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor) -> dict:
    """One update. Returns the scalars to log under `train/`."""
    model.train()
    loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": loss.item()}


@torch.no_grad()
def evaluate_split(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    logits = model(x)
    return {"loss": nn.functional.binary_cross_entropy_with_logits(logits, y).item(),
            "auroc": auroc(logits.cpu().numpy(), y.cpu().numpy())}


def train(cfg: RunConfig, out_dir: Path, exp: str, run: str) -> dict:
    torch.manual_seed(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = RunLog(out_dir, cfg.to_dict(), exp, run, total_steps=cfg.train.steps, use_wandb=cfg.track.wandb)
    step = 0
    try:
        cohort = load_cohort(cfg.data)
        hub = Hub(exp, run) if cfg.track.hub else None
        log.note(device=device, data=describe(cohort), **({"hub_url": hub.url} if hub else {}))
        print(f"[run] {exp}/{run} on {device} | data {describe(cohort)}", flush=True)
        if log.info.get("wandb_url"):
            print(f"[run] live: {log.info['wandb_url']}", flush=True)

        x, y = torch.from_numpy(cohort.x).to(device), torch.from_numpy(cohort.y).to(device)
        train_idx, val_idx = torch.from_numpy(cohort.train).to(device), torch.from_numpy(cohort.val).to(device)
        model = build_model(cfg.model, x.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        sampler = torch.Generator(device="cpu").manual_seed(cfg.train.seed)
        last_eval: dict = {}
        probes, monitor = Probes(model, cfg.track.probe_every), Monitor()
        pending: list[dict] = []  # train scalars since the last metrics row

        for step in range(1, cfg.train.steps + 1):
            started = time.perf_counter()
            batch = train_idx[torch.randint(len(train_idx), (cfg.train.batch_size,), generator=sampler).to(device)]
            probing = probes.due(step)
            if probing:
                probes.before_step()
            step_scalars = {f"train/{k}": v for k, v in training_step(model, optimizer, x[batch], y[batch]).items()}
            step_scalars["perf/step_seconds"] = time.perf_counter() - started
            pending.append(step_scalars)
            evaluating = step % cfg.train.eval_every == 0 or step == cfg.train.steps
            saving = hub is not None and (step % cfg.train.checkpoint_every == 0 or step == cfg.train.steps)
            if not (probing or evaluating or saving or step % cfg.track.log_every == 0):
                continue  # between metrics rows nothing is written
            scalars = {k: sum(p[k] for p in pending) / len(pending) for k in pending[0]}
            pending = []
            if probing:
                scalars |= probes.after_step()
            if evaluating:
                last_eval = {f"val/{k}": v for k, v in evaluate_split(model, x[val_idx], y[val_idx]).items()}
                last_eval |= {f"train_eval/{k}": v for k, v in evaluate_split(model, x[train_idx], y[train_idx]).items()}
                scalars |= last_eval | probes.on_eval(x[train_idx], y[train_idx], x[val_idx], y[val_idx])
                print(f"[step {step}/{cfg.train.steps}] " + " ".join(f"{k}={v:.4f}" for k, v in last_eval.items()), flush=True)
            log.log(step, **scalars)
            if probing or evaluating:
                for event in monitor.update(log.rows):
                    log.event(event)
            if saving:
                log.note(hub_commit=hub.save(model, out_dir, step, {"exp": exp, "run": run, "config": cfg.to_dict(),
                                                                    "feature_names": cohort.feature_names, **last_eval}))
            log.status("running", step)

        summary = {"steps": step, **last_eval}
        if hub:  # the saved model can be read back and is the model we trained
            state = hub.load_state(out_dir / ".hub_check")
            summary["hub_roundtrip_ok"] = all(torch.equal(v.cpu(), state[k]) for k, v in model.state_dict().items())
        log.finish(step, summary)
        print(f"[run] finished: {summary}", flush=True)
        return summary
    except BaseException as error:  # the run folder must say that the run died, and why
        log.fail(step, f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one training run from one config.")
    parser.add_argument("--config", help="TOML file under configs/")
    parser.add_argument("--run", required=True, help="Run folder name: <nn>-<short-name>, e.g. 01-smoke")
    parser.add_argument("--exp", default=None, help="Experiment name (default: from the lab's environment)")
    parser.add_argument("--out", type=Path, default=None, help="Run folder (default: notes/<exp>/runs/<run>)")
    parser.add_argument("overrides", nargs="*", help="section.key=value")
    args = parser.parse_args()
    exp = args.exp or current_exp()
    train(load_config(args.config, args.overrides), args.out or run_dir_for(exp, args.run), exp, args.run)


if __name__ == "__main__":
    main()

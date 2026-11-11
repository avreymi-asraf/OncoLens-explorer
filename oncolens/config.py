"""The config object of a run. A variant of an experiment is a value here, never a copied script.

    cfg = load_config("configs/connect.toml", ["train.steps=50", "model.hidden=[32]"])
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class DataConfig:
    study: str = "msk_chord_2024"
    n_patients: int = 2000
    val_fraction: float = 0.2
    seed: int = 0


@dataclass
class ModelConfig:
    name: str = "mlp"
    hidden: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.1


@dataclass
class TrainConfig:
    steps: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    eval_every: int = 25
    checkpoint_every: int = 100
    seed: int = 0


@dataclass
class TrackConfig:
    wandb: bool = True  # live metrics to Weights & Biases
    hub: bool = True    # checkpoints to the Hugging Face Hub (private repo)
    log_every: int = 1  # write a metrics row every N steps (train scalars are averaged over the N steps)
    probe_every: int = 0  # take the in-loop probes (oncolens/probes.py) every N steps; 0 = off


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    track: TrackConfig = field(default_factory=TrackConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _apply(section: object, values: dict, where: str) -> None:
    known = {f.name for f in fields(section)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"unknown config key '{where}.{key}' (known: {sorted(known)})")
        setattr(section, key, value)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> RunConfig:
    """Defaults, then the TOML file, then `section.key=value` overrides (values parsed as JSON, else kept as text)."""
    cfg = RunConfig()
    if path:
        for name, values in tomllib.loads(Path(path).read_text(encoding="utf-8")).items():
            if not hasattr(cfg, name) or not is_dataclass(getattr(cfg, name)):
                raise KeyError(f"unknown config section '{name}'")
            _apply(getattr(cfg, name), values, name)
    for item in overrides or []:
        dotted, _, raw = item.partition("=")
        name, _, key = dotted.partition(".")
        if not key or not hasattr(cfg, name):
            raise KeyError(f"override must be section.key=value, got '{item}'")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        _apply(getattr(cfg, name), {key: value}, name)
    return cfg

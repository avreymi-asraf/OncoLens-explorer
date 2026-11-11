"""Models. EXTENSION POINT: a new architecture is one function registered in MODELS and chosen with `model.name`.

A model maps features [batch, n_features] to one logit per patient [batch].
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from oncolens.config import ModelConfig

MODELS: dict[str, Callable[[ModelConfig, int], nn.Module]] = {}


def register(name: str):
    def wrap(builder):
        MODELS[name] = builder
        return builder
    return wrap


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: list[int], dropout: float):
        super().__init__()
        layers, width = [], n_features
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)


@register("mlp")
def build_mlp(cfg: ModelConfig, n_features: int) -> nn.Module:
    return MLP(n_features, list(cfg.hidden), cfg.dropout)


def build_model(cfg: ModelConfig, n_features: int) -> nn.Module:
    if cfg.name not in MODELS:
        raise KeyError(f"unknown model '{cfg.name}' (registered: {sorted(MODELS)})")
    return MODELS[cfg.name](cfg, n_features)

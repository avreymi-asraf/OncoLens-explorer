"""Probes: measurements taken inside the training loop, so a run can be seen from inside while it runs.

Turned on with `track.probe_every = N` (0 = off). Everything is logged through `RunLog` like any other scalar, so
it reaches the run folder and W&B as it happens. Keys:

    grad/<layer>      gradient norm of the layer's weights          grad/total   norm over all parameters
    weight/<layer>    norm of the layer's weights
    update/<layer>    |change made by this step| / |weights|        (how fast the layer is moving)
    act/<layer>/dead  fraction of the layer's units (after its ReLU) that are zero for every validation patient
    act/<layer>/mean  mean activation of the layer's units over validation patients
    pred/mean, pred/std   the predicted probabilities on validation
    pred/ece          expected calibration error on validation (10 bins)
    gap/loss          validation loss minus training loss           (how far the two have parted)
    patients/train_right, patients/val_right     fraction of patients on the right side of 0.5
    patients/train_forgotten   patients that were right at an earlier evaluation and are wrong now (count so far)
    patients/val_loss_p50, patients/val_loss_p90   spread of the per-patient loss

Patient-level values never leave this module: only aggregates are returned.
`<layer>` is the name of a module that has weights (`body.0`, `body.3`, `head` for the MLP).
"""

from __future__ import annotations

import torch
from torch import nn


def weight_layers(model: nn.Module) -> dict[str, torch.nn.Parameter]:
    """{layer name: its weight tensor} for every module with a weight matrix."""
    return {name.removesuffix(".weight"): p for name, p in model.named_parameters() if name.endswith(".weight") and p.dim() > 1}


class Probes:
    """`before_step` / `after_step` around one update on probe steps; `on_eval` at every evaluation."""

    def __init__(self, model: nn.Module, every: int):
        self.model, self.every = model, every
        self.layers = weight_layers(model)
        self._before: dict[str, torch.Tensor] = {}
        self._was_right: torch.Tensor | None = None
        self._forgotten = 0
        self._acts: dict[str, torch.Tensor] = {}
        relus = [m for m in model.modules() if isinstance(m, nn.ReLU)]
        for name, module in zip(self.layers, relus):  # the i-th ReLU is the output of the i-th weight layer
            module.register_forward_hook(lambda _m, _i, out, name=name: self._acts.__setitem__(name, out.detach()))

    def due(self, step: int) -> bool:
        return self.every > 0 and (step % self.every == 0 or step == 1)

    def before_step(self) -> None:
        self._before = {name: p.detach().clone() for name, p in self.layers.items()}

    def after_step(self) -> dict[str, float]:
        """Call right after the update: gradients of that step are still on the parameters."""
        out, total = {}, 0.0
        for name, p in self.layers.items():
            weight_norm = p.detach().norm().item()
            out[f"weight/{name}"] = weight_norm
            out[f"update/{name}"] = (p.detach() - self._before[name]).norm().item() / (weight_norm + 1e-12)
            if p.grad is not None:
                out[f"grad/{name}"] = p.grad.norm().item()
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.pow(2).sum().item()
        out["grad/total"] = total ** 0.5
        return out

    @torch.no_grad()
    def on_eval(self, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor) -> dict[str, float]:
        if self.every <= 0:
            return {}
        self.model.eval()
        out: dict[str, float] = {}
        train_logits = self.model(x_train)
        val_logits = self.model(x_val)  # hooks now hold validation activations
        for name, act in self._acts.items():
            out[f"act/{name}/dead"] = (act <= 0).all(dim=0).float().mean().item()
            out[f"act/{name}/mean"] = act.mean().item()

        prob = torch.sigmoid(val_logits)
        out["pred/mean"], out["pred/std"] = prob.mean().item(), prob.std().item()
        out["pred/ece"] = _ece(prob, y_val)

        loss = nn.functional.binary_cross_entropy_with_logits
        val_losses = loss(val_logits, y_val, reduction="none")
        out["gap/loss"] = val_losses.mean().item() - loss(train_logits, y_train).item()
        out["patients/val_loss_p50"] = val_losses.quantile(0.5).item()
        out["patients/val_loss_p90"] = val_losses.quantile(0.9).item()
        out["patients/val_right"] = ((val_logits > 0) == (y_val > 0.5)).float().mean().item()
        right = (train_logits > 0) == (y_train > 0.5)
        out["patients/train_right"] = right.float().mean().item()
        if self._was_right is not None:
            self._forgotten += int((self._was_right & ~right).sum().item())
        self._was_right = right
        out["patients/train_forgotten"] = float(self._forgotten)
        return out


def _ece(prob: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    """Expected calibration error: mean |predicted probability - observed rate| over equal-width bins, weighted by size."""
    edges = torch.linspace(0, 1, bins + 1, device=prob.device)
    index = torch.bucketize(prob, edges[1:-1])
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.float().mean().item() * abs(prob[mask].mean().item() - y[mask].mean().item())
    return total

"""Model checkpoints go to the Hugging Face Hub, in a private repo, under <exp-name>/<run>/.

Weights are never committed to git and never left only on a rented machine. One repo holds every run:
`$HF_MODEL_REPO`, default `<token owner>/oncolens-explorer`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars
from safetensors.torch import load_file, save_file

disable_progress_bars()  # run logs are read by people and agents; progress bars drown them


class Hub:
    def __init__(self, exp: str, run: str):
        if not os.environ.get("HF_TOKEN"):
            raise SystemExit("missing secret: HF_TOKEN (set it, or run with track.hub=false). This is a setup failure, not a result.")
        self.api = HfApi(token=os.environ["HF_TOKEN"])
        self.repo = os.environ.get("HF_MODEL_REPO") or f"{self.api.whoami()['name']}/oncolens-explorer"
        self.prefix = f"{exp}/{run}"
        self.api.create_repo(self.repo, repo_type="model", private=True, exist_ok=True)

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/tree/main/{self.prefix}"

    def save(self, model: torch.nn.Module, out_dir: Path, step: int, meta: dict) -> str:
        """Write model.safetensors + model.json next to the run's files and upload both. Returns the Hub commit."""
        save_file(model.state_dict(), out_dir / "model.safetensors")
        (out_dir / "model.json").write_text(json.dumps({"step": step, **meta}, indent=2) + "\n", encoding="utf-8")
        commit = None
        for name in ("model.safetensors", "model.json"):
            commit = self.api.upload_file(path_or_fileobj=out_dir / name, path_in_repo=f"{self.prefix}/{name}",
                                          repo_id=self.repo, commit_message=f"{self.prefix} step {step}")
        return commit.oid

    def load_state(self, to_dir: Path) -> dict:
        """Download the run's latest weights from the Hub (used to check that what was saved can be read back)."""
        path = self.api.hf_hub_download(self.repo, f"{self.prefix}/model.safetensors", local_dir=to_dir)
        return load_file(path)

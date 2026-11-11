"""The runner: send a run to a machine, mirror its run folder back while it runs, and clean up.

    uv run -m oncolens.runner --where local --run 01-smoke --config configs/connect.toml [section.key=value ...]
    uv run -m oncolens.runner --where colab --run 02-colab --config configs/connect.toml [--gpu T4]

Wherever the run goes, its run folder is `notes/<exp-name>/runs/<run>/` in this checkout. For a Colab run that
folder is a mirror, refreshed every `--poll` seconds, so `oncolens.watch` works on it while the run is going.

A Colab run ships COMMITTED code only (`git archive HEAD`), gets its secrets from an uploaded env file that never
enters a command line, runs in the background on the VM, and the VM is always released at the end.
Needs the Colab CLI (`colab`) and `ssh` on this machine; see the lab's `colab-cli` skill.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from oncolens.runlog import current_exp, run_dir_for

SECRETS = ("HF_TOKEN", "WANDB_TOKEN")            # sent to the VM in an env file
PASSED = ("HF_MODEL_REPO", "WANDB_PROJECT")      # optional, not secret
ADC_PATH = Path.home() / ".config" / "colab-cli" / "adc.json"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
MIRRORED = ("config.json", "metrics.jsonl", "events.jsonl", "status.json", "summary.json", "model.json", "train.log")


def train_command(args: argparse.Namespace, exp: str, out: str) -> list[str]:
    command = ["uv", "run", "-m", "oncolens.train", "--run", args.run, "--exp", exp, "--out", out]
    if args.config:
        command += ["--config", args.config]
    return command + args.overrides


def run_local(args: argparse.Namespace, exp: str) -> int:
    return subprocess.run(train_command(args, exp, str(run_dir_for(exp, args.run))), check=False).returncode


class Colab:
    """One Colab session, driven through the Colab CLI with headless credentials."""

    def __init__(self, session: str):
        self.session = session
        if not ADC_PATH.exists():  # headless login: the user's authorized_user JSON, from the environment
            token = os.environ.get("COLAB_TOKEN_B64")
            if not token:
                raise SystemExit("missing secret: COLAB_TOKEN_B64 (no Colab login on this machine). This is a setup failure, not a result.")
            ADC_PATH.parent.mkdir(parents=True, exist_ok=True)
            ADC_PATH.touch(mode=0o600)
            ADC_PATH.write_bytes(base64.b64decode(token))
        if not SSH_KEY.exists():
            SSH_KEY.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(SSH_KEY)], check=True)
        self.env = {**os.environ, "GOOGLE_APPLICATION_CREDENTIALS": str(ADC_PATH)}

    def cli(self, *args: str, **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(["colab", "--auth=adc", *args], env=self.env, **kwargs)

    def ssh(self, command: str, stdin: bytes | None = None, check: bool = True, timeout: int = 600) -> bytes:
        """Run one shell line on the VM (in /content). Returns its stdout. Keep commands short: a long silent
        command loses its connection, so long work is launched in the background on the VM and polled."""
        for attempt in range(5):
            try:
                done = self.cli("ssh", "-s", self.session, "-i", str(SSH_KEY), "--", command, input=stdin,
                                stdin=None if stdin is not None else subprocess.DEVNULL, stdout=subprocess.PIPE,
                                timeout=timeout, check=False)
            except subprocess.TimeoutExpired:
                done = None
            if done is not None and done.returncode != 255:  # 255 is the connection (or the VM still closing the previous one), not the command
                break
            time.sleep(5 * (attempt + 1))
        if done is None or (check and done.returncode != 0):
            if check:
                raise RuntimeError(f"command failed on the VM (exit {getattr(done, 'returncode', 'timeout')}): {command}")
            return b""
        return done.stdout

    def start(self, gpu: str | None) -> None:
        self.cli("new", "-s", self.session, *(["--gpu", gpu] if gpu else []), check=True)

    def stop(self) -> None:
        self.cli("stop", "-s", self.session, check=False)

    def upload_env(self, values: dict[str, str]) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.env"
            path.touch(mode=0o600)
            path.write_text("".join(f"{k}={shlex.quote(v)}\n" for k, v in values.items()), encoding="utf-8")
            self.cli("upload", "-s", self.session, str(path), "/content/run.env", check=True, stdout=subprocess.DEVNULL)

    def mirror(self, remote_dir: str, local_dir: Path) -> bool | None:
        """Copy the run's small files from the VM, each one replaced atomically, in ONE connection (the VM
        accepts one ssh connection at a time and frees it slowly). Returns whether the run's process is still
        alive, or None when the VM could not be reached."""
        names = " ".join(MIRRORED)
        blob = self.ssh(f"cd {remote_dir} && (kill -0 $(cat /content/train.pid) 2>/dev/null && echo RUNNING || echo EXITED) > run_state "
                        f"&& tar cz --ignore-failed-read run_state {names} 2>/dev/null", check=False, timeout=120)
        alive = None
        try:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
                local_dir.mkdir(parents=True, exist_ok=True)
                for member in archive.getmembers():
                    if member.name == "run_state":
                        alive = b"RUNNING" in archive.extractfile(member).read()
                    elif member.isfile() and member.name in MIRRORED:
                        tmp = local_dir / (member.name + ".tmp")
                        tmp.write_bytes(archive.extractfile(member).read())
                        tmp.replace(local_dir / member.name)
        except (tarfile.TarError, EOFError):
            return None  # nothing or a truncated transfer came back: the next poll brings a whole one
        return alive


def run_colab(args: argparse.Namespace, exp: str) -> int:
    missing = [name for name in SECRETS if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"missing secrets: {missing}. This is a setup failure, not a result.")
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip():
        print("[runner] WARNING: uncommitted changes are NOT shipped. The VM runs the committed code.", flush=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    local_dir = run_dir_for(exp, args.run)
    session = re.sub(r"[^a-z0-9-]+", "-", f"oncolens-{exp}-{args.run}".lower()).strip("-")
    code, out = "/content/code", "/content/code/run_out"
    colab = Colab(session)

    print(f"[runner] session {session}" + (f" (GPU {args.gpu})" if args.gpu else " (CPU)"), flush=True)
    colab.start(args.gpu)
    try:
        archive = subprocess.run(["git", "archive", "--format=tar.gz", "HEAD"], capture_output=True, check=True).stdout
        colab.ssh(f"rm -rf {code} && mkdir -p {code} && tar xz -C {code}", stdin=archive)
        env = {name: os.environ[name] for name in (*SECRETS, *PASSED) if os.environ.get(name)}
        colab.upload_env({**env, "ONCOLENS_COMMIT": commit})
        # Install and train in one background job on the VM: both outlive any single connection.
        launch = " ".join(shlex.quote(part) for part in train_command(args, exp, out))
        colab.ssh(f"cd {code} && mkdir -p {out} && set -a && . /content/run.env && set +a && "
                  f"(nohup sh -c 'uv sync -q && {launch}' > {out}/train.log 2>&1 & echo $! > /content/train.pid)")
        print(f"[runner] launched (the VM installs the environment first). Watch it: uv run -m oncolens.watch {local_dir} --follow", flush=True)

        seen_step = -1
        while True:
            alive = colab.mirror(out, local_dir)
            status = _read_status(local_dir)
            if status.get("step", -1) != seen_step:
                seen_step = status.get("step", -1)
                print(f"[runner] {status.get('state', 'installing')} step {seen_step}/{status.get('total_steps')}"
                      + (f"  {status['wandb_url']}" if status.get("wandb_url") else ""), flush=True)
            if alive is False:  # that mirror was taken after the process ended, so it is the final one
                break
            time.sleep(args.poll)
        log_path = local_dir / "train.log"
        if log_path.exists():
            print("[runner] last lines of train.log:\n" + "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]), flush=True)
        print(f"[runner] run ended: {status.get('state')}", flush=True)
        return 0 if status.get("state") == "finished" else 1
    finally:
        colab.stop()
        print(f"[runner] session {session} released", flush=True)


def _read_status(local_dir: Path) -> dict:
    path = local_dir / "status.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except json.JSONDecodeError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one run to a machine and bring its run folder back.")
    parser.add_argument("--where", choices=("local", "colab"), required=True)
    parser.add_argument("--run", required=True, help="Run folder name: <nn>-<short-name>")
    parser.add_argument("--exp", default=None, help="Experiment name (default: from the lab's environment)")
    parser.add_argument("--config", help="TOML file under configs/")
    parser.add_argument("--gpu", default=None, help="Colab accelerator, e.g. T4, L4, A100 (default: CPU)")
    parser.add_argument("--poll", type=float, default=10.0, help="Seconds between mirrors of a Colab run")
    parser.add_argument("overrides", nargs="*", help="section.key=value")
    args = parser.parse_args()
    exp = args.exp or current_exp()
    sys.exit(run_local(args, exp) if args.where == "local" else run_colab(args, exp))


if __name__ == "__main__":
    main()

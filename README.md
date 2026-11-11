# OncoLens-explorer

Experiment code for OncoLens: data in, a watched training run, a model out. The default branch is empty on purpose:
experiments live on `exp/<exp-name>` branches and are run from the lab repository (`OncoLens-expriments`), whose page
`cortex/systems/OncoLens-explorer.md` describes this repo's shared components and extension points.

```bash
uv run -m oncolens.runner --where local --run 01-smoke --config configs/connect.toml train.steps=40   # one run, here
uv run -m oncolens.runner --where colab --run 02-colab --config configs/connect.toml                  # one run, on a Colab VM
uv run -m oncolens.watch notes/<exp-name>/runs/<run> --follow                                         # read it while it runs
```

Secrets come from the environment: `HF_TOKEN`, `WANDB_TOKEN`, and `COLAB_TOKEN_B64` for Colab runs. Python is `uv` only.
Dataset rows are never committed; `notes/<exp-name>/` holds each experiment's plan, round log and small run files.

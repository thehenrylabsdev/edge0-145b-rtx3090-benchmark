# Contributing

Thanks for helping. The most useful contribution is **another data point**: the same harness on a different
3090 host, another 24 GB card, or a newer engine commit.

## Add your own result

1. Follow [docs/reproduce.md](docs/reproduce.md) and keep the pinned commits, unless testing a newer engine is
   the point. In that case say so.
2. Run `scripts/host_facts.sh` and keep its output. Results from a host without its facts cannot be compared.
3. Open an issue with the **New result** template and attach `results.csv`, `results/host/` and
   `versions.txt`. Or open a pull request that adds `results/contrib/<gpu>-<date>/`.

Please do not change the measurement rules in [docs/methodology.md](docs/methodology.md) inside a result PR.
Propose a methodology change in its own issue, so that old and new numbers stay comparable.

## Code changes

- Python is stdlib-only on the pod side. Keep it that way, so the harness runs on a bare CUDA image.
- Run `ruff check .` and `shellcheck scripts/*.sh` before opening a PR. CI runs both.
- Keep secrets out: the RunPod helpers read `RUNPOD_API_KEY` from the environment and never print it.

## Corrections

If a number, a claim about the engine, or a statement about the model is wrong, open an issue with the
evidence (log line, source line, file). Corrections are listed in [CHANGELOG.md](CHANGELOG.md).

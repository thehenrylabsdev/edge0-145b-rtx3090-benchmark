## What this changes

## Checks

- [ ] `ruff check .` passes
- [ ] `shellcheck scripts/*.sh` passes
- [ ] If results changed: `python3 charts/build_charts.py` and `(cd charts && npm run render)` were re-run
- [ ] No secrets, tokens or personal data in the diff

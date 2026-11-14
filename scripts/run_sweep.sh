#!/usr/bin/env bash
# Run every spec in a JSONL file in order; one config failing does not stop the rest.
# Usage: run_sweep.sh specs.jsonl   (progress: /results/sweep-<name>.log)
set -uo pipefail
SPECS="$1"; NAME="$(basename "$SPECS" .jsonl)"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p /results/specs
n=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  n=$((n+1)); f="/results/specs/${NAME}-${n}.json"; printf '%s\n' "$line" > "$f"
  echo "[$(date -u +%H:%M:%S)] start $(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["label"])' "$f")"
  python3 "$HERE/run_config.py" "$f"
done < "$SPECS"
echo "SWEEP_DONE $NAME"

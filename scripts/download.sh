#!/usr/bin/env bash
# Download the Edge0-145B Q3_K pack (public repo, no token needed). Logs to /results/download.log.
set -euo pipefail
DEST=/models/Edge0-145B-q3k
mkdir -p "$DEST" /results
python3 -m venv /opt/hfvenv
/opt/hfvenv/bin/pip install -q "huggingface_hub[hf_xet]"
/opt/hfvenv/bin/hf download Edge0/Edge0-145B-q3k --local-dir "$DEST" \
  --exclude "assets/*" --exclude "latent_dense_q4.bin"
du -sh "$DEST"
# size check against the HF listing (bytes)
python3 - "$DEST" <<'EOF'
import os, sys
d = sys.argv[1]
want = {"experts.bin": 40076574720, "ple-fp8.gguf": 51200246112, "model-skeleton.gguf": 5991569248,
        "dense.bin": 1538625024, "latent_dense_q8.bin": 3743416320, "mtp/experts.bin": 707788800,
        "mtp/dense.bin": 116099072, "profile.bin": 196632, "labels.bin": 98304}
bad = [f for f, n in want.items() if not os.path.exists(os.path.join(d, f)) or os.path.getsize(os.path.join(d, f)) != n]
print("SIZE_CHECK", "FAIL " + ",".join(bad) if bad else "OK")
EOF
echo DOWNLOAD_DONE

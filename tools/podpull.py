#!/usr/bin/env python3
"""Copy a file out of a pod via REST exec, in base64 chunks, and verify its sha256.

  podpull.py POD_ID REMOTE LOCAL

The exec endpoint returns at most 1 MiB of output per call, so the file is read in 600 KiB slices.
"""
import base64
import hashlib
import os
import subprocess
import sys

CHUNK = 600 * 1024
pod, remote, local = sys.argv[1:4]
here = os.path.dirname(os.path.abspath(__file__))


def run(cmd: str) -> str:
    return subprocess.run([f"{here}/podexec.py", pod, cmd, "25"], check=True, capture_output=True, text=True).stdout


size = int(run(f"stat -c %s {remote}").strip())
want = run(f"sha256sum {remote}").split()[0]
data = bytearray()
for off in range(0, size, CHUNK):
    part = run(f"dd if={remote} bs={CHUNK} skip={off // CHUNK} count=1 status=none | base64 -w0")
    data += base64.b64decode(part)
got = hashlib.sha256(data).hexdigest()
if got != want:
    sys.exit(f"sha256 mismatch: {got} != {want}")
open(local, "wb").write(data)
print(f"{local}: {len(data)} bytes, sha256 {got[:16]}... OK")

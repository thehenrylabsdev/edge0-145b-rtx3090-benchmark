#!/usr/bin/env python3
"""Run a shell command in a Runpod pod via REST v2 /pods/{id}/exec. Key from RUNPOD_API_KEY (never printed).

  podexec.py POD_ID 'shell command' [timeout_s]

Prints the command's stdout (stderr to stderr) and exits with its code. timeout_s is at most 25 (API limit).
"""
import json
import os
import sys
import urllib.error
import urllib.request

pod, cmd = sys.argv[1], sys.argv[2]
timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 20
req = urllib.request.Request(f"https://api.runpod.io/v2/pods/{pod}/exec",
                             data=json.dumps({"command": ["sh", "-c", cmd], "timeoutSeconds": timeout}).encode(),
                             headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
                                      "Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=timeout + 30) as r:
        d = json.load(r)
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}", file=sys.stderr)
    sys.exit(2)
sys.stdout.write(d.get("stdout", ""))
sys.stderr.write(d.get("stderr", ""))
sys.exit(d.get("exitCode", 0) or 0)

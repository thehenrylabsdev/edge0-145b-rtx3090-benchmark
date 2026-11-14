#!/usr/bin/env python3
"""Copy a local file into a pod via REST exec (base64 chunks). podpush.py POD_ID LOCAL REMOTE"""
import base64
import os
import subprocess
import sys

pod, local, remote = sys.argv[1:4]
b = base64.b64encode(open(local, "rb").read()).decode()
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([f"{here}/podexec.py", pod, f": > {remote}.b64"], check=True)
for i in range(0, len(b), 3500):
    subprocess.run([f"{here}/podexec.py", pod, f"printf %s '{b[i:i+3500]}' >> {remote}.b64"], check=True)
finish = f"base64 -d {remote}.b64 > {remote} && rm {remote}.b64 && sha256sum {remote}"
subprocess.run([f"{here}/podexec.py", pod, finish], check=True)

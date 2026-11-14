#!/usr/bin/env python3
"""Run one benchmark configuration end to end on the pod.

  python3 run_config.py spec.json        (spec: see SPEC_KEYS below)

1. Derive a config from the pack's base config (model.json / model-bench.json): set / remove / add engine args,
   "parallel" (batch slots), env.
2. Start serve/server.py (engine strata) on it, wait until a tiny request succeeds; record the load time.
3. Run the phases with loadgen (decode / prefill), snapshotting /metrics before and after each.
4. Save server log excerpts (batch slots, expert cache), nvidia-smi, and stop the server and engine.
Results: /results/runs/<label>/result.json (+ server.log, cfg.json, metrics-*.txt).
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loadgen  # noqa: E402

SPEC_KEYS = {"label", "base", "set", "remove", "add", "parallel", "env", "phases"}
PACK = "/models/Edge0-145B-q3k"
STRATA_DIR = "/opt/edge0/edge0-strata"
PORT = 8123
LOAD_TIMEOUT_S = 1800
LOG_PATTERNS = re.compile(r"batch_slots|batch-mtp|expert[ -]cache|expert_cache|slots|VRAM|vram|free|resident|"
                          r"prefill|refus|cannot|error|ERROR|WARN|warn|INFO", re.I)


def split_flag(args: list[str], flag: str) -> tuple[int, int]:
    """Index of flag and how many words it spans (1 for a switch, 2 when a value follows)."""
    i = args.index(flag)
    has_value = i + 1 < len(args) and not args[i + 1].startswith("--")
    return i, 2 if has_value else 1


def derive_config(spec: dict) -> dict:
    cfg = json.load(open(os.path.join(PACK, spec.get("base", "model.json"))))
    args = list(cfg["args"])
    for flag in spec.get("remove", []):
        if flag in args:
            i, n = split_flag(args, flag)
            del args[i:i + n]
    for flag, value in spec.get("set", {}).items():
        if flag in args:
            i, n = split_flag(args, flag)
            args[i:i + n] = [flag, str(value)]
        else:
            args += [flag, str(value)]
    for flag in spec.get("add", []):
        if flag not in args:
            args.append(flag)
    # Shipped configs assume 24 pool workers; never oversubscribe a smaller host.
    if "--pool-workers" in args:
        i, _ = split_flag(args, "--pool-workers")
        args[i + 1] = str(min(int(args[i + 1]), os.cpu_count() or 1))
    cfg["args"] = args
    if spec.get("parallel", 1) > 1:
        cfg["parallel"] = int(spec["parallel"])
    else:
        cfg.pop("parallel", None)
    cfg["env"] = {**cfg.get("env", {}), **spec.get("env", {})}
    cfg["exe"] = f"{STRATA_DIR}/build/strata"  # server.py resolves a bare "strata" against the pack folder
    cfg["model_name"] = "edge0"
    cfg["port"] = PORT
    cfg["log"] = f"./serve-{spec['label']}.log"
    return cfg


def http_get(path: str, timeout: float = 10) -> str:
    """Best effort: a missing endpoint must not fail the run."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
            return r.read().decode(errors="replace")
    except OSError as e:
        return f"unavailable: {e}"


def gpu_snapshot() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,clocks.sm,"
                           "pcie.link.gen.current,pcie.link.width.current,power.draw", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    subprocess.run(["pkill", "-9", "-f", f"{STRATA_DIR}/build/strata"], check=False)
    for _ in range(30):  # wait for VRAM to be released before the next config loads
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip()
        if used and int(used.split()[0]) < 1000:
            break
        time.sleep(2)


def main() -> None:
    spec = json.load(open(sys.argv[1]))
    unknown = set(spec) - SPEC_KEYS
    if unknown:
        sys.exit(f"unknown spec keys: {unknown}")
    label = spec["label"]
    out = f"/results/runs/{label}"
    os.makedirs(out, exist_ok=True)
    cfg = derive_config(spec)
    cfg_path = os.path.join(PACK, f"cfg-{label}.json")
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    json.dump(cfg, open(f"{out}/cfg.json", "w"), indent=1)

    env = dict(os.environ)
    env["PATH"] = f"{STRATA_DIR}/build:{env['PATH']}"
    log = open(f"{out}/server.log", "w")
    t0 = time.time()
    proc = subprocess.Popen([f"{STRATA_DIR}/.venv/bin/python", f"{STRATA_DIR}/serve/server.py", "--engine", "strata",
                             "--config", cfg_path, "--port", str(PORT), "--no-api-key"],
                            cwd=PACK, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result = {"label": label, "spec": spec, "phases": [], "load_s": None, "status": "starting"}
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with {proc.returncode} during load")
            if time.time() - t0 > LOAD_TIMEOUT_S:
                raise RuntimeError("load timeout")
            try:
                r = loadgen.one_request("127.0.0.1", PORT, [{"role": "user", "content": "Say OK."}], 4, timeout=120)
                if r["status"] == 200 and r["t_first"]:
                    break
            except OSError:
                pass
            time.sleep(10)
        result["load_s"] = time.time() - t0
        result["gpu_after_load"] = gpu_snapshot()
        for k, ph in enumerate(spec["phases"]):
            reps = []
            for rep in range(ph.get("reps", 3)):
                m0 = http_get("/metrics")
                if ph["mode"] == "decode":
                    res = loadgen.run_decode("127.0.0.1", PORT, ph.get("concurrency", 1), ph.get("duration", 60),
                                             ph.get("warmup", 20), ph.get("max_tokens", 256),
                                             ph.get("prompt_words", 380), seed=rep)
                elif ph["mode"] == "strata_bench":
                    sb = subprocess.run([f"{STRATA_DIR}/.venv/bin/python", f"{STRATA_DIR}/tools/strata_bench.py",
                                         "--url", f"http://127.0.0.1:{PORT}", "--runs", str(ph.get("runs", 3)),
                                         "--out", f"{out}/strata_bench", "--quiet"],
                                        capture_output=True, text=True, timeout=3600)
                    res = {"mode": "strata_bench", "rc": sb.returncode, "stdout": sb.stdout[-6000:],
                           "stderr": sb.stderr[-3000:]}
                else:
                    res = loadgen.run_prefill("127.0.0.1", PORT, ph["prompt_tokens"], ph.get("per_rep", 1), seed=rep)
                m1 = http_get("/metrics")
                open(f"{out}/metrics-p{k}-r{rep}-before.txt", "w").write(m0)
                open(f"{out}/metrics-p{k}-r{rep}-after.txt", "w").write(m1)
                res["gpu"] = gpu_snapshot()
                reps.append(res)
                print(json.dumps({"label": label, "phase": k, "rep": rep,
                                  **{x: res.get(x) for x in ("agg_tps", "prompt_tps_median", "errors")}}), flush=True)
            result["phases"].append({"phase": ph, "reps": reps})
        try:
            result["status_json"] = json.loads(http_get("/v1/status"))
        except ValueError:
            pass
        result["status"] = "ok"
    except Exception as e:  # recorded, the sweep continues with the next config
        result["status"] = f"failed: {e}"
    finally:
        stop_server(proc)
        log.close()
        lines = open(f"{out}/server.log", errors="replace").read().splitlines()
        result["log_excerpt"] = [ln for ln in lines if LOG_PATTERNS.search(ln)][:400]
        result["wall_s"] = time.time() - t0
        json.dump(result, open(f"{out}/result.json", "w"), indent=1)
        os.remove(cfg_path)
        print(json.dumps({"label": label, "status": result["status"], "load_s": result["load_s"],
                          "wall_s": result["wall_s"]}), flush=True)


if __name__ == "__main__":
    main()

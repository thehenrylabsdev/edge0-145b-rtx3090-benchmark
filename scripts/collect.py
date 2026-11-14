#!/usr/bin/env python3
"""Summarise /results/runs/*/result.json into results.csv and print a table.

  python3 collect.py RESULTS_DIR

Per config and phase: median over reps of aggregate TPS (decode) or prompt TPS (prefill), the spread, per-stream
TPS, TPOT/TTFT, MTP acceptance and engine-side rates (from the /metrics JSON totals), and the batch slots
the engine reported.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import statistics
import sys


def totals_delta(run_dir: str, phase: int, reps: int) -> dict:
    """Engine-side totals from GET /metrics (JSON "totals"), summed over the phase's reps."""
    keys = ("output_tokens", "decode_ms", "prompt_tokens", "prompt_ms", "drafts_offered", "drafts_accepted")
    acc = dict.fromkeys(keys, 0.0)
    for r in range(reps):
        try:
            a = json.load(open(f"{run_dir}/metrics-p{phase}-r{r}-before.txt"))["totals"]
            b = json.load(open(f"{run_dir}/metrics-p{phase}-r{r}-after.txt"))["totals"]
        except (OSError, ValueError, KeyError):
            continue
        for k in keys:
            acc[k] += b.get(k, 0) - a.get(k, 0)
    return {"accept_rate": acc["drafts_accepted"] / acc["drafts_offered"] if acc["drafts_offered"] else None,
            # decode_ms is summed per request, so with concurrent requests this is the mean per-stream rate
            "engine_stream_decode_tps": acc["output_tokens"] / acc["decode_ms"] * 1000 if acc["decode_ms"] else None,
            "engine_prefill_tps": acc["prompt_tokens"] / acc["prompt_ms"] * 1000 if acc["prompt_ms"] else None}


def engine_info(run_dir: str) -> dict:
    """The engine block of the first /metrics snapshot (batch_slots, expert_slots, spec, drafting)."""
    try:
        return json.load(open(f"{run_dir}/metrics-p0-r0-before.txt")).get("engine", {})
    except (OSError, ValueError):
        return {}


def slots_reported(log_lines: list[str], eng: dict) -> str:
    if eng.get("batch_slots") is not None:
        return str(eng["batch_slots"])
    for ln in log_lines:
        m = re.search(r"batch_slots=(\d+)", ln)
        if m:
            return m.group(1)
    return ""


def main() -> None:
    root = sys.argv[1]
    rows = []
    for path in sorted(glob.glob(f"{root}/runs/*/result.json")):
        res = json.load(open(path))
        run_dir = os.path.dirname(path)
        eng = engine_info(run_dir)
        slots = slots_reported(res.get("log_excerpt", []), eng)
        for k, ph in enumerate(res.get("phases", [])):
            mode = ph["phase"]["mode"]
            reps = ph["reps"]
            if mode == "decode":
                vals = [r["agg_tps"] for r in reps if r.get("agg_tps")]
                row = {"label": res["label"], "mode": "decode", "concurrency": ph["phase"]["concurrency"],
                       "prompt_words": ph["phase"].get("prompt_words"),
                       "tps_median": statistics.median(vals) if vals else None,
                       "tps_min": min(vals) if vals else None, "tps_max": max(vals) if vals else None,
                       "per_stream_tps": statistics.median([r["per_stream_tps_p50"] for r in reps
                                                            if r.get("per_stream_tps_p50")] or [0]),
                       "tpot_p50_ms": statistics.median([r["tpot_p50_ms"] for r in reps] or [0]),
                       "ttft_p50_s": statistics.median([r["ttft_p50_s"] for r in reps if r.get("ttft_p50_s")] or [0]),
                       "errors": sum(r.get("errors", 0) for r in reps),
                       **totals_delta(run_dir, k, len(reps))}
            elif mode == "prefill":
                vals = [r["prompt_tps_median"] for r in reps if r.get("prompt_tps_median")]
                row = {"label": res["label"], "mode": f"prefill-{ph['phase']['prompt_tokens']}",
                       "tps_median": statistics.median(vals) if vals else None,
                       "tps_min": min(vals) if vals else None, "tps_max": max(vals) if vals else None,
                       "prompt_tokens": statistics.median([r["prompt_tokens_median"] for r in reps
                                                           if r.get("prompt_tokens_median")] or [0]),
                       **totals_delta(run_dir, k, len(reps))}
            else:
                continue
            row.update({"status": res["status"], "load_s": res.get("load_s"), "slots_reported": slots,
                        "expert_slots": eng.get("expert_slots"), "engine_spec": eng.get("spec"),
                        "drafting": eng.get("drafting"), "gpu_after_load": res.get("gpu_after_load", "")})
            rows.append(row)
    if not rows:
        sys.exit("no results")
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "label", k != "mode", k))
    with open(f"{root}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        tps = f"{r['tps_median']:.1f}" if r.get("tps_median") else "-"
        extra = f"N={r.get('concurrency')} slots={r['slots_reported'] or '-'} acc={r.get('accept_rate')}" \
            if r["mode"] == "decode" else f"tokens={r.get('prompt_tokens')}"
        print(f"{r['label']:22} {r['mode']:14} {tps:>9}  {extra}  {r['status']}")


if __name__ == "__main__":
    main()

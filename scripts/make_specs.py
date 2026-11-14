#!/usr/bin/env python3
"""Write the JSONL spec files for each sweep phase (PLAN.md Epics 2-5).

  python3 make_specs.py OUTDIR [--base model.json] [--prefill X] [--spec K] [--refine 3,5,6] [--best 4,8]

Later phases take the winners of earlier ones as arguments, so the sweep is run phase by phase.
"""
from __future__ import annotations

import argparse
import json
import os

DECODE = {"mode": "decode", "duration": 45, "warmup": 15, "reps": 3, "max_tokens": 256, "prompt_words": 380}
PREFILL_4K = {"mode": "prefill", "prompt_tokens": 4096, "per_rep": 1, "reps": 3}
PREFILL_16K = {"mode": "prefill", "prompt_tokens": 16384, "per_rep": 1, "reps": 3}
# A native (IQ) pack refuses to start without --spec T >= 2, so "MTP off" drops only the drafter head.
NO_MTP = ["--mtp"]


def decode(n: int, **kw) -> dict:
    return {**DECODE, "concurrency": n, **kw}


def write(path: str, specs: list[dict]) -> None:
    with open(path, "w") as f:
        for s in specs:
            f.write(json.dumps(s) + "\n")
    print(path, len(specs), "configs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--base", default="model.json")
    ap.add_argument("--prefill", default=None, help="winning --prefill chunk (Epic 3)")
    ap.add_argument("--spec", default=None, help="winning --spec window, 0 = MTP head off (Epic 4)")
    ap.add_argument("--refine", default="", help="extra N values for the batch refine pass (Epic 5.3)")
    ap.add_argument("--best", default="", help="best two N values for the long-context and KV arms (Epic 5.4/5.5)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    b = a.base

    # Epic 2: shipped config vs the vendor bench config, plus the vendor's own quick bench on the shipped one.
    write(f"{a.outdir}/p2-baseline.jsonl", [
        {"label": "base-model", "base": "model.json",
         "phases": [decode(1), PREFILL_4K, PREFILL_16K, {"mode": "strata_bench", "runs": 3, "reps": 1}]},
        {"label": "base-modelbench", "base": "model-bench.json", "phases": [decode(1), PREFILL_4K, PREFILL_16K]},
    ])

    # Epic 3: prefill chunk.
    write(f"{a.outdir}/p3-prefill.jsonl", [
        {"label": f"prefill-{c}", "base": b, "set": {"--prefill": c}, "phases": [PREFILL_4K, PREFILL_16K]}
        for c in ["256", "512", "1024", "2048", "4096", "8192", "auto"]])

    tuned_set = {"--prefill": a.prefill} if a.prefill else {}

    # Epic 4: MTP verify window at N=1 (T >= 2 is the engine's minimum), plus the drafter head removed.
    p4 = [{"label": f"spec-{k}", "base": b, "set": {**tuned_set, "--spec": k}, "phases": [decode(1)]}
          for k in ["2", "3", "4", "6"]]
    p4 += [{"label": "spec-nomtp", "base": b, "set": tuned_set, "remove": NO_MTP, "phases": [decode(1)]}]
    write(f"{a.outdir}/p4-spec.jsonl", p4)

    if a.spec is None:
        return
    spec_args = {"remove": NO_MTP, "set": tuned_set} if a.spec == "0" else \
        {"set": {**tuned_set, "--spec": a.spec}}

    # Epic 5.2: coarse batch slots. The engine runs at most 8 slots on one GPU (PARALLEL_MAX in serve/server.py).
    write(f"{a.outdir}/p5-batch-coarse.jsonl", [
        {"label": f"batch-{n}", "base": b, "parallel": n, **spec_args, "phases": [decode(n)]} for n in [1, 2, 4, 8]])

    # Epic 5.3: refine around the best N, with MTP per slot (--batch-mtp, at most 4 slots) and with MTP off.
    p53 = [{"label": f"batch-{n}", "base": b, "parallel": n, **spec_args, "phases": [decode(n)]}
           for n in [int(x) for x in a.refine.split(",") if x]]
    if a.spec != "0":
        p53 += [{"label": f"batchmtp-{n}", "base": b, "parallel": n, **spec_args, "add": ["--batch-mtp"],
                 "phases": [decode(n)]} for n in [2, 3, 4]]
        p53 += [{"label": f"batch-nomtp-{n}", "base": b, "parallel": n, "set": tuned_set, "remove": NO_MTP,
                 "phases": [decode(n)]} for n in [int(x) for x in (a.best or "2,4").split(",")]]
    write(f"{a.outdir}/p5-batch-refine.jsonl", p53)

    # Epic 5.4/5.5: longer prompts (~4K tokens) and fewer resident KV cells at the best N values.
    best = [int(x) for x in (a.best or "2,4").split(",") if x]
    p54 = [{"label": f"long-{n}", "base": b, "parallel": n, **spec_args,
            "phases": [decode(n, prompt_words=3000)]} for n in best]
    p54 += [{"label": f"kvres8k-{n}", "base": b, "parallel": n,
             **{**spec_args, "set": {**spec_args["set"], "--kv-resident": "8192"}}, "phases": [decode(n)]}
            for n in best]
    write(f"{a.outdir}/p5-long-kv.jsonl", p54)


if __name__ == "__main__":
    main()

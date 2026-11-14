#!/usr/bin/env python3
"""Stdlib-only load generator for an OpenAI-compatible streaming chat endpoint.

decode  : closed loop, N concurrent workers, each sends unique prompts back to back for warmup+duration seconds.
          Aggregate TPS = completion tokens falling inside the measured window / window length (tokens of a request are
          spread uniformly between its first and last content chunk), so requests straddling the window edges count
          only their in-window share.
prefill : one request at a time, unique prompt of ~L tokens, max_tokens=1; prompt tok/s from server timings
          when present,
          else prompt_tokens / TTFT.
"""
from __future__ import annotations

import argparse
import http.client
import json
import random
import statistics
import threading
import time
import uuid

WORDS = ("system memory kernel latency bandwidth cache expert router token layer attention model engine stream "
         "buffer vector matrix tensor compute shader thread scheduler queue packet socket server client request "
         "response window budget profile sample entropy gradient weight bias scale offset table index record "
         "segment page block chunk frame signal channel device driver bus lane clock power thermal limit "
         "river mountain forest ocean desert city harbor bridge tower garden market library museum station").split()
TOPICS = ["the history of bridges", "how GPUs schedule work", "the ecology of coral reefs", "medieval trade routes",
          "the design of programming languages", "weather forecasting", "the economics of railways",
          "deep sea exploration", "the physics of music", "urban planning", "the evolution of writing systems",
          "volcanoes", "the life of honeybees", "space telescopes", "ancient astronomy", "the chemistry of cooking"]


def filler(n_words: int, rng: random.Random) -> str:
    out = []
    for i in range(n_words):
        out.append(rng.choice(WORDS))
        if i % 14 == 13:
            out[-1] += "."
    return " ".join(out)


def one_request(host: str, port: int, messages: list, max_tokens: int, timeout: float = 1800) -> dict:
    body = {"model": "edge0", "stream": True, "max_tokens": max_tokens, "temperature": 0, "messages": messages,
            "stream_options": {"include_usage": True}, "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    r = {"t_start": t0, "status": resp.status, "t_first": None, "t_last": None, "chunks": 0,
         "prompt_tokens": None, "completion_tokens": None, "timings": None, "text_head": ""}
    if resp.status != 200:
        r["error"] = resp.read(2000).decode(errors="replace")
        conn.close()
        return r
    text = []
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        try:
            ev = json.loads(data)
        except ValueError:
            continue
        now = time.time()
        for ch in ev.get("choices") or []:
            c = (ch.get("delta") or {}).get("content")
            if c:
                if r["t_first"] is None:
                    r["t_first"] = now
                r["t_last"] = now
                r["chunks"] += 1
                if len(text) < 40:
                    text.append(c)
        if ev.get("usage"):
            r["prompt_tokens"] = ev["usage"].get("prompt_tokens")
            r["completion_tokens"] = ev["usage"].get("completion_tokens")
        if ev.get("timings"):
            r["timings"] = ev["timings"]
    r["t_end"] = time.time()
    r["text_head"] = "".join(text)[:200]
    conn.close()
    return r


def decode_messages(rng: random.Random, prompt_words: int) -> list:
    topic = rng.choice(TOPICS)
    doc = filler(prompt_words, rng)
    return [{"role": "user", "content": f"Request {uuid.uuid4()}. Background notes (ignore their wording):\n{doc}\n\n"
             f"Now write a long, detailed essay about {topic}. Write at least 1500 words. Do not stop early."}]


def run_decode(host, port, concurrency, duration, warmup, max_tokens, prompt_words, seed=0) -> dict:
    t_begin = time.time()
    w0, w1 = t_begin + warmup, t_begin + warmup + duration
    results, lock = [], threading.Lock()

    def worker(i):
        rng = random.Random(seed * 1000 + i)
        while time.time() < w1:
            try:
                r = one_request(host, port, decode_messages(rng, prompt_words), max_tokens)
            except OSError as e:  # dropped connection: count it as an error and keep the worker alive
                r = {"t_start": time.time(), "status": -1, "error": repr(e), "t_first": None, "chunks": 0}
                time.sleep(1)
            r["worker"] = i
            with lock:
                results.append(r)

    ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(concurrency)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    in_window = 0.0
    tpots, ttfts, per_stream = [], [], []
    errors = [r for r in results if r["status"] != 200]
    for r in results:
        n = r.get("completion_tokens") or r["chunks"]
        if r["status"] != 200 or not r["t_first"] or not n:
            continue
        a, b = r["t_first"], max(r["t_last"], r["t_first"] + 1e-6)
        ov = max(0.0, min(b, w1) - max(a, w0))
        in_window += n * ov / (b - a) if n > 1 else (1.0 if w0 <= a <= w1 else 0.0)
        if r["t_start"] >= w0 - 1e-9:
            ttfts.append(r["t_first"] - r["t_start"])
        if n > 1:
            tpot = (b - a) / (n - 1)
            tpots.append(tpot)
            per_stream.append(1.0 / tpot)

    def pct(xs, p):
        if not xs:
            return None
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]

    return {"mode": "decode", "concurrency": concurrency, "duration": duration, "warmup": warmup,
            "max_tokens": max_tokens, "requests": len(results), "errors": len(errors),
            "error_sample": errors[0].get("error") if errors else None,
            "agg_tps": in_window / duration,
            "per_stream_tps_p50": pct(per_stream, 50), "tpot_p50_ms": (pct(tpots, 50) or 0) * 1000,
            "tpot_p99_ms": (pct(tpots, 99) or 0) * 1000, "ttft_p50_s": pct(ttfts, 50), "ttft_p99_s": pct(ttfts, 99),
            "mean_completion_tokens": statistics.mean([r.get("completion_tokens") or r["chunks"] for r in results
                                                       if r["status"] == 200]) if results else None,
            "text_head": results[0]["text_head"] if results else ""}


def run_prefill(host, port, prompt_tokens, reps, seed=0) -> dict:
    rng = random.Random(seed)
    rows = []
    words = int(prompt_tokens / 1.35)
    for _ in range(reps):
        msg = [{"role": "user", "content": f"Request {uuid.uuid4()}.\n{filler(words, rng)}\n\nReply with one word."}]
        try:
            r = one_request(host, port, msg, 1)
        except OSError as e:
            r = {"status": -1, "error": repr(e)}
        if r["status"] != 200:
            rows.append({"error": r.get("error")})
            continue
        t = r.get("timings") or {}
        ttft = (r["t_first"] or r["t_end"]) - r["t_start"]
        pps = t.get("prompt_per_second") or ((r["prompt_tokens"] or 0) / ttft if ttft else None)
        rows.append({"prompt_tokens": r["prompt_tokens"], "ttft_s": ttft, "prompt_tps": pps, "timings": t})
        # ratio correction so later reps land closer to the target length
        if r["prompt_tokens"]:
            words = max(16, int(words * prompt_tokens / r["prompt_tokens"]))
    ok = [x for x in rows if x.get("prompt_tps")]
    return {"mode": "prefill", "target_tokens": prompt_tokens, "reps": reps, "rows": rows,
            "prompt_tps_median": statistics.median([x["prompt_tps"] for x in ok]) if ok else None,
            "prompt_tokens_median": statistics.median([x["prompt_tokens"] or 0 for x in ok]) if ok else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--mode", choices=["decode", "prefill"], required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--warmup", type=float, default=20)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt-words", type=int, default=380)
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if a.mode == "decode":
        print(json.dumps(run_decode(a.host, a.port, a.concurrency, a.duration, a.warmup, a.max_tokens, a.prompt_words)))
    else:
        print(json.dumps(run_prefill(a.host, a.port, a.prompt_tokens, a.reps)))


if __name__ == "__main__":
    main()

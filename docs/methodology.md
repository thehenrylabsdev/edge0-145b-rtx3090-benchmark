# Measurement methodology

This page defines every number in the report: what was sent, what was timed, how it was aggregated, and what
the numbers do not show. The code that implements it is in [`scripts/`](../scripts).

## The question

> On one RTX 3090, what is the **largest batch size that still gives the highest tokens per second**?

"Batch size" has three meanings in this engine, and each one gets its own sweep:

| Knob | Engine flag | What it batches | Metric |
|---|---|---|---|
| Decode batch slots | `--batch N` (config `"parallel": N`) | N conversations decoded together, one token each per step, sharing weight and expert reads | **aggregate output tok/s** |
| Prefill chunk | `--prefill CHUNK` | prompt tokens processed per step | prompt tok/s |
| MTP verify window | `--spec T` | drafted tokens verified per step (a micro-batch inside one stream) | single-stream output tok/s, draft acceptance |

The headline answer, **B\***, is about decode batch slots.

## B\*: the decision rule

**B\* is the largest N whose median aggregate tok/s is within 2 % of the peak median.** Three repetitions
cannot separate configurations closer than about 2 %, so a larger N within that band counts as tied with the
peak, and the rule picks the larger N. Picking the larger N favours capacity: more users served at the same
throughput.

The rule is applied in [`charts/build_charts.py`](../charts/build_charts.py) (`b_star()`), in two ways:

1. over the plain `--batch N` family, which is the main sweep, and
2. over the best variant at each N, which also includes `--batch-mtp`, MTP head removed and fewer resident
   KV cells.

Both results are in `results/summary.json`. The per-stream cost at B\* (per-request speed and time to
first token) is reported next to it, because a larger N can raise total throughput while making each user
slower.

## Search order

Each phase used the winners of the phase before it.

1. **Baseline (Epic 2).** The shipped `model.json` against the vendor's `model-bench.json`, plus the
   vendor's own `tools/strata_bench.py` on the shipped config. The winner became the base for everything
   after.
2. **Prefill chunk (Epic 3).** 256, 512, 1024, 2048, 4096, 8192 and `auto`, each with a ~4K-token and a
   ~16K-token prompt.
3. **MTP window (Epic 4).** `--spec` 2, 3, 4 and 6, plus the MTP head removed, at N = 1. `--spec` 0 and 1
   were planned, but this pack refuses to start below 2 (see [engine notes](engine-and-build.md)).
4. **Batch slots, coarse (Epic 5.2).** N = 1, 2, 4, 8 with the winning `--spec`. 8 is the engine's
   per-GPU maximum (`PARALLEL_MAX`).
5. **Batch slots, refine (Epic 5.3).** N = 3, 5, 6, 7, plus `--batch-mtp` at N = 2, 3, 4 and the MTP head
   removed at N = 4–8.
6. **Pressure arms (Epic 5.4/5.5).** Long prompts at N = 4, 8 (3,000 filler words, which came to about
   3.3K tokens), and `--kv-resident 8192` (down from 32768) at N = 4, 8.
7. **Combined (added during the run).** The MTP head removed plus `--kv-resident 8192` at N = 4, 6, 8,
   because each change alone freed VRAM for experts and raised throughput. N = 5 and 7 were not measured
   for this combination.

**One setting differs between arms.** The plain `--batch N`, `--batch-mtp`, long-prompt and
`--kv-resident 8192` arms use the Epic 4 winner, `--spec 2`. The two MTP-head-removed families keep the
shipped `--spec 4`, because `make_specs.py` applies the winning window only when MTP is on. In batch mode
these arms drafted nothing (0 drafts offered in their engine logs), so the window should not affect them,
but this was not tested separately. The recommended config in the report uses exactly the measured
settings.

## Decode measurement

Harness: [`scripts/loadgen.py`](../scripts/loadgen.py) `run_decode()`. It uses only the Python standard
library and talks to the engine's OpenAI-compatible `/v1/chat/completions` endpoint with streaming.

**Closed loop.** N worker threads each send one request, wait for it to finish, then send the next, for
warmup + window seconds. Concurrency equals the slot count, so every slot is busy whenever a request is in
flight.

**Each request:**

- **Prompt:** a unique ID (`Request <uuid4>`), about 380 filler words (about 490 tokens), and an
  instruction to write a long essay on one of 16 topics. The UUID prevents any prompt-cache reuse.
- **Output:** `max_tokens = 256` at `temperature = 0`. The essay prompt makes almost every request run to
  the cap.
- **Thinking off:** `reasoning_effort: "none"` and `chat_template_kwargs.enable_thinking: false`.
- **Usage counts:** `stream_options.include_usage` makes the server return exact token counts.

**Window.** 15 s warmup, then a 45 s measured window. The baseline configs (`base-model`,
`base-modelbench`) and the smoke run used 20 s + 60 s, before the window was shortened to fit the time
budget.

**Aggregate tok/s** = output tokens produced *inside the window* ÷ window length:

- A request's tokens are spread evenly between its first and last content chunk.
- A request that straddles a window edge contributes only its in-window share.
- So start-up and drain at the edges do not inflate the number.

Aggregate tok/s **includes the prefill time of every request**, because a slot is busy reading the next
prompt between outputs. At N = 1 that makes it about 12 tok/s lower than the pure decode rate.

**Per-stream tok/s** = 1 / TPOT, where TPOT = (last chunk − first chunk) / (tokens − 1). The value reported
is the median over all requests. This is what one user experiences once their answer has started.

**TTFT** = first content chunk − request start, taken only from requests that started inside the window.

**Repetitions.** Three reps per config. The report shows the median of the three, with min and max in
`results/results.csv`. A confirmation pass (three more reps for each B\* candidate) was planned. It was
dropped when the MTP-head-removed and reduced-KV arms turned out faster and needed the remaining pod time.
`charts/build_charts.py` still pools any `confirm-<label>` run it finds. Workers are seeded from the rep number, so the topic and filler sequence repeats
across configs, while the UUIDs keep every prompt unique.

**Engine cross-check.** `GET /metrics` is saved before and after every rep. The difference in the engine's
`totals` gives:

- `engine_stream_decode_tps`: output tokens ÷ decode ms, summed per request, so it is a per-stream rate.
- `accept_rate`: drafts accepted ÷ drafts offered, for MTP or n-gram drafting.

The engine block of the same snapshot records:

- `batch_slots`: the slots that actually started.
- `expert_slots`: experts cached in VRAM.
- `spec`: the window, which the engine reports as T + 2.
- `drafting`.

## Prefill measurement

Harness: `loadgen.run_prefill()`. One request per rep:

- **Prompt:** a unique filler prompt sized for the target length, ending with "Reply with one word."
- **Output:** `max_tokens = 1`.
- **Prompt tok/s:** the engine's `timings.prompt_per_second` when present, otherwise prompt tokens ÷ TTFT.

**Known bias: the prompts came out shorter than targeted.** The word-to-token correction only applies from
the second request of a rep, and each rep sent one request. The prompts were therefore about **3.3K and
13.1K tokens** instead of 4K and 16K. Charts and tables label the actual lengths.

For a like-for-like comparison with the vendor's 4K/16K numbers, use the vendor tool run below. Its
prompts were 4,019 and 16,058 tokens.

## The vendor's own benchmark

`tools/strata_bench.py` (suite 1, 3 runs) was run against the shipped config. This is the closest available
comparison with the RTX 4090 numbers on the model card.

| Test | What it does |
|---|---|
| Chat decode | Three short questions, 100 new tokens each |
| 4K / 16K prefill | Fresh prompts, 24 new tokens |
| Agent decode | 5-turn coding-agent sessions with tools, about 14K tokens of context by the last turn, 64 new tokens per turn |

Its full output is in `results/runs/base-model/strata_bench/`.

## Controls

- **Fresh engine per config.** Each configuration starts a new server process. Readiness is confirmed with
  a real 4-token request, and the load time is recorded. After the run, the engine is killed and the next
  config waits until GPU memory falls below 1 GB.
- **One host for everything:**
  - GPU: RTX 3090 24 GB, PCIe gen4 x16, 350 W limit.
  - CPU: AMD EPYC 7H12 (Zen 2, AVX-2, no AVX-512).
  - Memory: 116.4 GiB cgroup limit.
  - Disk: NVMe at 9.1 GB/s sequential and 98K IOPS at 4 KiB.

  Host facts are in `results/host/`.
- **Nothing else ran on the GPU.** `nvidia-smi` is recorded after load and after every rep.
- **Pinned software.** Edge0 commit `da9cc526`, ggml at llama.cpp `3cf03257`, CUDA arch 86. See
  `results/versions.txt`.

## What these numbers do not show

- **Closed loop, not open-loop arrivals.** These are capacity numbers with N users always waiting. They
  are not latency under a Poisson arrival rate.
- **Short windows.** Three 45 s windows per config. Differences under about 2 % are noise. That is why
  B\* uses a 2 % band.
- **The first window of a batched config often reads low.** At N ≥ 3, rep 0 was often 2–6 % below reps 1
  and 2 (for example 52.0, 54.4 and 55.4 tok/s at N = 6). This fits the engine's adaptive expert cache
  still settling after the 15 s warmup. The median of three already discounts one low rep. A longer warmup would remove
  it.
- **A single host on a cloud.** Other 3090 hosts will differ, especially in CPU (AVX-512), PCIe and disk.
  Each of these affects a model that streams experts from RAM and reads its PLE table from SSD on every
  token.
- **Synthetic prompts.** Filler text plus an essay instruction, with thinking off. MTP acceptance depends
  on the text, so code or chat workloads will accept a different share of drafts.
- **No quality evaluation.** Outputs were checked for coherence, not scored. Every configuration uses the
  same Q3_K weights, and none changes the math, apart from speculative drafting, which is verified by the
  model.
- **Vendor numbers are self-reported.** The RTX 4090 figures come from the model card and were measured
  on a different host.

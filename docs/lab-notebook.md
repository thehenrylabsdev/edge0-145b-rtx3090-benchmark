# Lab notebook

What happened, in order, including the dead ends. Times are UTC+7 on 2026-11-14.

## Planning (no spend)

- **The original target changed.** It was Qwen3.8-27B on llama.cpp, then switched to Edge0's newest
  release, **Edge0-145B**. For a 24 GB card the right build is the **Q3_K** pack.
- **Feasibility check before renting a GPU.** The skeleton GGUF header was read over HTTP range requests,
  together with upstream llama.cpp's `qwen4exp` loader. That showed stock llama.cpp cannot load the pack:
  there are no `ffn_*_exps` tensors and no latent path. The benchmark therefore runs on Edge0's own
  ggml-based engine, edge0-strata.
- **Harness dry-run on a laptop.** The load generator was run against a local llama-server with a tiny
  model first. That run found one bug, a worker thread dying on a dropped connection, which was fixed
  before any money was spent.

## Pod

- **05:5x — first host rejected.** A Community Cloud RTX 3090 pod at $0.22/hr came up with only 30 GB of
  RAM. Below the 48 GB floor the experts would page. The pod was deleted within minutes.
- **05:58 — second host accepted.** Secure Cloud, EU-CZ-1, $0.50/hr:
  - 125 GB RAM, 32 vCPU (EPYC 7H12).
  - RTX 3090 on PCIe gen4 x16.
  - NVMe at 9.1 GB/s sequential.
- **Download fix.** The first download fetched only the one file it was meant to skip, because the `hf`
  CLI takes one pattern per `--exclude`. After the fix, 103.4 GB downloaded in about 2 minutes and
  matched the HF listing byte for byte.
- **Build.** The engine built cleanly for sm_86 against ggml `3cf03257`.

## Smoke test

- **First start failed:** `FileNotFoundError: .../Edge0-145B-q3k/strata`. `server.py` resolves a bare exe
  name against the pack folder. Fixed with an absolute `exe` in the derived config.
- **Then it worked.** Load took 71 s. 7358 experts (11.17 GiB) were cached in VRAM, and the output text was
  coherent. N = 1 decode was about 57 tok/s engine-side.
- **`/metrics` returns JSON, not Prometheus text.** The collector was rewritten to read the JSON
  `totals`.

## Baseline (Epic 2)

- **The shipped `model.json` beat the vendor's `model-bench.json` everywhere:** decode 43.8 vs 36.4, prefill
  1180 vs 394 tok/s.
- **The vendor's own suite** gave 54.1 chat and 57.3 agent decode, 1273 and 1470 tok/s prefill (4K and 16K).
  That is 77 %, 62 %, 62 % and 46 % of the RTX 4090 figures on the model card.
- **One load took 351 s instead of about 30 s.** The engine read 37.3 GiB of experts from disk at 3.6 GiB/s,
  because the page cache was cold. Later loads stayed at 30–130 s.

## Prefill chunk (Epic 3)

- **Larger chunks are faster up to the engine's 8192 cap.** `auto` and 8192 are within 1 %.
- **`--prefill 1024` dips below 512**, repeatably. The cause was not found.

## MTP window (Epic 4)

- **`spec-0` and `spec-1` failed at load.** The pack needs `--spec T ≥ 2`, so the plan's "MTP off" arm was
  wrong.
  - Correct form: remove only `--mtp`.
  - The spec generator was fixed, and a corrected arm was queued behind the running sweep.
- **The shortest window wins:** `--spec 2` 45.7 tok/s (70 % acceptance), 3: 44.9, 4: 44.4, 6: 40.3. The
  MTP head removed gave 41.3.
- **Found by reading `/metrics`:** the engine reports the window as T + 2.

## Batch slots (Epic 5)

- **Coarse result:** 46.8 (N=1), 50.2 (2), 53.5 (4), 48.1 (8) tok/s.
- **Plain `--batch N` stops MTP drafting:** 1 draft offered in 17 requests. Each slot also takes VRAM
  from the expert cache: 7360, 6397, 5434 and 3507 experts at N = 1, 2, 4, 8.
- **Refine pass:** N = 3, 5, 6, 7, plus `--batch-mtp` at 2–4 and the MTP head removed at N = 4–8.
- **Pressure arms:** long prompts (about 3.3K tokens) and `--kv-resident 8192` at N = 4 and 8.
- **Engine logs explained the slot cost.** The engine's own log (not `server.py`'s console) says `--batch N`
  allocates N slot sessions of 0.73 GiB each, taken from the expert cache.
- **Change of plan mid-run.** The MTP-head-removed arm (56.5 tok/s at N = 4) and `--kv-resident 8192`
  (57.9 tok/s at N = 4) both beat every plain `--batch` point. Each one frees VRAM for experts, so a
  combined arm was added at N = 4, 6, 8. To pay for it, the planned confirmation reps were dropped.
- **Result:** the combined arm at N = 6 gave 60.2 tok/s, the best of the study. The pod was deleted
  right after the final pull, about 3 h after it was created.
- **Missed during the run:** the MTP-head-removed arms kept the shipped `--spec 4`, while the other arms
  used `--spec 2`. Batch mode drafts nothing in these arms, so it should not matter, but it is disclosed
  in the methodology.

Final numbers are in the [report](../REPORT.md).

## Operations notes

- **Remote control.** The pod was driven from a laptop through RunPod's REST exec endpoint, which has a
  25 s cap per call. Every long job ran under `nohup` with a log file.
- **Monitoring.** A local monitor polled the sweep logs every 60 s and treated 15 minutes of silence as a
  stall. No stall happened. One poll failed transiently.
- **Chaining.** Phases were chained on the pod, each waiting for the previous `SWEEP_DONE`, so the GPU was
  never idle between phases.
- **Time budget.** Decode windows were shortened from 20 + 60 s to 15 + 45 s to fit the 6 h / $2.00 cap
  set before the run.

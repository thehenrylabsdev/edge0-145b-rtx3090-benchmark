# Engine, build and what we learned about it

## Lineage

| Component | What it is |
|---|---|
| **edge0-strata** | Edge0's fork of [Strata](https://github.com/Niko1221/Strata) (MIT), in the [Edge0 repo](https://github.com/Edge0-AI/Edge0). It is the only engine that runs the 120B/145B Edge0 packs. Its CUDA path reuses llama.cpp's MMQ kernels, and it builds against ggml from llama.cpp commit `3cf03257`. |
| `serve/server.py` | The OpenAI- and Anthropic-compatible HTTP server. It starts the `strata` engine binary with the arguments from a JSON config (`model.json`) and exposes `/v1/chat/completions`, `/metrics` and `/v1/status`. |
| `tools/strata_bench.py` | The vendor's benchmark suite (chat, 4K/16K prompts, agent session). |
| **edge0-infer** | Edge0's own C++ runtime. It vendors ggml at llama.cpp `7ab4ee7`. For 145B it only launches edge0-strata, so it was not used here. |

## Build (RTX 3090, sm_86)

[`scripts/setup.sh`](../scripts/setup.sh) does the following, pinned:

```bash
git clone https://github.com/Edge0-AI/Edge0.git /opt/edge0 && git -C /opt/edge0 checkout da9cc526747759227714ddde0728763d95306c04
cd /opt/edge0/edge0-strata
mkdir -p third_party/llama.cpp
curl -fsSL https://github.com/ggml-org/llama.cpp/archive/3cf03257f219afbe7334045ff7c6a06ac68c627d.tar.gz \
  | tar -xz --strip-components=1 -C third_party/llama.cpp
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$(command -v nvcc) \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
cmake --build build --target strata -j
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Toolchain: nvcc 13.0, host driver 595.91.07 (CUDA 13.2). Set `CMAKE_CUDA_ARCHITECTURES=86` explicitly,
because the default list targets newer GPUs. The engine accepts compute capability 7.5 and up.

At start, the engine turns on nine sm_86 speed switches by default (`STRATA_SM86_PERF=0` disables them):

```text
ADAPT_NOWAIT ADAPT_LAG CPU_IQ_ALL_EXACT1 SIDE_MIXER ATTN_DAILY GR_UP75 GU_SWIGLU QSA_ROWS B6_MMVQ_ROWS
```

## The shipped configuration (`model.json`)

```text
--pack . --native ./model-skeleton.gguf --native-dense-gguf ./model-skeleton.gguf
--ple-gguf ./ple-fp8.gguf --ple-io direct --expert-profile ./profile.bin --expert-cache auto
--prefill auto --pcie-frac 0.1 --spec 4 --spec-min-p 0.5 --mtp ./mtp --max-context 131072
--kv int8 --kv-resident 32768 --vram-reserve-mib 700 --pool-workers 24 --pcie-mode dma
```

The vendor's `model-bench.json` adds `--resident-experts`, `--no-prefill-borrow` and
`--conversation-cache-mib 0`, and sets `--kv-resident 0`. **On this 3090 host it was much slower:**

| Config | Decode, N=1 | Prefill, ~3.3K-token prompt | Prefill, ~13.1K-token prompt |
|---|---|---|---|
| `model.json` | 43.8 tok/s | 1180 tok/s | 1420 tok/s |
| `model-bench.json` | 36.4 tok/s | 394 tok/s | 497 tok/s |

It also loaded in 11 s instead of 30–350 s, which suggests it streams experts from disk instead of holding
them in RAM. Every later sweep uses `model.json` as the base.

## Engine behaviour this study found

These came out of failed runs and engine logs. None of them is in the Edge0 docs.

1. **A native (IQ) pack needs `--spec T` with T ≥ 2, and `--prefill CHUNK`.**

   - Removing `--spec`, or setting `--spec 1`, makes the engine exit at load with: "is a native (IQ) pack:
     it needs --native SHARD1, --spec T (T >= 2) and --prefill CHUNK" (`src/program/generate.cpp`, around
     line 4481).
   - So "speculation off" is not a valid setting for this model.
   - **To turn MTP off, remove `--mtp` only.** The engine then reports `drafting: off`. Its n-gram lookup
     still proposes drafts: 32 % were accepted in our run.

2. **The engine reports the window as T + 2.** `--spec 2` shows `spec: 4` in `/metrics`, and the shipped
   `--spec 4` shows `spec: 6`.

3. **Batch mode turns MTP drafting off unless `--batch-mtp` is given.**

   - With `--batch 2` and no `--batch-mtp`, a run offered 1 draft across 17 requests.
   - `--batch-mtp` needs `--batch` ≥ 2, `--mtp` and `--spec` ≥ 2, and allows at most 4 slots.
   - With plain `--batch N`, the MTP head still occupies VRAM it does not use.

4. **Each batch slot shrinks the VRAM expert cache.** At `--kv-resident 32768`:

   | `--batch` | Experts cached in VRAM | Expert cache |
   |---|---|---|
   | 1 (no `--batch`) | 7360 | 11.2 GiB |
   | 2 | 6397 | 9.7 GiB |
   | 4 | 5434 | 8.3 GiB |
   | 8 | 3507 | 5.3 GiB |

   The engine log states the cause: `--batch N` allocates **N slot sessions of 0.73 GiB each** (per-slot
   KV and buffers), "that the expert cache would otherwise hold". `--batch 2` takes 1.46 GiB, and
   `--batch 8` takes 5.82 GiB. A run without `--batch` has no slot sessions. One slot session equals about
   481 cached experts. Removing the MTP head at N = 1 raises the cache to 8079 experts.

5. **This CPU has no AVX-512.** The engine says: "the expert kernels run on AVX-2". The host is an EPYC 7H12
   (Zen 2). Many desktop 3090 hosts are in the same position, while recent server CPUs have AVX-512. This
   affects work the engine does on the CPU (PLE dequantisation, expert work on the CPU).

6. **Memory guard.** The engine reads the container's cgroup limit (116.4 GiB here) and credits reclaimable
   page cache toward it. The shipped config committed a 38.2 GiB arena.

7. **Load time depends on the page cache.**

   - Cold, the engine read 37.3 GiB of experts at 3.6 GiB/s, which took 334 s.
   - With the pack in page cache, a load took about 30 s.
   - Most configs here loaded in 30–130 s.

8. **There are two logs.** `server.py`'s console output is short. The engine's own log (`"log"` in the config,
   for example `serve-<label>.log` in the pack folder) carries the useful detail: slot-session sizes, expert-cache
   sizing, MTP drafter setup, and one line per request with prompt reuse, prefill speed and "drafts
   accepted X of Y". All of them are in `results/runs/<label>/engine.log`.

9. **`server.py` resolves a bare `"exe": "strata"` against the pack folder.** Running from a separate
   build tree needs an absolute path in the config. `scripts/run_config.py` sets it.

10. **Prefill chunk:** `auto` reports `prefill_cap 8192` and performed the same as `--prefill 8192`.
   `--prefill 1024` was consistently slower than both 512 and 2048 (632 vs 872 and 875 tok/s on ~3.3K-token
   prompts). The cause was not found. The expert cache was the same size in all three runs.

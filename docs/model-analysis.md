# Model analysis: Edge0-145B (Q3_K)

Facts below come from the Hugging Face model card, the pack's own files (`model.json`, the
`model-skeleton.gguf` header read by byte range) and the engine source at the pinned commit. Where two
sources disagree, both values are shown.

## What it is

| Item | Value | Source |
|---|---|---|
| Base model | Qwen3.8-Flash-Next (`qwen4exp` architecture), converted to a "latent sparse MoE" | model card |
| Parameters | 145B total, **10.12B active per token** (`model.json` says 6.6B, which looks stale) | card vs `model.json` |
| Layers | 48. Three Gated-DeltaNet (linear attention) layers per full-attention layer. Sparse QSA indexer, hyper-connections (`hc_count` 4) | base config, GGUF header |
| Experts | 512 per layer, 10 routed + 1 shared, SwiGLU width 640, run in a **latent width of 1792** through shared per-layer down/up projections (2560 ↔ 1792) | card |
| Expert record | Fixed 1,630,720 bytes per expert (Q3_K gate/up plus column scales). The engine caches thousands in VRAM and streams the rest from host RAM over PCIe | card, pack index |
| PLE table | A 51.2B-parameter fp8 n-gram embedding table (`ple-fp8.gguf`, 51 GB), read unbuffered from SSD on every token. Dequantised on the CPU, so the 3090's lack of FP8 hardware does not matter | card, `ngram.cpp` |
| MTP head | A 1.26B-parameter multi-token-prediction draft head with its own 512 experts | card, `mtp/` |
| KV cache | int8. 32K cells resident in VRAM by default, the rest streamed from pinned RAM. Context up to 256K (the shipped config uses 131,072) | `model.json` |
| Weights (Q3_K) | 40.81 GiB. On a 24 GB card the vendor plans **23.14 GiB VRAM + 31.28 GiB RAM** | card |
| License | Edge0 Community Model License 1.0: research use allowed, commercial use needs a separate license | `LICENSE` in the model repo |

## Files in the pack (downloaded: 103.4 GB)

| File | Size | Role |
|---|---|---|
| `experts.bin` | 40.08 GB | All routed experts, fixed-size records ("strata pack index v3") |
| `ple-fp8.gguf` | 51.20 GB | The n-gram embedding table, read directly from disk |
| `model-skeleton.gguf` | 5.99 GB | Everything except the experts. **It has no `ffn_*_exps` tensors** |
| `dense.bin`, `latent_dense_q8.bin` | 1.54 GB, 3.74 GB | Dense and latent projection weights |
| `mtp/` | 0.82 GB | MTP draft head and its experts |
| `profile.bin`, `labels.bin` | small | Expert-usage profile used to seed the VRAM cache |

The `assets/` folder and `latent_dense_q4.bin` (an alternative latent build) were excluded from the
download.

## Why stock llama.cpp cannot run it

The skeleton GGUF declares `general.architecture = "qwen4exp"`, which upstream llama.cpp supports
(`src/models/qwen4exp.cpp` at v1.0.0). That loader, though, requires `ffn_gate_exps` / `ffn_down_exps`
tensors shaped `{n_embd, n_ff_exp, n_expert}` and has no latent down/up path. The Edge0 skeleton has
neither: its experts live in `experts.bin` in a different layout. A load in stock llama.cpp therefore fails
on missing tensors. The only llama.cpp issue about Edge0 support (#29061) is closed, and no PR adds it.

The model runs on **edge0-strata**, Edge0's fork of the Strata engine (MIT). edge0-strata builds on ggml
pinned to llama.cpp commit `3cf03257` and uses llama.cpp's MMQ CUDA kernels. See
[engine-and-build.md](engine-and-build.md).

## What limits throughput on a 3090

Each decoded token draws on three resources at once:

1. **GPU memory bandwidth** for cached experts, attention and KV. The 3090 has 936 GB/s, close to the
   4090's 1008 GB/s.
2. **PCIe** for experts that miss the VRAM cache. This host runs gen4 x16.
3. **SSD and CPU** for the PLE table, which is read from disk and dequantised on the CPU every token.

Prefill is compute-bound, and there the 3090 has about 2.4× less compute than a 4090. So the expectation
going in was decode close to the 4090 and prefill well behind it, which is what the vendor-tool comparison
shows (77 % / 62 % of decode, 62 % / 46 % of prefill).

**VRAM is the other limit.** Every byte not used by experts is one fewer cached expert, and each batch slot
or resident KV cell costs expert cache. That trade-off is what bends the batch curve.

## Published speeds before this study

The model card lists RTX 4090 and RTX 5090 numbers for the Q3_K pack:

- 4090: chat decode 69.9 tok/s, agent decode 91.7 tok/s, 4K prefill 2042 tok/s, 16K prefill 3179 tok/s.

There were **no published RTX 3090 numbers**. One Edge0 issue (#205) mentions a 16 GB RTX A4000, also
sm_86, running the model "slowly", and a layer-split failure on two sm_86 cards. This study is single-GPU,
so that failure does not apply.

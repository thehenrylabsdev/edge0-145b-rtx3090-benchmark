# X thread

Attach `assets/social-card.png` (1200×675) to post 1, `assets/batch-curve.png` to post 3 and
`assets/vs-4090.png` to post 5. Every post is under 280 characters. Each paragraph is one line, so it pastes cleanly.

---

**1/6**

How many requests should one RTX 3090 batch to get the most tokens/s out of a 145B model?

I benchmarked Edge0-145B (Q3_K, ~10B active) on a single 24 GB 3090.

Answer: 6 requests → 60.2 tok/s aggregate, +29% over one at a time.

Method, scripts, raw data below 🧵

**2/6**

Edge0-145B is a latent sparse MoE with 512 experts per layer. It fits in 24 GB by caching experts in VRAM, streaming the rest from RAM over PCIe, and reading a 51B-param embedding table from SSD every token.

Stock llama.cpp can't load it, so I used Edge0's ggml-based engine.

**3/6**

The catch: every batch slot reserves 0.73 GiB of VRAM, taken from the expert cache.

Plain --batch N: 46.8 → 54.4 tok/s from N=1 to N=6, then down to 48.1 at N=8.

Removing the unused MTP head and shrinking resident KV gives experts that VRAM back: 60.2 tok/s at N=6.

**4/6**

Batching isn't free for users. At N=6 each request decodes at ~11 tok/s with a 2.1 s first token, vs 58.7 tok/s alone.

For a single user, the best setting was the shortest MTP window (--spec 2): 3% faster than the shipped --spec 4.

**5/6**

Vs the vendor's self-reported RTX 4090 numbers, the 3090 got:
• 77% of chat decode (54.1 vs 69.9 tok/s)
• 62% of agent decode
• 46% of 16K-token prefill (1470 vs 3179 tok/s)

Decode holds up. Prefill is compute-bound and falls behind.

**6/6**

Method: closed-loop load, unique prompts, fresh engine per config, 3×45 s windows, 37 configs. About 3 GPU-hours, about $1.60 on RunPod.

Everything is open: scripts, configs, engine logs, raw CSV.

https://github.com/thehenrylabsdev/edge0-145b-rtx3090-benchmark

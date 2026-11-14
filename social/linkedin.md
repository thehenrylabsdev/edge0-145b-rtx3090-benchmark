# LinkedIn post

Attach `assets/social-card-square.png` (1080×1080). Each paragraph is one line, so it pastes cleanly. Optionally add `assets/batch-curve.png` and
`assets/vs-4090.png` as a carousel.

---

How many requests should a single RTX 3090 batch to get the most tokens per second out of a 145B-parameter model?

I ran a reproducible benchmark of Edge0-145B (Q3_K, a latent sparse mixture-of-experts with about 10B parameters active per token) on one 24 GB RTX 3090.

The answer: batch 6 requests. That gave 60.2 tokens/s aggregate, 29% more than serving one request at a time (46.8 tokens/s).

What I learned:

1. Batch slots compete with the model for VRAM. This model keeps thousands of experts cached on the GPU and streams the rest over PCIe. Every batch slot reserves 0.73 GiB, taken from that cache. With the shipped config, throughput peaks at 54.4 tokens/s at 6 slots and drops to 48.1 at 8.

2. Free VRAM elsewhere and the curve rises. Batch mode doesn't use the speculative-decoding (MTP) head, so removing it, and keeping fewer KV cells resident, gives the cache its memory back. Together they lifted the peak to 60.2 tokens/s.

3. Throughput and latency pull in opposite directions. At 6 slots each user sees about 11 tokens/s and a 2-second first token. Alone, a request streams at 58.7 tokens/s. The right batch size depends on whether you serve many users or one.

4. A 3090 still holds up. Against the vendor's self-reported RTX 4090 numbers, it reached 77% of chat decode speed and 46% of 16K-token prefill.

Method: closed-loop load generator with unique prompts, a fresh engine process per configuration, three 45-second measured windows each, 37 configurations in total. About 3 GPU-hours, about $1.60 on RunPod.

Every script, config, engine log and raw number is in the repository (MIT code, CC BY 4.0 data): https://github.com/thehenrylabsdev/edge0-145b-rtx3090-benchmark

If you have a different GPU, the repo has an issue template for sharing results.

#LLM #GPU #Benchmarking #MachineLearning #OpenSource

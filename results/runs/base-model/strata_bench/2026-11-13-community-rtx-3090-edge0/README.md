# Community benchmark on NVIDIA GeForce RTX 3090

Measured on 2026-11-13 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3090, 24 GiB VRAM
- CPU: AMD EPYC 7H12 64-Core Processor (128 cores, 256 threads); installed RAM: 1007.6 GiB; storage: sda     Micron_5300_MTFD                         sata; nvme2n1 Seagate ZP4000GM30063                    nvme; nvme1n1 Seagate ZP4000GM30063                    nvme; nvme0n1 Seagate ZP4000GM30063                    nvme; nvme3n1 KINGSTON SFYRD4000G                      nvme
- OS: Linux 6.8.0-138-generic; GPU driver: 595.91.07; CUDA (driver supports up to): 13.2
- Strata: project 0.1.55, commit not a git checkout; engine 0.1.55
- GPU detail (after the run, near idle): power_limit_w=350.00, power_draw_w=329.02, pcie=gen4 x16, pstate=P2, temp_c=63
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: edge0
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 54.1 (47.8-62.1) | 0.79 (0.75-0.93) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1273.4 (1269.9-1280.7) | not measured | 3.19 (3.17-3.20) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1470.0 (1465.6-1480.2) | not measured | 10.96 (10.90-11.00) |
| agent session, follow-up turns | 14399 (14373-14407) at the last turn | see below | up to 64 | 3 | see below | 57.3 (51.6-73.6) | 2.34 (1.81-3.93) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 65.3 % (65.3-65.7); earlier-turn tokens read again per session: 143 (2-144) (earlier turns that the cache did not hold; the 13514 (13513-13515) new tokens of the files are read once); first turn (nothing cached) first token: 2.01 s (1.74-2.10).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 357.1 | 1.738 | 44.8 |
| 1 | 2096 | 628 | 1468 | 797.0 | 1.878 | 52.2 |
| 2 | 5340 | 2089 | 3251 | 1080.7 | 3.051 | 51.6 |
| 3 | 7401 | 5396 | 2005 | 973.7 | 2.101 | 59.7 |
| 4 | 10129 | 7447 | 2682 | 1164.0 | 2.338 | 67.9 |
| 5 | 14373 | 10122 | 4251 | 1093.4 | 3.929 | 59.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 135 s.

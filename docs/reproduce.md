# Reproduce it

Everything runs on one Linux machine with an RTX 3090. The scripts assume three paths: `/opt/edge0` (engine),
`/models/Edge0-145B-q3k` (pack) and `/results` (output). Edit the constants at the top of
`scripts/run_config.py` if yours differ.

## 1. A machine

| Need | Used here | Why |
|---|---|---|
| GPU | 1× RTX 3090 24 GB | the question |
| NVIDIA driver | 595.91.07 (CUDA 13.2) | the engine needs CUDA 13 |
| Image / OS | `nvidia/cuda:13.0.3-devel-ubuntu24.04` | nvcc 13.0 for the build |
| RAM | 125 GB allocation (116.4 GiB cgroup limit) | about 31 GiB of experts stay in RAM, plus page cache for a fast reload. Below 48 GB the host will page |
| Disk | 200 GB NVMe (fio: 9.1 GB/s sequential, 98K IOPS at 4 KiB) | the pack is 103.4 GB, and the 51 GB PLE table is read from disk on every token |
| CPU | 32 vCPU, AMD EPYC 7H12 | pool workers. AVX-512 would help, and this CPU does not have it |

We used a RunPod **Secure Cloud** pod in EU-CZ-1 at **$0.50/hr**: `sleep infinity` entrypoint, no
exposed ports, minimum CUDA 13.0. A Community Cloud 3090 is cheaper ($0.22/hr), but the only host
available that day had 30 GB of RAM, so it was rejected. Any Linux box with a 3090 and enough RAM works.

## 2. Build, download, host facts

The download and the build can run in parallel. Together they took about 20 minutes on this host.

```bash
bash scripts/setup.sh        # toolchain, Edge0 @ da9cc526, ggml @ 3cf03257, strata for sm_86, server venv
bash scripts/download.sh     # hf download Edge0/Edge0-145B-q3k (public), size check against the HF listing
bash scripts/host_facts.sh   # nvidia-smi, lscpu, free, cgroup limit, fio sequential and random reads
```

`download.sh` excludes `assets/*` and `latent_dense_q4.bin`. Note that the `hf` CLI takes one pattern per
`--exclude`. A space-separated list silently matches nothing useful.

## 3. Run the sweeps

Each phase takes the winners of the previous one as arguments:

```bash
cd scripts
python3 make_specs.py /results/speclists                       # Epics 2-4 (baseline, prefill, spec)
bash run_sweep.sh /results/speclists/p2-baseline.jsonl > /results/sweep-p2.log
bash run_sweep.sh /results/speclists/p3-prefill.jsonl  > /results/sweep-p3.log
bash run_sweep.sh /results/speclists/p4-spec.jsonl     > /results/sweep-p4.log

# Epic 5 with this study's winners: --spec 2, refine N = 3,5,6, pressure arms at N = 4,8
python3 make_specs.py /results/speclists --spec 2 --refine 3,5,6 --best 4,8
bash run_sweep.sh /results/speclists/p5-batch-coarse.jsonl > /results/sweep-p5c.log
bash run_sweep.sh /results/speclists/p5-batch-refine.jsonl > /results/sweep-p5r.log
bash run_sweep.sh /results/speclists/p5-long-kv.jsonl      > /results/sweep-p5k.log

# Two lists were written by hand during the run; copies are in results/speclists/:
#   p4b-nomtp.jsonl   the corrected "MTP head removed" arm at N = 1 (spec-0/spec-1 cannot load)
#   p6-confirm.jsonl  MTP head removed at N = 5, 6, 7; MTP head removed + --kv-resident 8192 at N = 4, 6, 8;
#                     batch-7. It replaced the planned confirmation reps, hence the name.
bash run_sweep.sh /results/speclists/p4b-nomtp.jsonl  > /results/sweep-p4b.log
bash run_sweep.sh /results/speclists/p6-confirm.jsonl > /results/sweep-p6.log

python3 collect.py /results     # -> /results/results.csv plus a printed table
```

A spec line is a small JSON object:

```json
{"label": "batch-4", "base": "model.json", "parallel": 4, "set": {"--spec": "2"},
 "phases": [{"mode": "decode", "concurrency": 4, "duration": 45, "warmup": 15, "reps": 3,
             "max_tokens": 256, "prompt_words": 380}]}
```

- `set` and `remove` edit engine flags.
- `add` appends a switch.
- `parallel` sets batch slots.
- `env` adds environment variables.

`run_config.py` derives the config, starts the server, waits for a real response, runs the phases, saves
`/metrics` before and after each rep, then stops the engine. Results go to `/results/runs/<label>/`.

Wall time per config is about 4 minutes for decode (load + 15 s warmup + 3 × 45 s) and 1–3 minutes for
prefill. The whole study is 37 measured configs plus a smoke test, about 3 hours of pod time (about $1.60 at
$0.50/hr).

## 4. Charts and summary

```bash
python3 charts/build_charts.py          # results/results.csv -> charts/specs/*.chart.json, results/summary.json
cd charts && npm install && npm run render   # -> assets/*.svg, assets/*.png (ECharts SSR + resvg, no browser)
```

## Running it from a laptop against a RunPod pod

`tools/podexec.py` runs one shell command in a pod through RunPod's REST v2 exec endpoint, and
`tools/podpush.py` copies a file in. Both read `RUNPOD_API_KEY` from the environment and never print it.
The exec timeout is capped at 25 s, so start long jobs with `nohup ... &` and poll their logs.

```bash
tools/podpush.py <pod-id> scripts/run_config.py /opt/bench/scripts/run_config.py
tools/podexec.py <pod-id> 'cd /opt/bench/scripts && nohup bash run_sweep.sh /results/speclists/p5-batch-coarse.jsonl > /results/sweep-p5c.log 2>&1 &'
tools/podexec.py <pod-id> 'tail -5 /results/sweep-p5c.log' 20
```

Delete the pod when you are done. A stopped pod still bills for its disk.

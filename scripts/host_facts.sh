#!/usr/bin/env bash
# Record host facts that drive Edge0 throughput: GPU/PCIe, CPU, RAM, disk read speed.
set -uo pipefail
OUT=/results/host
mkdir -p "$OUT"
nvidia-smi -q > "$OUT/nvidia-smi-q.txt" 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,power.limit,clocks.max.sm,clocks.max.mem \
  --format=csv > "$OUT/gpu.csv"
lscpu > "$OUT/lscpu.txt"; nproc > "$OUT/nproc.txt"
free -g > "$OUT/free.txt"; df -h /models / > "$OUT/df.txt" 2>&1
cat /sys/fs/cgroup/memory.max > "$OUT/cgroup-memory-max.txt" 2>/dev/null || true
# Disk: the PLE table is read unbuffered from here on every token.
mkdir -p /models/fio
fio --name=seq1m --directory=/models/fio --rw=read --bs=1M --size=4G --direct=1 --ioengine=libaio --iodepth=32 \
    --runtime=20 --time_based --group_reporting --output-format=json > "$OUT/fio-seq1m.json" 2>&1
fio --name=rand4k --directory=/models/fio --rw=randread --bs=4k --size=4G --direct=1 --ioengine=libaio --iodepth=32 \
    --runtime=20 --time_based --group_reporting --output-format=json > "$OUT/fio-rand4k.json" 2>&1
rm -rf /models/fio
python3 - <<'EOF'
import json
for n in ("seq1m", "rand4k"):
    j = json.load(open(f"/results/host/fio-{n}.json"))["jobs"][0]["read"]
    print(f"fio {n}: {j['bw']/1024:.0f} MiB/s, {j['iops']:.0f} IOPS")
EOF
cat "$OUT/gpu.csv"; cat "$OUT/free.txt"; echo "nproc=$(nproc)"
echo HOST_FACTS_DONE

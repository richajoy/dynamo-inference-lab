#!/usr/bin/env bash
# Summarize one nsys report with the HOST nsys. Runs ON the node.
#   ./profile/nsys-report.sh /var/lib/nsys-reports/<label>-<shape>.nsys-rep
# Writes next to the report:
#   <name>_cuda_gpu_kern_sum.csv  : per-kernel total time, count, avg  -> what the GPU spent time on
#   <name>_cuda_api_sum.csv       : CPU-side CUDA API time (launch overhead, graph launches, syncs)
#   <name>_cuda_gpu_mem_time_sum.csv : memcpy/memset time
#   <name>_nvtx_sum.csv           : vLLM per-step ranges execute_context_N(tok)_generation_N(tok) = step time by mix
# and prints the top 25 kernels by total time. The kernel names are the input for the prefill/decode classifier.
set -euo pipefail
REP="${1:?usage: nsys-report.sh <file.nsys-rep>}"
NSYS=/opt/nvidia/nsys-current/bin/nsys
BASE="${REP%.nsys-rep}"
"$NSYS" stats --force-export=true --format csv --output "$BASE" \
  --report cuda_gpu_kern_sum --report cuda_api_sum --report cuda_gpu_mem_time_sum --report nvtx_sum "$REP" >/dev/null
echo "== top kernels by GPU time: $(basename "$REP")"
python3 - "${BASE}_cuda_gpu_kern_sum.csv" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
tot = sum(float(r["Total Time (ns)"]) for r in rows)
print(f"{'%':>6} {'total ms':>9} {'count':>7} {'avg us':>8}  kernel")
for r in rows[:25]:
    t = float(r["Total Time (ns)"])
    print(f"{100*t/tot:6.1f} {t/1e6:9.1f} {int(r['Instances']):7d} {float(r['Avg (ns)'])/1e3:8.1f}  {r['Name'][:110]}")
print(f"kernels: {len(rows)} distinct, {tot/1e6:.1f} ms total GPU kernel time")
EOF

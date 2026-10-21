#!/usr/bin/env bash
# Phased load for the per-pool autoscaling test (deploy/pdp-mig-qwen3-8b.yaml). Runs INSIDE the aiperf pod:
#   kubectl -n dynamo-system cp bench/autoscale-phases.sh aiperf:/tmp/autoscale-phases.sh
#   kubectl -n dynamo-system exec aiperf -- bash /tmp/autoscale-phases.sh
# Pair with bench/watch-replicas.sh on the node (logs the P:D split every 10 s).
# Phases (PHASE_SECS each, default 300 s = enough for planner observations + a ~1.5-2 min worker cold start):
#   1 prefill-heavy : ISL 1500 / OSL 32,   c=48  -> prefill queue grows, decode KV low      -> expect P up, D down
#   2 decode-heavy  : ISL 256  / OSL 1024, c=128 -> decode KV high, prefill queue empty     -> expect P down, D up
#   3 mixed         : both shapes at once (c=24 prefill-heavy + c=64 decode-heavy)          -> expect a middle split
set -euo pipefail
URL="${URL:-http://pdp-mig-qwen3-8b-frontend.dynamo-system.svc:8000}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
PHASE_SECS="${PHASE_SECS:-300}"
PHASES="${PHASES:-prefill decode mixed}"
OUT="/results/$(date +%Y%m%d-%H%M)-autoscale"
mkdir -p "$OUT"
load() {  # $1=label $2=isl $3=osl $4=concurrency
  aiperf profile --model "$MODEL" --url "$URL" --endpoint-type completions --streaming \
    --concurrency "$4" --benchmark-duration "$PHASE_SECS" --benchmark-grace-period 30 \
    --synthetic-input-tokens-mean "$2" --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean "$3" --output-tokens-stddev 0 \
    --extra-inputs "max_tokens:$3" --extra-inputs "min_tokens:$3" --extra-inputs ignore_eos:true \
    --artifact-dir "$OUT/$1" --ui simple > "$OUT/$1.log" 2>&1 || tail -20 "$OUT/$1.log"
}
for p in $PHASES; do
  echo "$(date +%H:%M:%S) phase $p start"
  case $p in
    prefill) load prefill 1500 32 48 ;;   # 1500 not 3000: MIG P/D moves KV over TCP (~420 MiB/s), keep the planner signal compute-driven
    decode)  load decode 256 1024 128 ;;
    mixed)   load mixed-p 1500 32 24 & load mixed-d 256 1024 64; wait ;;
  esac
  echo "$(date +%H:%M:%S) phase $p end"
done
echo "done: $OUT"

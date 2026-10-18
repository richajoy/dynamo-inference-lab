#!/usr/bin/env bash
# Concurrency sweep for one deployment. Runs INSIDE the aiperf pod:
#   kubectl -n dynamo-system cp bench/sweep.sh aiperf:/tmp/sweep.sh
#   kubectl -n dynamo-system exec -it aiperf -- bash /tmp/sweep.sh <run-label>
# Two workload shapes isolate the two phases:
#   prefill-heavy : long input, short output  -> TTFT dominated by prefill compute
#   decode-heavy  : short input, long output  -> ITL dominated by decode (memory-bandwidth bound)
# Uses /v1/completions (raw prompt, no chat template) so Qwen3 <think> mode cannot change output length;
# ignore_eos + min_tokens force exact output lengths so runs are comparable.
set -euo pipefail

LABEL="${1:?usage: sweep.sh <run-label>}"
URL="${URL:-http://agg-qwen3-8b-frontend.dynamo-system.svc:8000}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
CONCURRENCIES="${CONCURRENCIES:-1 4 16 64}"
SHAPES="${SHAPES:-prefill-heavy decode-heavy}"
# AIPerf defaults to DCGM on localhost:9400 and scrapes only the frontend /metrics; in-cluster both live elsewhere.
DCGM_URL="${DCGM_URL:-http://nvidia-dcgm-exporter.gpu-operator.svc:9400/metrics}"
WORKER_METRICS="${WORKER_METRICS:-http://agg-qwen3-8b-vllmdecodeworker.dynamo-system.svc:9090/metrics}"
OUT="/results/$(date +%Y%m%d-%H%M)-$LABEL"

declare -A ISL=( [prefill-heavy]=3000 [decode-heavy]=256 [long]=6000 )
declare -A OSL=( [prefill-heavy]=128  [decode-heavy]=1024 [long]=1024 )

curl -sf "$URL/v1/models" >/dev/null || { echo "frontend not reachable at $URL"; exit 1; }
mkdir -p "$OUT"
echo "results -> $OUT (node: /var/lib/bench-results/${OUT#/results/})"

for shape in $SHAPES; do
  for c in $CONCURRENCIES; do
    n=$(( c * 4 > 32 ? c * 4 : 32 ))           # enough requests for stable percentiles at each level
    echo "== $shape  ISL=${ISL[$shape]} OSL=${OSL[$shape]}  concurrency=$c  requests=$n"
    aiperf profile \
      --model "$MODEL" --url "$URL" \
      --endpoint-type completions --streaming \
      --concurrency "$c" --request-count "$n" --warmup-request-count 4 \
      --synthetic-input-tokens-mean "${ISL[$shape]}" --synthetic-input-tokens-stddev 0 \
      --output-tokens-mean "${OSL[$shape]}" --output-tokens-stddev 0 \
      --extra-inputs "max_tokens:${OSL[$shape]}" \
      --extra-inputs "min_tokens:${OSL[$shape]}" \
      --extra-inputs ignore_eos:true \
      --gpu-telemetry "$DCGM_URL" --server-metrics "$WORKER_METRICS" \
      --artifact-dir "$OUT/$shape/c$c" --ui simple
  done
done
echo "done: $OUT"

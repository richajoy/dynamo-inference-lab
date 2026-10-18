#!/usr/bin/env bash
# Dynamo router features under load (2x H100). Runs ON the node with deploy/kvr-qwen3-4b.yaml up (2 workers).
# Same prefix workload as the A/B (40 x 2048 + 64, OSL 64) but CONC=64 so workers saturate and the router has to
# trade locality against load, and queueing can engage. Each arm: bench/ab-arm.sh + bench/prom-window.py.
#   ./bench/router-features.sh
# Arms (all kv mode with events):
#   rf-base      : --router-prefill-load-scale 10                          (best locality setting from the A/B)
#   rf-queue     : + --router-queue-threshold 0.5   router holds requests when every worker's active prefill exceeds
#                                                    50% of max_num_batched_tokens (queue instead of overloading)
#   rf-decay     : + --router-kv-overlap-score-credit-decay 1.0   overlap credit halves at 1 request-equivalent of
#                                                    excess prefill backlog (soft locality vs balance)
#   rf-temp      : + --router-temperature 0.5       softmax sampling instead of argmin (spreads hot spots)
#   rf-default   : kv defaults at c=64 (reference)
set -uo pipefail
export AB_ENV="CONC=64"
run() { label=$1; shift; ./bench/ab-arm.sh "$label" "$@" | tail -1
  w=$(python3 -c "import json;print(*json.load(open('results/kv-router/$label.json'))['summary']['window_ms'])")
  python3 bench/prom-window.py $w > "results/kv-router/$label.prom.json"; cat "results/kv-router/$label.prom.json"; }
run rf-default kv
run rf-base    kv --router-prefill-load-scale 10
run rf-queue   kv --router-prefill-load-scale 10 --router-queue-threshold 0.5
run rf-decay   kv --router-prefill-load-scale 10 --router-kv-overlap-score-credit-decay 1.0
run rf-temp    kv --router-prefill-load-scale 10 --router-temperature 0.5
echo RF_DONE

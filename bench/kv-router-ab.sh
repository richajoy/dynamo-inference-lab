#!/usr/bin/env bash
# KV-routing A/B: one prefix-sharing workload, run once per router mode. Runs INSIDE the aiperf pod:
#   kubectl -n dynamo-system cp bench/kv-router-ab.sh aiperf:/tmp/kv-router-ab.sh
#   (node) ./bench/router-mode.sh round-robin
#   kubectl -n dynamo-system exec -it aiperf -- bash /tmp/kv-router-ab.sh rr
#   (node) ./bench/router-mode.sh kv          ; ... kv-router-ab.sh kv
#   (node) ./bench/router-mode.sh kv-approx   ; ... kv-router-ab.sh kv-approx
#
# Reports per run:
#   engine hit rate (ACTUAL)  = delta vllm:prefix_cache_hits_total / delta vllm:prefix_cache_queries_total, per worker
#   router hit rate (PREDICTED) = router_kv_hit_rate histogram = overlap blocks in the router's index / ISL blocks
#                                 (absent in round-robin: no index)
#   UNITS CAVEAT: router rate is blocks/blocks averaged per request; engine rate is tokens/tokens summed. Treat the
#   aggregate comparison as indicative; the per-request pairing comes from the frontend request trace
#   (/var/lib/request-trace on the node: kv_hit_rate vs floor(cached_tokens/16)/ceil(input_tokens/16)).
#   routing balance           = delta vllm:request_success_total per worker
#   TTFT p50/p90, ITL         = from AIPerf
# --num-dataset-entries N: AIPerf attaches the prefix when it BUILDS an entry (synthetic.py:215) and defaults to
#   100 entries, so without it 800 requests replay 100 identical full prompts (suffix cached too -> inflated hits).
# Fresh random prefixes per run (SEED defaults to epoch) so caches warmed by the previous mode don't leak in.
set -euo pipefail

LABEL="${1:?usage: kv-router-ab.sh <label>}"
URL="${URL:-http://kvr-qwen3-4b-frontend.dynamo-system.svc:8000}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
PROM="${PROM:-http://prometheus-kube-prometheus-prometheus.monitoring.svc:9090}"
DCGM_URL="${DCGM_URL:-http://nvidia-dcgm-exporter.gpu-operator.svc:9400/metrics}"
SEED="${SEED:-$(date +%s)}"
CONC="${CONC:-16}"

# ---- Workload sizing ------------------------------------------------------------------------------------
# Fixed facts (deploy/kvr-qwen3-4b.yaml): 2 workers, each with a KV cache of 50,960 tokens (3,185 blocks of 16),
# CONC concurrent requests split across them; vLLM max-model-len 8192.
# AIPerf builds each prompt as: one prefix drawn uniformly at random from a pool of NUM_PREFIXES
# (PREFIX_LEN tokens each) + a unique random suffix (SUFFIX_LEN tokens).
#
# Sizing: prefix working set 40 x 2048 = 81,920 tokens = 5,120 blocks
#   = 1.61x ONE worker (3,185 blocks) -> round-robin must thrash (each worker sees all 40 prefixes)
#   = 0.80x the FLEET (6,370 blocks)  -> KV routing can pin ~20 prefixes per worker (2,560 blocks) and keep them
# PREFIX_LEN is a multiple of 16 (128 full blocks); SUFFIX_LEN 64 keeps prefix share of the prompt at 97%.
NUM_PREFIXES="${NUM_PREFIXES:-40}"
PREFIX_LEN="${PREFIX_LEN:-2048}"
SUFFIX_LEN="${SUFFIX_LEN:-64}"
# ---------------------------------------------------------------------------------------------------------
: "${NUM_PREFIXES:?set NUM_PREFIXES}" "${PREFIX_LEN:?set PREFIX_LEN}" "${SUFFIX_LEN:?set SUFFIX_LEN}"
OSL="${OSL:-64}"
N="${N:-$(( NUM_PREFIXES * 20 ))}"   # ~20 uses per prefix: cold misses = NUM_PREFIXES/N = 5%
OUT="/results/$(date +%Y%m%d-%H%M)-kvr-$LABEL"
mkdir -p "$OUT"

snapshot() {  # Prometheus instant query of the counters we diff; prints JSON
  python3 - "$PROM" <<'EOF'
import json, sys, urllib.parse, urllib.request
prom = sys.argv[1]
q = {
  "hits":    'sum by (pod) (vllm:prefix_cache_hits_total{pod=~"kvr-qwen3-4b.*"})',
  "queries": 'sum by (pod) (vllm:prefix_cache_queries_total{pod=~"kvr-qwen3-4b.*"})',
  "reqs":    'sum by (pod) (vllm:request_success_total{pod=~"kvr-qwen3-4b.*"})',
  "rhr_sum": 'sum({__name__=~".*router_kv_hit_rate_sum"})',
  "rhr_cnt": 'sum({__name__=~".*router_kv_hit_rate_count"})',
}
out = {}
for k, expr in q.items():
    r = json.load(urllib.request.urlopen(f"{prom}/api/v1/query?query={urllib.parse.quote(expr)}", timeout=10))
    out[k] = {s["metric"].get("pod", "_"): float(s["value"][1]) for s in r["data"]["result"]}
print(json.dumps(out))
EOF
}

curl -sf "$URL/v1/models" >/dev/null || { echo "frontend not reachable at $URL"; exit 1; }
echo "run=$LABEL seed=$SEED prefixes=$NUM_PREFIXES x $PREFIX_LEN, suffix=$SUFFIX_LEN, osl=$OSL, c=$CONC, n=$N -> $OUT"
sleep 6; snapshot > "$OUT/before.json"      # >1 scrape interval (5 s) after any previous run

T0=$(date +%s%3N)   # run window (epoch ms) -> joins this run to the frontend request trace
aiperf profile --model "$MODEL" --url "$URL" --endpoint-type completions --streaming \
  --concurrency "$CONC" --request-count "$N" --warmup-request-count 4 --random-seed "$SEED" \
  --num-dataset-entries "$N" \
  --num-prefix-prompts "$NUM_PREFIXES" --prefix-prompt-length "$PREFIX_LEN" \
  --synthetic-input-tokens-mean "$SUFFIX_LEN" --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean "$OSL" --output-tokens-stddev 0 \
  --extra-inputs "max_tokens:$OSL" --extra-inputs "min_tokens:$OSL" --extra-inputs ignore_eos:true \
  --gpu-telemetry "$DCGM_URL" --artifact-dir "$OUT" --ui simple > "$OUT/aiperf.out" 2>&1 \
  || { tail -30 "$OUT/aiperf.out"; exit 1; }
T1=$(date +%s%3N)

sleep 12; snapshot > "$OUT/after.json"      # let Prometheus scrape the final counter values

python3 - "$OUT" "$LABEL" "$SEED" "$T0" "$T1" <<'EOF'
import json, sys, pathlib
out, label, seed = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
b, a = json.load(open(out/"before.json")), json.load(open(out/"after.json"))
d = lambda k: {p: a[k].get(p, 0) - b[k].get(p, 0) for p in a[k]}
hits, qs, reqs = d("hits"), d("queries"), d("reqs")
def find(o, key):                      # locate a metric block anywhere in the AIPerf JSON export
    if isinstance(o, dict):
        if key in o and isinstance(o[key], dict): return o[key]
        for v in o.values():
            r = find(v, key)
            if r: return r
    return None
prof = json.load(open(out/"profile_export_aiperf.json"))
ttft, itl = find(prof, "time_to_first_token") or {}, find(prof, "inter_token_latency") or {}
rc = d("rhr_cnt").get("_", 0)
summary = {
  "label": label, "seed": seed, "window_ms": [int(sys.argv[4]), int(sys.argv[5])],
  "engine_hit_rate_total": round(sum(hits.values()) / max(sum(qs.values()), 1), 4),
  "engine_hit_rate_by_worker": {p[-5:]: round(hits[p] / max(qs.get(p, 0), 1), 4) for p in hits},
  "requests_by_worker": {p[-5:]: int(v) for p, v in reqs.items()},
  "router_predicted_hit_rate": round(d("rhr_sum").get("_", 0) / rc, 4) if rc else None,
  "ttft_ms": {k: round(ttft[k], 1) for k in ("p50", "p90", "avg") if k in ttft},
  "itl_ms_avg": round(itl["avg"], 2) if "avg" in itl else None,
}
json.dump(summary, open(out/"summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
EOF

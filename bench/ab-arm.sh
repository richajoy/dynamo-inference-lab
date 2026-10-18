#!/usr/bin/env bash
# One A/B arm end to end. Runs ON the node from the repo checkout:
#   AB_ENV="CONC=16" ./bench/ab-arm.sh <label> <router-mode> [extra frontend flags...]
#   e.g. ./bench/ab-arm.sh kv-pls4 kv --router-prefill-load-scale 4
# 1. switch the frontend (bench/router-mode.sh), 2. run bench/kv-router-ab.sh in the aiperf pod (AB_ENV passed as
# env, e.g. NUM_PREFIXES/PREFIX_LEN/SUFFIX_LEN/OSL/CONC/N), 3. join the run window with the frontend request trace.
# Writes results/kv-router/<label>.json = {"summary": <metrics-based>, "trace": <per-request based>}.
set -euo pipefail
LABEL="${1:?usage: ab-arm.sh <label> <router-mode> [flags...]}"; shift
NS=dynamo-system
mkdir -p results/kv-router
./bench/router-mode.sh "$@"
kubectl -n $NS cp bench/kv-router-ab.sh aiperf:/tmp/kv-router-ab.sh
# shellcheck disable=SC2086
kubectl -n $NS exec aiperf -- env ${AB_ENV:-} bash /tmp/kv-router-ab.sh "$LABEL" > "/tmp/ab-$LABEL.out" 2>&1 \
  || { tail -30 "/tmp/ab-$LABEL.out"; exit 1; }
python3 - "/tmp/ab-$LABEL.out" "$LABEL" "$*" "${AB_ENV:-}" <<'EOF'
import json, subprocess, sys
raw = open(sys.argv[1]).read()
summary = json.loads(raw[raw.index("\n{") + 1:])          # the summary JSON is the last block printed
t0, t1 = summary["window_ms"]
trace = json.loads(subprocess.check_output(
    ["python3", "bench/trace-join.py", "/var/lib/request-trace/kvr", str(t0), str(t1), sys.argv[2]]))
res = {"label": sys.argv[2], "router_args": sys.argv[3], "ab_env": sys.argv[4], "run_line": raw.splitlines()[0],
       "summary": summary, "trace": trace}
json.dump(res, open(f"results/kv-router/{sys.argv[2]}.json", "w"), indent=2)
s, t = summary, trace
print(f"{sys.argv[2]:>14}  hit(engine)={s['engine_hit_rate_total']:.3f}  hit(router-pred)={s['router_predicted_hit_rate']}  "
      f"affinity={t['prefix_affinity']}  reqs={s['requests_by_worker']}  "
      f"TTFT p50/p90={s['ttft_ms'].get('p50')}/{s['ttft_ms'].get('p90')} ms  ITL={s['itl_ms_avg']} ms  "
      f"miss TTFT p50={t['ttft_cache_miss']['ttft_p50']}  hit TTFT p50={t['ttft_cache_hit']['ttft_p50']}")
EOF

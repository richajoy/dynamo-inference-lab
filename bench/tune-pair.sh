#!/usr/bin/env bash
# Run two single-GPU tuning variants side by side (one per H100) and sweep both at once. Runs ON the node:
#   SHAPES="prefill-heavy decode-heavy long" CONCURRENCIES="16 64" ./bench/tune-pair.sh agg-mbt512 agg-mbt8k
# Variants come from deploy/variants/<name>.yaml (deploy/gen-variant.py). Deletes them afterwards.
# Caveat: two sweeps share one aiperf pod (4 CPU request). NOT safe: the fp8 and spec-decode pairs kept only
# ~47-54 of 64 requests in flight (docs/06-engine-tuning.md, 'Caveats'). Run one sweep per aiperf pod.
set -uo pipefail
NS=dynamo-system; SHAPES="${SHAPES:-prefill-heavy decode-heavy long}"; CONCURRENCIES="${CONCURRENCIES:-16 64}"
names=("$@")
for n in "${names[@]}"; do kubectl -n $NS apply -f "deploy/variants/$n.yaml" >/dev/null; done
for i in $(seq 1 120); do
  ok=0; for n in "${names[@]}"; do [ "$(kubectl -n $NS get pods --no-headers | grep "^$n-0-" | grep -c '1/1')" = 2 ] && ok=$((ok+1)); done
  [ $ok = ${#names[@]} ] && break; sleep 5
done
kubectl -n $NS get pods | grep -E "$(IFS='|'; echo "${names[*]}")"
kubectl -n $NS cp bench/sweep.sh aiperf:/tmp/sweep.sh
for n in "${names[@]}"; do
  kubectl -n $NS exec aiperf -- env URL=http://$n-frontend.$NS.svc:8000 WORKER_METRICS=http://$n-vllmdecodeworker.$NS.svc:9090/metrics \
    CONCURRENCIES="$CONCURRENCIES" SHAPES="$SHAPES" bash /tmp/sweep.sh "h100-$n" > ~/sweep-$n.log 2>&1 &
done
wait
for n in "${names[@]}"; do
  W=$(kubectl -n $NS get pods -o name | grep "$n-0-vllmdecodeworker" | head -1)
  kubectl -n $NS logs ${W#pod/} > ~/worker-$n.log 2>&1   # keep engine logs (spec-decode acceptance, fp8 notes)
  python3 bench/sweep-table.py "$(ls -d /var/lib/bench-results/*-h100-$n | tail -1)" 1
done
for n in "${names[@]}"; do kubectl -n $NS delete dgd "$n" --wait=false >/dev/null; done
until [ -z "$(kubectl -n $NS get pods --no-headers | grep -E "$(IFS='|'; echo "${names[*]}")")" ]; do sleep 3; done
echo PAIR_DONE

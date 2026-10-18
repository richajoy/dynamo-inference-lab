#!/usr/bin/env bash
# Switch the kvr-qwen3-4b frontend router mode. Runs ON the node (needs kubectl).
#   ./bench/router-mode.sh round-robin | kv | kv-approx  [extra frontend flags...]
#   e.g. ./bench/router-mode.sh kv --router-prefill-load-scale 4   (tuning runs; extra flags are appended verbatim)
# Only the frontend pod restarts; workers (and their warm prefix caches) stay up.
# kv        = router indexes worker KV events (ZMQ) -> knows what each worker actually cached
# kv-approx = --no-router-kv-events (canonical in 1.5.1; --no-kv-events is the obsolete alias, kv_router_args.py:403): router predicts cache contents from its own routing decisions (TTL/LRU)
set -euo pipefail
NS=dynamo-system; DGD=kvr-qwen3-4b
case "${1:-}" in
  round-robin) ARGS='["-m","dynamo.frontend","--router-mode","round-robin"]' ;;
  kv)          ARGS='["-m","dynamo.frontend","--router-mode","kv"]' ;;
  kv-approx)   ARGS='["-m","dynamo.frontend","--router-mode","kv","--no-router-kv-events"]' ;;
  *) sed -n '2,7p' "$0"; exit 1 ;;
esac
shift
for a in "$@"; do ARGS="${ARGS%]},\"$a\"]"; done   # append each extra flag as a JSON string
kubectl -n $NS patch dgd $DGD --type=json \
  -p="[{\"op\":\"replace\",\"path\":\"/spec/components/0/podTemplate/spec/containers/0/args\",\"value\":$ARGS}]"
echo "waiting for exactly one Ready frontend pod running the new args..."
for _ in $(seq 1 60); do
  pods=$(kubectl -n $NS get pods -o name | grep "$DGD.*frontend" || true)
  # exact args match: a substring check would accept the OLD pod (kv vs kv-approx) before the operator replaces it
  if [[ $(wc -w <<<"$pods") == 1 ]] && [[ $(kubectl -n $NS get $pods -o jsonpath='{.spec.containers[0].args}') == "$ARGS" ]] \
     && [[ $(kubectl -n $NS get $pods -o jsonpath='{.status.containerStatuses[0].ready}') == true ]]; then
    echo "ready: $pods"; kubectl -n $NS get $pods -o jsonpath='{.spec.containers[0].args}'; echo
    # Pod Ready and /v1/models 200 do NOT mean a worker is routable (seen once: a load run started at that moment failed 800/800 in 0.15 s
    # right after a frontend restart). Wait for a real completion.
    for _ in $(seq 1 60); do
      kubectl -n $NS exec aiperf -- curl -sf -m 10 "http://$DGD-frontend.$NS.svc:8000/v1/completions" -H 'Content-Type: application/json' \
        -d '{"model":"Qwen/Qwen3-4B","prompt":"ping","max_tokens":1}' >/dev/null 2>&1 && { echo "routable"; exit 0; }
      sleep 2
    done
    echo "frontend Ready but never served a completion"; exit 1
  fi
  sleep 5
done
echo "frontend not ready after 5 min"; kubectl -n $NS get pods | grep "$DGD"; exit 1

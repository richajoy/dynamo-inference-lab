#!/usr/bin/env bash
# Log the prefill/decode split of a DGD every 10 s. Runs ON the node:
#   ./bench/watch-replicas.sh pdp-mig-qwen3-8b > /tmp/replicas.log &
# Columns: time, desired replicas per component (DGD spec, i.e. what the planner wrote), Running+Ready pods per
# component, pending pods (scheduled-not-ready = cold start), and the MIG slices in use.
DGD="${1:?usage: watch-replicas.sh <dgd>}"; NS=dynamo-system
echo "time desired(P,D) ready(P,D) notready(P,D)"
while true; do
  spec=$(kubectl -n $NS get dgd "$DGD" -o json 2>/dev/null | python3 -c '
import json,sys
d=json.load(sys.stdin); r={c["name"]:c.get("replicas") for c in d["spec"]["components"]}
print(r.get("VllmPrefillWorker"), r.get("VllmDecodeWorker"))' 2>/dev/null)
  pods=$(kubectl -n $NS get pods --no-headers 2>/dev/null | grep "$DGD")
  rp=$(grep -c 'prefillworker.* 1/1 *Running' <<<"$pods"); rd=$(grep -c 'decodeworker.* 1/1 *Running' <<<"$pods")
  np=$(grep 'prefillworker' <<<"$pods" | grep -vc ' 1/1 *Running'); nd=$(grep 'decodeworker' <<<"$pods" | grep -vc ' 1/1 *Running')
  echo "$(date +%H:%M:%S) desired=(${spec// /,}) ready=($rp,$rd) notready=($np,$nd)"
  sleep 10
done

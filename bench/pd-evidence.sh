#!/usr/bin/env bash
# Evidence that KV blocks really move prefill -> decode in a P/D DGD. Runs ON the node; meant to be recorded:
#   asciinema rec -q -c "./bench/pd-evidence.sh pd-qwen3-8b" evidence/casts/pd-kv-transfer.cast
# Shows, in order:
#   1. pods and which GPU / MIG device each worker holds (nvidia-smi inside each pod)
#   2. NIXL / UCX lines from both worker logs (agent created, backend, handshake, transfers)
#   3. counters BEFORE and AFTER 8 long-prompt requests:
#        prefill worker: prompt tokens computed (vllm:prompt_tokens_total) grows by ~8 x ISL
#        decode worker : generation tokens grow by ~8 x OSL while its OWN prompt-token compute stays ~0
#        decode worker : NIXL/KV-connector transfer counters (any metric matching nixl|kv_transfer|kv_connector)
#   If decode computed the prompts itself, P/D silently degraded to "decode does everything".
set -uo pipefail
DGD="${1:?usage: pd-evidence.sh <dgd>}"; NS=dynamo-system
P=$(kubectl -n $NS get pods -o name | grep "$DGD.*prefillworker" | head -1); D=$(kubectl -n $NS get pods -o name | grep "$DGD.*decodeworker" | head -1)
h() { printf '\n\033[1;36m### %s\033[0m\n' "$*"; sleep 1; }
h "1. Pods and GPUs"
kubectl -n $NS get pods -o wide | grep "$DGD"
for w in $P $D; do echo "-- $w"; kubectl -n $NS exec ${w#pod/} -- nvidia-smi -L; done
h "2. NIXL / UCX in worker logs"
for w in $P $D; do echo "-- $w"; kubectl -n $NS logs ${w#pod/} | grep -i -E "nixl|ucx|kv_transfer|kv connector|handshake" | grep -v -i "warn.*deprecat" | head -8; done
m() { kubectl -n $NS exec aiperf -- curl -s "http://$1.$NS.svc:9090/metrics"; }
svcP=$(kubectl -n $NS get svc -o name | grep "$DGD-vllmprefillworker" | head -1); svcD=$(kubectl -n $NS get svc -o name | grep "$DGD-vllmdecodeworker" | head -1)
snap() {
  echo "prefill: $(m ${svcP#service/} | grep -E '^vllm:(prompt_tokens_total|generation_tokens_total|request_success_total)' | awk '{printf "%s=%s  ", $1, $2}')"
  echo "decode : $(m ${svcD#service/} | grep -E '^vllm:(prompt_tokens_total|generation_tokens_total|request_success_total)' | awk '{printf "%s=%s  ", $1, $2}')"
  echo "decode KV-transfer metrics:"; m ${svcD#service/} | grep -i -E 'nixl|kv_transfer|kv_connector' | grep -v '^#' | head -12
}
h "3a. Counters BEFORE"
snap
h "3b. 8 requests, ISL ~6000 tokens, OSL 64"
PROMPT=$(python3 -c "print(' '.join(['The quick brown fox jumps over the lazy dog.']*600))")
for i in $(seq 1 8); do
  kubectl -n $NS exec aiperf -- curl -s "http://$DGD-frontend.$NS.svc:8000/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"Qwen/Qwen3-8B\",\"prompt\":\"$i $PROMPT\",\"max_tokens\":64,\"min_tokens\":64,\"ignore_eos\":true}" \
    | python3 -c "import json,sys; u=json.load(sys.stdin)['usage']; print('req', $i, 'prompt_tokens', u['prompt_tokens'], 'completion_tokens', u['completion_tokens'])"
done
sleep 3
h "3c. Counters AFTER"
snap
h "4. Decode-side log lines during the requests"
kubectl -n $NS logs ${D#pod/} --since=60s | grep -i -E "nixl|transfer|remote|kv" | tail -8
echo; echo "EVIDENCE_DONE"

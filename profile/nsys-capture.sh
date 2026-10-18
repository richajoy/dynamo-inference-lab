#!/usr/bin/env bash
# Capture short nsys windows from the live profiled worker (deploy/prof-agg-qwen3-8b.yaml). Runs ON the node.
#   ./profile/nsys-capture.sh <label>
# Mechanism (verified in vendor/dynamo v1.5.1 + vLLM 0.28 by the upstream-vllm agent and checked here):
#   1. ARM once:  nsys start --session=vllm -c cudaProfilerApi --capture-range-end=repeat
#                 -> nsys records only between cudaProfilerStart/Stop, and re-arms after each range (one report each).
#   2. Per window: POST worker:9090/engine/control/start_profile -> vLLM CUDA profiler -> cudaProfilerStart,
#                  plus per-step NVTX ranges execute_context_<n>(<tok>)_generation_<n>(<tok>).
#                  drive a short AIPerf burst, then POST .../stop_profile -> cudaProfilerStop -> report written.
#   3. DISARM:    nsys stop --session=vllm
# Windows are short on purpose: --cuda-graph-trace=node records every kernel inside every decode graph replay.
# Reports land on the node in /var/lib/nsys-reports/ (pod path /reports); windows.txt maps report files -> shapes.
#
# Shapes (same ISL/OSL family as bench/sweep.sh, so kernels map onto the baseline numbers):
#   prefill : ISL 3000 / OSL 32,  c=4  -> prefill steps (H100 default budget 8192: a 3,000-token prompt runs in one step) mixed with a few decodes
#   decode  : ISL 256  / OSL 256, c=16 -> steady CUDA-graph decode steps, batch ~16
#   mixed   : ISL 3000 / OSL 128, c=16 -> the interference case (H100 untraced prefill-heavy c16 TPOT 13.76 ms)
set -euo pipefail
LABEL="${1:?usage: nsys-capture.sh <label>}"
NS=dynamo-system
URL=http://agg-qwen3-8b-frontend.dynamo-system.svc:8000
CTRL=http://agg-qwen3-8b-vllmdecodeworker.dynamo-system.svc:9090/engine/control
MODEL=Qwen/Qwen3-8B
SHAPES="${SHAPES:-prefill decode mixed}"
REPORTS=/var/lib/nsys-reports
declare -A ISL=( [prefill]=3000 [decode]=256 [mixed]=3000 )
declare -A OSL=( [prefill]=32   [decode]=256 [mixed]=128 )
declare -A CONC=( [prefill]=4   [decode]=16  [mixed]=16 )
declare -A N=( [prefill]=8      [decode]=16  [mixed]=24 )

W=$(kubectl -n $NS get pods -o name | grep 'agg-qwen3-8b.*vllmdecodeworker' | head -1)
[[ -n "$W" ]] || { echo "worker pod not found"; exit 1; }
nsys() { kubectl -n $NS exec "${W#pod/}" -- /opt/nsight-systems/bin/nsys "$@"; }
ctl()  { kubectl -n $NS exec aiperf -- curl -sf -X POST -H 'Content-Type: application/json' -d '{}' "$CTRL/$1"; echo; }

burst() {  # $1=shape $2=requests
  kubectl -n $NS exec aiperf -- aiperf profile --model "$MODEL" --url "$URL" \
    --endpoint-type completions --streaming --concurrency "${CONC[$1]}" --request-count "$2" \
    --warmup-request-count 1 \
    --synthetic-input-tokens-mean "${ISL[$1]}" --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean "${OSL[$1]}" --output-tokens-stddev 0 \
    --extra-inputs "max_tokens:${OSL[$1]}" --extra-inputs "min_tokens:${OSL[$1]}" --extra-inputs ignore_eos:true \
    --artifact-dir "/results/nsys-$LABEL/$1" --ui simple | tail -12
}

nsys sessions list   # expect session "vllm" (from `nsys launch`); fails here if the wrapper is not active
nsys start --session=vllm -c cudaProfilerApi --capture-range-end=repeat --output="/reports/$LABEL" --force-overwrite=true
: > "$REPORTS/$LABEL-windows.txt" 2>/dev/null || sudo touch "$REPORTS/$LABEL-windows.txt"

for s in $SHAPES; do
  echo "== $s: warm-up (not recorded)"; burst "$s" 4 >/dev/null
  echo "== $s: recording ISL=${ISL[$s]} OSL=${OSL[$s]} c=${CONC[$s]} n=${N[$s]}"
  ctl start_profile
  burst "$s" "${N[$s]}"
  ctl stop_profile
  i=$((${i:-0}+1))
  # MEASURED (nsys 2026.3.2): with --capture-range-end=repeat each range becomes <label>.<N>.nsys-rep,
  # N counting from 1, and files are only flushed when the session stops -> map by index, not by mtime.
  echo "$s $REPORTS/$LABEL.$i.nsys-rep" | sudo tee -a "$REPORTS/$LABEL-windows.txt"
  sleep 5
done
nsys stop --session=vllm || true
echo "windows:"; cat "$REPORTS/$LABEL-windows.txt"
echo "next: ./profile/nsys-report.sh <file.nsys-rep>   (one per window)"

#!/usr/bin/env bash
# One-shot bring-up of a GPU node (used for the 2x H100 runs). Runs ON the node from the repo checkout, logs to ~/bringup.log:
#   nohup ./infra/bringup.sh > ~/bringup.log 2>&1 &
# Same stack as the A100 node (k3s 1.34.6, GPU Operator v26.7.1 mixed MIG, kube-prometheus-stack, Dynamo 1.5.1 +
# KAI + Grove, AIPerf 0.12.0 pod) so results are comparable. Slow downloads run in PARALLEL with the k8s install:
#   - vllm-runtime image pulled into k3s containerd as soon as k3s is up
#   - Qwen3-8B + Qwen3-4B pre-downloaded into /var/lib/hf-cache BEFORE any worker starts, which also avoids the
#     cold-start lock race (two workers downloading into one cache -> "Lock acquisition failed" restarts).
# Expects hf-token-secret.yaml in ~ (copied from the A100 cluster; never printed).
set -euo pipefail
cd "$(dirname "$0")/.."
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
t() { printf '\n\033[1;35m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }

t "k3s + runtime + helm + dirs"
./infra/k3s/install.sh preflight
./infra/k3s/install.sh k3s
./infra/k3s/install.sh runtime
./infra/k3s/install.sh helm
./infra/k3s/install.sh dirs
sudo mkdir -p /var/lib/request-trace /var/lib/nsys-reports && sudo chown 1000:1000 /var/lib/request-trace /var/lib/nsys-reports

t "background: pull vllm-runtime image + download models"
( sudo k3s ctr images pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.1 > ~/pull.log 2>&1; echo "PULL_DONE rc=$?" >> ~/pull.log ) &
( python3 -m venv ~/hfenv >/dev/null && ~/hfenv/bin/pip -q install "huggingface_hub[hf_transfer]" \
  && HF_HOME=/var/lib/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 ~/hfenv/bin/hf download Qwen/Qwen3-8B > ~/dl.log 2>&1 \
  && HF_HOME=/var/lib/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 ~/hfenv/bin/hf download Qwen/Qwen3-4B >> ~/dl.log 2>&1; \
  echo "DL_DONE rc=$?" >> ~/dl.log ) &

t "monitoring: kube-prometheus-stack"
./platform/monitoring/install.sh prometheus

t "GPU Operator (dcgm ConfigMap first)"
kubectl create ns gpu-operator --dry-run=client -o yaml | kubectl apply -f -
kubectl -n gpu-operator create configmap dcgm-metrics --from-file=dcgm-metrics.csv=platform/monitoring/dcgm-metrics.csv \
  --dry-run=client -o yaml | kubectl apply -f -
./infra/k3s/install.sh gpu-operator

t "Dynamo platform 1.5.1 + HF secret"
kubectl create ns dynamo-system --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f ~/hf-token-secret.yaml
./platform/dynamo/install.sh platform

t "AIPerf pod"
kubectl apply -f bench/aiperf-pod.yaml

t "wait for background downloads"
wait
tail -n 2 ~/pull.log ~/dl.log
./infra/k3s/install.sh verify
kubectl get node -o jsonpath='{.items[0].status.allocatable}'; echo
t "BRINGUP_DONE"

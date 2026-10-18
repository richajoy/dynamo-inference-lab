#!/usr/bin/env bash
# Dynamo Platform (operator + CRDs + bundled KAI Scheduler + Grove). Run ON the node.
#   ./install.sh platform | secret | verify | all
# Requires: infra/k3s/install.sh and platform/monitoring/install.sh done (PodMonitor CRD must exist).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RELEASE_VERSION="${RELEASE_VERSION:-1.5.1}"
NAMESPACE="${NAMESPACE:-dynamo-system}"
CHART_URL="https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz"
CACHE="${CACHE:-$HOME/.cache/dynamo-charts}"
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

platform() {
  mkdir -p "$CACHE"
  tgz="$CACHE/dynamo-platform-${RELEASE_VERSION}.tgz"
  [[ -f $tgz ]] || { say "Fetching $CHART_URL"; curl -fsSL -o "$tgz" "$CHART_URL"; }
  say "helm upgrade -i dynamo-platform ($RELEASE_VERSION) in $NAMESPACE"
  # Known race (observed on 2 of 2 fresh installs, chart 1.5.1 + bundled KAI v0.17.0): the post-install hook
  # templates/kai.yaml creates KAI Queues before KAI's queue-controller webhook has endpoints
  # ("no endpoints available for service queue-controller"). The hook is post-install,post-upgrade,
  # so: wait for the webhook, then re-run helm to replay the hook.
  if ! helm upgrade -i dynamo-platform "$tgz" -n "$NAMESPACE" --create-namespace \
       -f "$HERE/values.yaml" --wait --timeout 15m; then
    say "Hook failed; waiting for KAI queue-controller endpoints, then replaying hooks"
    wait_queue_webhook
    helm upgrade dynamo-platform "$tgz" -n "$NAMESPACE" -f "$HERE/values.yaml" --wait --timeout 15m
  fi
  kubectl -n "$NAMESPACE" get pods
}

wait_queue_webhook() {
  for _ in $(seq 1 60); do
    ips=$(kubectl -n "$NAMESPACE" get endpointslices -l kubernetes.io/service-name=queue-controller \
          -o jsonpath='{.items[*].endpoints[?(@.conditions.ready==true)].addresses[*]}' 2>/dev/null || true)
    [[ -n $ips ]] && { echo "queue-controller ready at: $ips"; return 0; }
    sleep 5
  done
  echo "queue-controller never became ready"; kubectl -n "$NAMESPACE" get pods; return 1
}

secret() {
  # DGD examples reference envFrom secretRef hf-token-secret (key HF_TOKEN).
  [[ -n "${HF_TOKEN:-}" ]] || { echo "export HF_TOKEN first (Qwen models are public; token avoids rate limits)"; exit 1; }
  kubectl -n "$NAMESPACE" create secret generic hf-token-secret \
    --from-literal=HF_TOKEN="$HF_TOKEN" --dry-run=client -o yaml | kubectl apply -f -
}

verify() {
  say "Dynamo CRDs"
  kubectl get crd | grep -E 'nvidia.com|grove|scheduling.run.ai|kai' | awk '{print $1}'
  say "Pods: operator, KAI scheduler, Grove"
  kubectl get pods -A | grep -Ei 'dynamo|kai|grove' || true
  say "KAI queues (Dynamo creates its own when integration is on)"
  kubectl get queues.scheduling.run.ai 2>/dev/null || echo "no queues yet"
  say "Operator wiring: prometheus endpoint + podMonitors"
  helm -n "$NAMESPACE" get values dynamo-platform | grep -A3 metrics
  kubectl get podmonitor -A 2>/dev/null || true
}

case "${1:-}" in
  platform) platform ;; secret) secret ;; verify) verify ;;
  all) platform; secret; verify ;;
  *) sed -n '2,4p' "$0"; exit 1 ;;
esac

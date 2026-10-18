#!/usr/bin/env bash
# Monitoring layer: kube-prometheus-stack + custom DCGM metrics + Dynamo Grafana dashboards.
# Run ON the node after infra/k3s/install.sh. Steps:
#   ./install.sh prometheus | dcgm | dashboards | verify | all
# Order matters: prometheus (creates ServiceMonitor CRD) -> dcgm (GPU Operator references the CRD).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
KPS_VERSION="${KPS_VERSION:-92.3.0}"                 # pinned: the version every result in this repo used
GPU_OPERATOR_VERSION="${GPU_OPERATOR_VERSION:-v26.7.1}"
DYNAMO_SRC="${DYNAMO_SRC:-$HOME/dynamo}"             # v1.5.1 clone on the node (dashboards live here)
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

prometheus() {
  say "kube-prometheus-stack as release 'prometheus' in ns 'monitoring'"
  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
  helm repo update prometheus-community >/dev/null
  helm upgrade -i prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace \
    ${KPS_VERSION:+--version "$KPS_VERSION"} -f "$HERE/kube-prometheus-stack.values.yaml" --wait --timeout 10m
  helm -n monitoring list
  kubectl -n monitoring get pods
}

dcgm() {
  say "ConfigMap dcgm-metrics (custom field list) in gpu-operator"
  kubectl -n gpu-operator create configmap dcgm-metrics \
    --from-file=dcgm-metrics.csv="$HERE/dcgm-metrics.csv" --dry-run=client -o yaml | kubectl apply -f -
  say "Upgrade GPU Operator: point dcgm-exporter at it + enable ServiceMonitor"
  helm upgrade gpu-operator nvidia/gpu-operator -n gpu-operator --version "$GPU_OPERATOR_VERSION" \
    -f "$ROOT/platform/gpu-operator/values.yaml" --wait --timeout 10m
  kubectl -n gpu-operator rollout status ds/nvidia-dcgm-exporter --timeout=180s
}

dashboards() {
  say "Dynamo dashboards from $DYNAMO_SRC/dev/observability/grafana_dashboards -> ConfigMaps (grafana sidecar loads them)"
  for f in dynamo.json disagg-dashboard.json kvbm.json dcgm-metrics.json dynamo-operator.json; do
    src="$DYNAMO_SRC/dev/observability/grafana_dashboards/$f"
    [[ -f $src ]] || { echo "missing $src"; continue; }
    name="dash-$(basename "$f" .json)"
    kubectl -n monitoring create configmap "$name" --from-file="$f=$src" --dry-run=client -o yaml \
      | kubectl label --local -f - grafana_dashboard=1 -o yaml | kubectl apply -f -
  done
}

verify() {
  say "Prometheus targets: dcgm-exporter must be up"
  kubectl -n monitoring port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 >/dev/null 2>&1 &
  pf=$!; sleep 3
  curl -s localhost:9090/api/v1/targets | jq -r '.data.activeTargets[] | "\(.health)\t\(.labels.job)"' | sort | uniq -c
  say "Profiling metrics present in Prometheus (values ~0 while GPU idle is expected)"
  for m in DCGM_FI_PROF_SM_ACTIVE DCGM_FI_PROF_SM_OCCUPANCY DCGM_FI_PROF_PIPE_TENSOR_ACTIVE DCGM_FI_PROF_DRAM_ACTIVE DCGM_FI_DEV_GPU_UTIL; do
    printf '%-34s ' "$m"; curl -s "localhost:9090/api/v1/query?query=$m" | jq -r '.data.result[0].value[1] // "MISSING"'
  done
  kill $pf
  say "Grafana: tunnel port 3000 from your laptop (ssh -L 3000:localhost:3000 <node>) then open http://localhost:3000 (admin / dynamo-lab)"
  echo "this script's tunnel helper: kubectl -n monitoring port-forward svc/prometheus-grafana 3000:80"
}

case "${1:-}" in
  prometheus) prometheus ;; dcgm) dcgm ;; dashboards) dashboards ;; verify) verify ;;
  all) prometheus; dcgm; dashboards; verify ;;
  *) sed -n '2,5p' "$0"; exit 1 ;;
esac

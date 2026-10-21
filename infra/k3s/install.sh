#!/usr/bin/env bash
# k3s + NVIDIA GPU Operator bring-up for a single GPU node.
# Run ON the node, one step at a time, reading the output between steps:
#   ./install.sh preflight | k3s | runtime | helm | gpu-operator | dirs | verify | all
#
# Lessons carried over from an earlier k3s GPU node setup (they did not all apply here, see docs/01-stack-and-deployment.md):
#   - Podman's CNI conflist breaks k3s pod networking -> move it away.
#   - Never pre-create containerd config.toml.tmpl before k3s first start.
#   - nvidia-ctk may write SystemdCgroup=true; k3s containerd uses cgroupfs -> must be false.
set -euo pipefail

K3S_VERSION="${K3S_VERSION:-v1.34.6+k3s1}"          # known-good; >=1.34 is where DRA is GA
GPU_OPERATOR_VERSION="${GPU_OPERATOR_VERSION:-v26.7.1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
VALUES="${VALUES:-$HERE/../../platform/gpu-operator/values.yaml}"
CTD_DIR=/var/lib/rancher/k3s/agent/etc/containerd
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

preflight() {
  say "GPU must be idle (k3s pods will need it). Running GPU processes:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  say "Docker containers still running (stop dynamo-agg / etcd / nats before continuing):"
  sudo docker ps --format '{{.Names}}\t{{.Status}}' || true
  say "MIG mode (want Disabled unless you are testing MIG):"
  nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader
  say "Conflicting CNI configs:"
  ls -l /etc/cni/net.d/ 2>/dev/null || echo "none"
  if [[ -f /etc/cni/net.d/87-podman-bridge.conflist ]]; then
    sudo mv /etc/cni/net.d/87-podman-bridge.conflist /tmp/ && echo "moved podman conflist to /tmp/"
  fi
  say "Host NVIDIA toolkit (k3s auto-detects nvidia-container-runtime on PATH at start):"
  command -v nvidia-container-runtime && nvidia-ctk --version | head -1
}

k3s() {
  say "Installing k3s $K3S_VERSION (traefik + servicelb disabled; kubeconfig readable by non-root users)"
  curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="$K3S_VERSION" sudo -E sh -s - server \
    --write-kubeconfig-mode=644 --disable=traefik --disable=servicelb
  for _ in $(seq 1 30); do kubectl get nodes 2>/dev/null | grep -q ' Ready' && break; sleep 2; done
  # helm (unlike k3s's kubectl symlink) does not find k3s.yaml on its own
  grep -q 'KUBECONFIG=/etc/rancher/k3s/k3s.yaml' ~/.bashrc || echo 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml' >> ~/.bashrc
  kubectl get nodes -o wide
}

runtime() {
  say "Did k3s register the nvidia runtime in its own containerd?"
  if sudo grep -q 'nvidia' "$CTD_DIR/config.toml"; then
    sudo grep -n -A3 'runtimes.*nvidia' "$CTD_DIR/config.toml" | head -20
    echo "-> auto-detected; no manual patch needed"
  else
    echo "-> not detected; applying manual patch (copy generated config -> tmpl, nvidia-ctk, cgroupfs fix)"
    sudo cp "$CTD_DIR/config.toml" "$CTD_DIR/config.toml.tmpl"
    sudo nvidia-ctk runtime configure --runtime=containerd --config="$CTD_DIR/config.toml.tmpl"
    sudo sed -i '/nvidia-container-runtime/,+1 s/SystemdCgroup = true/SystemdCgroup = false/' "$CTD_DIR/config.toml.tmpl"
    sudo systemctl restart k3s && sleep 10
  fi
  say "RuntimeClass objects (pods select the nvidia runtime by name):"
  kubectl get runtimeclass 2>/dev/null || true
  kubectl get runtimeclass nvidia >/dev/null 2>&1 || kubectl apply -f - <<'EOF'
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata: { name: nvidia }
handler: nvidia
EOF
}

helm_install() {
  command -v helm >/dev/null && { helm version --short; return; }
  say "Installing helm 3"
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
  helm version --short
}

gpu_operator() {
  say "Installing GPU Operator $GPU_OPERATOR_VERSION with $VALUES"
  helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null 2>&1 || true
  helm repo update nvidia >/dev/null
  helm upgrade -i gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace \
    --version "$GPU_OPERATOR_VERSION" -f "$VALUES" --wait --timeout 10m
  kubectl -n gpu-operator get pods -o wide
}

dirs() {
  # Node-local caches mounted via hostPath. Owned by uid 1000 = `dynamo` user in vllm-runtime images
  # (verified: `id` -> uid=1000(dynamo) gid=0(root)). Kubernetes fsGroup is not applied to hostPath.
  say "Node cache dirs for hostPath mounts (owner 1000:1000)"
  for d in /var/lib/hf-cache /var/lib/bench-results /var/lib/profiles; do
    sudo mkdir -p "$d"; sudo chown 1000:1000 "$d"; sudo chmod 775 "$d"
  done
  ls -ldn /var/lib/hf-cache /var/lib/bench-results /var/lib/profiles
}

verify() {
  say "Node GPU capacity + GPU Feature Discovery labels"
  kubectl get node -o json | jq '.items[0].status.allocatable | with_entries(select(.key|test("nvidia")))'
  kubectl get node -o json | jq -r '.items[0].metadata.labels | to_entries[] | select(.key|test("nvidia.com/(gpu.product|gpu.memory|mig|cuda.driver)")) | "\(.key)=\(.value)"'
  say "Smoke pod: request 1 GPU, run nvidia-smi -L"
  # Don't use `run -i`: nvidia-smi exits before kubectl attaches and the output is lost.
  kubectl delete pod gpu-smoke --ignore-not-found >/dev/null
  kubectl run gpu-smoke --restart=Never --image=nvidia/cuda:12.9.1-base-ubuntu24.04 \
    --overrides='{"spec":{"runtimeClassName":"nvidia","containers":[{"name":"gpu-smoke","image":"nvidia/cuda:12.9.1-base-ubuntu24.04","command":["nvidia-smi","-L"],"resources":{"limits":{"nvidia.com/gpu":"1"}}}]}}'
  kubectl wait pod/gpu-smoke --for=jsonpath='{.status.phase}'=Succeeded --timeout=120s \
    || kubectl describe pod gpu-smoke | tail -20
  kubectl logs gpu-smoke
  kubectl delete pod gpu-smoke >/dev/null
  say "GPU Operator pods"
  kubectl -n gpu-operator get pods
}

case "${1:-}" in
  preflight) preflight ;; k3s) k3s ;; runtime) runtime ;; helm) helm_install ;;
  gpu-operator) gpu_operator ;; dirs) dirs ;; verify) verify ;;
  all) preflight; k3s; runtime; helm_install; gpu_operator; dirs; verify ;;
  *) sed -n '2,5p' "$0"; exit 1 ;;
esac

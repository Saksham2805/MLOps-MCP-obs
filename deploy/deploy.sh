#!/usr/bin/env bash
# ==============================================================================
# deploy.sh - LLM Observability Stack Deployment Script
# Namespace: llm-obs
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="llm-obs"
DEPLOY_TIMEOUT="300s"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_step()  { echo -e "\n${CYAN}==>${NC} $*"; }

# ------------------------------------------------------------------------------
# 1. Pre-flight checks
# ------------------------------------------------------------------------------
log_step "Running pre-flight checks..."

if ! command -v kubectl &>/dev/null; then
  log_error "kubectl is not installed or not in PATH."
  log_error "Install kubectl: https://kubernetes.io/docs/tasks/tools/"
  exit 1
fi
log_info "kubectl found: $(kubectl version --client 2>/dev/null | head -1)"

if ! kubectl cluster-info &>/dev/null; then
  log_error "Cannot reach the Kubernetes cluster."
  log_error "Check your KUBECONFIG / context: kubectl config current-context"
  exit 1
fi
log_info "Cluster reachable: $(kubectl config current-context)"

if [[ ! -f "${SCRIPT_DIR}/secrets.yaml" ]]; then
  log_error "secrets.yaml not found in ${SCRIPT_DIR}/"
  log_error "Create it from the example:"
  log_error "  cp ${SCRIPT_DIR}/secrets.yaml.example ${SCRIPT_DIR}/secrets.yaml"
  log_error "  # Edit ${SCRIPT_DIR}/secrets.yaml with real base64-encoded credentials"
  exit 1
fi
log_info "secrets.yaml found."

# ------------------------------------------------------------------------------
# 2. Namespace
# ------------------------------------------------------------------------------
log_step "Applying namespace..."
kubectl apply -f "${SCRIPT_DIR}/namespace.yaml"
log_info "Namespace '${NAMESPACE}' ready."

# ------------------------------------------------------------------------------
# 3. Secrets and ConfigMaps
# ------------------------------------------------------------------------------
log_step "Applying Secrets..."
kubectl apply -f "${SCRIPT_DIR}/secrets.yaml"
log_info "Secrets applied."

log_step "Applying ConfigMaps..."
kubectl apply -f "${SCRIPT_DIR}/configmap.yaml"
log_info "ConfigMaps applied."

# ------------------------------------------------------------------------------
# 4. Storage layer - TimescaleDB
# ------------------------------------------------------------------------------
log_step "Deploying TimescaleDB StatefulSet..."
kubectl apply -f "${SCRIPT_DIR}/timescaledb-statefulset.yaml"

log_info "Waiting for TimescaleDB to be ready (timeout: ${DEPLOY_TIMEOUT})..."
kubectl rollout status statefulset/timescaledb \
  -n "${NAMESPACE}" \
  --timeout="${DEPLOY_TIMEOUT}" || {
  log_error "TimescaleDB StatefulSet did not become ready in time."
  log_warn "Check logs: kubectl logs -n ${NAMESPACE} statefulset/timescaledb"
  exit 1
}
log_info "TimescaleDB is ready."

# ------------------------------------------------------------------------------
# 5. GPU metrics
# ------------------------------------------------------------------------------
log_step "Deploying DCGM Exporter DaemonSet..."
kubectl apply -f "${SCRIPT_DIR}/dcgm-exporter-daemonset.yaml"
log_info "DCGM Exporter DaemonSet applied (pods schedule only on nvidia.com/gpu=present nodes)."

# ------------------------------------------------------------------------------
# 6. Application workloads
# ------------------------------------------------------------------------------
log_step "Deploying Collector..."
kubectl apply -f "${SCRIPT_DIR}/collector-deployment.yaml"

log_step "Deploying MCP Server..."
kubectl apply -f "${SCRIPT_DIR}/mcp-server-deployment.yaml"

log_info "Waiting for Collector deployment to be ready..."
kubectl rollout status deployment/llm-obs-collector \
  -n "${NAMESPACE}" \
  --timeout="${DEPLOY_TIMEOUT}" || {
  log_error "Collector deployment did not become ready in time."
  log_warn "Check logs: kubectl logs -n ${NAMESPACE} deployment/llm-obs-collector"
  exit 1
}
log_info "Collector is ready."

log_info "Waiting for MCP Server deployment to be ready..."
kubectl rollout status deployment/llm-obs-mcp-server \
  -n "${NAMESPACE}" \
  --timeout="${DEPLOY_TIMEOUT}" || {
  log_error "MCP Server deployment did not become ready in time."
  log_warn "Check logs: kubectl logs -n ${NAMESPACE} deployment/llm-obs-mcp-server"
  exit 1
}
log_info "MCP Server is ready."

# ------------------------------------------------------------------------------
# 7. Visualization - Grafana
# ------------------------------------------------------------------------------
log_step "Deploying Grafana..."
kubectl apply -f "${SCRIPT_DIR}/grafana-deployment.yaml"

log_info "Waiting for Grafana deployment to be ready..."
kubectl rollout status deployment/grafana \
  -n "${NAMESPACE}" \
  --timeout="${DEPLOY_TIMEOUT}" || {
  log_error "Grafana deployment did not become ready in time."
  log_warn "Check logs: kubectl logs -n ${NAMESPACE} deployment/grafana"
  exit 1
}
log_info "Grafana is ready."

# ------------------------------------------------------------------------------
# 8. Print summary
# ------------------------------------------------------------------------------
MCP_CLUSTER_IP=$(kubectl get svc llm-obs-mcp-server-svc \
  -n "${NAMESPACE}" \
  -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "<pending>")

NODE_IP=$(kubectl get nodes \
  -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}' 2>/dev/null || true)
if [[ -z "${NODE_IP}" ]]; then
  NODE_IP=$(kubectl get nodes \
    -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null || echo "<node-ip>")
fi

echo -e "\n${GREEN}================================================================${NC}"
echo -e "${GREEN}  LLM Observability Stack deployed successfully!${NC}"
echo -e "${GREEN}================================================================${NC}"
echo
echo -e "${CYAN}MCP Server endpoint (in-cluster):${NC}"
echo -e "  http://${MCP_CLUSTER_IP}:8080/mcp"
echo -e "  http://llm-obs-mcp-server-svc.${NAMESPACE}.svc.cluster.local:8080/mcp"
echo
echo -e "${CYAN}Port-forward MCP Server for local access:${NC}"
echo -e "  kubectl port-forward -n ${NAMESPACE} svc/llm-obs-mcp-server-svc 8080:8080"
echo -e "  # Then connect MCP client to: http://localhost:8080/mcp"
echo
echo -e "${CYAN}Grafana:${NC}"
echo -e "  NodePort URL : http://${NODE_IP}:30300"
echo -e "  Port-forward : kubectl port-forward -n ${NAMESPACE} svc/grafana-svc 3000:3000"
echo -e "  Credentials  : admin / changeme123"
echo
echo -e "${YELLOW}Import the LLM Observability dashboard:${NC}"
echo -e "  1. Open Grafana at http://${NODE_IP}:30300"
echo -e "  2. Log in with admin / changeme123"
echo -e "  3. Dashboards -> Import -> Upload JSON file"
echo -e "  4. Select: $(cd "${SCRIPT_DIR}/.." && pwd)/dashboards/grafana/llm-observability.json"
echo -e "  5. Choose datasource 'TimescaleDB' and click Import"
echo -e "  OR via Grafana API:"
DASH_JSON="$(cd "${SCRIPT_DIR}/.." && pwd)/dashboards/grafana/llm-observability.json"
echo -e "    curl -s -u admin:changeme123 \\"
echo -e "      -X POST http://${NODE_IP}:30300/api/dashboards/import \\"
echo -e "      -H 'Content-Type: application/json' \\"
echo -e "      -d \"{\\\"dashboard\\\":\$(cat ${DASH_JSON}),\\\"overwrite\\\":true,\\\"folderId\\\":0}\""
echo
echo -e "${CYAN}Useful commands:${NC}"
echo -e "  kubectl get all -n ${NAMESPACE}"
echo -e "  kubectl logs -n ${NAMESPACE} deployment/llm-obs-collector -f"
echo -e "  kubectl logs -n ${NAMESPACE} deployment/llm-obs-mcp-server -f"
echo -e "  kubectl logs -n ${NAMESPACE} statefulset/timescaledb"
echo

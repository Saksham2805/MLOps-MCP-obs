# Deployment Guide

Step-by-step instructions to deploy `llm-obs-mcp` on any Kubernetes cluster that already has vLLM model pods running.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Kubernetes cluster with kubeconfig access | `kubectl config current-context` should return your cluster |
| vLLM pods already running | GPT-OSS 20B, Gemma4, Svara TTS served via vLLM |
| NVIDIA DCGM Exporter | Deployed as DaemonSet on GPU nodes (optional — GPU metrics will be null without it) |
| Docker | For building and pushing the two application images |
| A container registry | Docker Hub, ECR, GCR, or any registry your cluster can pull from |

---

## Overview

There are **6 steps**. Only **4 files need your input** — everything else deploys as-is.

| Step | Action | Your input needed? |
|---|---|---|
| 1 | Clone the repo | No |
| 2 | Edit `collector/config.yaml` | **Yes — vLLM endpoints + GPU node names** |
| 3 | Edit `deploy/secrets.yaml` | **Yes — Postgres credentials** |
| 4 | Build + push Docker images | **Yes — your registry name** |
| 5 | Run `deploy.sh` | No — fully automated |
| 6 | Verify + access services | No |

---

## Step 1 — Clone the repo on the target machine

```bash
git clone https://github.com/Saksham2805/MLOps-MCP-obs.git
cd MLOps-MCP-obs
```

---

## Step 2 — Edit `collector/config.yaml`

Copy the example and fill in your real vLLM service endpoints:

```bash
cp collector/config.yaml.example collector/config.yaml
```

**Lines you must change** (marked with `← change this`):

```yaml
models:
  - name: gpt-oss-20b
    vllm_endpoint: http://<YOUR-GPT-OSS-SERVICE>.<NAMESPACE>.svc.cluster.local:8000/metrics  # ← change
    gpu_node_selector: <YOUR-GPU-NODE-NAME>   # ← change (e.g. gpu-node-1)

  - name: gemma4
    vllm_endpoint: http://<YOUR-GEMMA4-SERVICE>.<NAMESPACE>.svc.cluster.local:8000/metrics   # ← change
    gpu_node_selector: <YOUR-GPU-NODE-NAME>   # ← change

  - name: svara-tts
    vllm_endpoint: http://<YOUR-SVARA-SERVICE>.<NAMESPACE>.svc.cluster.local:8000/metrics    # ← change
    gpu_node_selector: <YOUR-GPU-NODE-NAME>   # ← change

collector:
  k8s_namespace: <NAMESPACE-WHERE-VLLM-PODS-RUN>               # ← change (e.g. inference)
  dcgm_endpoint: http://<DCGM-SVC>.<NS>.svc.cluster.local:9400/metrics  # ← change if DCGM is not in 'monitoring'
```

**How to find your values:**

```bash
# Find your vLLM service names and namespaces
kubectl get svc -A | grep -i 'vllm\|gpt\|gemma\|svara'

# Find your GPU node names
kubectl get nodes -l nvidia.com/gpu=present

# Find your DCGM exporter service
kubectl get svc -A | grep -i dcgm
```

---

## Step 3 — Create `deploy/secrets.yaml`

```bash
cp deploy/secrets.yaml.example deploy/secrets.yaml
```

All values must be **base64-encoded**. Run these commands to generate them:

```bash
# Choose your own username and password
echo -n 'obs_user'         | base64    # → POSTGRES_USER
echo -n 'yourpassword'     | base64    # → POSTGRES_PASSWORD

# Full connection string (timescaledb-svc is created by the StatefulSet in namespace llm-obs)
echo -n 'postgresql://obs_user:yourpassword@timescaledb-svc.llm-obs.svc.cluster.local:5432/llm_obs' | base64
# → TIMESCALEDB_URL

# Alertmanager URL (optional — anomaly alerts still work without it)
echo -n 'http://alertmanager.monitoring.svc.cluster.local:9093' | base64
# → ALERTMANAGER_URL
```

Paste the outputs into `deploy/secrets.yaml`:

```yaml
data:
  POSTGRES_USER:     <paste base64 here>
  POSTGRES_PASSWORD: <paste base64 here>
  TIMESCALEDB_URL:   <paste base64 here>
  ALERTMANAGER_URL:  <paste base64 here>
```

> `deploy/secrets.yaml` is in `.gitignore` and will never be committed to Git.

---

## Step 4 — Build and push Docker images

```bash
# Set your container registry prefix
REGISTRY=docker.io/saksham2805   # ← change to your registry

# Build + push collector (scrapes vLLM, DCGM, K8s metrics every 15s)
docker build -f collector/Dockerfile -t $REGISTRY/llm-obs-collector:latest .
docker push $REGISTRY/llm-obs-collector:latest

# Build + push MCP server (FastMCP + 6 tools + analytics engine)
docker build -f mcp_server/Dockerfile -t $REGISTRY/llm-obs-mcp-server:latest .
docker push $REGISTRY/llm-obs-mcp-server:latest
```

**Update the image names in the K8s manifests:**

```bash
sed -i "s|<YOUR_REGISTRY>/llm-obs-collector:latest|$REGISTRY/llm-obs-collector:latest|g" \
    deploy/collector-deployment.yaml

sed -i "s|<YOUR_REGISTRY>/llm-obs-mcp-server:latest|$REGISTRY/llm-obs-mcp-server:latest|g" \
    deploy/mcp-server-deployment.yaml
```

---

## Step 5 — Run the deploy script

```bash
cd deploy
chmod +x deploy.sh
./deploy.sh
```

**What the script does automatically:**

| # | Action |
|---|---|
| Pre-flight | Checks `kubectl` is in PATH |
| Pre-flight | Checks cluster is reachable (`kubectl cluster-info`) |
| Pre-flight | Checks `deploy/secrets.yaml` exists (exits loudly if not) |
| 1 | Creates `llm-obs` namespace |
| 2 | Applies Secrets + ConfigMaps |
| 3 | Deploys TimescaleDB StatefulSet — **waits until Ready** |
| 4 | Deploys DCGM Exporter DaemonSet |
| 5 | Deploys Collector — **waits until Ready** |
| 6 | Deploys MCP Server — **waits until Ready** |
| 7 | Deploys Grafana — **waits until Ready** |
| 8 | Prints MCP endpoint, Grafana URL, and useful `kubectl` commands |

If any step fails, the script exits immediately with a clear error and the exact `kubectl logs` command to investigate.

---

## Step 6 — Verify and access services

**Check all pods are running:**

```bash
kubectl get all -n llm-obs
```

Expected:
```
NAME                                      READY   STATUS
pod/timescaledb-0                         1/1     Running
pod/llm-obs-collector-xxx                 1/1     Running
pod/llm-obs-mcp-server-xxx                1/1     Running
pod/grafana-xxx                           1/1     Running
```

**Verify the collector is scraping data:**

```bash
kubectl logs -n llm-obs deployment/llm-obs-collector -f
```

You should see lines like:
```
[INFO] scraper: Inserted metrics for gpt-oss-20b: latency_p95=820.3ms tput=2.40rps
[INFO] scraper: Inserted metrics for gemma4: latency_p95=450.1ms tput=3.10rps
[INFO] scraper: Inserted metrics for svara-tts: latency_p95=310.2ms tput=1.80rps
```

**Verify the MCP server registered all 6 tools:**

```bash
kubectl logs -n llm-obs deployment/llm-obs-mcp-server
```

You should see:
```
[INFO] Registered MCP tools (6): ['get_model_health', 'detect_drift', 'get_root_cause', 'compare_models', 'get_alerts', 'predict_saturation']
```

---

## Accessing the services

### MCP Server (for AI agents)

```bash
kubectl port-forward -n llm-obs svc/llm-obs-mcp-server-svc 8080:8080
# MCP endpoint: http://localhost:8080/mcp
```

**Claude Desktop config** (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "llm-obs": {
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

**In-cluster URL** (for other pods):
```
http://llm-obs-mcp-server-svc.llm-obs.svc.cluster.local:8080/mcp
```

### Grafana Dashboard

```bash
kubectl port-forward -n llm-obs svc/grafana-svc 3000:3000
# Open: http://localhost:3000
# Credentials: admin / changeme123
```

**Import the dashboard:**
1. Open Grafana → Dashboards → Import
2. Upload `dashboards/grafana/llm-observability.json`
3. Select datasource `TimescaleDB` → click Import

Or via API:
```bash
curl -s -u admin:changeme123 \
  -X POST http://localhost:3000/api/dashboards/import \
  -H 'Content-Type: application/json' \
  -d "{\"dashboard\":$(cat dashboards/grafana/llm-observability.json),\"overwrite\":true,\"folderId\":0}"
```

---

## Troubleshooting

**TimescaleDB pod stuck in Init:**
```bash
kubectl logs -n llm-obs statefulset/timescaledb -c init-schema
# The init container runs schema.sql — check for SQL syntax errors
```

**Collector CrashLoopBackOff:**
```bash
kubectl logs -n llm-obs deployment/llm-obs-collector
# Common causes:
# - Wrong vllm_endpoint URL (connection refused)
# - TIMESCALEDB_URL secret wrong (auth failed)
# - vLLM pods not yet ready to serve /metrics
```

**MCP server not responding:**
```bash
kubectl logs -n llm-obs deployment/llm-obs-mcp-server
# Check asyncpg pool creation succeeded at startup
# Check TIMESCALEDB_URL is correct in the secret
```

**No data in Grafana:**
```bash
# Verify the collector is inserting rows
kubectl exec -n llm-obs statefulset/timescaledb -- \
  psql -U obs_user -d llm_obs -c \
  "SELECT model_name, COUNT(*), MAX(time) FROM inference_metrics GROUP BY model_name;"
```

**DCGM metrics all null:**
```bash
# Verify DCGM exporter is running on GPU nodes
kubectl get pods -A | grep dcgm
# Check dcgm_endpoint URL in collector/config.yaml
```

---

## Useful commands post-deployment

```bash
# Watch all pods in llm-obs namespace
kubectl get pods -n llm-obs -w

# Tail collector logs
kubectl logs -n llm-obs deployment/llm-obs-collector -f

# Tail MCP server logs
kubectl logs -n llm-obs deployment/llm-obs-mcp-server -f

# Query TimescaleDB directly
kubectl exec -n llm-obs statefulset/timescaledb -- \
  psql -U obs_user -d llm_obs -c \
  "SELECT model_name, latency_p95_ms, kv_cache_usage_pct, requests_waiting \
   FROM inference_metrics ORDER BY time DESC LIMIT 10;"

# Port-forward MCP server
kubectl port-forward -n llm-obs svc/llm-obs-mcp-server-svc 8080:8080

# Port-forward Grafana
kubectl port-forward -n llm-obs svc/grafana-svc 3000:3000

# Restart collector after config change
kubectl rollout restart deployment/llm-obs-collector -n llm-obs

# Restart MCP server
kubectl rollout restart deployment/llm-obs-mcp-server -n llm-obs
```

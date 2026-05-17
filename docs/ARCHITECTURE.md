# Architecture — llm-obs-mcp

This document describes the full technical architecture of `llm-obs-mcp`: how the four layers interact, how data flows from vLLM pods to MCP tool responses, and how each component is deployed on Kubernetes.

---

## 1. System Overview

The system has four layers with a strictly downward data dependency:

```
AI Agent / LLM Client
        |
        | MCP protocol (JSON-RPC over HTTP or stdio)
        v
+---------------------------------------+
|  LAYER 4: MCP SERVER (FastMCP)        |
|  server.py + tools/                   |
|  Translates tool calls into           |
|  analytics engine queries             |
+---------------------------------------+
        |
        | Python function calls
        v
+---------------------------------------+
|  LAYER 3: ML ANALYTICS ENGINE         |
|  drift.py | anomaly.py               |
|  forecaster.py | root_cause.py        |
|  Reads from TimescaleDB               |
|  Runs models in-process               |
+---------------------------------------+
        |
        | SQL (psycopg2 / asyncpg)
        v
+---------------------------------------+
|  TimescaleDB                          |
|  Hypertables: inference_metrics,      |
|  anomaly_events, drift_scores,        |
|  forecast_snapshots                   |
+---------------------------------------+
        ^
        | INSERT every 15s
        |
+---------------------------------------+
|  LAYER 2: METRICS COLLECTOR           |
|  scraper.py                           |
|  Polls vLLM /metrics (Prometheus)     |
|  Polls DCGM Exporter                  |
|  Polls K8s metrics-server             |
+---------------------------------------+
        ^
        | HTTP GET /metrics
        |
+---------------------------------------+
|  LAYER 1: K8S CLUSTER                 |
|  vLLM pods: GPT-OSS 20B, Gemma4,     |
|  Svara TTS                            |
|  NVIDIA DCGM Exporter (DaemonSet)     |
|  K8s metrics-server                   |
+---------------------------------------+
```

---

## 2. Layer 1 — K8s Cluster

### 2.1 vLLM Model Pods

Each model runs as a Kubernetes Deployment with vLLM as the inference engine.

| Model | Type | Typical Resource Profile |
|---|---|---|
| GPT-OSS 20B | Large LLM | 2-4 x A100/H100 GPUs, high KV-cache usage |
| Gemma 4 | Mid-size LLM | 1-2 x A100 GPUs, moderate KV-cache |
| Svara TTS | Text-to-Speech | 1 x GPU, latency measured in Real-Time Factor |

vLLM exposes a Prometheus-format metrics endpoint at `http://<pod-ip>:8000/metrics` by default. No configuration changes required.

### 2.2 NVIDIA DCGM Exporter

Deployed as a DaemonSet so it runs on every GPU node. Exposes per-GPU metrics:
- `DCGM_FI_DEV_GPU_UTIL` - GPU compute utilization %
- `DCGM_FI_DEV_MEM_COPY_UTIL` - memory bandwidth utilization %
- `DCGM_FI_DEV_FB_USED` - framebuffer memory used (MB)
- `DCGM_FI_DEV_FB_FREE` - framebuffer memory free (MB)
- `DCGM_FI_DEV_GPU_TEMP` - GPU temperature (Celsius)
- `DCGM_FI_DEV_POWER_USAGE` - power draw (Watts)

### 2.3 K8s metrics-server

Standard K8s component. Provides pod-level CPU and memory usage via the metrics API. Used to correlate CPU-side bottlenecks (tokenizer thread saturation) with GPU-side metrics.

---

## 3. Layer 2 — Metrics Collector

### 3.1 scraper.py

A long-running Python process (or K8s CronJob for simpler deployments). Every 15 seconds it:

1. Reads `config.yaml` for the list of model endpoints
2. HTTP GETs each vLLM `/metrics` endpoint
3. Parses the Prometheus text format using `prometheus_client.parser`
4. HTTP GETs the DCGM Exporter endpoint and maps GPU metrics to pod labels
5. Calls the K8s metrics API for pod CPU/memory
6. Normalizes all metrics into a flat row per `(model_name, timestamp)`
7. Bulk-INSERTs into TimescaleDB `inference_metrics` hypertable

### 3.2 config.yaml structure

```yaml
models:
  - name: gpt-oss-20b
    vllm_endpoint: http://gpt-oss-svc.inference.svc.cluster.local:8000/metrics
    model_type: llm
    gpu_node_selector: gpu-node-1

  - name: gemma4
    vllm_endpoint: http://gemma4-svc.inference.svc.cluster.local:8000/metrics
    model_type: llm
    gpu_node_selector: gpu-node-2

  - name: svara-tts
    vllm_endpoint: http://svara-svc.inference.svc.cluster.local:8000/metrics
    model_type: tts
    gpu_node_selector: gpu-node-3

collector:
  scrape_interval_seconds: 15
  timescaledb_url: postgresql://obs_user:obs_pass@timescaledb:5432/llm_obs
  dcgm_endpoint: http://dcgm-exporter.monitoring.svc.cluster.local:9400/metrics
```

### 3.3 TimescaleDB Schema

See `collector/schema.sql` for the full DDL. The primary hypertable:

```sql
CREATE TABLE inference_metrics (
    time                    TIMESTAMPTZ NOT NULL,
    model_name              TEXT NOT NULL,
    latency_p50_ms          FLOAT,
    latency_p95_ms          FLOAT,
    latency_p99_ms          FLOAT,
    ttft_p50_ms             FLOAT,   -- time to first token
    ttft_p95_ms             FLOAT,
    tpot_ms                 FLOAT,   -- time per output token
    throughput_rps          FLOAT,
    prompt_tokens_total     BIGINT,
    completion_tokens_total BIGINT,
    requests_running        INT,
    requests_waiting        INT,
    kv_cache_usage_pct      FLOAT,
    gpu_util_pct            FLOAT,
    gpu_mem_used_mb         FLOAT,
    gpu_mem_free_mb         FLOAT,
    gpu_temp_celsius        FLOAT,
    gpu_power_watts         FLOAT,
    pod_cpu_millicores      INT,
    pod_mem_mb              FLOAT,
    rtf                     FLOAT    -- TTS only: Real-Time Factor
);
SELECT create_hypertable('inference_metrics', 'time');
```

Additional hypertables: `anomaly_events`, `drift_scores`, `forecast_snapshots` - see `collector/schema.sql`.

---

## 4. Layer 3 — ML Analytics Engine

See [`ML_DESIGN.md`](ML_DESIGN.md) for mathematical detail. Summary of runtime behavior:

| Module | Runs | Reads from | Writes to |
|---|---|---|---|
| `drift.py` | Every 5 min | `inference_metrics` | `drift_scores` |
| `anomaly.py` | Every 1 min | `inference_metrics` | `anomaly_events` |
| `forecaster.py` | Every 30 min | `inference_metrics` | `forecast_snapshots` |
| `root_cause.py` | On-demand (MCP call) | `anomaly_events` + `inference_metrics` | (returns result) |

The anomaly model is retrained nightly on the rolling 7-day window.

---

## 5. Layer 4 — MCP Server

### 5.1 FastMCP setup

`mcp_server/server.py` initializes a FastMCP instance and registers all 6 tools:

```python
from mcp.server.fastmcp import FastMCP
from tools.health import get_model_health
from tools.drift import detect_drift
from tools.root_cause import get_root_cause
from tools.compare import compare_models
from tools.alerts import get_alerts
from tools.forecast import predict_saturation

mcp = FastMCP("llm-obs")
mcp.tool()(get_model_health)
mcp.tool()(detect_drift)
mcp.tool()(get_root_cause)
mcp.tool()(compare_models)
mcp.tool()(get_alerts)
mcp.tool()(predict_saturation)

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
```

### 5.2 Transport

The server supports both:
- `streamable-http` for K8s deployment (agents connect over HTTP)
- `stdio` for local development and Claude Desktop integration

---

## 6. Kubernetes Deployment

### Component Map

```
Namespace: llm-obs

Deployments:
  collector          1 replica, no GPU
  mcp-server         1 replica (scale to 2+ for HA), no GPU

StatefulSet:
  timescaledb        1 replica, PVC 50Gi

DaemonSet (Namespace: monitoring):
  dcgm-exporter      1 pod per GPU node (pre-existing)

Services:
  timescaledb-svc    ClusterIP :5432
  mcp-server-svc     ClusterIP :8080 (expose via Ingress or LoadBalancer)
```

### Environment Variables (mcp-server and collector)

| Variable | Description |
|---|---|
| `TIMESCALEDB_URL` | Full PostgreSQL connection string |
| `DCGM_ENDPOINT` | DCGM exporter metrics URL |
| `K8S_API_SERVER` | K8s API server (in-cluster: auto-detected) |
| `SCRAPE_INTERVAL` | Seconds between scrapes (default: 15) |
| `ANOMALY_RETRAIN_CRON` | Cron expression for nightly model retrain |

---

## 7. Grafana Dashboard

Pre-built dashboard at `dashboards/grafana/llm-observability.json`. Import via Grafana UI (Dashboard -> Import -> Upload JSON).

Panels included:
- Per-model latency P50/P95/P99 time series
- GPU utilization heatmap (all models, 24h)
- KV cache usage per model
- Request queue depth with saturation forecast overlay
- Drift score trend (PSI per model)
- Anomaly event timeline
- Real-Time Factor for Svara TTS

---

## 8. Data Flow — Single MCP Tool Call

Example: Agent calls `get_root_cause(model_name="gpt-oss-20b")`

```
1. Agent sends MCP tool call to mcp-server:8080
2. FastMCP routes to tools/root_cause.py
3. root_cause.py queries TimescaleDB:
   a. Fetches latest anomaly_event for gpt-oss-20b
   b. Fetches inference_metrics for last 30 min
4. Passes (anomaly_dimensions, metric_snapshot) to root_cause correlator
5. Correlator applies rule tree -> produces diagnosis string + evidence
6. FastMCP serializes result as MCP tool response
7. Agent receives structured JSON with diagnosis, evidence, recommendation
```

Total latency: typically 50-200ms (dominated by TimescaleDB queries).

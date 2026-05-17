# llm-obs-mcp

> **ML-native observability for production LLM inference clusters, exposed as an MCP server.**

A production-grade monitoring and analytics platform for LLMs and TTS models served via [vLLM](https://github.com/vllm-project/vllm) on Kubernetes. It goes beyond infrastructure metrics, applying real ML (drift detection, anomaly detection, predictive forecasting) to inference telemetry, and surfacing everything through a [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server so any AI agent can query cluster health in natural language.

---

## The Problem

Running LLMs in production is not like running a web server. A REST API either works or it does not. An LLM degrades silently, gradually, in ways CPU/RAM dashboards never show:

| What goes wrong | Why standard monitoring misses it |
|---|---|
| Input distribution shifts (users send longer prompts) | CPU/RAM fine, GPU is the bottleneck |
| KV cache saturation on large models | Latency climbs 3x, no error thrown |
| TTS Real-Time Factor degrades after a traffic spike | No SLA metric for audio generation speed |
| Queue depth builds before it becomes a user timeout | P50 looks healthy; P99 is on fire |
| A model fast last week is slow this week | No baseline to compare against |

Prometheus and Grafana tell you what the numbers are. They do not tell you whether those numbers are normal, why they changed, or what to do about it. That gap is what this project fills.

---

## Architecture

```
+--------------------------------------------------------------+
|  LAYER 4 - MCP SERVER                                        |
|  6 tools callable by any AI agent over MCP protocol          |
|  Is GPT-OSS healthy?        -> get_model_health()            |
|  Why is Gemma4 slow?        -> get_root_cause()              |
|  Will Svara TTS saturate?   -> predict_saturation()          |
+--------------------------------------------------------------+
                          ^ queries
+--------------------------------------------------------------+
|  LAYER 3 - ML ANALYTICS ENGINE                               |
|  Drift Detector | Anomaly Detector | Root Cause | Forecaster |
|  KS test, PSI, Isolation Forest, Prophet                     |
+--------------------------------------------------------------+
                          ^ reads from
+--------------------------------------------------------------+
|  LAYER 2 - METRICS COLLECTOR                                 |
|  Scrapes vLLM /metrics, K8s metrics-server, NVIDIA DCGM      |
|  Stores time-series in TimescaleDB (PostgreSQL extension)    |
+--------------------------------------------------------------+
                          ^ scrapes from
+--------------------------------------------------------------+
|  LAYER 1 - K8S CLUSTER                                       |
|  +-------------+  +-------------+  +----------------+       |
|  | GPT-OSS 20B |  |   Gemma 4   |  |   Svara TTS    |       |
|  | (vLLM)      |  |   (vLLM)    |  |   (vLLM)       |       |
|  +-------------+  +-------------+  +----------------+       |
+--------------------------------------------------------------+
```

---

## Key Features

- **Zero-instrumentation scraping** - vLLM exposes Prometheus metrics at `/metrics` out of the box. No model code changes required.
- **Statistical drift detection** - KS test + PSI on input/output distributions. Catches silent behavioral shifts before they become incidents.
- **Unsupervised anomaly detection** - Isolation Forest on a rolling 7-day baseline. Flags multi-dimensional anomalies and identifies which metric dimension drove the score.
- **Predictive saturation forecasting** - Prophet forecasts queue depth and KV-cache utilization up to 2 hours ahead with automated scaling recommendations.
- **Root cause analysis** - Correlates metric patterns into human-readable diagnoses instead of raw numbers.
- **MCP interface** - All analytics exposed as typed MCP tools callable by any AI agent.

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Model Serving | vLLM | Serves GPT-OSS 20B, Gemma4, Svara TTS on K8s |
| Orchestration | Kubernetes | Runs models and all observability components |
| GPU Metrics | NVIDIA DCGM Exporter | GPU util, memory, temperature per pod |
| Scraping | Python + prometheus_client | Polls vLLM /metrics every 15s |
| Storage | TimescaleDB (PostgreSQL) | Hypertable-optimized time-series storage |
| Drift | scipy.stats + custom PSI | Distribution comparison over rolling windows |
| Anomaly | scikit-learn IsolationForest | Multi-variate unsupervised anomaly scoring |
| Forecast | Facebook Prophet | Queue depth + KV-cache utilization prediction |
| MCP Server | Python mcp SDK (FastMCP) | Agent-callable tool interface |
| Dashboards | Grafana | Visual layer reading from TimescaleDB |
| Alerting | Prometheus Alertmanager | Threshold and anomaly-based alert routing |

---

## MCP Tools

| Tool | Question it answers | Backed by |
|---|---|---|
| `get_model_health` | Current health snapshot of one or all models | TimescaleDB query + live anomaly score |
| `detect_drift` | Has this model's input/output distribution shifted? | KS test + PSI |
| `get_root_cause` | Why is this model degraded right now? | Rule-based correlator |
| `compare_models` | Side-by-side performance of two models | Aggregates + Welch t-test |
| `get_alerts` | Active and recent alerts with severity | Alertmanager + anomaly history |
| `predict_saturation` | Will this model hit capacity in N hours? | Prophet per model |

Full input/output schemas with examples: [`docs/MCP_TOOLS.md`](docs/MCP_TOOLS.md)

---

## ML Methods

| Method | Applied To | Why This Method |
|---|---|---|
| Kolmogorov-Smirnov test | Token length drift, latency distribution drift | Non-parametric. Catches shape changes, not just mean shifts. No distribution assumption. |
| Population Stability Index | Prompt length buckets over rolling windows | Industry standard for model monitoring. Single interpretable score with established thresholds. |
| Isolation Forest | (latency_p95, gpu_util, kv_cache_usage, queue_depth) | Unsupervised, no labels needed, handles correlated multi-dimensional signals, fast. |
| Facebook Prophet | Queue depth and KV-cache utilization forecasting | Handles daily/weekly seasonality. Robust to data gaps. Interpretable trend decomposition. |
| Rule-based correlator | Root cause analysis on flagged anomalies | Domain rules encode why something is wrong. Anomaly model finds that something is wrong. |

Full derivations and threshold rationale: [`docs/ML_DESIGN.md`](docs/ML_DESIGN.md)

---

## Project Structure

```
llm-obs-mcp/
|-- README.md
|-- docs/
|   |-- ARCHITECTURE.md        <- Full layer design, data flow, component interactions
|   |-- METRICS_REFERENCE.md   <- Every metric: meaning, source, alert thresholds
|   |-- MCP_TOOLS.md           <- All 6 tool specs with input/output schemas and examples
|   |-- ML_DESIGN.md           <- Drift, anomaly, forecasting: math and design rationale
|   +-- BUILD_PHASES.md        <- 5-phase plan with tasks and acceptance criteria
|-- collector/
|   |-- scraper.py             <- Polls vLLM /metrics + DCGM exporter endpoints
|   |-- schema.sql             <- TimescaleDB hypertable schema
|   +-- config.yaml            <- Scrape targets and model registry
|-- analytics/
|   |-- drift.py               <- KS test + PSI drift detector
|   |-- anomaly.py             <- Isolation Forest detector and trainer
|   |-- forecaster.py          <- Prophet saturation forecaster
|   +-- root_cause.py          <- Rule-based root cause correlator
|-- mcp_server/
|   |-- server.py              <- FastMCP server entry point
|   +-- tools/
|       |-- health.py          <- get_model_health
|       |-- drift.py           <- detect_drift
|       |-- root_cause.py      <- get_root_cause
|       |-- compare.py         <- compare_models
|       |-- alerts.py          <- get_alerts
|       +-- forecast.py        <- predict_saturation
|-- deploy/
|   |-- collector-deployment.yaml
|   |-- mcp-server-deployment.yaml
|   +-- timescaledb-statefulset.yaml
+-- dashboards/
    +-- grafana/
        +-- llm-observability.json
```

---

## Build Phases

| Phase | Name | Deliverable | Duration |
|---|---|---|---|
| 1 | Metrics Foundation | vLLM scraper, TimescaleDB schema, Grafana dashboard | 1 week |
| 2 | ML Analytics Engine | Drift + anomaly detectors, alert integration | 1.5 weeks |
| 3 | MCP Server Core | get_model_health, detect_drift, get_root_cause, compare_models | 1 week |
| 4 | Forecasting + Full MCP | Prophet forecaster, predict_saturation, get_alerts | 1.5 weeks |
| 5 | K8s Deploy + Docs | Manifests, end-to-end demo, documentation | 1 week |

**Total: ~6 weeks** - Full breakdown with per-task details: [`docs/BUILD_PHASES.md`](docs/BUILD_PHASES.md)

---

## Getting Started

```bash
git clone https://github.com/Saksham2805/MLOps-MCP-obs.git
cd MLOps-MCP-obs

# 1. Fill in your vLLM endpoints
cp collector/config.yaml.example collector/config.yaml

# 2. Fill in your credentials
cp deploy/secrets.yaml.example deploy/secrets.yaml

# 3. Build images, update manifest image names, then:
cd deploy && ./deploy.sh
```

**Full step-by-step deployment guide:** [`DEPLOYMENT.md`](DEPLOYMENT.md)

Covers: vLLM endpoint discovery, base64 credential encoding, Docker build + push, `deploy.sh` walkthrough, service access, and troubleshooting.

---

## Resume Bullet

> *Built an MLOps observability platform for a vLLM multi-model K8s cluster (20B LLM, mid-size LLM, TTS model) implementing statistical drift detection (KS+PSI), unsupervised multi-variate anomaly detection (Isolation Forest), and predictive saturation forecasting (Prophet); exposed as an MCP server enabling natural-language model health queries from AI agents.*

---

## Status

**Complete** — All code, manifests, and documentation written. Deploy with `cd deploy && ./deploy.sh`.

---

## License

MIT

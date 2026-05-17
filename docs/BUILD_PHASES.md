# Build Phases

Detailed delivery plan for `llm-obs-mcp`. Each phase has a goal, list of tasks, and acceptance criteria. Phases are sequential — each builds on the previous.

---

## Phase 1: Metrics Foundation

**Goal:** Collect real inference telemetry from all three vLLM model pods and store it in TimescaleDB. Build a Grafana dashboard so you can see the data visually before any ML is applied.

**Duration:** 1 week

### Tasks

| # | Task | Detail |
|---|---|---|
| 1.1 | Deploy TimescaleDB | Run TimescaleDB as a K8s StatefulSet with a 50Gi PVC. Apply `collector/schema.sql` to create all hypertables. |
| 1.2 | Write `collector/schema.sql` | Create hypertables: `inference_metrics`, `anomaly_events`, `drift_scores`, `forecast_snapshots`, `model_artifacts`. Add retention policy: auto-drop data older than 90 days. |
| 1.3 | Write `collector/config.yaml` | Define all three model endpoints, DCGM exporter URL, scrape interval, TimescaleDB connection string. |
| 1.4 | Write `collector/scraper.py` | Poll vLLM `/metrics`, parse Prometheus text format, poll DCGM exporter, join GPU metrics to model by node label, compute derived metrics (throughput_rps, avg_prompt_tokens, RTF for TTS), bulk INSERT to TimescaleDB every 15s. |
| 1.5 | Verify scraper output | Run scraper locally against real cluster. Query TimescaleDB to confirm all fields populated correctly for all three models. |
| 1.6 | Import Grafana dashboard | Create `dashboards/grafana/llm-observability.json` with panels: latency time series, GPU heatmap, KV cache usage, queue depth, RTF for Svara TTS. |
| 1.7 | Write collector K8s manifest | `deploy/collector-deployment.yaml`: 1 replica, env vars from K8s Secret, readiness probe. |

### Acceptance Criteria

- TimescaleDB `inference_metrics` table has rows for all three models updating every 15 seconds
- All fields in the schema have non-null values (or null only for TTS-specific fields on LLM rows)
- Grafana dashboard renders without errors and shows live data for the past 1 hour
- Scraper pod runs stably in K8s for 24 hours without crashing or memory growth

---

## Phase 2: ML Analytics Engine

**Goal:** Implement drift detection, anomaly detection, and alert integration. The system should automatically flag when something unusual is happening with any model.

**Duration:** 1.5 weeks

### Tasks

| # | Task | Detail |
|---|---|---|
| 2.1 | Write `analytics/drift.py` | Implement `run_drift_check(model_name, metric, reference_window, comparison_window)`. Query TimescaleDB for both windows, run KS test and PSI, map to severity, write result to `drift_scores` hypertable. Schedule to run every 5 minutes via APScheduler. |
| 2.2 | Write `analytics/anomaly.py` | Implement `train_model(model_name)` and `score_current_window(model_name)`. Fetch 7-day baseline, train Isolation Forest, score latest 1-minute window, write anomaly_score and contributing_dimensions to `anomaly_events`. Schedule scoring every 60s, retraining nightly at 02:00 UTC. |
| 2.3 | Handle cold start for anomaly | For the first 7 days, the Isolation Forest cannot be trained. Fall back to threshold-based z-score anomaly detection during this period. Log a warning in the anomaly_events table that the model is in cold-start mode. |
| 2.4 | Persist model artifacts | Serialize trained Isolation Forest models with `joblib`. Store as BLOB in `model_artifacts` TimescaleDB table with columns: `model_name`, `trained_at`, `artifact`. Load latest artifact on startup. |
| 2.5 | Write `analytics/root_cause.py` | Implement `analyze(model_name, anomaly_event)`. Apply rule tree (see MCP_TOOLS.md). Return diagnosis, evidence list, recommendation, confidence score. |
| 2.6 | Write Alertmanager rules | Create Prometheus Alertmanager rules for all threshold alerts defined in METRICS_REFERENCE.md. Add a webhook receiver that writes fired alerts to `anomaly_events` table with `type='threshold'`. |
| 2.7 | Add drift alerts | When PSI > 0.2 for any model/metric combination, write a drift alert to `anomaly_events` with `type='drift'` and severity `medium`. When PSI > 0.25, severity `high`. |
| 2.8 | Add PSI panel to Grafana | Add a PSI trend panel to the Grafana dashboard. One line per model. Horizontal reference lines at 0.1 (warning) and 0.25 (critical). |

### Acceptance Criteria

- `drift_scores` table populates every 5 minutes for all three models and all configured metrics
- `anomaly_events` table populates every 60 seconds with anomaly scores
- When a model is artificially overloaded (e.g., by sending a burst of requests), an anomaly event with a negative score appears within 2 minutes
- Alertmanager fires a `critical` alert when `kv_cache_usage_pct` is manually set above 90% in a test
- Root cause correlator returns a diagnosis matching the artificial overload condition
- Isolation Forest cold-start fallback works without crashing during the first 7 days

---

## Phase 3: MCP Server (Core Tools)

**Goal:** Build the MCP server and implement the four core tools: `get_model_health`, `detect_drift`, `get_root_cause`, `compare_models`.

**Duration:** 1 week

### Tasks

| # | Task | Detail |
|---|---|---|
| 3.1 | Set up FastMCP server | Write `mcp_server/server.py`. Initialize FastMCP, register tools, configure transport (streamable-http for K8s, stdio for local). Add database connection pool (asyncpg) shared across all tools. |
| 3.2 | Write `tools/health.py` | Query `inference_metrics` for the requested time window, compute aggregates, fetch latest `anomaly_events` score, fetch latest `drift_scores` severity. Map to `healthy/degraded/critical`. Handle the `model_name=null` case (return all models). |
| 3.3 | Write `tools/drift.py` | Call `analytics/drift.py` `run_drift_check()` directly (or read from `drift_scores` cache). Return full schema including interpretation string and recommendation. |
| 3.4 | Write `tools/root_cause.py` | Fetch latest anomaly event from `anomaly_events`. Fetch 30-minute metric snapshot. Call `analytics/root_cause.py` `analyze()`. Return full schema. Handle case where no anomaly is detected (return `anomaly_detected: false` with explanation). |
| 3.5 | Write `tools/compare.py` | Query `inference_metrics` for both models over the requested window. Compute mean, p50, p95, p99. Run Welch t-test (`scipy.stats.ttest_ind`) on the raw samples. Return comparison schema with significance flag and summary string. |
| 3.6 | Write MCP server K8s manifest | `deploy/mcp-server-deployment.yaml`: 1 replica, env vars from Secret, liveness probe on `/health`, expose port 8080. |
| 3.7 | Test all four tools locally | Use `mcp dev mcp_server/server.py` to open the MCP inspector. Manually call each tool and verify responses match the schemas in MCP_TOOLS.md. |

### Acceptance Criteria

- All four tools return valid JSON matching their output schemas
- `get_model_health` returns different statuses for a healthy vs artificially degraded model
- `detect_drift` correctly returns `drift_detected: true` when comparing a 24h window with intentionally different prompt lengths to a 7d baseline
- `get_root_cause` returns a matching diagnosis when the model is under the artificial overload from Phase 2 testing
- `compare_models` returns `statistically_significant: true` when one model is loaded and the other is idle
- MCP server runs stably in K8s for 24 hours

---

## Phase 4: Forecasting + Full MCP

**Goal:** Add the Prophet forecasting model and the two remaining MCP tools: `predict_saturation` and `get_alerts`.

**Duration:** 1.5 weeks

### Tasks

| # | Task | Detail |
|---|---|---|
| 4.1 | Write `analytics/forecaster.py` | Implement `train_forecast_model(model_name, metric)` and `get_forecast(model_name, metric, horizon_hours)`. Use Prophet with configuration from ML_DESIGN.md. Serialize models with joblib. Store forecasts in `forecast_snapshots` table. Schedule reforecast every 30 minutes. |
| 4.2 | Handle Prophet cold start | Prophet requires at least 2 periods of data for seasonality. During the first 14 days, return a forecast with `confidence: low` and a note that the model is still learning seasonal patterns. Use a simple linear extrapolation as fallback. |
| 4.3 | Write `tools/forecast.py` | Read latest forecast from `forecast_snapshots` table (or trigger a fresh forecast if stale). Apply `find_saturation_time()`. Return full `predict_saturation` schema. |
| 4.4 | Write `tools/alerts.py` | Query `anomaly_events` table filtered by model_name, severity, and resolved status. Enrich with Alertmanager API for currently firing alerts. Return sorted alert list with `total_active` and `total_critical` summary. |
| 4.5 | Register remaining tools in server.py | Add `predict_saturation` and `get_alerts` to the FastMCP server. |
| 4.6 | Add forecast overlay to Grafana | Add a panel to the dashboard showing queue depth history plus the Prophet forecast as a dashed line with confidence interval band. |
| 4.7 | End-to-end agent test | Use a local MCP client (or Claude Desktop configured with this MCP server) to run a natural-language session: ask about health, drift, root cause, and saturation for each model. Verify all responses are coherent and useful. |

### Acceptance Criteria

- `predict_saturation` returns a forecast with at least 4 hourly data points and confidence intervals
- After 14+ days of data, Prophet-based forecast shows a visible daily seasonality pattern in the Grafana overlay
- `get_alerts` returns currently active Alertmanager alerts within 30 seconds of them firing
- A full natural-language agent session covering all 6 tools completes without errors
- All 6 tools registered and visible in MCP server tool listing

---

## Phase 5: K8s Deployment + Documentation

**Goal:** Make the full system deployable from scratch on any compatible K8s cluster. Write final documentation and a demo script.

**Duration:** 1 week

### Tasks

| # | Task | Detail |
|---|---|---|
| 5.1 | Write TimescaleDB StatefulSet | `deploy/timescaledb-statefulset.yaml`: StatefulSet with PVC, liveness + readiness probes, resource limits. Include an init container that runs `schema.sql` on first boot. |
| 5.2 | Write K8s Secrets template | Document which secrets are required and how to create them: `TIMESCALEDB_URL`, `DCGM_ENDPOINT`, `K8S_API_SERVER`. Provide a `secrets.yaml.example`. |
| 5.3 | Write `requirements.txt` | Pin all Python dependencies: `prometheus_client`, `psycopg2-binary`, `asyncpg`, `scipy`, `scikit-learn`, `prophet`, `pandas`, `numpy`, `mcp`, `apscheduler`, `pyyaml`, `joblib`, `kubernetes`. |
| 5.4 | Write Dockerfiles | Two Dockerfiles: one for `collector/` and one for `mcp_server/`. Use `python:3.11-slim` base. Multi-stage build to minimize image size. |
| 5.5 | Write `config.yaml.example` | Full example configuration file with comments on every field. Covers all three model types (LLM large, LLM mid, TTS). |
| 5.6 | End-to-end deployment test | Deploy the full stack to a fresh K8s namespace from scratch using only the manifests and README instructions. Verify all components reach Ready state. |
| 5.7 | Write demo script | A markdown walkthrough of a realistic demo session: starting from a fresh deployment, triggering an artificial overload on GPT-OSS 20B, then using an MCP agent to detect, diagnose, and get a recommendation for the incident. |
| 5.8 | Final README review | Ensure README contains no references to internal systems or proprietary tools. Confirm all links to docs/ files resolve correctly. |

### Acceptance Criteria

- A team member with no prior context can deploy the full stack in under 30 minutes following the README
- All K8s manifests are valid (`kubectl apply --dry-run=client` passes)
- Demo script produces a coherent end-to-end narrative that can be recorded as a screen capture
- `requirements.txt` installs without conflicts on Python 3.11 in a fresh virtual environment
- All 5 doc files in `docs/` are complete and cross-referenced correctly

---

## Milestone Summary

| Milestone | End of Phase | What you can show |
|---|---|---|
| M1 | Phase 1 | Live Grafana dashboard with real vLLM metrics for all 3 models |
| M2 | Phase 2 | Automatic anomaly and drift alerts firing on real cluster events |
| M3 | Phase 3 | AI agent answering health, drift, root cause, and comparison questions via MCP |
| M4 | Phase 4 | AI agent predicting when a model will saturate and listing active alerts |
| M5 | Phase 5 | Full deployable system with demo video and complete documentation |

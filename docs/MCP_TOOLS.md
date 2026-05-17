# MCP Tools Reference

Full specification for all 6 tools exposed by the `llm-obs-mcp` MCP server.
Each entry includes: description, input schema, output schema, and a worked example.

---

## Tool 1: `get_model_health`

**What it answers:** What is the current health status of one or all models?

### Input Schema

```json
{
  "model_name": "string | null",
  "time_window": "string"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `model_name` | string or null | No | Name of the model (e.g. `gpt-oss-20b`). If null, returns health for all models. |
| `time_window` | string | No (default: `15m`) | Aggregation window. Accepts `5m`, `15m`, `1h`, `6h`, `24h`. |

### Output Schema

```json
{
  "models": [
    {
      "model_name": "string",
      "status": "healthy | degraded | critical",
      "latency_p50_ms": "float",
      "latency_p95_ms": "float",
      "latency_p99_ms": "float",
      "ttft_p95_ms": "float",
      "throughput_rps": "float",
      "requests_waiting": "int",
      "kv_cache_usage_pct": "float",
      "gpu_util_pct": "float",
      "gpu_mem_free_mb": "float",
      "anomaly_score": "float",
      "drift_status": "none | low | medium | high",
      "as_of": "ISO8601 timestamp"
    }
  ]
}
```

`status` is derived as:
- `healthy` — anomaly_score > -0.05 AND no critical threshold breached
- `degraded` — anomaly_score between -0.05 and -0.15 OR a warning threshold breached
- `critical` — anomaly_score < -0.15 OR a critical threshold breached

### Example

Agent asks: *"Is GPT-OSS 20B healthy right now?"*

Tool call:
```json
{ "model_name": "gpt-oss-20b", "time_window": "15m" }
```

Response:
```json
{
  "models": [{
    "model_name": "gpt-oss-20b",
    "status": "degraded",
    "latency_p50_ms": 980,
    "latency_p95_ms": 3420,
    "latency_p99_ms": 5800,
    "ttft_p95_ms": 640,
    "throughput_rps": 1.2,
    "requests_waiting": 8,
    "kv_cache_usage_pct": 82.4,
    "gpu_util_pct": 96.1,
    "gpu_mem_free_mb": 3200,
    "anomaly_score": -0.09,
    "drift_status": "low",
    "as_of": "2026-05-17T10:23:00Z"
  }]
}
```

---

## Tool 2: `detect_drift`

**What it answers:** Has the input or output distribution of a model shifted compared to its baseline?

### Input Schema

```json
{
  "model_name": "string",
  "metric": "string",
  "reference_window": "string",
  "comparison_window": "string"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `model_name` | string | Yes | Model to check. |
| `metric` | string | No (default: `prompt_tokens`) | One of: `prompt_tokens`, `completion_tokens`, `latency`, `throughput`, `rtf` (TTS only). |
| `reference_window` | string | No (default: `7d`) | Baseline period: `7d`, `14d`, `30d`. |
| `comparison_window` | string | No (default: `1h`) | Current period: `1h`, `6h`, `24h`. |

### Output Schema

```json
{
  "model_name": "string",
  "metric": "string",
  "drift_detected": "bool",
  "severity": "none | low | medium | high",
  "ks_statistic": "float",
  "ks_p_value": "float",
  "psi_score": "float",
  "reference_mean": "float",
  "comparison_mean": "float",
  "reference_p95": "float",
  "comparison_p95": "float",
  "interpretation": "string",
  "recommendation": "string"
}
```

Severity mapping:
- `none`: PSI < 0.1 AND p_value > 0.05
- `low`: PSI 0.1-0.2 OR p_value < 0.05
- `medium`: PSI 0.2-0.25
- `high`: PSI > 0.25

### Example

Agent asks: *"Has Gemma4's input distribution changed this week?"*

Tool call:
```json
{ "model_name": "gemma4", "metric": "prompt_tokens", "reference_window": "7d", "comparison_window": "24h" }
```

Response:
```json
{
  "model_name": "gemma4",
  "metric": "prompt_tokens",
  "drift_detected": true,
  "severity": "medium",
  "ks_statistic": 0.31,
  "ks_p_value": 0.003,
  "psi_score": 0.21,
  "reference_mean": 312.4,
  "comparison_mean": 487.9,
  "reference_p95": 890,
  "comparison_p95": 1640,
  "interpretation": "Prompt token length has increased significantly. The current distribution is shifted right — users are sending longer inputs than the 7-day baseline.",
  "recommendation": "Monitor KV cache usage closely. If latency degrades, consider reducing max_num_seqs or increasing GPU memory allocation."
}
```

---

## Tool 3: `get_root_cause`

**What it answers:** Why is a specific model currently degraded?

### Input Schema

```json
{
  "model_name": "string",
  "lookback_minutes": "int"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `model_name` | string | Yes | Model to diagnose. |
| `lookback_minutes` | int | No (default: 30) | How far back to look for anomaly evidence. |

### Output Schema

```json
{
  "model_name": "string",
  "anomaly_detected": "bool",
  "root_cause": "string",
  "confidence": "float",
  "evidence": ["string"],
  "recommendation": "string",
  "contributing_dimensions": [
    { "metric": "string", "current_value": "float", "baseline_value": "float", "deviation_pct": "float" }
  ]
}
```

### Diagnosis Rule Tree

The correlator applies rules in priority order:

```
IF kv_cache_usage_pct > 90 AND requests_waiting > 5:
    -> "KV cache saturation: too many concurrent long-context requests"

IF latency_p95 > 2x_baseline AND gpu_util_pct < 30:
    -> "CPU-side bottleneck: tokenizer or scheduler thread saturation"

IF requests_waiting > 20 AND gpu_util_pct > 90:
    -> "Model at full GPU capacity: insufficient replicas for current traffic"

IF ttft_p95 > 2x_baseline AND tpot_ms unchanged:
    -> "Prefill bottleneck: incoming prompts are getting longer"

IF gpu_temp_celsius > 85:
    -> "GPU thermal throttling: sustained high temperature reducing clock speeds"

IF gpu_mem_free_mb < 2048:
    -> "VRAM near exhaustion: OOM risk, reduce batch size or context length"

IF rtf > 1.0 (TTS only):
    -> "TTS Real-Time Factor exceeded: audio generation slower than playback speed"

IF anomaly_score < -0.15 AND no specific rule matches:
    -> "Multi-dimensional anomaly: unusual combination of metrics, manual inspection recommended"
```

### Example

Agent asks: *"Why is GPT-OSS slow right now?"*

Response:
```json
{
  "model_name": "gpt-oss-20b",
  "anomaly_detected": true,
  "root_cause": "KV cache saturation: too many concurrent long-context requests",
  "confidence": 0.91,
  "evidence": [
    "kv_cache_usage_pct: 88.4% (threshold: 75%)",
    "requests_waiting: 11 (threshold: 5)",
    "latency_p95_ms: 3420ms (baseline: 1100ms, +211%)"
  ],
  "recommendation": "Reduce vLLM max_num_seqs from 256 to 128, or add a second replica. Consider enabling prefix caching to reduce KV cache pressure from repeated system prompts.",
  "contributing_dimensions": [
    { "metric": "kv_cache_usage_pct", "current_value": 88.4, "baseline_value": 51.2, "deviation_pct": 72.7 },
    { "metric": "requests_waiting", "current_value": 11, "baseline_value": 0.3, "deviation_pct": 3566 },
    { "metric": "latency_p95_ms", "current_value": 3420, "baseline_value": 1100, "deviation_pct": 210.9 }
  ]
}
```

---

## Tool 4: `compare_models`

**What it answers:** How does model A perform vs model B over a given time window?

### Input Schema

```json
{
  "model_a": "string",
  "model_b": "string",
  "metric": "string",
  "time_window": "string"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `model_a` | string | Yes | First model name. |
| `model_b` | string | Yes | Second model name. |
| `metric` | string | No (default: `latency_p95_ms`) | Metric to compare. |
| `time_window` | string | No (default: `24h`) | Comparison window. |

### Output Schema

```json
{
  "metric": "string",
  "time_window": "string",
  "model_a": { "name": "string", "mean": "float", "p50": "float", "p95": "float", "p99": "float" },
  "model_b": { "name": "string", "mean": "float", "p50": "float", "p95": "float", "p99": "float" },
  "difference_pct": "float",
  "statistically_significant": "bool",
  "t_statistic": "float",
  "p_value": "float",
  "summary": "string"
}
```

---

## Tool 5: `get_alerts`

**What it answers:** What alerts are currently active or recently fired?

### Input Schema

```json
{
  "model_name": "string | null",
  "severity": "string | null",
  "include_resolved": "bool",
  "limit": "int"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `model_name` | string or null | No | Filter to one model. Null = all models. |
| `severity` | string or null | No | One of `warning`, `critical`. Null = all. |
| `include_resolved` | bool | No (default: false) | Include alerts resolved in the last 24h. |
| `limit` | int | No (default: 20) | Max alerts to return. |

### Output Schema

```json
{
  "alerts": [
    {
      "id": "string",
      "model_name": "string",
      "severity": "warning | critical",
      "type": "threshold | anomaly | drift",
      "title": "string",
      "description": "string",
      "fired_at": "ISO8601",
      "resolved_at": "ISO8601 | null",
      "duration_minutes": "int | null"
    }
  ],
  "total_active": "int",
  "total_critical": "int"
}
```

---

## Tool 6: `predict_saturation`

**What it answers:** Will this model hit capacity in the next N hours, and when?

### Input Schema

```json
{
  "model_name": "string",
  "metric": "string",
  "horizon_hours": "int"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `model_name` | string | Yes | Model to forecast. |
| `metric` | string | No (default: `requests_waiting`) | One of: `requests_waiting`, `kv_cache_usage_pct`, `gpu_util_pct`, `rtf`. |
| `horizon_hours` | int | No (default: 2) | Forecast horizon. Max 6. |

### Output Schema

```json
{
  "model_name": "string",
  "metric": "string",
  "horizon_hours": "int",
  "current_value": "float",
  "saturation_threshold": "float",
  "saturation_risk": "low | medium | high",
  "predicted_saturation_at": "ISO8601 | null",
  "forecast": [
    { "timestamp": "ISO8601", "predicted_value": "float", "lower_bound": "float", "upper_bound": "float" }
  ],
  "recommendation": "string"
}
```

`saturation_risk`:
- `low` — predicted value stays below 80% of threshold in the horizon
- `medium` — predicted value reaches 80-100% of threshold
- `high` — predicted value exceeds threshold within the horizon

### Example

Agent asks: *"Will Svara TTS hit capacity tonight?"*

Response:
```json
{
  "model_name": "svara-tts",
  "metric": "requests_waiting",
  "horizon_hours": 4,
  "current_value": 2.1,
  "saturation_threshold": 20,
  "saturation_risk": "low",
  "predicted_saturation_at": null,
  "forecast": [
    { "timestamp": "2026-05-17T14:00:00Z", "predicted_value": 3.2, "lower_bound": 1.1, "upper_bound": 5.8 },
    { "timestamp": "2026-05-17T16:00:00Z", "predicted_value": 4.7, "lower_bound": 2.0, "upper_bound": 8.1 },
    { "timestamp": "2026-05-17T18:00:00Z", "predicted_value": 6.1, "lower_bound": 2.8, "upper_bound": 11.4 }
  ],
  "recommendation": "No action required. Predicted queue depth remains well below saturation threshold over the next 4 hours."
}
```

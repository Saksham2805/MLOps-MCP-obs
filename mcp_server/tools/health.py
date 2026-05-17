import logging
from typing import Any, Dict, List, Optional

from mcp_server import db as _db

log = logging.getLogger(__name__)

TIME_WINDOW_MAP: Dict[str, str] = {
    "5m": "5 minutes",
    "15m": "15 minutes",
    "1h": "1 hour",
    "6h": "6 hours",
    "24h": "24 hours",
}

ANOMALY_SCORE_WARNING = -0.05
ANOMALY_SCORE_CRITICAL = -0.15


def _derive_status(
    anomaly_score: Optional[float],
    kv_cache: Optional[float],
    requests_waiting: Optional[float],
    gpu_mem_free: Optional[float],
    gpu_temp: Optional[float],
    drift_severity: Optional[str],
) -> str:
    if anomaly_score is not None and anomaly_score < ANOMALY_SCORE_CRITICAL:
        return "critical"
    if kv_cache is not None and kv_cache >= 90.0:
        return "critical"
    if requests_waiting is not None and requests_waiting >= 20:
        return "critical"
    if gpu_mem_free is not None and gpu_mem_free < 2048:
        return "critical"
    if gpu_temp is not None and gpu_temp >= 87.0:
        return "critical"
    if drift_severity in ("high",):
        return "critical"
    if anomaly_score is not None and anomaly_score < ANOMALY_SCORE_WARNING:
        return "degraded"
    if kv_cache is not None and kv_cache >= 75.0:
        return "degraded"
    if requests_waiting is not None and requests_waiting >= 5:
        return "degraded"
    if gpu_mem_free is not None and gpu_mem_free < 4096:
        return "degraded"
    if gpu_temp is not None and gpu_temp >= 80.0:
        return "degraded"
    if drift_severity in ("medium",):
        return "degraded"
    return "healthy"


async def get_model_health(
    model_name: Optional[str] = None,
    time_window: str = "15m",
) -> Dict[str, Any]:
    pg_interval = TIME_WINDOW_MAP.get(time_window, "15 minutes")

    if model_name:
        model_filter = "AND model_name = $2"
        metrics_args = [pg_interval, model_name]
    else:
        model_filter = ""
        metrics_args = [pg_interval]

    metrics_query = (
        "SELECT model_name, model_type, "
        "    AVG(latency_p95_ms) AS latency_p95_ms, "
        "    AVG(latency_p99_ms) AS latency_p99_ms, "
        "    AVG(ttft_p95_ms) AS ttft_p95_ms, "
        "    AVG(tpot_mean_ms) AS tpot_mean_ms, "
        "    AVG(throughput_rps) AS throughput_rps, "
        "    AVG(requests_running) AS requests_running, "
        "    AVG(requests_waiting) AS requests_waiting, "
        "    AVG(kv_cache_usage_pct) AS kv_cache_usage_pct, "
        "    AVG(gpu_util_pct) AS gpu_util_pct, "
        "    AVG(gpu_mem_free_mb) AS gpu_mem_free_mb, "
        "    AVG(gpu_temp_celsius) AS gpu_temp_celsius, "
        "    AVG(gpu_power_watts) AS gpu_power_watts, "
        "    AVG(rtf) AS rtf "
        "FROM inference_metrics "
        f"WHERE time >= NOW() - INTERVAL $1 {model_filter} "
        "GROUP BY model_name, model_type "
        "ORDER BY model_name"
    )

    if model_name:
        metrics_rows = await _db.fetch_rows(metrics_query, pg_interval, model_name)
    else:
        metrics_rows = await _db.fetch_rows(metrics_query, pg_interval)

    models_out = []
    for row in metrics_rows:
        mname = row["model_name"]

        anomaly_row = await _db.fetch_one(
            "SELECT anomaly_score, is_anomaly, contributing_dimensions, cold_start_mode "
            "FROM anomaly_events WHERE model_name = $1 "
            "ORDER BY computed_at DESC LIMIT 1",
            mname,
        )
        anomaly_score: Optional[float] = None
        is_anomaly = False
        contributing_dims = []
        cold_start_mode = False
        if anomaly_row:
            anomaly_score = anomaly_row.get("anomaly_score")
            is_anomaly = bool(anomaly_row.get("is_anomaly", False))
            raw_dims = anomaly_row.get("contributing_dimensions")
            if isinstance(raw_dims, str):
                import json
                try:
                    contributing_dims = json.loads(raw_dims)
                except Exception:
                    contributing_dims = []
            elif isinstance(raw_dims, list):
                contributing_dims = raw_dims
            cold_start_mode = bool(anomaly_row.get("cold_start_mode", False))

        drift_row = await _db.fetch_one(
            "SELECT severity FROM drift_scores WHERE model_name = $1 "
            "ORDER BY computed_at DESC LIMIT 1",
            mname,
        )
        drift_severity: Optional[str] = drift_row["severity"] if drift_row else None

        def _f(v: Any) -> Optional[float]:
            return round(float(v), 4) if v is not None else None

        status = _derive_status(
            anomaly_score=anomaly_score,
            kv_cache=_f(row.get("kv_cache_usage_pct")),
            requests_waiting=_f(row.get("requests_waiting")),
            gpu_mem_free=_f(row.get("gpu_mem_free_mb")),
            gpu_temp=_f(row.get("gpu_temp_celsius")),
            drift_severity=drift_severity,
        )

        models_out.append({
            "model_name": mname,
            "model_type": row.get("model_type", "llm"),
            "status": status,
            "time_window": time_window,
            "metrics": {
                "latency_p95_ms": _f(row.get("latency_p95_ms")),
                "latency_p99_ms": _f(row.get("latency_p99_ms")),
                "ttft_p95_ms": _f(row.get("ttft_p95_ms")),
                "tpot_mean_ms": _f(row.get("tpot_mean_ms")),
                "throughput_rps": _f(row.get("throughput_rps")),
                "requests_running": _f(row.get("requests_running")),
                "requests_waiting": _f(row.get("requests_waiting")),
                "kv_cache_usage_pct": _f(row.get("kv_cache_usage_pct")),
                "gpu_util_pct": _f(row.get("gpu_util_pct")),
                "gpu_mem_free_mb": _f(row.get("gpu_mem_free_mb")),
                "gpu_temp_celsius": _f(row.get("gpu_temp_celsius")),
                "gpu_power_watts": _f(row.get("gpu_power_watts")),
                "rtf": _f(row.get("rtf")),
            },
            "anomaly": {
                "score": round(float(anomaly_score), 6) if anomaly_score is not None else None,
                "is_anomaly": is_anomaly,
                "cold_start_mode": cold_start_mode,
                "contributing_dimensions": contributing_dims[:5],
            },
            "drift_severity": drift_severity,
        })

    return {"models": models_out}

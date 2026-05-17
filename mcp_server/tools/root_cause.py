import json
import logging
import datetime
from typing import Any, Dict, List, Optional

from mcp_server import db as _db

log = logging.getLogger(__name__)


async def get_root_cause(
    model_name: str,
    lookback_minutes: int = 30,
) -> Dict[str, Any]:
    anomaly_row = await _db.fetch_one(
        "SELECT id, computed_at, anomaly_score, is_anomaly, contributing_dimensions, cold_start_mode "
        "FROM anomaly_events "
        "WHERE model_name = $1 "
        "AND computed_at >= NOW() - ($2 * INTERVAL '1 minute') "
        "ORDER BY computed_at DESC LIMIT 1",
        model_name,
        lookback_minutes,
    )

    if anomaly_row is None:
        return {
            "status": "no_anomaly",
            "model_name": model_name,
            "message": f"No anomaly events found for {model_name} in the last {lookback_minutes} minutes.",
            "lookback_minutes": lookback_minutes,
        }

    model_type_row = await _db.fetch_one(
        "SELECT DISTINCT model_type FROM inference_metrics WHERE model_name = $1 LIMIT 1",
        model_name,
    )
    model_type = model_type_row["model_type"] if model_type_row else "llm"

    metrics_row = await _db.fetch_one(
        "SELECT "
        "    AVG(latency_p95_ms) AS latency_p95_ms, "
        "    AVG(ttft_p95_ms) AS ttft_p95_ms, "
        "    AVG(tpot_mean_ms) AS tpot_mean_ms, "
        "    AVG(kv_cache_usage_pct) AS kv_cache_usage_pct, "
        "    AVG(requests_waiting) AS requests_waiting, "
        "    AVG(gpu_util_pct) AS gpu_util_pct, "
        "    AVG(gpu_temp_celsius) AS gpu_temp_celsius, "
        "    AVG(gpu_mem_free_mb) AS gpu_mem_free_mb, "
        "    AVG(rtf) AS rtf "
        "FROM inference_metrics "
        "WHERE model_name = $1 "
        "AND time >= NOW() - ($2 * INTERVAL '1 minute')",
        model_name,
        lookback_minutes,
    )
    metrics_snapshot: Dict[str, Any] = dict(metrics_row) if metrics_row else {}

    artifact_row = await _db.fetch_one(
        "SELECT baseline_means FROM model_artifacts WHERE model_name = $1",
        model_name,
    )
    if artifact_row:
        raw_bm = artifact_row.get("baseline_means")
        if isinstance(raw_bm, str):
            try:
                baseline_means = json.loads(raw_bm)
            except Exception:
                baseline_means = {}
        elif isinstance(raw_bm, dict):
            baseline_means = raw_bm
        else:
            baseline_means = {}
        metrics_snapshot["baseline_latency_p95_ms"] = baseline_means.get("latency_p95_ms")
        metrics_snapshot["baseline_ttft_p95_ms"] = baseline_means.get("ttft_p95_ms")
        metrics_snapshot["baseline_tpot_mean_ms"] = baseline_means.get("tpot_mean_ms")

    anomaly_event: Dict[str, Any] = dict(anomaly_row)
    raw_dims = anomaly_event.get("contributing_dimensions")
    if isinstance(raw_dims, str):
        try:
            anomaly_event["contributing_dimensions"] = json.loads(raw_dims)
        except Exception:
            anomaly_event["contributing_dimensions"] = []

    from analytics import root_cause as _rc
    analysis = _rc.analyze(
        model_name=model_name,
        metrics_snapshot=metrics_snapshot,
        anomaly_event=anomaly_event,
        model_type=model_type,
    )

    computed_at_val = anomaly_row.get("computed_at")
    computed_at_str = (
        computed_at_val.isoformat() + "Z"
        if hasattr(computed_at_val, "isoformat")
        else str(computed_at_val)
    )

    return {
        "status": "ok",
        "model_name": model_name,
        "model_type": model_type,
        "lookback_minutes": lookback_minutes,
        "anomaly_score": anomaly_row.get("anomaly_score"),
        "is_anomaly": bool(anomaly_row.get("is_anomaly", False)),
        "cold_start_mode": bool(anomaly_row.get("cold_start_mode", False)),
        "anomaly_detected": analysis["anomaly_detected"],
        "root_cause": analysis["root_cause"],
        "confidence": analysis["confidence"],
        "evidence": analysis["evidence"],
        "recommendation": analysis["recommendation"],
        "contributing_dimensions": analysis["contributing_dimensions"],
        "metrics_snapshot": {k: (round(float(v), 4) if v is not None else None) for k, v in metrics_snapshot.items() if not k.startswith("baseline_")},
        "computed_at": computed_at_str,
    }

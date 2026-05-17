import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from mcp_server import db as _db

log = logging.getLogger(__name__)

MAX_HORIZON_HOURS = 6
SNAPSHOT_STALE_MINUTES = 35

SATURATION_THRESHOLDS: Dict[str, float] = {
    "requests_waiting": 20.0,
    "kv_cache_usage_pct": 90.0,
    "gpu_util_pct": 95.0,
    "rtf": 1.0,
}

SATURATION_RECOMMENDATIONS: Dict[str, Dict[str, str]] = {
    "requests_waiting": {
        "low": "Queue depth is stable. No immediate action required.",
        "medium": "Monitor request queue trends. Consider pre-scaling replicas.",
        "high": "Request queue approaching saturation. Plan capacity increase within hours.",
        "critical": "Request queue will saturate imminently. Scale replicas now or enable rate limiting.",
    },
    "kv_cache_usage_pct": {
        "low": "KV cache usage is stable.",
        "medium": "KV cache approaching capacity. Review max_model_len settings.",
        "high": "KV cache saturation expected soon. Reduce context length or add GPU memory.",
        "critical": "KV cache will saturate imminently. Reduce max_model_len or scale now.",
    },
    "gpu_util_pct": {
        "low": "GPU utilisation is sustainable.",
        "medium": "GPU utilisation trending up. Monitor serving capacity.",
        "high": "GPU saturation expected. Add replicas or migrate to larger GPU.",
        "critical": "GPU saturation imminent. Add replicas immediately.",
    },
    "rtf": {
        "low": "RTF is within real-time bounds.",
        "medium": "RTF approaching 1.0. TTS service may slow below real-time soon.",
        "high": "RTF saturation expected. Consider scaling TTS replicas.",
        "critical": "RTF will exceed 1.0 imminently. Scale TTS replicas or reduce concurrency now.",
    },
}


def _get_recommendation(metric: str, saturation_risk: str) -> str:
    metric_recs = SATURATION_RECOMMENDATIONS.get(metric, {})
    return metric_recs.get(saturation_risk, "Monitor the metric trend and plan capacity accordingly.")


async def predict_saturation(
    model_name: str,
    metric: str = "requests_waiting",
    horizon_hours: int = 2,
) -> Dict[str, Any]:
    horizon_hours = min(int(horizon_hours), MAX_HORIZON_HOURS)
    horizon_hours = max(horizon_hours, 1)

    if metric not in SATURATION_THRESHOLDS:
        return {
            "status": "error",
            "error": f"Unknown metric: {metric}. Valid: {list(SATURATION_THRESHOLDS.keys())}",
        }

    snapshot = await _db.fetch_one(
        "SELECT computed_at, saturation_time, saturation_risk, saturation_threshold, "
        "    n_training_points, cold_start_mode, forecast_points_json "
        "FROM forecast_snapshots "
        "WHERE model_name = $1 AND metric = $2 "
        "AND computed_at >= NOW() - INTERVAL '35 minutes' "
        "ORDER BY computed_at DESC LIMIT 1",
        model_name, metric,
    )

    needs_refresh = snapshot is None

    if needs_refresh:
        loop = asyncio.get_event_loop()
        try:
            from analytics import forecaster as _forecaster
            fresh_result = await loop.run_in_executor(
                None,
                lambda: _forecaster.train_and_forecast(
                    model_name=model_name,
                    metric=metric,
                    horizon_hours=MAX_HORIZON_HOURS,
                ),
            )
            if fresh_result.get("status") not in ("ok",):
                return fresh_result
            snapshot = await _db.fetch_one(
                "SELECT computed_at, saturation_time, saturation_risk, saturation_threshold, "
                "    n_training_points, cold_start_mode, forecast_points_json "
                "FROM forecast_snapshots "
                "WHERE model_name = $1 AND metric = $2 "
                "ORDER BY computed_at DESC LIMIT 1",
                model_name, metric,
            )
            if snapshot is None:
                return fresh_result
        except Exception as exc:
            log.error("predict_saturation: fresh forecast failed: %s", exc)
            return {
                "status": "error",
                "error": str(exc),
                "model_name": model_name,
                "metric": metric,
            }

    raw_points = snapshot.get("forecast_points_json")
    if isinstance(raw_points, str):
        try:
            all_points = json.loads(raw_points)
        except Exception:
            all_points = []
    elif isinstance(raw_points, list):
        all_points = raw_points
    else:
        all_points = []

    filtered_points = all_points[:horizon_hours * 4]

    threshold = SATURATION_THRESHOLDS[metric]
    saturation_time = snapshot.get("saturation_time")
    saturation_risk = snapshot.get("saturation_risk", "low")

    if saturation_time and hasattr(saturation_time, "isoformat"):
        saturation_time_str = saturation_time.isoformat() + "Z"
    elif saturation_time:
        saturation_time_str = str(saturation_time)
    else:
        saturation_time_str = None

    computed_at_val = snapshot.get("computed_at")
    computed_at_str = (
        computed_at_val.isoformat() + "Z"
        if hasattr(computed_at_val, "isoformat")
        else str(computed_at_val)
    )

    return {
        "status": "ok",
        "model_name": model_name,
        "metric": metric,
        "horizon_hours": horizon_hours,
        "saturation_threshold": threshold,
        "saturation_time": saturation_time_str,
        "saturation_risk": saturation_risk,
        "n_training_points": snapshot.get("n_training_points"),
        "cold_start_mode": bool(snapshot.get("cold_start_mode", False)),
        "forecast_points": filtered_points,
        "recommendation": _get_recommendation(metric, saturation_risk),
        "computed_at": computed_at_str,
    }

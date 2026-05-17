import asyncio
import datetime
import logging
from typing import Any, Dict, Optional

from mcp_server import db as _db

log = logging.getLogger(__name__)

CACHE_STALE_MINUTES = 6

METRIC_INTERPRETATIONS: Dict[str, Dict[str, str]] = {
    "prompt_tokens": {
        "none": "Prompt token distribution is stable. No significant drift detected.",
        "low": "Minor shift in prompt token distribution. Monitor for continued changes.",
        "medium": "Moderate drift in prompt token counts. Input workload may be changing.",
        "high": "Significant prompt token distribution shift. Likely a change in input workload or client behaviour.",
    },
    "completion_tokens": {
        "none": "Completion token distribution is stable.",
        "low": "Minor shift in completion token lengths. Possibly sampling parameter changes.",
        "medium": "Moderate drift in completion lengths. Review max_tokens settings or prompt changes.",
        "high": "Major drift in completion lengths. Workload or model behaviour has changed significantly.",
    },
    "latency": {
        "none": "Latency distribution is stable.",
        "low": "Minor latency shift detected. Continue monitoring.",
        "medium": "Moderate latency drift. Consider investigating serving infrastructure.",
        "high": "Major latency regression detected. Immediate investigation recommended.",
    },
    "throughput": {
        "none": "Throughput distribution is stable.",
        "low": "Minor throughput variation. Normal traffic fluctuations.",
        "medium": "Moderate throughput drift. Check for traffic pattern or resource changes.",
        "high": "Major throughput change. Possible capacity issue or traffic spike.",
    },
    "rtf": {
        "none": "Real-Time Factor is stable.",
        "low": "Minor RTF variation. Within acceptable range.",
        "medium": "Moderate RTF drift. TTS synthesis speed may be degrading.",
        "high": "Major RTF degradation. Real-time synthesis may be at risk.",
    },
}

METRIC_RECOMMENDATIONS: Dict[str, Dict[str, str]] = {
    "latency": {
        "medium": "Profile GPU utilisation and KV cache usage. Check for queue depth increases.",
        "high": "Escalate immediately. Run root cause analysis. Consider rolling back recent deployments.",
    },
    "prompt_tokens": {
        "medium": "Review client application for prompt template changes.",
        "high": "Enforce prompt length limits or update capacity planning.",
    },
    "completion_tokens": {
        "medium": "Review max_tokens settings and sampling parameters.",
        "high": "Adjust generation constraints or scale serving capacity.",
    },
    "throughput": {
        "medium": "Monitor resource utilisation trends.",
        "high": "Evaluate auto-scaling policies and capacity limits.",
    },
    "rtf": {
        "medium": "Monitor GPU utilisation for the TTS service.",
        "high": "Consider scaling TTS replicas or reducing concurrent synthesis requests.",
    },
}


def _build_interpretation(metric: str, severity: str) -> str:
    metric_map = METRIC_INTERPRETATIONS.get(metric, {})
    return metric_map.get(severity, f"Drift severity: {severity} for metric {metric}.")


def _build_recommendation(metric: str, severity: str) -> str:
    if severity in ("none", "low"):
        return "No action required. Continue monitoring."
    recs = METRIC_RECOMMENDATIONS.get(metric, {})
    return recs.get(severity, "Investigate the metric changes and review recent deployments.")


async def detect_drift(
    model_name: str,
    metric: str = "prompt_tokens",
    reference_window: str = "7d",
    comparison_window: str = "1h",
) -> Dict[str, Any]:
    cached = await _db.fetch_one(
        "SELECT * FROM drift_scores "
        "WHERE model_name = $1 AND metric = $2 "
        "AND reference_window = $3 AND comparison_window = $4 "
        "ORDER BY computed_at DESC LIMIT 1",
        model_name, metric, reference_window, comparison_window,
    )

    needs_refresh = True
    if cached:
        computed_at = cached.get("computed_at")
        if computed_at is not None:
            import datetime as _dt
            if hasattr(computed_at, "utcoffset"):
                age_seconds = (_dt.datetime.now(_dt.timezone.utc) - computed_at).total_seconds()
            else:
                age_seconds = (_dt.datetime.utcnow() - computed_at).total_seconds()
            if age_seconds < CACHE_STALE_MINUTES * 60:
                needs_refresh = False

    if needs_refresh:
        loop = asyncio.get_event_loop()
        try:
            from analytics import drift as _drift_module
            result = await loop.run_in_executor(
                None,
                lambda: _drift_module.run_drift_check(
                    model_name=model_name,
                    metric=metric,
                    reference_window=reference_window,
                    comparison_window=comparison_window,
                ),
            )
            cached = await _db.fetch_one(
                "SELECT * FROM drift_scores "
                "WHERE model_name = $1 AND metric = $2 "
                "AND reference_window = $3 AND comparison_window = $4 "
                "ORDER BY computed_at DESC LIMIT 1",
                model_name, metric, reference_window, comparison_window,
            )
            if not cached:
                return result
        except Exception as exc:
            log.error("detect_drift: inline run_drift_check failed: %s", exc)
            if not cached:
                return {
                    "status": "error",
                    "error": str(exc),
                    "model_name": model_name,
                    "metric": metric,
                }

    severity = cached.get("severity", "none")
    return {
        "status": "ok",
        "model_name": model_name,
        "metric": metric,
        "severity": severity,
        "ks_statistic": cached.get("ks_statistic"),
        "ks_p_value": cached.get("ks_p_value"),
        "psi": cached.get("psi"),
        "reference_mean": cached.get("reference_mean"),
        "comparison_mean": cached.get("comparison_mean"),
        "drift_pct": cached.get("drift_pct"),
        "n_reference": cached.get("n_reference"),
        "n_comparison": cached.get("n_comparison"),
        "reference_window": reference_window,
        "comparison_window": comparison_window,
        "interpretation": _build_interpretation(metric, severity),
        "recommendation": _build_recommendation(metric, severity),
        "computed_at": cached.get("computed_at").isoformat() + "Z"
            if hasattr(cached.get("computed_at"), "isoformat")
            else str(cached.get("computed_at")),
    }

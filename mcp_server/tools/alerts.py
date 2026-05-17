import os
import json
import logging
import datetime
from typing import Any, Dict, List, Optional

import httpx

from mcp_server import db as _db

log = logging.getLogger(__name__)


def _alertmanager_url() -> Optional[str]:
    return os.environ.get("ALERTMANAGER_URL")


async def _fetch_alertmanager_alerts() -> List[Dict[str, Any]]:
    url = _alertmanager_url()
    if not url:
        return []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            base = url.rstrip("/")
            resp = await client.get(f"{base}/api/v2/alerts")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        log.warning("alerts: alertmanager fetch failed: %s", exc)
        return []


def _am_alert_to_unified(am_alert: Dict[str, Any]) -> Dict[str, Any]:
    labels = am_alert.get("labels", {})
    annotations = am_alert.get("annotations", {})
    status_info = am_alert.get("status", {})
    fired_at = am_alert.get("startsAt", "")
    ends_at = am_alert.get("endsAt", "")
    model_name = labels.get("model_name", labels.get("job", "unknown"))
    severity = labels.get("severity", "warning")
    title = annotations.get("summary", labels.get("alertname", "unknown"))
    description = annotations.get("description", "")
    state = status_info.get("state", "active")
    resolved = state == "resolved"
    return {
        "source": "alertmanager",
        "model_name": model_name,
        "title": title,
        "description": description,
        "severity": severity,
        "fired_at": fired_at,
        "resolved": resolved,
        "ends_at": ends_at,
        "labels": labels,
    }


async def get_alerts(
    model_name: Optional[str] = None,
    severity: Optional[str] = None,
    include_resolved: bool = False,
    limit: int = 20,
) -> Dict[str, Any]:
    conditions = ["is_anomaly = TRUE"]
    params: List[Any] = []
    param_idx = 1

    if model_name:
        conditions.append(f"model_name = ${param_idx}")
        params.append(model_name)
        param_idx += 1

    anomaly_score_threshold = -0.05
    if severity == "critical":
        anomaly_score_threshold = -0.15
        conditions.append(f"anomaly_score <= ${param_idx}")
        params.append(anomaly_score_threshold)
        param_idx += 1
    elif severity == "warning":
        conditions.append(f"anomaly_score > ${param_idx} AND anomaly_score <= ${param_idx + 1}")
        params.extend([-0.15, -0.05])
        param_idx += 2

    where_clause = " AND ".join(conditions)
    query = (
        "SELECT id, computed_at, model_name, anomaly_score, is_anomaly, "
        "    contributing_dimensions, cold_start_mode "
        "FROM anomaly_events "
        f"WHERE {where_clause} "
        "ORDER BY computed_at DESC "
        f"LIMIT ${param_idx}"
    )
    params.append(limit)

    db_rows = await _db.fetch_rows(query, *params)

    db_alerts = []
    for row in db_rows:
        score = row.get("anomaly_score")
        if score is not None and score <= -0.15:
            sev = "critical"
        elif score is not None and score <= -0.05:
            sev = "warning"
        else:
            sev = "info"

        dims = row.get("contributing_dimensions")
        if isinstance(dims, str):
            try:
                dims = json.loads(dims)
            except Exception:
                dims = []

        top_dim = dims[0].get("feature", "unknown") if dims else "unknown"
        fired_at_val = row.get("computed_at")
        fired_at_str = (
            fired_at_val.isoformat() + "Z"
            if hasattr(fired_at_val, "isoformat")
            else str(fired_at_val)
        )
        db_alerts.append({
            "source": "anomaly_detector",
            "model_name": row["model_name"],
            "title": f"Anomaly detected: {top_dim} deviation",
            "description": f"IsolationForest score {score:.4f}" if score is not None else "Anomaly detected",
            "severity": sev,
            "fired_at": fired_at_str,
            "resolved": False,
            "anomaly_score": score,
            "contributing_dimensions": dims[:3],
        })

    am_raw = await _fetch_alertmanager_alerts()
    am_alerts = [_am_alert_to_unified(a) for a in am_raw]

    if model_name:
        am_alerts = [a for a in am_alerts if a["model_name"] == model_name]
    if severity:
        am_alerts = [a for a in am_alerts if a["severity"] == severity]
    if not include_resolved:
        am_alerts = [a for a in am_alerts if not a["resolved"]]

    seen: set = set()
    all_alerts: List[Dict[str, Any]] = []
    for alert in db_alerts + am_alerts:
        key = (alert["model_name"], alert["title"], alert["fired_at"])
        if key not in seen:
            seen.add(key)
            all_alerts.append(alert)

    all_alerts = all_alerts[:limit]

    total_active = sum(1 for a in all_alerts if not a.get("resolved", False))
    total_critical = sum(1 for a in all_alerts if a.get("severity") == "critical" and not a.get("resolved", False))

    return {
        "status": "ok",
        "total_active": total_active,
        "total_critical": total_critical,
        "alerts": all_alerts,
        "filters": {
            "model_name": model_name,
            "severity": severity,
            "include_resolved": include_resolved,
            "limit": limit,
        },
    }

import os
import sys
import logging
import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from analytics import db as _db

log = logging.getLogger(__name__)

METRIC_COLUMN_MAP: Dict[str, str] = {
    "prompt_tokens": "prompt_tokens_total",
    "completion_tokens": "generation_tokens_total",
    "latency": "latency_p95_ms",
    "throughput": "throughput_rps",
    "rtf": "rtf",
}

ALL_METRICS = list(METRIC_COLUMN_MAP.keys())
MIN_SAMPLES = 30
PSI_LOW = 0.1
PSI_MEDIUM = 0.2
PSI_HIGH = 0.25


def _severity_from_psi_and_pvalue(psi: float, p_value: float) -> str:
    if psi < PSI_LOW:
        return "none"
    if psi < PSI_MEDIUM:
        if p_value < 0.01:
            return "medium"
        return "low"
    if psi < PSI_HIGH:
        return "medium"
    return "high"


def _compute_psi(reference: np.ndarray, comparison: np.ndarray, n_bins: int = 10) -> float:
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, percentiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0
    bin_edges[-1] += 1e-9
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    comp_counts, _ = np.histogram(comparison, bins=bin_edges)
    ref_pct = np.clip(ref_counts / len(reference), 1e-6, None)
    comp_pct = np.clip(comp_counts / len(comparison), 1e-6, None)
    return float(np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct)))


def _to_pg_interval(window: str) -> str:
    if window.endswith("d"):
        return f"{window[:-1]} days"
    if window.endswith("h"):
        return f"{window[:-1]} hours"
    if window.endswith("m"):
        return f"{window[:-1]} minutes"
    return window


def run_drift_check(
    model_name: str,
    metric: str,
    reference_window: str = "7d",
    comparison_window: str = "1h",
    db=None,
) -> Dict[str, Any]:
    if metric not in METRIC_COLUMN_MAP:
        return {"status": "error", "error": f"Unknown metric '{metric}'. Valid: {list(METRIC_COLUMN_MAP.keys())}"}

    col = METRIC_COLUMN_MAP[metric]
    ref_interval = _to_pg_interval(reference_window)
    comp_interval = _to_pg_interval(comparison_window)

    ref_query = (
        f"SELECT {col} AS val FROM inference_metrics "
        f"WHERE model_name = %s "
        f"AND time >= NOW() - INTERVAL '{ref_interval}' "
        f"AND time < NOW() - INTERVAL '{comp_interval}' "
        f"AND {col} IS NOT NULL ORDER BY time"
    )
    comp_query = (
        f"SELECT {col} AS val FROM inference_metrics "
        f"WHERE model_name = %s "
        f"AND time >= NOW() - INTERVAL '{comp_interval}' "
        f"AND {col} IS NOT NULL ORDER BY time"
    )

    ref_rows = _db.read_rows(ref_query, (model_name,), pool=db)
    comp_rows = _db.read_rows(comp_query, (model_name,), pool=db)
    n_ref = len(ref_rows)
    n_comp = len(comp_rows)
    computed_at = datetime.datetime.utcnow().isoformat() + "Z"

    if n_ref < MIN_SAMPLES or n_comp < MIN_SAMPLES:
        log.info("drift: insufficient data for %s/%s (ref=%d comp=%d)", model_name, metric, n_ref, n_comp)
        return {
            "status": "insufficient_data",
            "model_name": model_name,
            "metric": metric,
            "n_reference": n_ref,
            "n_comparison": n_comp,
            "minimum_required": MIN_SAMPLES,
            "computed_at": computed_at,
        }

    ref_vals = np.array([r["val"] for r in ref_rows], dtype=float)
    comp_vals = np.array([r["val"] for r in comp_rows], dtype=float)

    ks_stat, ks_p = stats.ks_2samp(ref_vals, comp_vals)
    psi = _compute_psi(ref_vals, comp_vals)
    severity = _severity_from_psi_and_pvalue(psi, ks_p)
    ref_mean = float(np.mean(ref_vals))
    comp_mean = float(np.mean(comp_vals))
    drift_pct = ((comp_mean - ref_mean) / ref_mean * 100.0) if ref_mean != 0 else 0.0

    result: Dict[str, Any] = {
        "status": "ok",
        "model_name": model_name,
        "metric": metric,
        "severity": severity,
        "ks_statistic": round(float(ks_stat), 6),
        "ks_p_value": round(float(ks_p), 6),
        "psi": round(psi, 6),
        "reference_mean": round(ref_mean, 4),
        "comparison_mean": round(comp_mean, 4),
        "drift_pct": round(drift_pct, 2),
        "n_reference": n_ref,
        "n_comparison": n_comp,
        "reference_window": reference_window,
        "comparison_window": comparison_window,
        "computed_at": computed_at,
    }

    log.info(
        "drift: %s/%s severity=%s psi=%.4f ks_p=%.4f drift_pct=%.1f%%",
        model_name, metric, severity, psi, ks_p, drift_pct,
    )
    _write_drift_score(result, db=db)
    return result


def _write_drift_score(result: Dict[str, Any], db=None) -> None:
    insert_sql = (
        "INSERT INTO drift_scores ("
        "    computed_at, model_name, metric, "
        "    reference_window, comparison_window, "
        "    severity, ks_statistic, ks_p_value, "
        "    psi, reference_mean, comparison_mean, "
        "    drift_pct, n_reference, n_comparison"
        ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (model_name, metric, reference_window, comparison_window) "
        "DO UPDATE SET "
        "    computed_at = EXCLUDED.computed_at, "
        "    severity = EXCLUDED.severity, "
        "    ks_statistic = EXCLUDED.ks_statistic, "
        "    ks_p_value = EXCLUDED.ks_p_value, "
        "    psi = EXCLUDED.psi, "
        "    reference_mean = EXCLUDED.reference_mean, "
        "    comparison_mean = EXCLUDED.comparison_mean, "
        "    drift_pct = EXCLUDED.drift_pct, "
        "    n_reference = EXCLUDED.n_reference, "
        "    n_comparison = EXCLUDED.n_comparison"
    )
    try:
        _db.execute(
            insert_sql,
            (
                result["computed_at"], result["model_name"], result["metric"],
                result.get("reference_window", "7d"), result.get("comparison_window", "1h"),
                result.get("severity", "none"), result.get("ks_statistic"),
                result.get("ks_p_value"), result.get("psi"),
                result.get("reference_mean"), result.get("comparison_mean"),
                result.get("drift_pct"), result.get("n_reference"), result.get("n_comparison"),
            ),
            pool=db,
        )
    except Exception as exc:
        log.warning("drift: failed to write drift_score: %s", exc)


def _get_all_model_names(db=None) -> List[str]:
    rows = _db.read_rows("SELECT DISTINCT model_name FROM inference_metrics ORDER BY model_name", pool=db)
    return [r["model_name"] for r in rows]


def run_all_models_drift(
    db=None,
    reference_window: str = "7d",
    comparison_window: str = "1h",
) -> List[Dict[str, Any]]:
    model_names = _get_all_model_names(db=db)
    results = []
    for model_name in model_names:
        for metric in ALL_METRICS:
            try:
                result = run_drift_check(
                    model_name=model_name,
                    metric=metric,
                    reference_window=reference_window,
                    comparison_window=comparison_window,
                    db=db,
                )
                results.append(result)
            except Exception as exc:
                log.error("drift: unhandled error for %s/%s: %s", model_name, metric, exc, exc_info=True)
    return results


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(
        stream=sys.stdout,
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from apscheduler.schedulers.blocking import BlockingScheduler

    interval_minutes = int(os.environ.get("DRIFT_INTERVAL_MINUTES", "5"))
    log.info("Starting drift scheduler (every %d min)", interval_minutes)

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        func=run_all_models_drift,
        trigger="interval",
        minutes=interval_minutes,
        id="drift_check",
        name="Run drift checks for all models",
        replace_existing=True,
        misfire_grace_time=60,
    )
    run_all_models_drift()
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Drift scheduler stopped.")

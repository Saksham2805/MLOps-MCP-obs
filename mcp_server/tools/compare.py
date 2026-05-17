import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from mcp_server import db as _db

log = logging.getLogger(__name__)

TIME_WINDOW_MAP: Dict[str, str] = {
    "5m": "5 minutes",
    "15m": "15 minutes",
    "1h": "1 hour",
    "6h": "6 hours",
    "24h": "24 hours",
}

VALID_METRICS = [
    "latency_p95_ms", "latency_p99_ms", "latency_p50_ms",
    "ttft_p95_ms", "tpot_mean_ms",
    "throughput_rps", "kv_cache_usage_pct",
    "requests_waiting", "gpu_util_pct", "rtf",
]


def _percentile(arr: np.ndarray, p: float) -> float:
    if len(arr) == 0:
        return 0.0
    return float(np.percentile(arr, p))


async def compare_models(
    model_a: str,
    model_b: str,
    metric: str = "latency_p95_ms",
    time_window: str = "24h",
) -> Dict[str, Any]:
    if metric not in VALID_METRICS:
        return {
            "status": "error",
            "error": f"Invalid metric '{metric}'. Valid options: {VALID_METRICS}",
        }

    pg_interval = TIME_WINDOW_MAP.get(time_window, "24 hours")

    rows_a = await _db.fetch_rows(
        f"SELECT {metric} AS val FROM inference_metrics "
        f"WHERE model_name = $1 "
        f"AND time >= NOW() - INTERVAL $2 "
        f"AND {metric} IS NOT NULL ",
        model_a, pg_interval,
    )
    rows_b = await _db.fetch_rows(
        f"SELECT {metric} AS val FROM inference_metrics "
        f"WHERE model_name = $1 "
        f"AND time >= NOW() - INTERVAL $2 "
        f"AND {metric} IS NOT NULL ",
        model_b, pg_interval,
    )

    n_a = len(rows_a)
    n_b = len(rows_b)

    if n_a == 0 and n_b == 0:
        return {
            "status": "no_data",
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "time_window": time_window,
        }

    def _stats(rows: list) -> Dict[str, Any]:
        if not rows:
            return {"mean": None, "p50": None, "p95": None, "p99": None, "n": 0}
        arr = np.array([r["val"] for r in rows], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return {"mean": None, "p50": None, "p95": None, "p99": None, "n": 0}
        return {
            "mean": round(float(np.mean(arr)), 4),
            "p50": round(_percentile(arr, 50), 4),
            "p95": round(_percentile(arr, 95), 4),
            "p99": round(_percentile(arr, 99), 4),
            "n": len(arr),
        }

    stats_a = _stats(rows_a)
    stats_b = _stats(rows_b)

    t_stat: Optional[float] = None
    p_value: Optional[float] = None
    statistically_significant = False

    if stats_a["n"] >= 2 and stats_b["n"] >= 2:
        arr_a = np.array([r["val"] for r in rows_a], dtype=float)
        arr_b = np.array([r["val"] for r in rows_b], dtype=float)
        arr_a = arr_a[np.isfinite(arr_a)]
        arr_b = arr_b[np.isfinite(arr_b)]
        if len(arr_a) >= 2 and len(arr_b) >= 2:
            t_result = stats.ttest_ind(arr_a, arr_b, equal_var=False)
            t_stat = round(float(t_result.statistic), 6)
            p_value = round(float(t_result.pvalue), 6)
            statistically_significant = p_value < 0.05

    mean_a = stats_a.get("mean")
    mean_b = stats_b.get("mean")
    difference_pct: Optional[float] = None
    if mean_a is not None and mean_b is not None and mean_b != 0:
        difference_pct = round((mean_a - mean_b) / mean_b * 100.0, 2)

    if difference_pct is not None:
        direction = "higher" if difference_pct > 0 else "lower"
        abs_diff = abs(difference_pct)
        sig_str = "(statistically significant)" if statistically_significant else "(not statistically significant)"
        summary = (
            f"{model_a} has {abs_diff:.1f}% {direction} {metric} than {model_b} {sig_str}"
        )
    else:
        summary = f"Insufficient data to compare {metric} between {model_a} and {model_b}."

    return {
        "status": "ok",
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "time_window": time_window,
        "model_a_stats": stats_a,
        "model_b_stats": stats_b,
        "difference_pct": difference_pct,
        "t_statistic": t_stat,
        "p_value": p_value,
        "statistically_significant": statistically_significant,
        "summary": summary,
    }

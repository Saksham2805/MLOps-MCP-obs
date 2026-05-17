import io
import os
import sys
import json
import logging
import datetime
import contextlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analytics import db as _db

log = logging.getLogger(__name__)

SATURATION_THRESHOLDS: Dict[str, float] = {
    "requests_waiting": 20.0,
    "kv_cache_usage_pct": 90.0,
    "gpu_util_pct": 95.0,
    "rtf": 1.0,
}

ALL_FORECAST_METRICS = list(SATURATION_THRESHOLDS.keys())
MIN_TRAINING_POINTS = 672


def _to_pg_interval(window: str) -> str:
    if window.endswith("d"):
        return f"{window[:-1]} days"
    if window.endswith("h"):
        return f"{window[:-1]} hours"
    if window.endswith("m"):
        return f"{window[:-1]} minutes"
    return window


def find_saturation_time(
    forecast_df: "pd.DataFrame",
    threshold: float,
    ds_col: str = "ds",
    yhat_col: str = "yhat_upper",
) -> Tuple[Optional[str], str]:
    future_rows = forecast_df[forecast_df[yhat_col] >= threshold]
    if future_rows.empty:
        return None, "low"
    sat_time = future_rows.iloc[0][ds_col]
    now = pd.Timestamp.utcnow().tz_localize(None)
    if hasattr(sat_time, "tz_localize") and sat_time.tzinfo is not None:
        sat_time = sat_time.tz_localize(None)
    hours_until = (sat_time - now).total_seconds() / 3600.0
    if hours_until <= 1.0:
        risk = "critical"
    elif hours_until <= 3.0:
        risk = "high"
    elif hours_until <= 6.0:
        risk = "medium"
    else:
        risk = "low"
    return sat_time.isoformat() + "Z", risk


def _linear_extrapolation_forecast(
    series: "pd.Series",
    timestamps: "pd.Series",
    horizon_hours: int,
) -> "pd.DataFrame": 
    if len(series) < 2:
        last_val = float(series.iloc[-1]) if len(series) > 0 else 0.0
        future_times = pd.date_range(
            start=timestamps.iloc[-1] + pd.Timedelta(minutes=15),
            periods=horizon_hours * 4,
            freq="15min",
        )
        return pd.DataFrame({"ds": future_times, "yhat": last_val, "yhat_upper": last_val, "yhat_lower": last_val})
    x = np.arange(len(series), dtype=float)
    y = series.values.astype(float)
    valid = np.isfinite(y)
    if valid.sum() < 2:
        slope, intercept = 0.0, float(np.nanmean(y))
    else:
        slope, intercept = np.polyfit(x[valid], y[valid], 1)
    n_future = horizon_hours * 4
    future_x = np.arange(len(series), len(series) + n_future, dtype=float)
    future_vals = slope * future_x + intercept
    std = float(np.nanstd(y[valid])) if valid.sum() > 1 else 0.0
    freq = timestamps.diff().median()
    if pd.isna(freq) or freq.total_seconds() == 0:
        freq = pd.Timedelta(minutes=15)
    future_times = pd.date_range(
        start=timestamps.iloc[-1] + freq,
        periods=n_future,
        freq=freq,
    )
    return pd.DataFrame({
        "ds": future_times,
        "yhat": future_vals,
        "yhat_upper": future_vals + 2 * std,
        "yhat_lower": future_vals - 2 * std,
    })


def train_and_forecast(
    model_name: str,
    metric: str,
    horizon_hours: int = 6,
    db=None,
) -> Dict[str, Any]:
    if metric not in SATURATION_THRESHOLDS:
        return {"status": "error", "error": f"Unknown metric: {metric}. Valid: {ALL_FORECAST_METRICS}"}
    threshold = SATURATION_THRESHOLDS[metric]
    query = (
        f"SELECT time_bucket('15 minutes', time) AS bucket, AVG({metric}) AS val "
        f"FROM inference_metrics "
        f"WHERE model_name = %s AND time >= NOW() - INTERVAL '14 days' "
        f"AND {metric} IS NOT NULL "
        f"GROUP BY bucket ORDER BY bucket"
    )
    rows = _db.read_rows(query, (model_name,), pool=db)
    computed_at = datetime.datetime.utcnow().isoformat() + "Z"

    cold_start_mode = len(rows) < MIN_TRAINING_POINTS
    if not rows:
        return {
            "status": "no_data",
            "model_name": model_name,
            "metric": metric,
            "computed_at": computed_at,
        }

    timestamps = pd.Series([r["bucket"] for r in rows])
    values = pd.Series([float(r["val"]) for r in rows])

    if cold_start_mode:
        log.info("forecast: cold_start for %s/%s (n=%d < %d required)",
                 model_name, metric, len(rows), MIN_TRAINING_POINTS)
        forecast_df = _linear_extrapolation_forecast(values, timestamps, horizon_hours)
    else:
        try:
            from prophet import Prophet
        except ImportError:
            log.warning("forecast: prophet not installed, falling back to linear extrapolation")
            cold_start_mode = True
            forecast_df = _linear_extrapolation_forecast(values, timestamps, horizon_hours)
        else:
            df = pd.DataFrame({"ds": timestamps, "y": values})
            df["ds"] = pd.to_datetime(df["ds"])
            df = df.dropna(subset=["y"])
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                seasonality_mode="multiplicative",
                daily_seasonality=True,
                weekly_seasonality=True,
                interval_width=0.95,
            )
            suppress_stdout = open(os.devnull, "w")
            suppress_stderr = open(os.devnull, "w")
            with contextlib.redirect_stdout(suppress_stdout), contextlib.redirect_stderr(suppress_stderr):
                model.fit(df)
            suppress_stdout.close()
            suppress_stderr.close()
            future = model.make_future_dataframe(periods=horizon_hours * 4, freq="15min")
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                forecast_df = model.predict(future)
            forecast_df = forecast_df[forecast_df["ds"] > df["ds"].max()].copy()

    saturation_time, saturation_risk = find_saturation_time(forecast_df, threshold)

    forecast_points = []
    for _, row in forecast_df.iterrows():
        ds_val = row["ds"]
        if hasattr(ds_val, "isoformat"):
            ds_str = ds_val.isoformat() + "Z"
        else:
            ds_str = str(ds_val)
        forecast_points.append({
            "timestamp": ds_str,
            "yhat": round(float(row["yhat"]), 4),
            "yhat_lower": round(float(row["yhat_lower"]), 4),
            "yhat_upper": round(float(row["yhat_upper"]), 4),
        })

    result: Dict[str, Any] = {
        "status": "ok",
        "model_name": model_name,
        "metric": metric,
        "horizon_hours": horizon_hours,
        "saturation_threshold": threshold,
        "saturation_time": saturation_time,
        "saturation_risk": saturation_risk,
        "n_training_points": len(rows),
        "cold_start_mode": cold_start_mode,
        "forecast_points": forecast_points,
        "computed_at": computed_at,
    }

    _write_forecast_snapshot(result, db=db)
    log.info(
        "forecast: %s/%s risk=%s saturation=%s cold_start=%s",
        model_name, metric, saturation_risk, saturation_time, cold_start_mode,
    )
    return result


def _write_forecast_snapshot(result: Dict[str, Any], db=None) -> None:
    insert_sql = (
        "INSERT INTO forecast_snapshots "
        "    (computed_at, model_name, metric, horizon_hours, "
        "     saturation_threshold, saturation_time, saturation_risk, "
        "     n_training_points, cold_start_mode, forecast_points_json) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
    try:
        _db.execute(
            insert_sql,
            (
                result["computed_at"],
                result["model_name"],
                result["metric"],
                result["horizon_hours"],
                result["saturation_threshold"],
                result.get("saturation_time"),
                result.get("saturation_risk"),
                result["n_training_points"],
                result["cold_start_mode"],
                json.dumps(result.get("forecast_points", [])),
            ),
            pool=db,
        )
    except Exception as exc:
        log.warning("forecast: failed to write forecast_snapshot: %s", exc)


def _get_all_models(db=None) -> List[Dict[str, str]]:
    rows = _db.read_rows(
        "SELECT DISTINCT model_name FROM inference_metrics ORDER BY model_name",
        pool=db,
    )
    return [{"model_name": r["model_name"]} for r in rows]


def run_all_forecasts(db=None) -> List[Dict[str, Any]]:
    models = _get_all_models(db=db)
    results = []
    for m in models:
        for metric in ALL_FORECAST_METRICS:
            try:
                result = train_and_forecast(
                    model_name=m["model_name"],
                    metric=metric,
                    horizon_hours=6,
                    db=db,
                )
                results.append(result)
            except Exception as exc:
                log.error(
                    "forecast: error for %s/%s: %s",
                    m["model_name"], metric, exc, exc_info=True,
                )
    return results


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(
        stream=sys.stdout,
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from apscheduler.schedulers.blocking import BlockingScheduler

    interval_minutes = int(os.environ.get("FORECAST_INTERVAL_MINUTES", "30"))
    log.info("Starting forecast scheduler (every %d min)", interval_minutes)

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        func=run_all_forecasts,
        trigger="interval",
        minutes=interval_minutes,
        id="forecast",
        name="Run all model forecasts",
        replace_existing=True,
        misfire_grace_time=120,
    )
    run_all_forecasts()
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Forecast scheduler stopped.")

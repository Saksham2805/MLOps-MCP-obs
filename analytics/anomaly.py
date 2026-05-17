import io
import os
import sys
import json
import pickle
import logging
import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psycopg2
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from analytics import db as _db

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature definitions per model type
# ---------------------------------------------------------------------------
FEATURES_LLM = ["latency_p95_ms", "gpu_util_pct", "kv_cache_usage_pct", "requests_waiting"]
FEATURES_TTS = ["latency_p95_ms", "gpu_util_pct", "rtf", "requests_waiting"]


def _feature_names(model_type: str) -> List[str]:
    if model_type == "tts":
        return FEATURES_TTS
    return FEATURES_LLM


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_model(model_name: str, model_type: str, db=None) -> Dict[str, Any]:
    feature_names = _feature_names(model_type)
    cols = ", ".join(f"COALESCE({c}, 0) AS {c}" for c in feature_names)
    query = (
        f"SELECT {cols} FROM inference_metrics "
        f"WHERE model_name = %s "
        f"AND time >= NOW() - INTERVAL '7 days' "
        f"ORDER BY time"
    )
    rows = _db.read_rows(query, (model_name,), pool=db)
    if len(rows) < 10:
        log.warning("anomaly: too few samples (%d) for %s, skipping training", len(rows), model_name)
        return {"status": "insufficient_data", "n_samples": len(rows)}

    X = np.array([[row[f] for f in feature_names] for row in rows], dtype=float)
    col_means = np.nanmean(X, axis=0)
    for ci in range(X.shape[1]):
        mask = ~np.isfinite(X[:, ci])
        X[mask, ci] = col_means[ci]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    forest = IsolationForest(
        contamination=0.05,
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    forest.fit(X_scaled)

    scaler_blob = pickle.dumps(scaler)
    model_blob = pickle.dumps(forest)
    baseline_means = {f: round(float(np.mean(X[:, i])), 6) for i, f in enumerate(feature_names)}
    baseline_stds = {f: round(float(np.std(X[:, i])), 6) for i, f in enumerate(feature_names)}
    n_samples = len(rows)
    trained_at = datetime.datetime.utcnow().isoformat() + "Z"

    upsert_sql = (
        "INSERT INTO model_artifacts "
        "    (model_name, model_type, trained_at, scaler_blob, model_blob, "
        "     feature_names, baseline_means, baseline_stds, n_training_samples) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (model_name) DO UPDATE SET "
        "    model_type = EXCLUDED.model_type, "
        "    trained_at = EXCLUDED.trained_at, "
        "    scaler_blob = EXCLUDED.scaler_blob, "
        "    model_blob = EXCLUDED.model_blob, "
        "    feature_names = EXCLUDED.feature_names, "
        "    baseline_means = EXCLUDED.baseline_means, "
        "    baseline_stds = EXCLUDED.baseline_stds, "
        "    n_training_samples = EXCLUDED.n_training_samples"
    )
    try:
        _db.execute(
            upsert_sql,
            (
                model_name, model_type, trained_at,
                psycopg2.Binary(scaler_blob), psycopg2.Binary(model_blob),
                json.dumps(feature_names), json.dumps(baseline_means),
                json.dumps(baseline_stds), n_samples,
            ),
            pool=db,
        )
        log.info("anomaly: trained model for %s (n=%d features=%s)", model_name, n_samples, feature_names)
    except Exception as exc:
        log.error("anomaly: failed to persist model for %s: %s", model_name, exc)
        raise

    return {
        "status": "ok",
        "model_name": model_name,
        "model_type": model_type,
        "n_training_samples": n_samples,
        "feature_names": feature_names,
        "baseline_means": baseline_means,
        "trained_at": trained_at,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(
    model_name: str,
    db=None,
) -> Optional[Tuple[StandardScaler, IsolationForest, List[str], Dict[str, float]]]:
    row = _db.read_one(
        "SELECT scaler_blob, model_blob, feature_names, baseline_means "
        "FROM model_artifacts WHERE model_name = %s",
        (model_name,),
        pool=db,
    )
    if row is None:
        return None
    scaler: StandardScaler = pickle.loads(bytes(row["scaler_blob"]))
    forest: IsolationForest = pickle.loads(bytes(row["model_blob"]))
    raw_fn = row["feature_names"]
    feature_names: List[str] = json.loads(raw_fn) if isinstance(raw_fn, str) else raw_fn
    raw_bm = row["baseline_means"]
    baseline_means: Dict[str, float] = json.loads(raw_bm) if isinstance(raw_bm, str) else raw_bm
    return scaler, forest, feature_names, baseline_means


# ---------------------------------------------------------------------------
# Cold-start z-score fallback
# ---------------------------------------------------------------------------
def _cold_start_score(
    feature_vec: np.ndarray,
    feature_names: List[str],
) -> Tuple[float, bool, List[Dict[str, Any]]]:
    anomaly_score = -0.1
    is_anomaly = False
    dims: List[Dict[str, Any]] = []
    for i, fname in enumerate(feature_names):
        val = feature_vec[i]
        if not np.isfinite(val):
            continue
        dims.append({"feature": fname, "value": round(float(val), 4), "deviation_pct": 0.0})
    return anomaly_score, is_anomaly, dims


# ---------------------------------------------------------------------------
# Score current window
# ---------------------------------------------------------------------------
def score_current_window(
    model_name: str,
    model_type: str,
    db=None,
) -> Dict[str, Any]:
    feature_names = _feature_names(model_type)
    cols = ", ".join(f"AVG({c}) AS {c}" for c in feature_names)
    query = (
        f"SELECT {cols} FROM inference_metrics "
        f"WHERE model_name = %s AND time >= NOW() - INTERVAL '60 seconds'"
    )
    rows = _db.read_rows(query, (model_name,), pool=db)
    computed_at = datetime.datetime.utcnow().isoformat() + "Z"

    if not rows or all(rows[0].get(f) is None for f in feature_names):
        log.debug("anomaly: no recent data for %s", model_name)
        return {"status": "no_data", "model_name": model_name, "computed_at": computed_at}

    row = rows[0]
    raw_vec = np.array([row.get(f) or 0.0 for f in feature_names], dtype=float)
    raw_vec = np.where(np.isfinite(raw_vec), raw_vec, 0.0)

    artifact = load_model(model_name, db=db)
    cold_start_mode = artifact is None

    if cold_start_mode:
        anomaly_score, is_anomaly, contributing_dims = _cold_start_score(raw_vec, feature_names)
        baseline_means_dict: Dict[str, float] = {f: 0.0 for f in feature_names}
        log.info("anomaly: cold_start mode for %s score=%.4f", model_name, anomaly_score)
    else:
        scaler, forest, feat_names_stored, baseline_means_dict = artifact
        raw_vec_aligned = np.array([row.get(f) or 0.0 for f in feat_names_stored], dtype=float)
        raw_vec_aligned = np.where(np.isfinite(raw_vec_aligned), raw_vec_aligned, 0.0)
        X = raw_vec_aligned.reshape(1, -1)
        X_scaled = scaler.transform(X)
        anomaly_score = float(forest.decision_function(X_scaled)[0])
        prediction = forest.predict(X_scaled)[0]
        is_anomaly = bool(prediction == -1)
        contributing_dims = []
        for i, fname in enumerate(feat_names_stored):
            baseline_val = baseline_means_dict.get(fname, 0.0)
            current_val = float(raw_vec_aligned[i])
            if baseline_val != 0:
                deviation_pct = abs((current_val - baseline_val) / baseline_val) * 100.0
            else:
                deviation_pct = abs(current_val) * 100.0
            contributing_dims.append({
                "feature": fname,
                "value": round(current_val, 4),
                "baseline": round(baseline_val, 4),
                "deviation_pct": round(deviation_pct, 2),
            })
        contributing_dims.sort(key=lambda d: d["deviation_pct"], reverse=True)
        feature_names = feat_names_stored
        log.info(
            "anomaly: %s score=%.4f is_anomaly=%s top_dim=%s",
            model_name, anomaly_score, is_anomaly,
            contributing_dims[0]["feature"] if contributing_dims else "none",
        )

    result: Dict[str, Any] = {
        "status": "ok",
        "model_name": model_name,
        "model_type": model_type,
        "anomaly_score": round(anomaly_score, 6),
        "is_anomaly": is_anomaly,
        "contributing_dimensions": contributing_dims if not cold_start_mode else [],
        "cold_start_mode": cold_start_mode,
        "computed_at": computed_at,
    }
    _write_anomaly_event(result, db=db)
    return result


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------
def _write_anomaly_event(result: Dict[str, Any], db=None) -> None:
    insert_sql = (
        "INSERT INTO anomaly_events "
        "    (computed_at, model_name, anomaly_score, is_anomaly, "
        "     contributing_dimensions, cold_start_mode) "
        "VALUES (%s, %s, %s, %s, %s, %s)"
    )
    try:
        _db.execute(
            insert_sql,
            (
                result["computed_at"], result["model_name"],
                result["anomaly_score"], result["is_anomaly"],
                json.dumps(result["contributing_dimensions"]),
                result["cold_start_mode"],
            ),
            pool=db,
        )
    except Exception as exc:
        log.warning("anomaly: failed to write anomaly_event: %s", exc)


# ---------------------------------------------------------------------------
# Batch runners
# ---------------------------------------------------------------------------
def _get_all_models(db=None) -> List[Dict[str, str]]:
    rows = _db.read_rows(
        "SELECT DISTINCT model_name, model_type FROM inference_metrics ORDER BY model_name",
        pool=db,
    )
    return [{"model_name": r["model_name"], "model_type": r["model_type"]} for r in rows]


def run_all_models_scoring(db=None) -> List[Dict[str, Any]]:
    models = _get_all_models(db=db)
    results = []
    for m in models:
        try:
            result = score_current_window(
                model_name=m["model_name"],
                model_type=m["model_type"],
                db=db,
            )
            results.append(result)
        except Exception as exc:
            log.error("anomaly: scoring failed for %s: %s", m["model_name"], exc, exc_info=True)
    return results


def retrain_all_models(db=None) -> List[Dict[str, Any]]:
    models = _get_all_models(db=db)
    results = []
    for m in models:
        try:
            result = train_model(
                model_name=m["model_name"],
                model_type=m["model_type"],
                db=db,
            )
            results.append(result)
        except Exception as exc:
            log.error("anomaly: retrain failed for %s: %s", m["model_name"], exc, exc_info=True)
    return results


# ---------------------------------------------------------------------------
# Scheduler entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(
        stream=sys.stdout,
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from apscheduler.schedulers.blocking import BlockingScheduler

    scoring_interval_seconds = int(os.environ.get("ANOMALY_SCORE_INTERVAL_SECONDS", "60"))
    retrain_hour_utc = int(os.environ.get("ANOMALY_RETRAIN_HOUR_UTC", "2"))
    log.info(
        "Starting anomaly scheduler (scoring every %ds, retrain at %02d:00 UTC)",
        scoring_interval_seconds, retrain_hour_utc,
    )

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        func=run_all_models_scoring,
        trigger="interval",
        seconds=scoring_interval_seconds,
        id="anomaly_score",
        name="Score all models",
        replace_existing=True,
        misfire_grace_time=30,
    )
    scheduler.add_job(
        func=retrain_all_models,
        trigger="cron",
        hour=retrain_hour_utc,
        minute=0,
        id="anomaly_retrain",
        name="Retrain all models",
        replace_existing=True,
        misfire_grace_time=300,
    )
    try:
        retrain_all_models()
    except Exception as exc:
        log.warning("anomaly: initial retrain skipped: %s", exc)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Anomaly scheduler stopped.")

import logging
import math
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def compute_confidence(
    matched_conditions: int,
    max_conditions: int,
    total_deviation: float,
) -> float:
    if max_conditions <= 0:
        return 0.0
    condition_score = matched_conditions / max_conditions
    deviation_score = min(total_deviation / 100.0, 1.0)
    raw = 0.6 * condition_score + 0.4 * deviation_score
    return round(min(max(raw, 0.0), 1.0), 4)


def _safe_get(d: dict, key: str, default: float = 0.0) -> float:
    val = d.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def analyze(
    model_name: str,
    metrics_snapshot: Dict[str, Any],
    anomaly_event: Dict[str, Any],
    model_type: str = "llm",
) -> Dict[str, Any]:
    kv = _safe_get(metrics_snapshot, "kv_cache_usage_pct")
    rw = _safe_get(metrics_snapshot, "requests_waiting")
    lat = _safe_get(metrics_snapshot, "latency_p95_ms")
    gpu_util = _safe_get(metrics_snapshot, "gpu_util_pct")
    gpu_temp = _safe_get(metrics_snapshot, "gpu_temp_celsius")
    gpu_mem_free = _safe_get(metrics_snapshot, "gpu_mem_free_mb", default=99999.0)
    ttft = _safe_get(metrics_snapshot, "ttft_p95_ms")
    tpot = _safe_get(metrics_snapshot, "tpot_mean_ms")
    rtf = _safe_get(metrics_snapshot, "rtf")
    anomaly_score = _safe_get(anomaly_event, "anomaly_score", default=0.0)

    baseline_lat = _safe_get(metrics_snapshot, "baseline_latency_p95_ms", default=lat or 1.0)
    baseline_ttft = _safe_get(metrics_snapshot, "baseline_ttft_p95_ms", default=ttft or 1.0)
    baseline_tpot = _safe_get(metrics_snapshot, "baseline_tpot_mean_ms", default=tpot or 1.0)

    contributing_dims: List[Dict[str, Any]] = anomaly_event.get("contributing_dimensions", [])
    if isinstance(contributing_dims, str):
        import json
        try:
            contributing_dims = json.loads(contributing_dims)
        except Exception:
            contributing_dims = []

    total_deviation = sum(
        _safe_get(d, "deviation_pct") for d in contributing_dims
    ) if contributing_dims else 0.0

    # -- Rule 1: KV cache saturation --
    if kv > 90.0 and rw > 5:
        evidence = [
            f"KV cache usage at {kv:.1f}% (threshold: 90%)",
            f"{int(rw)} requests queued waiting for cache space",
        ]
        return {
            "anomaly_detected": True,
            "root_cause": "KV cache saturation causing request queuing",
            "confidence": compute_confidence(2, 2, total_deviation),
            "evidence": evidence,
            "recommendation": (
                "Reduce max_model_len or increase gpu_memory_utilization. "
                "Consider enabling prefix caching or adding more GPU replicas."
            ),
            "contributing_dimensions": contributing_dims,
        }

    # -- Rule 2: CPU-side bottleneck --
    if lat > 2.0 * baseline_lat and gpu_util < 30.0:
        evidence = [
            f"p95 latency {lat:.1f}ms is {lat/max(baseline_lat,1):.1f}x baseline ({baseline_lat:.1f}ms)",
            f"GPU utilisation low at {gpu_util:.1f}% indicating CPU/IO bottleneck",
        ]
        return {
            "anomaly_detected": True,
            "root_cause": "CPU-side bottleneck: tokenisation or pre/post-processing delay",
            "confidence": compute_confidence(2, 2, total_deviation),
            "evidence": evidence,
            "recommendation": (
                "Profile CPU usage in the vLLM process. Consider increasing CPU resources, "
                "checking tokeniser performance, or pinning inference threads."
            ),
            "contributing_dimensions": contributing_dims,
        }

    # -- Rule 3: Model at full GPU capacity --
    if rw > 20.0 and gpu_util > 90.0:
        evidence = [
            f"{int(rw)} requests waiting (threshold: 20)",
            f"GPU utilisation saturated at {gpu_util:.1f}%",
        ]
        return {
            "anomaly_detected": True,
            "root_cause": "Model at full GPU capacity: request throughput exceeds serving capacity",
            "confidence": compute_confidence(2, 2, total_deviation),
            "evidence": evidence,
            "recommendation": (
                "Scale horizontally by adding more replicas. "
                "Consider enabling tensor parallelism or deploying to additional GPUs."
            ),
            "contributing_dimensions": contributing_dims,
        }

    # -- Rule 4: Prefill bottleneck --
    tpot_deviation = abs(tpot - baseline_tpot) / max(baseline_tpot, 1e-6)
    if ttft > 2.0 * baseline_ttft and tpot_deviation < 0.2:
        evidence = [
            f"TTFT p95 {ttft:.1f}ms is {ttft/max(baseline_ttft,1):.1f}x baseline ({baseline_ttft:.1f}ms)",
            f"TPOT stable at {tpot:.1f}ms (deviation {tpot_deviation*100:.1f}% < 20%)",
        ]
        return {
            "anomaly_detected": True,
            "root_cause": "Prefill bottleneck: long prompts or chunked prefill queue congestion",
            "confidence": compute_confidence(2, 2, total_deviation),
            "evidence": evidence,
            "recommendation": (
                "Enable chunked prefill (--enable-chunked-prefill). "
                "Consider prompt caching or reducing max prompt length limits."
            ),
            "contributing_dimensions": contributing_dims,
        }

    # -- Rule 5: GPU thermal throttling --
    if gpu_temp > 85.0:
        evidence = [
            f"GPU temperature {gpu_temp:.1f}°C exceeds 85°C thermal threshold",
        ]
        return {
            "anomaly_detected": True,
            "root_cause": "GPU thermal throttling reducing effective compute throughput",
            "confidence": compute_confidence(1, 1, total_deviation),
            "evidence": evidence,
            "recommendation": (
                "Check datacenter cooling. Consider reducing GPU power limit (nvidia-smi -pl) "
                "or throttling request rate until temperature normalises."
            ),
            "contributing_dimensions": contributing_dims,
        }

    # -- Rule 6: VRAM near exhaustion --
    if gpu_mem_free < 2048.0:
        evidence = [
            f"GPU free VRAM only {gpu_mem_free:.0f} MB remaining (critical threshold: 2048 MB)",
        ]
        return {
            "anomaly_detected": True,
            "root_cause": "VRAM near exhaustion causing OOM risk and potential evictions",
            "confidence": compute_confidence(1, 1, total_deviation),
            "evidence": evidence,
            "recommendation": (
                "Reduce gpu_memory_utilization fraction, enable KV cache offloading, "
                "or restart the model server to reclaim fragmented memory."
            ),
            "contributing_dimensions": contributing_dims,
        }

    # -- Rule 7: TTS RTF exceeded --
    if model_type == "tts" and rtf > 1.0:
        evidence = [
            f"Real-Time Factor {rtf:.3f} > 1.0 (model generating slower than real-time)",
        ]
        return {
            "anomaly_detected": True,
            "root_cause": "TTS Real-Time Factor exceeded: synthesis speed below real-time requirement",
            "confidence": compute_confidence(1, 1, total_deviation),
            "evidence": evidence,
            "recommendation": (
                "Reduce batch size or concurrent TTS requests. "
                "Consider quantisation (INT8/FP8) or a faster TTS model variant."
            ),
            "contributing_dimensions": contributing_dims,
        }

    # -- Rule 8: Multi-dimensional anomaly (catch-all) --
    if anomaly_score < -0.15:
        top_dims = sorted(contributing_dims, key=lambda d: _safe_get(d, "deviation_pct"), reverse=True)[:3]
        dim_names = ", ".join(d.get("feature", "unknown") for d in top_dims)
        evidence = [
            f"IsolationForest anomaly score {anomaly_score:.4f} below critical threshold (-0.15)",
            f"Top deviating dimensions: {dim_names}",
        ]
        return {
            "anomaly_detected": True,
            "root_cause": "Multi-dimensional anomaly: correlated degradation across multiple metrics",
            "confidence": compute_confidence(1, 2, total_deviation),
            "evidence": evidence,
            "recommendation": (
                "Investigate correlated metric changes. Check for recent deployments, "
                "traffic pattern changes, or infrastructure events in the anomaly timeframe."
            ),
            "contributing_dimensions": contributing_dims,
        }

    # -- No rule matched: no anomaly detected --
    return {
        "anomaly_detected": False,
        "root_cause": None,
        "confidence": 0.0,
        "evidence": [],
        "recommendation": "No anomaly patterns detected in the current metrics window.",
        "contributing_dimensions": contributing_dims,
    }

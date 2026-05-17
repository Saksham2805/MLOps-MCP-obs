
import os
import sys
import time
import signal
import logging
import datetime
from typing import Any, Dict, Optional

import yaml
import requests
import psycopg2
import psycopg2.pool
from prometheus_client.parser import text_string_to_metric_families
from kubernetes import client as k8s_client, config as k8s_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("scraper")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_SHUTDOWN = False
_previous_counters: Dict[str, Dict[str, float]] = {}  # model_name -> {counter_name -> value}


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------
def _handle_sigterm(signum, frame):
    global _SHUTDOWN
    log.info("Received SIGTERM – shutting down gracefully.")
    _SHUTDOWN = True


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    # Apply env-var overrides
    collector = cfg.setdefault("collector", {})
    if os.environ.get("TIMESCALEDB_URL"):
        collector["timescaledb_url"] = os.environ["TIMESCALEDB_URL"]
    if os.environ.get("DCGM_ENDPOINT"):
        collector["dcgm_endpoint"] = os.environ["DCGM_ENDPOINT"]
    if os.environ.get("SCRAPE_INTERVAL"):
        collector["scrape_interval_seconds"] = int(os.environ["SCRAPE_INTERVAL"])
    return cfg


# ---------------------------------------------------------------------------
# DB pool
# ---------------------------------------------------------------------------
def make_pool(dsn: str) -> psycopg2.pool.ThreadedConnectionPool:
    return psycopg2.pool.ThreadedConnectionPool(1, 5, dsn=dsn)


def db_execute(pool: psycopg2.pool.ThreadedConnectionPool, query: str, params: tuple) -> None:
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ---------------------------------------------------------------------------
# Prometheus metric parsing helpers
# ---------------------------------------------------------------------------
def parse_prometheus_text(text: str) -> Dict[str, Any]:
    """Return flat dict: metric_name -> list of (labels_dict, value)."""
    result: Dict[str, list] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            result.setdefault(sample.name, []).append((sample.labels, sample.value))
    return result


def _quantile_value(samples: list, quantile: float) -> Optional[float]:
    """Extract a specific quantile value from a list of (labels, value) tuples."""
    for labels, value in samples:
        q = labels.get("quantile")
        if q is not None and abs(float(q) - quantile) < 1e-6:
            return value
    return None


def _sum_value(samples: list) -> Optional[float]:
    """Return the _sum sample value (usually the first entry with no quantile label)."""
    for labels, value in samples:
        if "quantile" not in labels:
            return value
    return None


def _count_value(samples: list) -> Optional[float]:
    for labels, value in samples:
        if "quantile" not in labels:
            return value
    return None


# ---------------------------------------------------------------------------
# vLLM metrics scraping
# ---------------------------------------------------------------------------
def scrape_vllm(endpoint: str, model_name: str, model_cfg: dict, interval: float) -> dict:
    """Scrape a single vLLM /metrics endpoint and return a flat metrics dict."""
    resp = requests.get(endpoint, timeout=10)
    resp.raise_for_status()
    parsed = parse_prometheus_text(resp.text)

    def get(metric_base: str):
        # vLLM metric names may have _bucket / _sum / _count variants.
        # Try exact name first, then with common suffixes.
        for candidate in [
            metric_base,
            metric_base + "_sum",
            metric_base.replace(":", "_"),
        ]:
            if candidate in parsed:
                return parsed[candidate]
        return []

    # --- Latency histogram (e2e_request_latency_seconds) ---
    e2e_samples = parsed.get("vllm:e2e_request_latency_seconds", [])
    latency_p50_s = _quantile_value(e2e_samples, 0.5)
    latency_p95_s = _quantile_value(e2e_samples, 0.95)
    latency_p99_s = _quantile_value(e2e_samples, 0.99)

    # --- TTFT ---
    ttft_samples = parsed.get("vllm:time_to_first_token_seconds", [])
    ttft_p50_s = _quantile_value(ttft_samples, 0.5)
    ttft_p95_s = _quantile_value(ttft_samples, 0.95)

    # --- TPOT (time per output token – use _sum/_count for mean) ---
    tpot_sum_samples = parsed.get("vllm:time_per_output_token_seconds_sum", [])
    tpot_count_samples = parsed.get("vllm:time_per_output_token_seconds_count", [])
    tpot_sum = None
    tpot_count = None
    # Also check directly as histogram samples with no quantile
    tpot_all = parsed.get("vllm:time_per_output_token_seconds", [])
    for labels, value in tpot_all:
        if "quantile" not in labels:
            tpot_sum = value  # summary _sum variant
    for labels, value in tpot_sum_samples:
        tpot_sum = value
        break
    for labels, value in tpot_count_samples:
        tpot_count = value
        break
    tpot_mean_s: Optional[float] = None
    if tpot_sum is not None and tpot_count is not None and tpot_count > 0:
        tpot_mean_s = tpot_sum / tpot_count

    # --- Queue / cache metrics ---
    def scalar(metric_name: str) -> Optional[float]:
        samples = parsed.get(metric_name, [])
        if samples:
            return samples[0][1]
        return None

    requests_running = scalar("vllm:num_requests_running")
    requests_waiting = scalar("vllm:num_requests_waiting")
    kv_cache_usage_pct = scalar("vllm:gpu_cache_usage_perc")
    if kv_cache_usage_pct is not None and kv_cache_usage_pct <= 1.0:
        kv_cache_usage_pct = kv_cache_usage_pct * 100.0  # normalise to 0-100

    # --- Counters (cumulative) ---
    request_success_total = scalar("vllm:request_success_total")
    prompt_tokens_total = scalar("vllm:request_prompt_tokens_total")
    generation_tokens_total = scalar("vllm:request_generation_tokens_total")

    # --- Derive throughput_rps via delta ---
    prev = _previous_counters.setdefault(model_name, {})
    throughput_rps: Optional[float] = None
    if request_success_total is not None:
        prev_val = prev.get("request_success_total")
        if prev_val is not None and request_success_total >= prev_val:
            delta = request_success_total - prev_val
            throughput_rps = delta / interval
        prev["request_success_total"] = request_success_total

    # --- RTF for TTS models ---
    rtf: Optional[float] = None
    tts_tokens_per_sec = model_cfg.get("tts_tokens_per_second_of_audio")
    if model_cfg.get("model_type") == "tts" and tts_tokens_per_sec and tts_tokens_per_sec > 0:
        if tpot_mean_s is not None and tpot_mean_s > 0:
            # tokens_per_second_generated = 1 / tpot_mean_s
            # RTF = audio_seconds_per_token / (1/tts_tokens_per_sec)
            # RTF > 1 means slower than real-time
            tokens_per_second_generated = 1.0 / tpot_mean_s
            rtf = tts_tokens_per_sec / tokens_per_second_generated

    return {
        "latency_p50_ms": latency_p50_s * 1000.0 if latency_p50_s is not None else None,
        "latency_p95_ms": latency_p95_s * 1000.0 if latency_p95_s is not None else None,
        "latency_p99_ms": latency_p99_s * 1000.0 if latency_p99_s is not None else None,
        "ttft_p50_ms": ttft_p50_s * 1000.0 if ttft_p50_s is not None else None,
        "ttft_p95_ms": ttft_p95_s * 1000.0 if ttft_p95_s is not None else None,
        "tpot_mean_ms": tpot_mean_s * 1000.0 if tpot_mean_s is not None else None,
        "requests_running": requests_running,
        "requests_waiting": requests_waiting,
        "kv_cache_usage_pct": kv_cache_usage_pct,
        "request_success_total": request_success_total,
        "prompt_tokens_total": prompt_tokens_total,
        "generation_tokens_total": generation_tokens_total,
        "throughput_rps": throughput_rps,
        "rtf": rtf,
    }


# ---------------------------------------------------------------------------
# DCGM metrics scraping
# ---------------------------------------------------------------------------
def scrape_dcgm(dcgm_endpoint: str, models: list) -> Dict[str, dict]:
    """Scrape DCGM exporter and return per-model GPU metrics keyed by model name."""
    try:
        resp = requests.get(dcgm_endpoint, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("DCGM scrape failed: %s", exc)
        return {}

    parsed = parse_prometheus_text(resp.text)
    node_selector_to_model: Dict[str, str] = {
        m["gpu_node_selector"]: m["name"] for m in models if m.get("gpu_node_selector")
    }

    def extract_for_node(metric_name: str, node: str) -> Optional[float]:
        samples = parsed.get(metric_name, [])
        for labels, value in samples:
            # DCGM labels: Hostname, GPU_I_ID, etc.
            hostname = labels.get("Hostname", labels.get("hostname", ""))
            if node in hostname:
                return value
        return None

    result: Dict[str, dict] = {}
    for node_selector, model_name in node_selector_to_model.items():
        result[model_name] = {
            "gpu_util_pct": extract_for_node("DCGM_FI_DEV_GPU_UTIL", node_selector),
            "gpu_mem_used_mb": extract_for_node("DCGM_FI_DEV_FB_USED", node_selector),
            "gpu_mem_free_mb": extract_for_node("DCGM_FI_DEV_FB_FREE", node_selector),
            "gpu_temp_celsius": extract_for_node("DCGM_FI_DEV_GPU_TEMP", node_selector),
            "gpu_power_watts": extract_for_node("DCGM_FI_DEV_POWER_USAGE", node_selector),
        }
    return result


# ---------------------------------------------------------------------------
# Kubernetes pod metrics
# ---------------------------------------------------------------------------
def scrape_k8s_pod_metrics(namespace: str, models: list) -> Dict[str, dict]:
    """Query the Kubernetes metrics API for pod CPU and memory."""
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        try:
            k8s_config.load_kube_config()
        except Exception as exc:
            log.warning("K8s config load failed: %s", exc)
            return {}

    custom_api = k8s_client.CustomObjectsApi()
    try:
        pod_metrics = custom_api.list_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace=namespace,
            plural="pods",
        )
    except Exception as exc:
        log.warning("K8s pod metrics API failed: %s", exc)
        return {}

    model_names = [m["name"] for m in models]
    result: Dict[str, dict] = {}

    for pod in pod_metrics.get("items", []):
        pod_name = pod["metadata"]["name"]
        matched_model = None
        for mname in model_names:
            # Match by model name substring in pod name (case-insensitive)
            if mname.lower().replace("-", "") in pod_name.lower().replace("-", ""):
                matched_model = mname
                break
        if matched_model is None:
            continue

        total_cpu_millicores = 0.0
        total_mem_bytes = 0.0
        for container in pod.get("containers", []):
            usage = container.get("usage", {})
            cpu_str = usage.get("cpu", "0")
            mem_str = usage.get("memory", "0")
            # CPU: may be in "250m" (millicores) or "1" (cores)
            if cpu_str.endswith("n"):
                total_cpu_millicores += float(cpu_str[:-1]) / 1_000_000.0
            elif cpu_str.endswith("u"):
                total_cpu_millicores += float(cpu_str[:-1]) / 1_000.0
            elif cpu_str.endswith("m"):
                total_cpu_millicores += float(cpu_str[:-1])
            else:
                total_cpu_millicores += float(cpu_str) * 1000.0
            # Memory: may be in Ki, Mi, Gi
            if mem_str.endswith("Ki"):
                total_mem_bytes += float(mem_str[:-2]) * 1024
            elif mem_str.endswith("Mi"):
                total_mem_bytes += float(mem_str[:-2]) * 1024 * 1024
            elif mem_str.endswith("Gi"):
                total_mem_bytes += float(mem_str[:-2]) * 1024 * 1024 * 1024
            else:
                total_mem_bytes += float(mem_str)

        existing = result.get(matched_model, {"pod_cpu_millicores": 0.0, "pod_mem_mb": 0.0})
        result[matched_model] = {
            "pod_cpu_millicores": existing["pod_cpu_millicores"] + total_cpu_millicores,
            "pod_mem_mb": existing["pod_mem_mb"] + total_mem_bytes / (1024 * 1024),
        }

    return result


# ---------------------------------------------------------------------------
# DB INSERT
# ---------------------------------------------------------------------------
INSERT_SQL = """
INSERT INTO inference_metrics (
    time, model_name, model_type,
    latency_p50_ms, latency_p95_ms, latency_p99_ms,
    ttft_p50_ms, ttft_p95_ms, tpot_mean_ms,
    throughput_rps,
    prompt_tokens_total, generation_tokens_total, request_success_total,
    requests_running, requests_waiting,
    kv_cache_usage_pct,
    gpu_util_pct, gpu_mem_used_mb, gpu_mem_free_mb,
    gpu_temp_celsius, gpu_power_watts,
    pod_cpu_millicores, pod_mem_mb,
    rtf
) VALUES (
    %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s,
    %s,
    %s, %s, %s,
    %s, %s,
    %s,
    %s, %s, %s,
    %s, %s,
    %s, %s,
    %s
)
"""


def insert_metrics(pool, ts, model_name, model_type, vllm_metrics, dcgm_metrics, k8s_metrics):
    dcgm = dcgm_metrics.get(model_name, {})
    k8s = k8s_metrics.get(model_name, {})
    params = (
        ts,
        model_name,
        model_type,
        vllm_metrics.get("latency_p50_ms"),
        vllm_metrics.get("latency_p95_ms"),
        vllm_metrics.get("latency_p99_ms"),
        vllm_metrics.get("ttft_p50_ms"),
        vllm_metrics.get("ttft_p95_ms"),
        vllm_metrics.get("tpot_mean_ms"),
        vllm_metrics.get("throughput_rps"),
        vllm_metrics.get("prompt_tokens_total"),
        vllm_metrics.get("generation_tokens_total"),
        vllm_metrics.get("request_success_total"),
        vllm_metrics.get("requests_running"),
        vllm_metrics.get("requests_waiting"),
        vllm_metrics.get("kv_cache_usage_pct"),
        dcgm.get("gpu_util_pct"),
        dcgm.get("gpu_mem_used_mb"),
        dcgm.get("gpu_mem_free_mb"),
        dcgm.get("gpu_temp_celsius"),
        dcgm.get("gpu_power_watts"),
        k8s.get("pod_cpu_millicores"),
        k8s.get("pod_mem_mb"),
        vllm_metrics.get("rtf"),
    )
    db_execute(pool, INSERT_SQL, params)


# ---------------------------------------------------------------------------
# Main scrape loop
# ---------------------------------------------------------------------------
def run_scrape_loop(config_path: str) -> None:
    cfg = load_config(config_path)
    collector_cfg = cfg["collector"]
    models = cfg["models"]
    interval = float(collector_cfg.get("scrape_interval_seconds", 15))
    dsn = collector_cfg["timescaledb_url"]
    dcgm_endpoint = collector_cfg.get("dcgm_endpoint", "")
    k8s_namespace = collector_cfg.get("k8s_namespace", "inference")

    log.info("Starting scraper. Models: %s  Interval: %ss", [m["name"] for m in models], interval)

    pool = make_pool(dsn)
    log.info("DB pool created.")

    while not _SHUTDOWN:
        loop_start = time.monotonic()
        ts = datetime.datetime.utcnow()

        # Scrape DCGM once per loop (shared across models)
        dcgm_metrics = scrape_dcgm(dcgm_endpoint, models) if dcgm_endpoint else {}

        # Scrape K8s pod metrics once per loop
        try:
            k8s_metrics = scrape_k8s_pod_metrics(k8s_namespace, models)
        except Exception as exc:
            log.warning("K8s pod metrics scrape error: %s", exc)
            k8s_metrics = {}

        for model_cfg in models:
            model_name = model_cfg["name"]
            model_type = model_cfg.get("model_type", "llm")
            endpoint = model_cfg["vllm_endpoint"]
            try:
                vllm_metrics = scrape_vllm(endpoint, model_name, model_cfg, interval)
            except Exception as exc:
                log.error("vLLM scrape failed for %s: %s", model_name, exc)
                continue

            try:
                insert_metrics(pool, ts, model_name, model_type, vllm_metrics, dcgm_metrics, k8s_metrics)
                log.info(
                    "Inserted metrics for %s: latency_p95=%.1fms tput=%.2frps",
                    model_name,
                    vllm_metrics.get("latency_p95_ms") or 0.0,
                    vllm_metrics.get("throughput_rps") or 0.0,
                )
            except Exception as exc:
                log.error("DB insert failed for %s: %s", model_name, exc)

        elapsed = time.monotonic() - loop_start
        sleep_time = max(0.0, interval - elapsed)
        log.debug("Loop took %.2fs, sleeping %.2fs", elapsed, sleep_time)
        # Interruptible sleep
        deadline = time.monotonic() + sleep_time
        while not _SHUTDOWN and time.monotonic() < deadline:
            time.sleep(0.5)

    log.info("Scraper shut down cleanly.")
    pool.closeall()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config_path = os.environ.get("CONFIG_PATH", "/etc/llm-obs/config.yaml")
    run_scrape_loop(config_path)

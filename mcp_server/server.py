import logging
import os
import sys
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_server import db as _db
from mcp_server.tools import health as _health
from mcp_server.tools import drift as _drift
from mcp_server.tools import root_cause as _root_cause
from mcp_server.tools import compare as _compare
from mcp_server.tools import alerts as _alerts
from mcp_server.tools import forecast as _forecast

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("llm-obs-mcp")

mcp = FastMCP("llm-obs-mcp")


@mcp.tool()
async def get_model_health(
    model_name: Optional[str] = None,
    time_window: str = "15m",
) -> Dict[str, Any]:
    """
    Get current health status for one or all LLM/TTS models.

    Args:
        model_name: Specific model to query (None = all models)
        time_window: Aggregation window. One of: 5m, 15m, 1h, 6h, 24h

    Returns:
        Dict with "models" list, each containing status, metrics, anomaly, and drift_severity.
    """
    return await _health.get_model_health(model_name=model_name, time_window=time_window)


@mcp.tool()
async def detect_drift(
    model_name: str,
    metric: str = "prompt_tokens",
    reference_window: str = "7d",
    comparison_window: str = "1h",
) -> Dict[str, Any]:
    """
    Detect statistical drift for a model metric using KS-test and PSI.

    Args:
        model_name: Exact model name to analyse.
        metric: One of: prompt_tokens, completion_tokens, latency, throughput, rtf
        reference_window: Historical baseline window (e.g. 7d, 14d)
        comparison_window: Recent window to compare against baseline (e.g. 1h, 6h)

    Returns:
        Dict with severity, ks_statistic, psi, drift_pct, interpretation, recommendation.
    """
    return await _drift.detect_drift(
        model_name=model_name,
        metric=metric,
        reference_window=reference_window,
        comparison_window=comparison_window,
    )


@mcp.tool()
async def get_root_cause(
    model_name: str,
    lookback_minutes: int = 30,
) -> Dict[str, Any]:
    """
    Run root cause analysis on the most recent anomaly for a model.

    Args:
        model_name: Model to analyse.
        lookback_minutes: How many minutes back to search for anomalies (default: 30).

    Returns:
        Dict with root_cause, confidence, evidence[], recommendation, contributing_dimensions[].
    """
    return await _root_cause.get_root_cause(
        model_name=model_name,
        lookback_minutes=lookback_minutes,
    )


@mcp.tool()
async def compare_models(
    model_a: str,
    model_b: str,
    metric: str = "latency_p95_ms",
    time_window: str = "24h",
) -> Dict[str, Any]:
    """
    Compare two models on a given metric using statistical tests.

    Args:
        model_a: First model name.
        model_b: Second model name.
        metric: Metric to compare. One of: latency_p95_ms, latency_p99_ms,
                ttft_p95_ms, tpot_mean_ms, throughput_rps, kv_cache_usage_pct,
                requests_waiting, gpu_util_pct, rtf.
        time_window: Window to compare over. One of: 5m, 15m, 1h, 6h, 24h.

    Returns:
        Dict with per-model stats, Welch t-test result, difference_pct, and summary.
    """
    return await _compare.compare_models(
        model_a=model_a,
        model_b=model_b,
        metric=metric,
        time_window=time_window,
    )


@mcp.tool()
async def get_alerts(
    model_name: Optional[str] = None,
    severity: Optional[str] = None,
    include_resolved: bool = False,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Retrieve active alerts from the anomaly detector and Alertmanager.

    Args:
        model_name: Filter by model (None = all models).
        severity: Filter by severity: critical, warning (None = all).
        include_resolved: Whether to include resolved alerts.
        limit: Maximum number of alerts to return (default: 20).

    Returns:
        Dict with total_active, total_critical, and alerts list.
    """
    return await _alerts.get_alerts(
        model_name=model_name,
        severity=severity,
        include_resolved=include_resolved,
        limit=limit,
    )


@mcp.tool()
async def predict_saturation(
    model_name: str,
    metric: str = "requests_waiting",
    horizon_hours: int = 2,
) -> Dict[str, Any]:
    """
    Predict when a model metric will saturate its threshold using time-series forecasting.

    Args:
        model_name: Model to forecast.
        metric: Metric to forecast. One of: requests_waiting, kv_cache_usage_pct,
                gpu_util_pct, rtf.
        horizon_hours: Forecast horizon in hours (1-6, default: 2).

    Returns:
        Dict with saturation_time, saturation_risk, forecast_points[], recommendation.
    """
    return await _forecast.predict_saturation(
        model_name=model_name,
        metric=metric,
        horizon_hours=horizon_hours,
    )


app = mcp.streamable_http_app()


@app.route("/health", methods=["GET"])
async def health_endpoint(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def _on_startup() -> None:
    pool = await _db.get_pool()
    log.info("MCP server startup: asyncpg pool ready (%s)", pool)
    registered = [t.name for t in mcp._tool_manager.list_tools()]
    log.info("Registered MCP tools (%d): %s", len(registered), registered)


async def _on_shutdown() -> None:
    await _db.close_pool()
    log.info("MCP server shutdown: pool closed")


app.add_event_handler("startup", _on_startup)
app.add_event_handler("shutdown", _on_shutdown)


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8080"))

    if transport == "stdio":
        log.info("Starting MCP server with stdio transport")
        import asyncio
        from mcp.server.stdio import stdio_server
        async def _run_stdio():
            async with stdio_server() as (read_stream, write_stream):
                await mcp._mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp._mcp_server.create_initialization_options(),
                )
        asyncio.run(_run_stdio())
    else:
        import uvicorn
        log.info("Starting MCP server on %s:%d (transport=streamable-http)", host, port)
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
        )

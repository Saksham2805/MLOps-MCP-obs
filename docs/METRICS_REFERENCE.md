# Metrics Reference

Every metric collected by the scraper, its source, what it means, and the alert threshold used by the anomaly detector and Alertmanager.

---

## vLLM Metrics (all models)

Source: `http://<vllm-pod>:8000/metrics` (Prometheus text format)

| Metric Name | Internal Field | Unit | What It Means |
|---|---|---|---|
| `vllm:e2e_request_latency_seconds` | `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms` | ms | End-to-end time from request received to last token sent. Includes queue wait + prefill + decode. The primary user-facing SLA metric. |
| `vllm:time_to_first_token_seconds` | `ttft_p50_ms`, `ttft_p95_ms` | ms | Time from request received to first token streamed back. Dominated by prompt prefill cost. Key for streaming UX. |
| `vllm:time_per_output_token_seconds` | `tpot_ms` | ms | Average time to generate each output token after the first. Dominated by GPU decode throughput. |
| `vllm:request_success_total` | (derived) `throughput_rps` | req/s | Requests completed per second. Computed as delta over the scrape interval. |
| `vllm:request_prompt_tokens_total` | `prompt_tokens_total` | tokens | Cumulative prompt tokens processed. Delta used to track input load. |
| `vllm:request_generation_tokens_total` | `completion_tokens_total` | tokens | Cumulative output tokens generated. |
| `vllm:num_requests_running` | `requests_running` | count | Requests actively being processed (prefill or decode). |
| `vllm:num_requests_waiting` | `requests_waiting` | count | Requests in the queue, waiting for a slot. Non-zero = model at capacity. |
| `vllm:gpu_cache_usage_perc` | `kv_cache_usage_pct` | % | Fraction of KV cache blocks currently occupied. When this hits 90%+, vLLM starts evicting older sequences, causing latency spikes. Critical for GPT-OSS 20B. |
| `vllm:gpu_prefix_cache_hit_rate` | (logged, not stored) | % | Cache hit rate for prefix sharing. Informational only. |

### Per-model notes

**GPT-OSS 20B:**
- `kv_cache_usage_pct` is the primary risk metric. At 20B parameters with large context windows, KV cache fills quickly under concurrent load.
- Watch `requests_waiting` and `ttft_p95_ms` together. If waiting > 5 and TTFT spikes, you have a scheduling bottleneck.

**Gemma 4:**
- Generally more cache-efficient than 20B. Watch `tpot_ms` - if generation slows without GPU util increase, suspect memory bandwidth saturation.

**Svara TTS:**
- Standard LLM latency metrics still apply (vLLM serves TTS via its API).
- Additional derived metric: `rtf` (Real-Time Factor) = audio_duration_generated / wall_clock_time. RTF < 1.0 means faster than real-time. RTF > 1.0 means the system cannot keep up.
- RTF is computed by the scraper from `completion_tokens_total` using the model's known tokens-per-second-of-audio mapping.

---

## NVIDIA DCGM Metrics (GPU-level)

Source: `http://<dcgm-exporter>:9400/metrics`
Label `modelName` mapped to vLLM pod via node selector in config.yaml.

| DCGM Metric | Internal Field | Unit | What It Means |
|---|---|---|---|
| `DCGM_FI_DEV_GPU_UTIL` | `gpu_util_pct` | % | SM (compute core) utilization. High = GPU busy. Low during high latency = CPU/memory bottleneck, not GPU. |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | (logged) | % | PCIe memory copy utilization. High = data transfer bottleneck. |
| `DCGM_FI_DEV_FB_USED` | `gpu_mem_used_mb` | MB | Framebuffer (VRAM) used. Model weights + KV cache + activations. |
| `DCGM_FI_DEV_FB_FREE` | `gpu_mem_free_mb` | MB | VRAM headroom. If this drops to < 2GB, OOM risk is high. |
| `DCGM_FI_DEV_GPU_TEMP` | `gpu_temp_celsius` | C | GPU die temperature. Sustained > 85C triggers throttling. |
| `DCGM_FI_DEV_POWER_USAGE` | `gpu_power_watts` | W | Power draw. Approaches TDP under full load. Sustained at TDP with low GPU util = memory-bound workload. |

---

## K8s metrics-server Metrics (pod-level)

Source: K8s metrics API (`/apis/metrics.k8s.io/v1beta1/namespaces/{ns}/pods/{pod}`)

| Field | Unit | What It Means |
|---|---|---|
| `pod_cpu_millicores` | m (millicores) | CPU consumed by the pod. vLLM tokenizer and scheduler are CPU-bound. High CPU + high latency = tokenizer bottleneck. |
| `pod_mem_mb` | MB | Pod RAM (system memory, not GPU VRAM). High values may indicate KV cache overflow to CPU RAM (slow path). |

---

## Alert Thresholds

Used by Prometheus Alertmanager rules and as reference inputs for the Isolation Forest baseline.

| Metric | Warning Threshold | Critical Threshold | Notes |
|---|---|---|---|
| `latency_p95_ms` | > 2x 7-day p95 baseline | > 3x baseline | Relative, not absolute. Different for each model. |
| `requests_waiting` | > 5 | > 20 | Absolute. Any waiting queue is a UX signal. |
| `kv_cache_usage_pct` | > 75% | > 90% | 90% triggers active eviction in vLLM. |
| `gpu_util_pct` | < 20% during high latency | - | Low GPU + high latency = not a GPU problem. |
| `gpu_mem_free_mb` | < 4096 MB | < 2048 MB | OOM prevention. |
| `gpu_temp_celsius` | > 80 C | > 87 C | Throttle risk. |
| `rtf` (TTS only) | > 0.8 | > 1.0 | RTF > 1.0 = system cannot produce audio in real time. |
| PSI (drift score) | > 0.1 | > 0.25 | PSI thresholds from industry standard. |
| Anomaly score | < -0.05 | < -0.15 | Isolation Forest decision_function output. More negative = more anomalous. |

---

## Derived Metrics (computed by the collector, not from raw scrape)

| Derived Field | Formula | Why Useful |
|---|---|---|
| `throughput_rps` | `delta(request_success_total) / scrape_interval` | Requests per second over last window |
| `avg_prompt_tokens` | `delta(prompt_tokens_total) / delta(request_success_total)` | Average input size trend |
| `avg_completion_tokens` | `delta(completion_tokens_total) / delta(request_success_total)` | Average output size trend |
| `gpu_mem_utilization_pct` | `gpu_mem_used_mb / (gpu_mem_used_mb + gpu_mem_free_mb) * 100` | Total VRAM pressure % |
| `rtf` (TTS) | `delta(completion_tokens_total) / tokens_per_second_of_audio / scrape_interval` | Real-Time Factor for audio generation |

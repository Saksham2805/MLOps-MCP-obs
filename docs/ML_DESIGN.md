# ML Design

This document covers the mathematical foundation, design decisions, and implementation rationale for every ML component in `llm-obs-mcp`.

---

## 1. Drift Detection

### 1.1 What is drift in the context of LLM inference?

Drift means the statistical distribution of a metric has changed compared to a historical baseline. In LLM serving, the most consequential drifts are:

- **Input drift**: users start sending much longer or shorter prompts. This directly changes KV cache pressure and prefill cost.
- **Latency drift**: the latency distribution shifts, indicating the model is handling requests differently (different load, different prompt types, infrastructure changes).
- **Throughput drift**: requests per second changes, signalling load pattern changes.
- **RTF drift** (TTS only): Real-Time Factor shifts, indicating audio generation speed is changing relative to playback speed.

Drift is insidious because it is gradual. It does not trigger threshold alerts. By the time you notice it, users have already been experiencing degraded quality for hours or days.

---

### 1.2 Kolmogorov-Smirnov (KS) Test

**What it measures:** Whether two samples come from the same underlying distribution.

**Formula:**

```
D = sup_x | F_ref(x) - F_curr(x) |
```

Where:
- `F_ref(x)` = empirical CDF of the reference window sample
- `F_curr(x)` = empirical CDF of the current window sample
- `D` = the KS statistic (maximum absolute difference between the two CDFs)
- `sup` = supremum (maximum over all x)

**Interpretation:**
- Large D = the two distributions differ in shape significantly
- The associated p-value tests H0: "both samples come from the same distribution"
- p-value < 0.05 = reject H0 = distributions are significantly different

**Why KS over a simple mean comparison?**
The mean can stay the same while the distribution shape changes dramatically. Example: if the baseline had mostly short prompts (50 tokens) with occasional long ones (2000 tokens), and the current window has mostly medium prompts (500 tokens), the means might be similar but the distributions are completely different. KS catches this because it compares the full shape of the CDF.

**Implementation:**
```python
from scipy import stats

def ks_test(reference_samples, current_samples):
    stat, p_value = stats.ks_2samp(reference_samples, current_samples)
    return {
        "ks_statistic": round(stat, 4),
        "p_value": round(p_value, 4),
        "drift_detected": p_value < 0.05
    }
```

---

### 1.3 Population Stability Index (PSI)

**What it measures:** How much a distribution has shifted across bucketed bins, expressed as a single score.

**Formula:**

```
PSI = sum_i [ (Actual_i - Expected_i) * ln(Actual_i / Expected_i) ]
```

Where:
- `Expected_i` = fraction of reference window samples in bucket i
- `Actual_i` = fraction of current window samples in bucket i
- Buckets are fixed quantile bins from the reference distribution (10 bins by default)

**Interpretation thresholds (industry standard):**

| PSI Score | Severity | Meaning |
|---|---|---|
| < 0.1 | None | No significant drift |
| 0.1 to 0.2 | Low | Minor drift, continue monitoring |
| 0.2 to 0.25 | Medium | Moderate drift, investigate |
| > 0.25 | High | Major drift, action required |

**Why PSI in addition to KS?**
KS gives a yes/no signal (with p-value). PSI gives a continuous score that is stable and interpretable over time. You can plot PSI as a time series and see drift happening gradually. PSI is the metric you show in dashboards and trend reports. KS is the metric you use for statistical significance testing.

**Implementation:**
```python
import numpy as np

def compute_psi(reference, current, n_bins=10):
    # Define bins from the reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Compute bin frequencies
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    curr_counts, _ = np.histogram(current, bins=breakpoints)

    # Convert to proportions, add small epsilon to avoid log(0)
    eps = 1e-6
    ref_pct = (ref_counts / len(reference)) + eps
    curr_pct = (curr_counts / len(current)) + eps

    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return round(float(psi), 4)
```

---

## 2. Anomaly Detection

### 2.1 Why Isolation Forest?

Anomaly detection on inference metrics has three constraints:

1. **No labeled data** - we do not have historical records of "this was an anomaly" / "this was normal". We only have metric time-series.
2. **Multi-dimensional** - a single metric being high is often not an anomaly. What matters is the unusual *combination*: high latency + low GPU util is different from high latency + high GPU util.
3. **Fast inference** - the detector runs every 60 seconds, must not add meaningful latency.

Isolation Forest satisfies all three:
- **Unsupervised**: trained only on normal data, no labels needed
- **Multi-dimensional**: naturally handles feature vectors of arbitrary dimensionality
- **Fast**: O(n log n) training, O(log n) inference per sample

### 2.2 How Isolation Forest Works

The core intuition: **anomalies are easier to isolate than normal points.**

An Isolation Tree is built by randomly selecting a feature and a random split value. This recursively partitions the data. Anomalies (points in sparse regions of the feature space) get isolated in fewer splits than normal points (which are in dense regions).

The anomaly score for a sample is the average path length across a forest of such trees, normalized:

```
score(x) = 2^(-E[h(x)] / c(n))
```

Where:
- `E[h(x)]` = average path length for sample x across all trees
- `c(n)` = expected path length for a sample in a dataset of size n (normalization)
- Score close to 1 = anomaly (isolated quickly)
- Score close to 0.5 = normal (takes many splits to isolate)

The `decision_function` output (what we store) is: `score - 0.5`, shifted so:
- Positive values = normal
- Negative values = anomalous
- More negative = more anomalous

### 2.3 Feature Vector

The Isolation Forest is trained and scored on 4-dimensional feature vectors:

```python
features = [
    latency_p95_ms,       # User-facing performance
    gpu_util_pct,         # GPU compute utilization
    kv_cache_usage_pct,   # Memory pressure
    requests_waiting      # Queue depth
]
```

For TTS (Svara), `kv_cache_usage_pct` is replaced with `rtf` (Real-Time Factor).

All features are standardized (zero mean, unit variance) using the reference window statistics before training and scoring.

### 2.4 Training Schedule

- **Initial training**: on 7 days of baseline data after the collector has run for 7+ days
- **Nightly retraining**: rolls the 7-day window forward, retrains from scratch
- **Contamination parameter**: set to `0.05` (we assume up to 5% of the training window may include mild anomalies)
- **Model persistence**: serialized with `joblib` and stored in the TimescaleDB `model_artifacts` table as a BLOB, versioned by model_name and training timestamp

### 2.5 Identifying Contributing Dimensions

Isolation Forest itself does not explain *which* features drove the anomaly score. We use a post-hoc method:

For each anomalous point, compute the per-feature deviation from the baseline mean:
```python
deviation_pct = abs(current_value - baseline_mean) / baseline_mean * 100
```

Sort dimensions by deviation_pct descending. The top 2-3 dimensions are reported as `contributing_dimensions` in the MCP response. This gives the rule-based correlator its inputs.

---

## 3. Predictive Saturation Forecasting

### 3.1 Why Prophet?

Prophet (by Meta/Facebook Research) is designed for business time-series that exhibit:
- **Trend**: gradual growth or decline over time
- **Seasonality**: daily and weekly patterns (API traffic always has these)
- **Holidays / events**: sudden spikes and troughs (deployments, traffic events)
- **Missing data**: gaps in the time-series (pod restarts, maintenance windows)

LLM inference traffic has all of these properties. Prophet handles them without manual feature engineering.

Alternative considered: ARIMA. Rejected because ARIMA requires stationarity, is sensitive to missing data, and does not handle multiple seasonality periods well without significant manual tuning.

### 3.2 Model Setup

```python
from prophet import Prophet

model = Prophet(
    changepoint_prior_scale=0.05,   # Conservative: don't overfit to recent spikes
    seasonality_prior_scale=10,     # Strong weekly and daily seasonality expected
    interval_width=0.80,            # 80% confidence interval in output
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False        # Insufficient data for yearly patterns
)
```

**Input data format:**
```python
# Prophet expects a DataFrame with columns 'ds' (datetime) and 'y' (value)
df = pd.DataFrame({
    'ds': timestamps,        # UTC timestamps from TimescaleDB
    'y': metric_values       # e.g. requests_waiting, kv_cache_usage_pct
})
```

### 3.3 Forecast Horizon and Granularity

- Training window: rolling 14 days of 15-minute aggregated data points
- Forecast horizon: up to 6 hours ahead (24 prediction points at 15-min granularity)
- Reforecast frequency: every 30 minutes
- Each model has its own Prophet instance per metric

### 3.4 Saturation Detection from Forecast

After getting the forecast DataFrame from Prophet:

```python
def find_saturation_time(forecast_df, threshold):
    # forecast_df has columns: ds, yhat, yhat_lower, yhat_upper
    breaching = forecast_df[forecast_df['yhat'] >= threshold]
    if breaching.empty:
        return None, 'low'
    first_breach = breaching.iloc[0]['ds']
    # If upper bound (pessimistic) also breaches -> high risk
    upper_breaching = forecast_df[forecast_df['yhat_upper'] >= threshold]
    risk = 'high' if not upper_breaching.empty else 'medium'
    return first_breach.isoformat(), risk
```

---

## 4. Root Cause Correlator

### 4.1 Design Philosophy

The correlator is intentionally **not** an ML model. It is a deterministic rule engine.

Reason: ML anomaly detection tells you *that* something is unusual. It does not tell you *why* in human-interpretable terms. A random forest trained to classify root causes would require labeled training data we do not have, and its outputs would be harder to trust and debug than explicit rules.

The hybrid approach:
- **Isolation Forest**: detects anomalies and identifies contributing dimensions (data-driven)
- **Rule engine**: maps dimension patterns to domain-specific diagnoses (expert knowledge)

This is maintainable, explainable, and correct for a system where the failure modes are well-understood (KV cache saturation, GPU capacity limits, CPU bottlenecks, thermal throttling).

### 4.2 Rule Confidence Scoring

Each rule has an associated confidence score based on how many conditions match and how strongly:

```python
def compute_confidence(matched_conditions, total_evidence_strength):
    # matched_conditions: number of conditions in the rule that fired
    # total_evidence_strength: sum of deviation percentages for contributing dims
    base_confidence = matched_conditions / max_conditions_in_rule
    evidence_boost = min(total_evidence_strength / 500, 0.2)  # cap boost at 0.2
    return min(base_confidence + evidence_boost, 0.99)
```

This gives outputs like `confidence: 0.91` for strong matches and `confidence: 0.52` for weak or ambiguous ones, giving the agent and user a signal about how much to trust the diagnosis.

---

## 5. Design Decisions Summary

| Decision | Choice | Rejected Alternative | Reason |
|---|---|---|---|
| Drift statistical test | KS test + PSI (both) | Only one test | KS for significance, PSI for interpretable trend. Complementary. |
| Drift binning | Quantile bins from reference | Equal-width bins | Equal-width bins are sensitive to outliers. Quantile bins are stable. |
| Anomaly model | Isolation Forest | One-Class SVM, Autoencoder | Isolation Forest is faster, requires no hyperparameter tuning per model, no GPU needed. |
| Anomaly features | 4 dimensions | All available metrics | More dimensions without feature selection causes the curse of dimensionality. 4 dimensions are the most operationally significant. |
| Forecasting model | Prophet | ARIMA, LSTM | Prophet handles multiple seasonality and missing data without manual tuning. LSTM requires GPU and labeled sequences. ARIMA requires stationarity. |
| Root cause | Rule-based correlator | ML classifier | No labeled root cause data exists. Rules are explainable, maintainable, and immediately correct. |
| Anomaly retraining | Nightly, rolling 7-day window | Online learning | Online learning risks contaminating the model with anomalous data. Nightly batch retrain is simple and reliable. |

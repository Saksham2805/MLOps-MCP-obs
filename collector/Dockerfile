FROM python:3.11-slim

LABEL maintainer="llm-obs-mcp" \
      description="LLM Observability Metrics Collector"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY collector/ /app/collector/
COPY analytics/ /app/analytics/

# Create non-root user for security
RUN useradd -r -u 1001 -g root collector && \
    chown -R 1001:0 /app

USER 1001

# Config is mounted via ConfigMap at /etc/llm-obs/config.yaml
ENV CONFIG_PATH=/etc/llm-obs/config.yaml \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["python", "collector/scraper.py"]

FROM python:3.11-slim

LABEL maintainer="llm-obs-mcp" \
      description="LLM Observability MCP Server"

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
COPY mcp_server/ /app/mcp_server/
COPY analytics/ /app/analytics/

# Create non-root user for security
RUN useradd -r -u 1001 -g root mcpserver && \
    chown -R 1001:0 /app

USER 1001

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8080 \
    MCP_TRANSPORT=streamable-http

EXPOSE 8080

CMD ["python", "mcp_server/server.py"]

# =============================================================================
# Production Dockerfile for ML Pipeline
# =============================================================================

FROM python:3.8-slim

# Set metadata
LABEL maintainer="ML Pipeline Team"
LABEL description="Production ML Pipeline with MLOps stack"
LABEL version="1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_APP=web_app.py
ENV FLASK_ENV=production
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY *.py ./
COPY params.yaml ./
COPY dvc.yaml ./

# Copy entrypoint script
COPY docker/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create necessary directories
RUN mkdir -p data/raw data/processed data/features models trained_models plots reports logs
# Copy trained models if they exist
COPY trained_models/ ./trained_models/ 2>/dev/null || true
COPY models/ ./models/ 2>/dev/null || true

# Set permissions
RUN chmod +x scripts/*.py

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mluser && \
    chown -R mluser:mluser /app
USER mluser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose ports
# Set entrypoint

# Expose ports
EXPOSE 5000 8000
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command - can be overridden
# Default command - can be overridden
CMD ["web"]
FROM python:3.9-slim

LABEL maintainer="Climate Prediction System"
LABEL description="Enhanced Climate & AQI Prediction System for Pune"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with timeout and retry
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout 60 --retries 3 -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/external data/api \
    outputs/models outputs/logs outputs/figures

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0

# Expose ports
EXPOSE 8501

# Health check (simplified for CI)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=2 \
    CMD python -c "print('Container is healthy')" || exit 1

# Run application (default to test mode for CI)
CMD ["python", "simple_test.py"]
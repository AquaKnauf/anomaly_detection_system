# ==================================
# Dockerfile
# ==================================

# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TF_CPP_MIN_LOG_LEVEL=3

# Install system dependencies including ps command for TensorRT
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    procps \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with specific order to avoid conflicts
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install TensorFlow without TensorRT to avoid the error
RUN pip3 install --no-cache-dir tensorflow==2.15.0

# Install other requirements (excluding tensorflow and torch)
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python3", "anomaly_detector.py"]

# ==================================
# requirements.txt
# ==================================



# ==================================
# docker-compose.yml
# ==================================


# ==================================
# prometheus.yml
# ==================================


# ==================================
# Kubernetes Deployment Files
# ==================================

# namespace.yaml



# configmap.yaml


# secrets.yaml



# postgres-deployment.yaml



# postgres-service.yaml



# postgres-pvc.yaml



# redis-deployment.yaml



# redis-service.yaml



# redis-pvc.yaml



# kafka-deployment.yaml



# kafka-service.yaml



# zookeeper-deployment.yaml



# zookeeper-service.yaml



# anomaly-detector-deployment.yaml



# anomaly-detector-service.yaml



# anomaly-detector-hpa.yaml



# model-pvc.yaml



# prometheus-deployment.yaml



# prometheus-service.yaml



# prometheus-configmap.yaml



# prometheus-pvc.yaml



# grafana-deployment.yaml



# grafana-service.yaml



# grafana-pvc.yaml

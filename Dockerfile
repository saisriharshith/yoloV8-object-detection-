# ==============================================================================
# YOLOv8 Real-Time Edge Analytics - Production Dockerfile for Render Deployment
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

WORKDIR /app

# Install system dependencies:
# - ffmpeg: robust audio/video decoding for YouTube HLS and network streams
# - libgl1 & libglib2.0-0: required by OpenCV for frame manipulation & encoding
# - build-essential: C/C++ compiler for lap / ByteTrack Cython extensions
# - curl: container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-optimized PyTorch first (shrinks Docker image from ~5GB to ~900MB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure working directories and cache YOLOv8n model weights during build
RUN mkdir -p data/inputs data/outputs models && \
    python3 -c "from ultralytics import YOLO; YOLO('models/yolov8n.pt')"

# Expose default port
EXPOSE 8000

# Healthcheck configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/ || exit 1

# Launch using dynamic Render PORT variable
CMD ["sh", "-c", "python3 run_app.py --host 0.0.0.0 --port ${PORT:-8000}"]

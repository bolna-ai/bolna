# =============================================================================
# Kallabot v2 — Multi-stage Dockerfile
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder — install Python dependencies into a virtual-env
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /build

# System packages needed to compile native extensions (e.g. scipy, onnxruntime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first so this layer is cached unless deps change
COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Install the bolna package itself (editable is not needed in the image)
COPY pyproject.toml README.md LICENSE ./
COPY bolna/ bolna/
RUN pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Stage 2: runtime — lean image with only what we need to run
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

WORKDIR /app

# Runtime system libraries (ffmpeg for audio processing, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtual-env from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application source code
COPY bolna/  bolna/
COPY app/    app/
COPY db/     db/
COPY services/   services/
COPY extensions/ extensions/

# Expose the application port
EXPOSE 8001

# Healthcheck — hit the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the FastAPI app via uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]

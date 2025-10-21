FROM python:3.10.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY telephony_server/requirements.txt /app/requirements.txt

# Install Python dependencies with BuildKit caching
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy application files
COPY telephony_server/plivo_api_server.py /app/

EXPOSE 8002

CMD ["uvicorn", "plivo_api_server:app", "--host", "0.0.0.0", "--port", "8002"]

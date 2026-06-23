FROM python:3.10.13-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY telephony_server/requirements.txt /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

COPY telephony_server/twilio_api_server.py /app/

EXPOSE 8001


CMD ["uvicorn", "twilio_api_server:app", "--host", "0.0.0.0", "--port", "8001"]

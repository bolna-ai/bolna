FROM python:3.10.13-slim

WORKDIR /app

RUN apt-get update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
    libgomp1 \
    git \
    ffmpeg \
    gcc \
    g++
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install git+https://github.com/bolna-ai/bolna@master
COPY quickstart_server.py /app/
COPY presets /app/presets

EXPOSE 5001

CMD ["uvicorn", "quickstart_server:app", "--host", "0.0.0.0", "--port", "5001"]

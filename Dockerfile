# If build containers cannot resolve domains, use:
#   docker build --network=host -t cfbox-api:latest .

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    HOME=/home/appuser \
    CFBOX_BACKEND=onnx \
    CFBOX_WIDGET_WEIGHTS=/models/onnx/widget.onnx \
    CFBOX_STATE_WEIGHTS=/models/onnx/state.onnx \
    CFBOX_MODEL_MANIFEST=/models/onnx/manifest.json

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-serve.txt

COPY cfbox_detector.py backends.py app.py ./
COPY LICENSE THIRD_PARTY_NOTICES.md ./
COPY models/onnx /models/onnx

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /models

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

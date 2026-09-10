# dlib is the only native dependency left, so it is compiled once in a builder
# stage and the toolchain is left behind.  The previous tensorflow:2.8.0-gpu
# base existed solely for DeepFace; nothing in the service uses TensorFlow now.
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
# insightface still builds through a legacy setup.py that expects these to be
# importable already, so they go in before the wheel pass.
RUN pip install --no-cache-dir "cython<3" "numpy<2"
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


FROM python:3.11-slim

# Runtime halves of what dlib was linked against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Bake the ArcFace weights in.  Left to first use, four Swarm replicas would
# each fetch ~300MB on the same cold start.
RUN python -c "from insightface.app import FaceAnalysis;     FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'],                  allowed_modules=['detection','recognition']).prepare(ctx_id=-1, det_size=(640,640))"

COPY . .

# Face templates live outside the image so they survive a redeploy.
ENV FACE_DB_PATH=/data/faces.db
RUN mkdir -p /data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

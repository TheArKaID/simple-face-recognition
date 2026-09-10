# Nothing here compiles any more.  The image once carried TensorFlow, CUDA and
# DeepFace (~7GB), then dlib and the insightface package (2.2GB); what remains is
# onnxruntime driving two ONNX files, which is most of the way back down.
#
# The builder stage exists only to keep the wheel cache and the weight download
# out of the final layers.
FROM python:3.11-slim AS builder

# build-essential is insurance: every dependency currently ships a manylinux
# wheel for cp311, so nothing is built from source, but a future dependency
# without one would fail the build rather than fall back.  It costs build time
# only - multi-stage means it never reaches the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Two files out of the buffalo_l pack.  Fetching them here rather than at first
# use matters for Swarm: four replicas on a cold start would otherwise each pull
# ~300MB at once.  The other three models in the pack - 3D landmarks, 2D
# landmarks, age/gender - are never loaded, so they are not extracted.
RUN mkdir -p /models/arcface \
    && curl -fsSL -o /tmp/buffalo_l.zip \
        https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
    && unzip -j /tmp/buffalo_l.zip '*det_10g.onnx' '*w600k_r50.onnx' -d /models/arcface \
    && rm /tmp/buffalo_l.zip \
    && ls -l /models/arcface


FROM python:3.11-slim

# onnxruntime uses OpenMP.  dlib's libopenblas is gone along with dlib itself.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
# onnxruntime declares sympy (and its mpmath) as a hard dependency but only
# reaches for them in shape-inference and transformer tooling, never in
# InferenceSession.  Verified by loading w600k_r50 and running a forward pass
# with both gone: ~100MB of image for no loss of function.
#
# These comments sit ABOVE the RUN on purpose.  A '#' line inside a backslash
# continuation ends the instruction and silently discards the rest - an earlier
# version put them mid-chain and the uninstall never ran, while the build still
# reported success.
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels \
    && pip uninstall -y sympy mpmath \
    && find /usr/local/lib/python3.11/site-packages -name __pycache__ -type d -exec rm -rf {} + || true

COPY --from=builder /models/arcface /app/models/arcface

COPY . .

# Face templates live outside the image so they survive a redeploy.
ENV FACE_DB_PATH=/data/faces.db
RUN mkdir -p /data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

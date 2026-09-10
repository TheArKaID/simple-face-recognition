# tools

One-off utilities. Neither runs in production, and neither is installed into
the runtime image.

## convert_minifas.py — anti-spoofing weights, PyTorch to ONNX

The MiniFASNet weights ship as `.pth`, but the service runs them through
onnxruntime, which is already present for InsightFace. Converting once in a
throwaway container keeps torch — several hundred megabytes — out of the image
entirely.

Source: [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
by minivision-ai. Third-party weights, fetched at conversion time rather than
vendored, so the provenance stays visible.

```sh
cat > /tmp/conv.sh <<'SH'
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq --no-install-recommends git
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir onnxscript onnx onnxruntime numpy
git clone --depth 1 https://github.com/minivision-ai/Silent-Face-Anti-Spoofing /repo
python /tools/convert_minifas.py
SH

docker run --rm \
  -v "$PWD/tools:/tools" \
  -v "$PWD/models/liveness:/out" \
  -v /tmp/conv.sh:/conv.sh \
  python:3.11-slim bash /conv.sh
```

It writes two `.onnx` files into `models/liveness/` and checks each against the
torch model it came from — a max absolute difference above `1e-4` means the
export is wrong and the file should not be used.

That check covers the export only. Whether the surrounding preprocessing in
`liveness.py` matches what the model expects is a separate question, and the
only thing that answers it is `measure_liveness.py` below.

## measure_liveness.py — is the liveness check worth anything?

```sh
docker exec <container> python /app/tools/measure_liveness.py
```

Needs presentation attacks in `tests/images/spoof/`: a phone or laptop screen
showing one of the faces in `tests/images/`, photographed as an employee would
hold it. Five to ten is enough to be informative.

Without them the script stops instead of printing anything, which is
deliberate. A liveness threshold picked without spoof samples is worse than no
liveness at all, because the service then reports a score that nobody has
reason to believe. Set `FACE_LIVENESS_MIN_SCORE` from its threshold sweep, and
weigh the two error types unequally: a refused live employee retakes a photo,
an accepted spoof records attendance that never happened.

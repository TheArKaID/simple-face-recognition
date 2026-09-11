"""Does INT8 keep the decisions, and does it actually run faster?

Size is the easy part and never the question.  Neither, it turns out, is drift.

Quantisation shifts every embedding, but it shifts them in a correlated way, so
relative distances partly survive - and the distances are what decisions are
made from.  A drift threshold still cannot tell a harmless shift from a harmful
one, and an earlier version of this script rejected a usable model for exactly
that reason.

Measured on tests/new-images (30 identities, 298 photos), separation gap:
FP32 0.108 -> recogniser INT8 0.092 -> both INT8 0.077.  Quantising costs gap
in both cases; the detector step costs almost as much as the recogniser step
while buying a quarter as much time.

An earlier note here claimed the recogniser INT8 WIDENED the gap from 0.105 to
0.117.  That was measured on the old 12-identity fixture and did not survive
30 identities - with ~3.6x more impostor pairs the closest one lands closer,
and the apparent widening was small-sample luck.  Gap figures from a small
fixture do not transfer; re-measure on the full set before trusting any of
them.

So this script measures drift and latency, and tests/calibrate.py decides.  Run
this to see where a change came from and what it costs in time; run calibrate to
see whether the separation gap survives.

    docker exec <c> env QUANT_REC=w600k_r50_int8.onnx \
        python /app/tools/compare_quantized.py

QUANT_DET and QUANT_REC each default to the FP32 file, so quantising one at a
time isolates which model a change belongs to.
"""
import glob
import os
import statistics
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

BASE_DET = os.getenv("BASE_DET", "det_2.5g.onnx")
BASE_REC = os.getenv("BASE_REC", "w600k_r50.onnx")
QUANT_DET = os.getenv("QUANT_DET", BASE_DET)
QUANT_REC = os.getenv("QUANT_REC", BASE_REC)

IMAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "tests", "images")


def pipeline(det_name, rec_name):
    """A fresh backend bound to one detector/recogniser pair."""
    import importlib

    config.ARCFACE_DET_MODEL = det_name
    config.ARCFACE_REC_MODEL = rec_name
    for mod in ("engines.arcface_onnx",):
        sys.modules.pop(mod, None)
    import engines
    if hasattr(engines, "arcface_onnx"):
        delattr(engines, "arcface_onnx")
    ax = importlib.import_module("engines.arcface_onnx")
    assert ax.DET_MODEL == det_name and ax.REC_MODEL == rec_name, (
        f"backend bound to {ax.DET_MODEL}/{ax.REC_MODEL}, wanted {det_name}/{rec_name}"
    )
    return ax


def run(ax, paths):
    from engines.common import downscale
    embs, det_ms, rec_ms, missing = {}, [], [], []
    for p in paths:
        stem = os.path.basename(p)
        img = downscale(Image.open(p).convert("RGB"))
        bgr = np.array(img)[:, :, ::-1].copy()
        t0 = time.perf_counter()
        boxes, scores, kpss = ax._detect(bgr)
        t1 = time.perf_counter()
        if len(boxes) == 0:
            missing.append(stem)
            continue
        i = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
        embs[stem] = ax._embed_face(bgr, kpss[i])
        t2 = time.perf_counter()
        det_ms.append((t1 - t0) * 1000)
        rec_ms.append((t2 - t1) * 1000)
    return embs, det_ms, rec_ms, missing


paths = sorted(glob.glob(os.path.join(IMAGES, "*.jpg")))
print(f"  baseline : {BASE_DET} + {BASE_REC}")
print(f"  quantised: {QUANT_DET} + {QUANT_REC}")
print(f"  photos   : {len(paths)}")

ax = pipeline(BASE_DET, BASE_REC)
base_emb, base_det, base_rec, base_missing = run(ax, paths)
ax = pipeline(QUANT_DET, QUANT_REC)
q_emb, q_det, q_rec, q_missing = run(ax, paths)

shared = [k for k in base_emb if k in q_emb]
drift = sorted((float(1 - np.dot(base_emb[k], q_emb[k])), k) for k in shared)

print()
print(f"  {'tahap':<22}{'FP32':>12}{'INT8':>12}{'speedup':>10}")
print(f"  {'deteksi (median)':<22}{statistics.median(base_det):>9.1f} ms"
      f"{statistics.median(q_det):>9.1f} ms"
      f"{statistics.median(base_det)/statistics.median(q_det):>9.2f}x")
print(f"  {'pengenalan (median)':<22}{statistics.median(base_rec):>9.1f} ms"
      f"{statistics.median(q_rec):>9.1f} ms"
      f"{statistics.median(base_rec)/statistics.median(q_rec):>9.2f}x")
total_b = statistics.median(base_det) + statistics.median(base_rec)
total_q = statistics.median(q_det) + statistics.median(q_rec)
print(f"  {'gabungan':<22}{total_b:>9.1f} ms{total_q:>9.1f} ms{total_b/total_q:>9.2f}x")

if base_missing or q_missing:
    print(f"\n  wajah tidak terdeteksi - FP32: {base_missing or '-'}, INT8: {q_missing or '-'}")

print(f"\n  drift embedding pada {len(shared)} foto:")
print(f"    maks   {drift[-1][0]:.5f}  ({drift[-1][1]})")
print(f"    p95    {drift[int(len(drift)*0.95)][0]:.5f}")
print(f"    median {statistics.median(d for d, _ in drift):.5f}")
print("    5 terbesar:")
for d, k in reversed(drift[-5:]):
    print(f"      {d:.5f}  {k}")

print("")
print(f"  margin impostor terkonfigurasi: {config.MIN_IMPOSTOR_MARGIN}")
print("")
print("  Drift is a diagnostic here, NOT the decision.  An earlier version of")
print("  this script failed a quantised model for exceeding a drift threshold,")
print("  and that was the wrong gate: quantisation shifts every embedding in a")
print("  correlated way, so relative distances partly survive.  Drift cannot")
print("  tell a harmless shift from a harmful one - only the gap can.")
print("")
print("  On tests/new-images (30 identities): gap 0.108 FP32, 0.092 with the")
print("  recogniser quantised, 0.077 with both.  Quantising always cost gap.")
print("")
print("  What decides adoption is the separation gap between the worst genuine")
print("  pair and the closest impostor pair, which comes from tests/calibrate.py:")
print("")
print(f"      docker exec <c> env FACE_ARCFACE_DET={QUANT_DET} \\\\")
print(f"          FACE_ARCFACE_REC={QUANT_REC} python /app/tests/calibrate.py")
print("")
print("  Read `separation` there and compare it against the FP32 baseline.  Use")
print("  the drift figures above to localise where a change came from, not to")
print("  accept or reject it.")

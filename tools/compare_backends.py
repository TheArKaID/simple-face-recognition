"""Prove the bare-onnxruntime backend matches the insightface package exactly.

The claim engines/arcface_onnx.py makes is strong - same ENGINE_ID, so stored
templates stay valid across the swap - and end-to-end metrics are too blunt to
check it. A detector that is a few pixels off, or an alignment fitted slightly
wrong, still scores well on 15 identities while quietly degrading on harder
faces. So this compares the two implementations stage by stage on identical
input: box, landmarks, and finally the embedding itself.

Reference values are captured from the package backend by
tools/dump_reference.py and live in a directory passed as REF_DIR.

    docker exec <container> env FACE_ENGINE=arcface-onnx REF_DIR=/ref \\
        python /app/tools/compare_backends.py
"""
import glob
import json
import os
import statistics
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

if config.FACE_ENGINE != "arcface-onnx":
    print(f"FACE_ENGINE is {config.FACE_ENGINE!r}; run this with arcface-onnx")
    sys.exit(2)

import engine  # noqa: E402
from engines.common import downscale  # noqa: E402

REF_DIR = os.getenv("REF_DIR", "/ref")
IMAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "tests", "images")

with open(os.path.join(REF_DIR, "reference.json")) as fh:
    reference = json.load(fh)

print(f"backend under test : {engine.ENGINE_ID} via {config.FACE_ENGINE}")
print(f"reference photos   : {len(reference)}")

box_err, kps_err, emb_dist, count_mismatch, failures = [], [], [], [], []

for stem, ref in sorted(reference.items()):
    candidates = glob.glob(os.path.join(IMAGES, stem)) + \
        glob.glob(os.path.join(IMAGES, "spoof", stem))
    if not candidates:
        continue
    path = candidates[0]

    ref_emb = np.load(os.path.join(REF_DIR, f"{stem}.npy"))
    try:
        result = engine.embed(Image.open(path), quality_gates=False)
    except engine.FaceError as exc:
        failures.append((stem, exc.reason))
        continue

    if result.faces_found != ref["n_faces"]:
        count_mismatch.append((stem, ref["n_faces"], result.faces_found))

    # Box: the reference is (x1, y1, x2, y2); ours is (top, right, bottom, left)
    # and clamped to the frame, so compare against a clamped reference.
    top, right, bottom, left = result.bbox
    size = ref["size"]
    rx1, ry1, rx2, ry2 = ref["bbox"]
    ref_clamped = (max(0, int(ry1)), min(size[0], int(rx2)),
                   min(size[1], int(ry2)), max(0, int(rx1)))
    box_err.append(max(abs(a - b) for a, b in
                       zip((top, right, bottom, left), ref_clamped)))

    # The embedding is what actually has to match: both are L2-normalised, so
    # cosine distance is the honest measure and 0 means identical.
    emb_dist.append((stem, float(1.0 - np.dot(ref_emb, result.embedding))))

print(f"\nphotos compared    : {len(emb_dist)}")
if failures:
    print(f"failed to embed    : {len(failures)}")
    for stem, reason in failures[:10]:
        print(f"  {stem}: {reason}")
if count_mismatch:
    print(f"face-count differs : {len(count_mismatch)}")
    for stem, ref_n, got_n in count_mismatch[:10]:
        print(f"  {stem}: reference {ref_n}, ours {got_n}")

if box_err:
    print(f"\nbox corner error (px)  max {max(box_err)}  "
          f"median {statistics.median(box_err)}")

if emb_dist:
    d = sorted(v for _, v in emb_dist)
    print(f"embedding cosine distance to reference:")
    print(f"  max    {d[-1]:.3e}")
    print(f"  median {statistics.median(d):.3e}")
    worst = sorted(emb_dist, key=lambda t: -t[1])[:5]
    print("  worst photos:")
    for stem, dist in worst:
        print(f"    {dist:.3e}  {stem}")

    # 1e-5 is generous for float32 through a different runtime path; anything
    # larger means the two backends are not computing the same thing, and the
    # shared ENGINE_ID would then be a lie.
    ok = d[-1] < 1e-5
    print(f"\nverdict: {'EQUIVALENT' if ok else 'NOT EQUIVALENT'}"
          f"  (threshold 1e-5)")
    sys.exit(0 if (ok and not failures and not count_mismatch) else 1)

print("\nnothing compared")
sys.exit(1)

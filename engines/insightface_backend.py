"""InsightFace / ArcFace backend - the candidate being evaluated against dlib.

Uses the buffalo_l pack: an SCRFD detector for boxes plus five-point landmarks,
and w600k_r50 for the 512-d ArcFace embedding. The reference package is used
rather than driving the ONNX files directly, deliberately: the question being
answered is whether ArcFace separates these faces better than dlib, and a
hand-rolled detector or alignment step would confound that answer with bugs of
my own. If it wins, reimplementing on bare onnxruntime is a later, separately
verifiable step - tests/calibrate.py would have to reproduce the same numbers.

Embeddings are L2-normalised, so cosine distance is 1 - dot product. That runs
0 (identical) to 2 (opposite), keeping the "lower means more similar" contract
the decision logic relies on - but the numbers share no scale with dlib's
Euclidean distances, so thresholds must be recalibrated from scratch.
"""
from __future__ import annotations

import os
import threading

import numpy as np
from PIL import Image

import config
from engines.common import (
    FaceError,
    FaceResult,
    apply_quality_gates,
    crop_metrics,
    downscale,
    select_subject,
)

ENGINE_ID = "insightface-buffalo-l"
EMBEDDING_DIM = 512

MODEL_PACK = os.getenv("FACE_INSIGHTFACE_PACK", "buffalo_l")
DET_SIZE = int(os.getenv("FACE_INSIGHTFACE_DET_SIZE", 640))
# SCRFD reports a detection score; a low one usually means something that is
# only face-shaped.  dlib's HOG offers no equivalent, so this gate exists only
# on this backend.
MIN_DET_SCORE = float(os.getenv("FACE_INSIGHTFACE_MIN_DET_SCORE", 0.5))

_app = None
_lock = threading.Lock()


def _app_instance():
    """Build the model once, lazily, and share it across request threads."""
    global _app
    if _app is None:
        with _lock:
            if _app is None:
                from insightface.app import FaceAnalysis

                app = FaceAnalysis(
                    name=MODEL_PACK,
                    providers=["CPUExecutionProvider"],
                    # Landmarks and the age/gender heads are dead weight here.
                    allowed_modules=["detection", "recognition"],
                )
                app.prepare(ctx_id=-1, det_size=(DET_SIZE, DET_SIZE))
                _app = app
    return _app


def embed(image: Image.Image, quality_gates: bool = True) -> FaceResult:
    """Detect the face in `image` and return its template plus quality metrics.

    Raises FaceError with the same reason codes the dlib backend uses, so the
    API contract does not change when the engine does.
    """
    image = downscale(image)
    # insightface follows OpenCV's channel order.
    bgr = np.array(image)[:, :, ::-1].copy()

    faces = _app_instance().get(bgr)
    if not faces:
        raise FaceError("no_face", "No face detected in the image")

    faces_found = len(faces)

    def area(f):
        x1, y1, x2, y2 = f.bbox
        return (x2 - x1) * (y2 - y1)

    primary_i, other_i = select_subject([area(f) for f in faces])
    face = faces[primary_i]

    x1, y1, x2, y2 = (int(v) for v in face.bbox)
    # Clamp: SCRFD boxes can extend past the frame on faces near an edge.
    left, top = max(0, x1), max(0, y1)
    right, bottom = min(image.width, x2), min(image.height, y2)
    if right <= left or bottom <= top:
        raise FaceError("no_face", "Detected face box falls outside the image")

    face_pixels = min(bottom - top, right - left)
    blur_variance, brightness = crop_metrics(image, (top, right, bottom, left))

    if quality_gates:
        if float(face.det_score) < MIN_DET_SCORE:
            raise FaceError(
                "low_confidence",
                f"Detection score {float(face.det_score):.2f} below {MIN_DET_SCORE}",
            )
        apply_quality_gates(face_pixels, blur_variance, brightness)

    embedding = getattr(face, "normed_embedding", None)
    if embedding is None:
        raise FaceError("encoding_failed", "Face detected but no embedding was produced")

    # app.get() already embedded every face it found, so bystanders come free.
    others = [
        np.asarray(faces[i].normed_embedding, dtype=np.float32)
        for i in other_i
        if getattr(faces[i], "normed_embedding", None) is not None
    ]

    return FaceResult(
        embedding=np.asarray(embedding, dtype=np.float32),
        others=others,
        faces_found=faces_found,
        face_pixels=int(face_pixels),
        blur_variance=blur_variance,
        brightness=brightness,
    )


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two templates.  Lower means more similar."""
    return float(1.0 - np.dot(a, b))


def distances(matrix: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """Distance from `probe` to every row of `matrix`.  Lower means more similar."""
    if matrix.size == 0:
        return np.empty(0, dtype=np.float32)
    return 1.0 - matrix @ probe

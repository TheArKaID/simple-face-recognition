"""dlib / face_recognition backend - the engine that has been in production.

Kept intact as the comparison baseline; see tests/baseline_dlib-resnet-v1.json
for the numbers it scores on tests/images.  Distances are Euclidean over 128
dimensions, so they are NOT comparable to another backend's numbers - only the
scale-free metrics in tests/calibrate.py are.
"""
from __future__ import annotations

from typing import List, Tuple

import face_recognition
import numpy as np
from PIL import Image

import config
from engines.common import (
    FaceError,
    FaceResult,
    apply_quality_gates,
    crop_metrics,
    downscale,
)

ENGINE_ID = "dlib-resnet-v1"
EMBEDDING_DIM = 128


def _largest(boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    # boxes are (top, right, bottom, left)
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))


def embed(image: Image.Image, quality_gates: bool = True) -> FaceResult:
    """Detect the face in `image` and return its template plus quality metrics.

    Raises FaceError with a stable reason code when the image is unusable.
    With `quality_gates` false, blur/brightness/size are measured but not enforced.
    """
    image = downscale(image)
    array = np.array(image)

    boxes = face_recognition.face_locations(
        array, number_of_times_to_upsample=config.UPSAMPLE
    )
    if not boxes:
        raise FaceError("no_face", "No face detected in the image")

    faces_found = len(boxes)
    if faces_found > 1 and not config.ALLOW_MULTIPLE_FACES:
        # Previously the first detection won by accident.  For an attendance
        # gate an extra face in frame (a bystander, or a phone held up showing
        # someone else) has to fail closed rather than be silently picked.
        raise FaceError("multiple_faces", f"{faces_found} faces detected; expected exactly one")

    box = _largest(boxes)
    top, right, bottom, left = box
    face_pixels = min(bottom - top, right - left)
    blur_variance, brightness = crop_metrics(image, box)

    # Metrics are always computed so they can be logged for calibration, but
    # they only block the request where the caller opted into the gates.
    if quality_gates:
        apply_quality_gates(face_pixels, blur_variance, brightness)

    encodings = face_recognition.face_encodings(
        array,
        known_face_locations=[box],
        num_jitters=config.NUM_JITTERS,
        model=config.LANDMARK_MODEL,
    )
    if not encodings:
        raise FaceError("encoding_failed", "Face detected but could not be encoded")

    return FaceResult(
        embedding=np.asarray(encodings[0], dtype=np.float32),
        faces_found=faces_found,
        face_pixels=int(face_pixels),
        blur_variance=blur_variance,
        brightness=brightness,
    )


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two templates.  Lower means more similar."""
    return float(np.linalg.norm(a - b))


def distances(matrix: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """Distance from `probe` to every row of `matrix`.  Lower means more similar."""
    if matrix.size == 0:
        return np.empty(0, dtype=np.float32)
    return np.linalg.norm(matrix - probe, axis=1)

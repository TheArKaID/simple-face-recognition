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
    select_subject,
)

ENGINE_ID = "dlib-resnet-v1"
EMBEDDING_DIM = 128


def _area(box: Tuple[int, int, int, int]) -> int:
    top, right, bottom, left = box
    return (bottom - top) * (right - left)


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
    # The original code took whichever detection came first.  select_subject
    # instead names the largest face as the subject and refuses frames that do
    # not clearly identify one.
    primary_i, other_i = select_subject([_area(b) for b in boxes])

    box = boxes[primary_i]
    top, right, bottom, left = box
    face_pixels = min(bottom - top, right - left)
    blur_variance, brightness = crop_metrics(image, box)

    # Metrics are always computed so they can be logged for calibration, but
    # they only block the request where the caller opted into the gates.
    if quality_gates:
        apply_quality_gates(face_pixels, blur_variance, brightness)

    # One call encodes the subject and any bystanders together, so tolerating
    # extra faces costs the encoder pass for them and nothing else.
    wanted = [box] + [boxes[i] for i in other_i]
    encodings = face_recognition.face_encodings(
        array,
        known_face_locations=wanted,
        num_jitters=config.NUM_JITTERS,
        model=config.LANDMARK_MODEL,
    )
    if not encodings:
        raise FaceError("encoding_failed", "Face detected but could not be encoded")

    return FaceResult(
        embedding=np.asarray(encodings[0], dtype=np.float32),
        others=[np.asarray(e, dtype=np.float32) for e in encodings[1:]],
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

"""A recogniser-free backend: test double, and the smallest example of the contract.

Selected with FACE_ENGINE=stub. It does everything a real backend does -
downscaling, subject selection, quality metrics, bystander embeddings - except
that instead of running a model it returns whatever vectors the caller put in
STATE. That lets the offline suites drive the decision logic at exact distances,
with no weights, no onnxruntime, and no model download.

It is also the file to read before writing a real backend: everything here is
required, and nothing here is model-specific.

STATE holds two things:

  boxes   detected faces as (top, right, bottom, left), largest wins the subject
  vector  either one array returned for every face, or a callable
          fn(image_array, face_index) -> array so a test can hand out a
          different vector per image or per face

Distance is plain Euclidean rather than the cosine the ArcFace backend uses.
That is deliberate: the logic under test - threshold bands, the impostor margin,
the 1:N cross-check - only requires "lower means more similar", and Euclidean
lets a test place a probe at an exactly known distance from a template.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from engines.common import (
    FaceError,
    FaceResult,
    apply_quality_gates,
    crop_metrics,
    downscale,
    select_subject,
)

ENGINE_ID = "stub-v1"
EMBEDDING_DIM = 512

STATE = {
    "boxes": [(10, 210, 210, 10)],
    "vector": None,
}


def _vector_for(array: np.ndarray, index: int) -> np.ndarray:
    vector = STATE["vector"]
    if vector is None:
        raise FaceError("encoding_failed", "stub backend: STATE['vector'] is not set")
    if callable(vector):
        vector = vector(array, index)
    return np.asarray(vector, dtype=np.float32)


def _area(box) -> int:
    top, right, bottom, left = box
    return (bottom - top) * (right - left)


def embed(image: Image.Image, quality_gates: bool = True) -> FaceResult:
    image = downscale(image)
    array = np.array(image)

    boxes = list(STATE["boxes"])
    if not boxes:
        raise FaceError("no_face", "No face detected in the image")

    faces_found = len(boxes)
    primary_i, other_i = select_subject([_area(b) for b in boxes])

    box = boxes[primary_i]
    top, right, bottom, left = box
    face_pixels = min(bottom - top, right - left)
    blur_variance, brightness = crop_metrics(image, box)

    if quality_gates:
        apply_quality_gates(face_pixels, blur_variance, brightness)

    return FaceResult(
        embedding=_vector_for(array, primary_i),
        others=[_vector_for(array, i) for i in other_i],
        bbox=(top, right, bottom, left),
        image=image,
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

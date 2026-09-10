"""Engine dispatcher.

Selects a recognition backend from FACE_ENGINE and re-exports its interface, so
storage, matching and the API layer never learn which model is running. Adding
a backend means writing one module under engines/ and listing it here.

The contract a backend must honour:

  * ENGINE_ID     - names the vector space; stored with every template so a
                    model change cannot silently mix incompatible vectors
  * EMBEDDING_DIM - length of the vectors it produces
  * embed()       - returns a FaceResult, or raises FaceError
  * distance()    - LOWER means MORE SIMILAR, whatever the underlying metric

That last point is what lets Euclidean and cosine backends share the decision
logic. It does NOT make their numbers comparable: thresholds are per-engine and
must be recalibrated after a switch. Use tests/calibrate.py, which reports
scale-free metrics for exactly this reason.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

import config
from engines.common import (  # noqa: F401  (re-exported)
    FaceError,
    FaceResult,
    decode_base64_image,
)

_BACKENDS = {"dlib", "insightface"}

if config.FACE_ENGINE == "dlib":
    from engines import dlib_backend as _backend
elif config.FACE_ENGINE == "insightface":
    from engines import insightface_backend as _backend
else:
    raise RuntimeError(
        f"FACE_ENGINE={config.FACE_ENGINE!r} is not one of {sorted(_BACKENDS)}"
    )

ENGINE_ID = _backend.ENGINE_ID
EMBEDDING_DIM = _backend.EMBEDDING_DIM


def embed(image: Image.Image, quality_gates: bool = True) -> FaceResult:
    return _backend.embed(image, quality_gates=quality_gates)


def embed_base64(base64_string: str, quality_gates: bool = True) -> FaceResult:
    return embed(decode_base64_image(base64_string), quality_gates=quality_gates)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two templates.  Lower means more similar."""
    return _backend.distance(a, b)


def distances(matrix: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """Distance from `probe` to every row of `matrix`.  Lower means more similar."""
    return _backend.distances(matrix, probe)

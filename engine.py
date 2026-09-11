"""Engine dispatcher.

Selects a recognition backend from FACE_ENGINE and re-exports its interface, so
storage, matching and the API layer never learn which model is running.

Adding a backend is two lines: a module under engines/ and an entry in
_BACKENDS below, plus its thresholds in config._ENGINE_THRESHOLDS. Imports are
lazy, so a backend whose dependencies are absent costs nothing until it is
selected. engines/stub_backend.py is the smallest complete example.

The contract a backend must honour:

  ENGINE_ID       names the vector space.  Stored with every template, so a
                  model change cannot silently mix incompatible vectors - the
                  store simply stops seeing templates from other spaces.  It
                  must cover everything that shapes the embedding, not just the
                  module: engines/arcface_onnx.py takes its id from
                  config._vector_space_id, which derives it from the detector
                  and recogniser files, because swapping a model file changes
                  the vectors while leaving the module name untouched.  Two
                  backends computing identical vectors may share an id, and
                  tools/compare_backends.py is what earns them the right to.
  EMBEDDING_DIM   length of the vectors it produces.
  embed(image, quality_gates=True)
                  returns a FaceResult, or raises FaceError with one of the
                  reason codes the API already documents.
  distance(a, b)  LOWER means MORE SIMILAR, whatever the underlying metric.
  distances(matrix, probe)
                  the same, vectorised over rows.

That last point is what lets Euclidean and cosine backends share one decision
path. It does NOT make their numbers comparable: thresholds are per-engine and
must be recalibrated after a switch. tests/calibrate.py reports scale-free
metrics for exactly that reason.
"""
from __future__ import annotations

import importlib

import numpy as np
from PIL import Image

import config
from engines.common import (  # noqa: F401  (re-exported)
    FaceError,
    FaceResult,
    decode_base64_image,
)

_BACKENDS = {
    "arcface-onnx": "engines.arcface_onnx",
    "stub": "engines.stub_backend",
}

if config.FACE_ENGINE not in _BACKENDS:
    raise RuntimeError(
        f"FACE_ENGINE={config.FACE_ENGINE!r} is not one of {sorted(_BACKENDS)}"
    )

_backend = importlib.import_module(_BACKENDS[config.FACE_ENGINE])

# The thresholds were picked for one vector space and the templates were stored
# under one id, so the backend and config must name the same space.  They are
# derived separately - config from the model filenames, the backend from itself
# - and a disagreement would mean decisions made with another space's numbers,
# which no test would fail on.  Cheaper to refuse to start.
if _backend.ENGINE_ID != config.ENGINE_ID:
    raise RuntimeError(
        f"backend {_BACKENDS[config.FACE_ENGINE]} reports ENGINE_ID "
        f"{_backend.ENGINE_ID!r} but config derived {config.ENGINE_ID!r}. "
        f"Thresholds and stored templates key off the id, so these must agree; "
        f"check config._KNOWN_SPACES against the backend."
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

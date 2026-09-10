"""Presentation-attack detection: is this a live face or a picture of one?

Kept out of the engines because it is a separate question from identity. An
engine answers "whose face is this"; this module answers "is there a face here
at all, or a screen showing one". Both have to pass before attendance is
recorded.

The model is MiniFASNet from Silent-Face-Anti-Spoofing (minivision-ai), run
through onnxruntime - already in the image for InsightFace, so this costs about
2MB of weights and a few milliseconds. Its pipeline is fiddly and the fiddly
part matters: each of the two models sees the face box expanded by its own
scale factor (2.7x and 4.0x), resized to 80x80, and their softmax outputs are
summed. Get the crop scale wrong and the model still returns confident-looking
numbers while discriminating nothing.

Which is why this module refuses to pretend. It runs only when weights are
actually present AND it has been calibrated - see tools/measure_liveness.py.
Until then `check()` returns None, meaning "not assessed", and the caller must
treat that as an unprotected request rather than a passing one.
"""
from __future__ import annotations

import glob
import os
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

import config


@dataclass
class LivenessResult:
    score: float          # 0..1, higher means more likely a live face
    is_live: bool
    detail: str

    def public_data(self) -> dict:
        return {"score": round(self.score, 4), "is_live": self.is_live}


# (filename fragment, crop scale) - the scale is baked into each model's name
# in the upstream repo and is not interchangeable between them.
_MODEL_SCALES = (("2.7_80x80", 2.7), ("4_0_0_80x80", 4.0))

_sessions = None
_lock = threading.Lock()


def available() -> bool:
    """True when weights are present and the mode asks for the model."""
    return config.LIVENESS_MODE == "model" and bool(_model_files())


def _model_files():
    if not os.path.isdir(config.LIVENESS_MODEL_DIR):
        return []
    found = []
    for fragment, scale in _MODEL_SCALES:
        matches = glob.glob(os.path.join(config.LIVENESS_MODEL_DIR, f"*{fragment}*.onnx"))
        if matches:
            found.append((sorted(matches)[0], scale))
    return found


def _load():
    global _sessions
    if _sessions is None:
        with _lock:
            if _sessions is None:
                import onnxruntime

                loaded = []
                for path, scale in _model_files():
                    session = onnxruntime.InferenceSession(
                        path, providers=["CPUExecutionProvider"]
                    )
                    loaded.append((session, scale, os.path.basename(path)))
                _sessions = loaded
    return _sessions


def _crop(image: Image.Image, bbox, scale: float) -> np.ndarray:
    """Expand the face box by `scale`, then resize to 80x80.

    Mirrors upstream CropImage._get_new_box, and the details are load-bearing -
    an earlier version of this function squared the crop, padded instead of
    shifting, and fed raw 0-255 pixels.  Real faces then scored 0.0002 on
    average, i.e. the model called almost every live face a spoof.  Three
    things matter:

      * width and height scale independently, keeping the box aspect ratio,
        and the 80x80 resize squashes it - the model was trained that way
      * a box running past the frame edge is SHIFTED back inside, not padded,
        so the crop never contains invented pixels
      * pixels are divided by 255, because upstream passes the crop through
        torchvision ToTensor() before the model sees it
    """
    top, right, bottom, left = bbox
    src_w, src_h = image.width, image.height
    box_w, box_h = right - left, bottom - top
    if box_w <= 0 or box_h <= 0:
        raise ValueError("empty face box")

    # Never ask for more context than the frame holds.
    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w, new_h = box_w * scale, box_h * scale
    centre_x, centre_y = left + box_w / 2.0, top + box_h / 2.0

    lt_x, lt_y = centre_x - new_w / 2.0, centre_y - new_h / 2.0
    rb_x, rb_y = centre_x + new_w / 2.0, centre_y + new_h / 2.0
    if lt_x < 0:
        rb_x -= lt_x
        lt_x = 0
    if lt_y < 0:
        rb_y -= lt_y
        lt_y = 0
    if rb_x > src_w - 1:
        lt_x -= rb_x - src_w + 1
        rb_x = src_w - 1
    if rb_y > src_h - 1:
        lt_y -= rb_y - src_h + 1
        rb_y = src_h - 1

    # Upstream slices inclusively, so the far edge is +1 here.
    patch = image.crop((int(lt_x), int(lt_y), int(rb_x) + 1, int(rb_y) + 1))
    arr = np.asarray(patch.resize((80, 80), Image.BILINEAR), dtype=np.float32)
    arr = arr[:, :, ::-1] / 255.0          # RGB -> BGR, and ToTensor's scaling
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...])


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def check(image: Image.Image, bbox) -> Optional[LivenessResult]:
    """Score one face for liveness, or return None when not assessed.

    None is not a pass. It means no model was available, and the caller has to
    decide what an unassessed request is worth - see config.LIVENESS_MODE.
    """
    if not available():
        return None

    sessions = _load()
    if not sessions:
        return None

    # Upstream sums the per-model softmax vectors and takes class 1 as "live".
    total = np.zeros(3, dtype=np.float64)
    used = []
    for session, scale, name in sessions:
        logits = session.run(None, {session.get_inputs()[0].name: _crop(image, bbox, scale)})[0]
        probs = _softmax(np.asarray(logits, dtype=np.float64).ravel())
        if probs.size < 3:
            probs = np.pad(probs, (0, 3 - probs.size))
        total += probs[:3]
        used.append(name)

    total /= len(sessions)
    score = float(total[1])
    return LivenessResult(
        score=score,
        is_live=score >= config.LIVENESS_MIN_SCORE,
        detail=f"{len(used)} model(s): {', '.join(used)}",
    )

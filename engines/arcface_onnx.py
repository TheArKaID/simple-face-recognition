"""ArcFace over bare onnxruntime - the same model, without the insightface package.

Produces byte-for-byte the same embeddings as engines/insightface_backend.py,
which is why it keeps the same ENGINE_ID: stored templates stay valid across the
swap. `tools/compare_backends.py` is what proves that claim, and it should be
re-run after any change here.

The reason for the duplication is size. The insightface package drags in scipy,
scikit-image, albumentations and matplotlib to do work this file does in numpy:
decode SCRFD's outputs, fit a similarity transform to five landmarks, and
normalise a 512-d vector. Those dependencies plus three model files nothing
loads account for most of a 2.15GB image.

Every constant below comes from the upstream implementation and none of them are
adjustable knobs:

  detection    SCRFD det_10g, 640x640 letterboxed, RGB, (x-127.5)/128
               three strides (8/16/32), two anchors per location, distance-coded
               boxes and five keypoints, then NMS at 0.4
  alignment    similarity transform from the five keypoints onto ArcFace's
               canonical 112x112 positions
  recognition  w600k_r50, 112x112, RGB, (x-127.5)/127.5, L2-normalised output
"""
from __future__ import annotations

import os
import threading

import cv2
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

# Same vector space as the package backend, so templates carry over.
ENGINE_ID = "insightface-buffalo-l"
EMBEDDING_DIM = 512

# Two files out of the buffalo_l pack, fetched during the build rather than
# committed - 184MB of weights does not belong in git.  Point this at
# ~/.insightface/models/buffalo_l instead if the package is installed locally.
MODEL_DIR = os.getenv(
    "FACE_ARCFACE_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "models", "arcface"),
)
DET_MODEL = os.getenv("FACE_ARCFACE_DET", "det_10g.onnx")
REC_MODEL = os.getenv("FACE_ARCFACE_REC", "w600k_r50.onnx")

DET_SIZE = int(os.getenv("FACE_INSIGHTFACE_DET_SIZE", 640))
DET_THRESH = float(os.getenv("FACE_INSIGHTFACE_MIN_DET_SCORE", 0.5))
NMS_THRESH = 0.4

# SCRFD det_10g topology: nine outputs, three strides, two anchors per cell.
_STRIDES = (8, 16, 32)
_NUM_ANCHORS = 2
_FMC = 3

# Where ArcFace expects the five landmarks to land in a 112x112 crop.
_ARCFACE_DST = np.array(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]],
    dtype=np.float32,
)

_sessions = None
_lock = threading.Lock()
_anchor_cache = {}


def _load():
    global _sessions
    if _sessions is None:
        with _lock:
            if _sessions is None:
                import onnxruntime

                opts = ["CPUExecutionProvider"]
                det_path = os.path.join(MODEL_DIR, DET_MODEL)
                rec_path = os.path.join(MODEL_DIR, REC_MODEL)
                for path in (det_path, rec_path):
                    if not os.path.exists(path):
                        raise FaceError(
                            "model_missing", f"ArcFace weights not found: {path}"
                        )
                det = onnxruntime.InferenceSession(det_path, providers=opts)
                rec = onnxruntime.InferenceSession(rec_path, providers=opts)
                _sessions = (
                    det,
                    [o.name for o in det.get_outputs()],
                    det.get_inputs()[0].name,
                    rec,
                    rec.get_inputs()[0].name,
                )
    return _sessions


def _anchor_centers(height: int, width: int, stride: int) -> np.ndarray:
    """Grid centres for one FPN level, repeated per anchor."""
    key = (height, width, stride)
    cached = _anchor_cache.get(key)
    if cached is not None:
        return cached
    centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
    centers = (centers * stride).reshape((-1, 2))
    if _NUM_ANCHORS > 1:
        centers = np.stack([centers] * _NUM_ANCHORS, axis=1).reshape((-1, 2))
    if len(_anchor_cache) < 100:
        _anchor_cache[key] = centers
    return centers


def _nms(dets: np.ndarray) -> list:
    """Standard IoU non-maximum suppression; input is sorted by score already."""
    x1, y1, x2, y2, scores = (dets[:, i] for i in range(5))
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= NMS_THRESH]
    return keep


def _detect(bgr: np.ndarray):
    """Return (boxes Nx4, scores N, keypoints Nx5x2) in `bgr` pixel coordinates."""
    det, out_names, in_name, _, _ = _load()

    # Letterbox into a DET_SIZE square, top-left aligned, as upstream does.
    img_h, img_w = bgr.shape[:2]
    im_ratio = float(img_h) / img_w
    if im_ratio > 1.0:
        new_h = DET_SIZE
        new_w = int(new_h / im_ratio)
    else:
        new_w = DET_SIZE
        new_h = int(new_w * im_ratio)
    det_scale = float(new_h) / img_h
    canvas = np.zeros((DET_SIZE, DET_SIZE, 3), dtype=np.uint8)
    canvas[:new_h, :new_w, :] = cv2.resize(bgr, (new_w, new_h))

    blob = cv2.dnn.blobFromImage(
        canvas, 1.0 / 128.0, (DET_SIZE, DET_SIZE), (127.5, 127.5, 127.5), swapRB=True
    )
    outs = det.run(out_names, {in_name: blob})

    scores_all, boxes_all, kps_all = [], [], []
    for idx, stride in enumerate(_STRIDES):
        scores = outs[idx].reshape(-1)
        bbox_preds = outs[idx + _FMC].reshape(-1, 4) * stride
        kps_preds = outs[idx + _FMC * 2].reshape(-1, 10) * stride

        height, width = DET_SIZE // stride, DET_SIZE // stride
        centers = _anchor_centers(height, width, stride)

        keep = np.where(scores >= DET_THRESH)[0]
        if keep.size == 0:
            continue

        # Boxes and keypoints are distances from the anchor centre.
        cx, cy = centers[:, 0], centers[:, 1]
        boxes = np.stack(
            [cx - bbox_preds[:, 0], cy - bbox_preds[:, 1],
             cx + bbox_preds[:, 2], cy + bbox_preds[:, 3]], axis=-1
        )
        kps = np.stack(
            [centers[:, i % 2] + kps_preds[:, i] for i in range(10)], axis=-1
        ).reshape(-1, 5, 2)

        scores_all.append(scores[keep])
        boxes_all.append(boxes[keep])
        kps_all.append(kps[keep])

    if not scores_all:
        return np.empty((0, 4)), np.empty(0), np.empty((0, 5, 2))

    scores = np.concatenate(scores_all)
    boxes = np.concatenate(boxes_all) / det_scale
    kps = np.concatenate(kps_all) / det_scale

    order = scores.argsort()[::-1]
    pre_det = np.hstack((boxes, scores[:, None])).astype(np.float32)[order]
    keep = _nms(pre_det)
    return pre_det[keep, :4], pre_det[keep, 4], kps[order][keep]


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (scale, rotation, translation).

    Umeyama's method.  insightface reaches for skimage's SimilarityTransform
    here, which is a large part of why scikit-image is a dependency; the maths
    is a mean, a covariance and one SVD.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num = src.shape[0]
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_demean, dst_demean = src - src_mean, dst - dst_mean

    A = dst_demean.T @ src_demean / num
    d = np.ones((2,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[1] = -1

    U, S, Vt = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        raise FaceError("alignment_failed", "Degenerate landmark configuration")
    if rank == 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            R = U @ Vt
        else:
            s = d[1]
            d[1] = -1
            R = U @ np.diag(d) @ Vt
            d[1] = s
    else:
        R = U @ np.diag(d) @ Vt

    var = src_demean.var(axis=0).sum()
    scale = 1.0 / var * (S @ d) if var else 1.0
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = scale * R
    matrix[:2, 2] = dst_mean - scale * R @ src_mean
    return matrix[:2, :]


def _embed_face(bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Align on the five landmarks, then run the recogniser."""
    _, _, _, rec, rec_in = _load()
    matrix = _similarity_transform(kps.astype(np.float32), _ARCFACE_DST)
    aligned = cv2.warpAffine(bgr, matrix, (112, 112), borderValue=0.0)
    blob = cv2.dnn.blobFromImage(
        aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
    )
    vec = rec.run(None, {rec_in: blob})[0].reshape(-1)
    norm = np.linalg.norm(vec)
    if not norm:
        raise FaceError("encoding_failed", "Recogniser returned a zero vector")
    return (vec / norm).astype(np.float32)


def embed(image: Image.Image, quality_gates: bool = True) -> FaceResult:
    """Detect the face in `image` and return its template plus quality metrics.

    Raises FaceError with the same reason codes as the other backends, so the
    API contract does not shift when the engine does.
    """
    image = downscale(image)
    bgr = np.array(image)[:, :, ::-1].copy()

    boxes, scores, kpss = _detect(bgr)
    if len(boxes) == 0:
        raise FaceError("no_face", "No face detected in the image")

    faces_found = len(boxes)
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    primary_i, other_i = select_subject(areas)

    x1, y1, x2, y2 = (int(v) for v in boxes[primary_i])
    left, top = max(0, x1), max(0, y1)
    right, bottom = min(image.width, x2), min(image.height, y2)
    if right <= left or bottom <= top:
        raise FaceError("no_face", "Detected face box falls outside the image")

    face_pixels = min(bottom - top, right - left)
    blur_variance, brightness = crop_metrics(image, (top, right, bottom, left))

    if quality_gates:
        if float(scores[primary_i]) < DET_THRESH:
            raise FaceError(
                "low_confidence",
                f"Detection score {float(scores[primary_i]):.2f} below {DET_THRESH}",
            )
        apply_quality_gates(face_pixels, blur_variance, brightness)

    return FaceResult(
        embedding=_embed_face(bgr, kpss[primary_i]),
        others=[_embed_face(bgr, kpss[i]) for i in other_i],
        bbox=(top, right, bottom, left),
        image=image,
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

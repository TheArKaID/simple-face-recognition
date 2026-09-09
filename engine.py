"""Face embedding engine.

The recognition model sits behind this module so that storage and decision
logic never touch it directly.  Swapping dlib for a stronger model later means
replacing this file and bumping ENGINE_ID; nothing else has to change, as long
as the replacement keeps the contract:

  * embed() returns a 1-D float32 vector, or raises FaceError
  * distance() returns a value where LOWER means MORE SIMILAR
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import List, Tuple

import face_recognition
import numpy as np
from PIL import Image, ImageOps

import config

# Identifies the vector space templates live in.  Stored alongside every
# template so stale templates are detectable after a model change.
ENGINE_ID = "dlib-resnet-v1"
EMBEDDING_DIM = 128


class FaceError(ValueError):
    """An image could not be turned into a usable face template.

    `reason` is a stable machine-readable code for the HRIS to branch on.
    """

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason


@dataclass
class FaceResult:
    embedding: np.ndarray
    faces_found: int
    face_pixels: int
    blur_variance: float
    brightness: float

    def quality(self) -> dict:
        return {
            "faces_found": self.faces_found,
            "face_pixels": self.face_pixels,
            "blur_variance": round(self.blur_variance, 2),
            "brightness": round(self.brightness, 1),
        }


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode a base64 payload, with or without a data-URL prefix."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        image = Image.open(io.BytesIO(base64.b64decode(base64_string)))
        # Phone cameras store orientation in EXIF rather than rotating pixels;
        # without this a portrait selfie is fed to the detector sideways.
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except Exception as exc:
        raise FaceError("invalid_image", f"Invalid base64 image: {exc}")


def _downscale(image: Image.Image) -> Image.Image:
    """Cap the longest side so detection cost and crop scale stay predictable."""
    longest = max(image.size)
    if longest <= config.MAX_IMAGE_DIMENSION:
        return image
    ratio = config.MAX_IMAGE_DIMENSION / longest
    new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
    return image.resize(new_size, Image.BILINEAR)


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian - the usual blur proxy.

    The crop is normalised to a fixed size first, otherwise the value tracks
    image resolution and no single threshold works across devices.
    """
    lap = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(lap.var())


def _largest(boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    # boxes are (top, right, bottom, left)
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))


def embed(image: Image.Image, quality_gates: bool = True) -> FaceResult:
    """Detect the face in `image` and return its template plus quality metrics.

    Raises FaceError with a stable reason code when the image is unusable.
    With `quality_gates` false, blur/brightness/size are measured but not enforced.
    """
    image = _downscale(image)
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

    crop = image.crop((left, top, right, bottom)).convert("L").resize((160, 160), Image.BILINEAR)
    gray = np.asarray(crop, dtype=np.float32)
    blur_variance = _laplacian_variance(gray)
    brightness = float(gray.mean())

    # Metrics are always computed so they can be logged for calibration, but
    # they only block the request where the caller opted into the gates.
    if quality_gates:
        if face_pixels < config.MIN_FACE_PIXELS:
            raise FaceError(
                "face_too_small",
                f"Face is {face_pixels}px across, minimum is {config.MIN_FACE_PIXELS}px",
            )
        if blur_variance < config.MIN_BLUR_VARIANCE:
            raise FaceError("low_quality_blur", f"Image too blurry (variance {blur_variance:.1f})")
        if brightness < config.MIN_BRIGHTNESS:
            raise FaceError("low_quality_dark", f"Face too dark (brightness {brightness:.0f})")
        if brightness > config.MAX_BRIGHTNESS:
            raise FaceError("low_quality_bright", f"Face overexposed (brightness {brightness:.0f})")

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


def embed_base64(base64_string: str, quality_gates: bool = True) -> FaceResult:
    return embed(decode_base64_image(base64_string), quality_gates=quality_gates)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two templates.  Lower means more similar."""
    return float(np.linalg.norm(a - b))


def distances(matrix: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """Distance from `probe` to every row of `matrix`.  Lower means more similar."""
    if matrix.size == 0:
        return np.empty(0, dtype=np.float32)
    return np.linalg.norm(matrix - probe, axis=1)

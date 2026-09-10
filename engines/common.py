"""Pieces every backend shares: error type, result type, image preparation.

Nothing here knows which recognition model is in use.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

import config


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
    embedding: np.ndarray          # the subject: the largest face in frame
    others: list                   # every other face's embedding, for cross-checking
    bbox: tuple                    # subject box (top, right, bottom, left); liveness needs it
    image: object                  # the downscaled PIL image the box refers to
    faces_found: int
    face_pixels: int
    blur_variance: float
    brightness: float

    def quality(self) -> dict:
        return {
            "faces_found": self.faces_found,
            "extra_faces": len(self.others),
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


def downscale(image: Image.Image) -> Image.Image:
    """Cap the longest side so detection cost and crop scale stay predictable."""
    longest = max(image.size)
    if longest <= config.MAX_IMAGE_DIMENSION:
        return image
    ratio = config.MAX_IMAGE_DIMENSION / longest
    new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
    return image.resize(new_size, Image.BILINEAR)


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian - the usual blur proxy.

    The crop is normalised to a fixed size first, otherwise the value tracks
    image resolution and no single threshold works across devices.
    """
    lap = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(lap.var())


def crop_metrics(image: Image.Image, box) -> tuple:
    """Blur and brightness for one face box, measured on a normalised crop."""
    top, right, bottom, left = box
    crop = (image.crop((left, top, right, bottom))
                 .convert("L")
                 .resize((160, 160), Image.BILINEAR))
    gray = np.asarray(crop, dtype=np.float32)
    return laplacian_variance(gray), float(gray.mean())


def apply_quality_gates(face_pixels: int, blur_variance: float, brightness: float) -> None:
    """Raise FaceError if the face fails any configured gate."""
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


def select_subject(areas):
    """Which detected face to verify, and which are bystanders.

    Returns (primary_index, other_indices).  The subject is the largest face -
    in a selfie, the person holding the phone.  Raises FaceError when the frame
    does not clearly identify a subject, rather than picking one arbitrarily as
    the original code did.
    """
    if not areas:
        raise FaceError("no_face", "No face detected in the image")

    order = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
    primary, rest = order[0], order[1:]
    extra = len(rest)

    # Order matters: the opt-out is reported as multiple_faces, the original
    # reason code, so a deployment that never enables bystanders keeps the
    # error the HRIS already handles.
    if extra and config.MAX_EXTRA_FACES == 0:
        raise FaceError(
            "multiple_faces", f"{len(areas)} faces detected; expected exactly one"
        )
    if extra > config.MAX_EXTRA_FACES:
        raise FaceError(
            "too_many_faces",
            f"{len(areas)} faces detected; at most {1 + config.MAX_EXTRA_FACES} allowed",
        )
    if extra:
        runner_up = areas[rest[0]]
        ratio = areas[primary] / runner_up if runner_up else float("inf")
        if ratio < config.PRIMARY_FACE_DOMINANCE:
            raise FaceError(
                "ambiguous_subject",
                f"Largest face is only {ratio:.2f}x the next one; "
                f"{config.PRIMARY_FACE_DOMINANCE}x is required to tell who is presenting",
            )
    return primary, rest

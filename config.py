"""Runtime configuration.

Every threshold is an environment variable so it can be retuned after
calibration without rebuilding the image.  Distances follow the dlib
convention used throughout the service: LOWER means MORE SIMILAR.
"""
import os


def _float(name, default):
    return float(os.getenv(name, default))


def _int(name, default):
    return int(os.getenv(name, default))


def _bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# --- Decision thresholds -----------------------------------------------------
# distance <= ACCEPT_MAX_DISTANCE            -> accept
# ACCEPT_MAX_DISTANCE < d <= REVIEW_MAX      -> review (accepted, flagged)
# distance > REVIEW_MAX_DISTANCE             -> reject
#
# Measured on tests/images (6 identities, 12 photos): genuine pairs ran
# 0.092-0.484, impostor pairs 0.601-1.076.  Accept sits just above the worst
# genuine pair, review just below the closest impostor pair.  Six identities is
# far too small a sample to settle these - retune from the production
# verify_log, where the closest impostor pair will be much closer than 0.601
# simply because there are vastly more pairs to draw from.
ACCEPT_MAX_DISTANCE = _float("FACE_ACCEPT_MAX_DISTANCE", 0.50)
REVIEW_MAX_DISTANCE = _float("FACE_REVIEW_MAX_DISTANCE", 0.56)

# Tolerance for the legacy /compare-fr endpoint (was face_recognition's 0.6).
LEGACY_TOLERANCE = _float("FACE_LEGACY_TOLERANCE", 0.50)

# --- 1:N impostor cross-check ------------------------------------------------
CROSS_CHECK_ENABLED = _bool("FACE_CROSS_CHECK", True)
# The claimed employee must beat the best other employee by at least this much.
MIN_IMPOSTOR_MARGIN = _float("FACE_MIN_IMPOSTOR_MARGIN", 0.05)

# --- Image quality gates -----------------------------------------------------
MAX_IMAGE_DIMENSION = _int("FACE_MAX_IMAGE_DIMENSION", 1600)
MIN_FACE_PIXELS = _int("FACE_MIN_FACE_PIXELS", 80)
MIN_BLUR_VARIANCE = _float("FACE_MIN_BLUR_VARIANCE", 40.0)
MIN_BRIGHTNESS = _float("FACE_MIN_BRIGHTNESS", 40.0)
MAX_BRIGHTNESS = _float("FACE_MAX_BRIGHTNESS", 225.0)
ALLOW_MULTIPLE_FACES = _bool("FACE_ALLOW_MULTIPLE_FACES", False)
# Off by default on /compare-fr so the endpoint already in production only
# changes in the two ways intended: stricter tolerance, and multi-face refusal.
LEGACY_QUALITY_GATES = _bool("FACE_LEGACY_QUALITY_GATES", False)

# --- Engine ------------------------------------------------------------------
LANDMARK_MODEL = os.getenv("FACE_LANDMARK_MODEL", "large")  # "large" = 68 points
NUM_JITTERS = _int("FACE_NUM_JITTERS", 1)
UPSAMPLE = _int("FACE_DETECT_UPSAMPLE", 1)

# --- Storage -----------------------------------------------------------------
DB_PATH = os.getenv(
    "FACE_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "faces.db"),
)
MAX_TEMPLATES_PER_EMPLOYEE = _int("FACE_MAX_TEMPLATES", 5)
LOG_VERIFICATIONS = _bool("FACE_LOG_VERIFICATIONS", True)

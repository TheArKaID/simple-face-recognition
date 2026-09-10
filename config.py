"""Runtime configuration.

Every threshold is an environment variable so it can be retuned after
calibration without rebuilding the image.  Distances follow one convention
throughout the service, whichever backend is loaded: LOWER means MORE SIMILAR.
"""
import os


def _float(name, default):
    return float(os.getenv(name, default))


def _int(name, default):
    return int(os.getenv(name, default))


def _bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# --- Engine ------------------------------------------------------------------
# Which backend under engines/ to load.  "dlib" is the engine that has been in
# production; "insightface" is the ArcFace candidate.  Templates carry their
# engine id, so switching cannot mix vector spaces - but it does invalidate
# every stored template, and every threshold below.
FACE_ENGINE = os.getenv("FACE_ENGINE", "dlib").strip().lower()

# Each backend measures distance on its own scale, so one set of thresholds
# cannot serve both.  These come from tests/calibrate.py over tests/images
# (15 identities, 63 usable photos) and are frozen in tests/baseline_*.json.
# Env vars still win, and any real deployment should retune from its own
# verify_log - a larger roster brings impostors closer than any fixture set can.
_ENGINE_THRESHOLDS = {
    # Euclidean over 128 dims.  Measured genuine 0.000-0.587 against impostor
    # 0.453-1.138: the two OVERLAP, so no threshold separates them.  These are
    # chosen for the genuine side and the 1:N margin carries the security.
    "dlib": {"accept": 0.50, "review": 0.56, "margin": 0.05},
    # Cosine distance over 512 dims, range 0-2.  Measured genuine 0.016-0.502
    # against impostor 0.677-1.202: fully separated, gap +0.174.  Accept sits
    # above the worst genuine with room, review below the closest impostor with
    # room, so the measured gap stays available as headroom for harder faces.
    "insightface": {"accept": 0.55, "review": 0.64, "margin": 0.10},
}
_T = _ENGINE_THRESHOLDS.get(FACE_ENGINE, _ENGINE_THRESHOLDS["dlib"])

# dlib backend only.
LANDMARK_MODEL = os.getenv("FACE_LANDMARK_MODEL", "large")  # "large" = 68 points
NUM_JITTERS = _int("FACE_NUM_JITTERS", 1)
UPSAMPLE = _int("FACE_DETECT_UPSAMPLE", 1)

# --- Decision thresholds -----------------------------------------------------
# distance <= ACCEPT_MAX_DISTANCE            -> accept
# ACCEPT_MAX_DISTANCE < d <= REVIEW_MAX      -> review (accepted, flagged)
# distance > REVIEW_MAX_DISTANCE             -> reject
ACCEPT_MAX_DISTANCE = _float("FACE_ACCEPT_MAX_DISTANCE", _T["accept"])
REVIEW_MAX_DISTANCE = _float("FACE_REVIEW_MAX_DISTANCE", _T["review"])

# Tolerance for the legacy /compare-fr endpoint (was face_recognition's 0.6).
# Only meaningful on the dlib scale, which is the engine that endpoint shipped
# with; set it explicitly if you ever point /compare-fr at another backend.
LEGACY_TOLERANCE = _float("FACE_LEGACY_TOLERANCE", 0.50)

# --- 1:N impostor cross-check ------------------------------------------------
CROSS_CHECK_ENABLED = _bool("FACE_CROSS_CHECK", True)
# The claimed employee must beat the best other employee by at least this much.
# Falling short only downgrades to review, never rejects, so raising it trades
# review volume for security without ever refusing a genuine employee outright.
MIN_IMPOSTOR_MARGIN = _float("FACE_MIN_IMPOSTOR_MARGIN", _T["margin"])

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

# --- Storage -----------------------------------------------------------------
DB_PATH = os.getenv(
    "FACE_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "faces.db"),
)
MAX_TEMPLATES_PER_EMPLOYEE = _int("FACE_MAX_TEMPLATES", 5)
LOG_VERIFICATIONS = _bool("FACE_LOG_VERIFICATIONS", True)

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
# Which backend under engines/ to load; engine.py documents the contract a new
# one has to honour.  "arcface-onnx" is the production engine: SCRFD detection
# and ArcFace w600k_r50 driven through onnxruntime directly.
#
# The dlib backend the service originally shipped with has been removed.  It
# lost on every measure that mattered - d-prime 3.07 against 8.20, and its
# genuine and impostor distance distributions overlapped, so no single threshold
# separated them - and it was the only thing left in the image needing a C++
# toolchain.
FACE_ENGINE = os.getenv("FACE_ENGINE", "arcface-onnx").strip().lower()

# Distance scales are per-engine and never interchangeable, so thresholds live
# beside the engine that produced them.  Adding an engine means adding an entry
# here and a module under engines/ - nothing else.
#
# These come from tests/calibrate.py over tests/images (15 identities, 68 usable
# photos) and are frozen in tests/baseline_*.json.  Env vars still win, and a
# real deployment should retune from its own verify_log: a larger roster brings
# impostors closer than any fixture set can.
_ENGINE_THRESHOLDS = {
    # Cosine distance over 512 dims, range 0-2.  Measured genuine 0.016-0.548
    # against impostor 0.677-1.202: fully separated.  Accept sits above the
    # worst genuine with room, review below the closest impostor with room, so
    # the measured gap stays available as headroom for harder faces.
    "arcface-onnx": {
        "accept": 0.55,
        "review": 0.64,
        "margin": 0.10,
        "blur": 10.0,
        # /compare-fr has no review band and no roster to cross-check against,
        # so it needs one line.  0.60 sits between the worst genuine pair and
        # the closest impostor pair - looser than `accept` because that endpoint
        # has no 1:N check behind it to catch what slips past.
        "legacy": 0.60,
    },
    # A recogniser-free double for the offline suites, and the smallest
    # complete example of the backend contract.
    "stub": {
        "accept": 0.55,
        "review": 0.64,
        "margin": 0.10,
        "blur": 10.0,
        "legacy": 0.60,
    },
}
if FACE_ENGINE not in _ENGINE_THRESHOLDS:
    raise RuntimeError(
        f"FACE_ENGINE={FACE_ENGINE!r} has no thresholds. Add an entry to "
        f"config._ENGINE_THRESHOLDS; known engines: {sorted(_ENGINE_THRESHOLDS)}"
    )
_T = _ENGINE_THRESHOLDS[FACE_ENGINE]

# onnxruntime sizes its intra-op thread pool from the HOST cpu count, not from
# the cgroup limit, so four Swarm replicas each capped at cpus: "2" would each
# open a pool for every core on the box and then fight over 8 CPU-equivalents.
# That does not corrupt anything, it just thrashes - and it thrashes hardest
# under exactly the morning attendance load it needs to survive.  Keep this in
# step with docker-stack.yml's cpus limit; 0 hands the decision back to
# onnxruntime.
ONNX_INTRA_OP_THREADS = _int("FACE_ONNX_THREADS", 2)

# --- Decision thresholds -----------------------------------------------------
# distance <= ACCEPT_MAX_DISTANCE            -> accept
# ACCEPT_MAX_DISTANCE < d <= REVIEW_MAX      -> review (accepted, flagged)
# distance > REVIEW_MAX_DISTANCE             -> reject
ACCEPT_MAX_DISTANCE = _float("FACE_ACCEPT_MAX_DISTANCE", _T["accept"])
REVIEW_MAX_DISTANCE = _float("FACE_REVIEW_MAX_DISTANCE", _T["review"])

# Single threshold for the legacy /compare-fr endpoint, which production still
# calls.  It was face_recognition's 0.6 on dlib's Euclidean scale originally;
# that number means nothing here, hence the per-engine value above.
LEGACY_TOLERANCE = _float("FACE_LEGACY_TOLERANCE", _T["legacy"])

# --- 1:N impostor cross-check ------------------------------------------------
CROSS_CHECK_ENABLED = _bool("FACE_CROSS_CHECK", True)
# The claimed employee must beat the best other employee by at least this much.
# Falling short only downgrades to review, never rejects, so raising it trades
# review volume for security without ever refusing a genuine employee outright.
#
# Worth keeping on even though ArcFace separates cleanly without it: the gap
# narrows as headcount grows, and the check costs one matrix multiply.
MIN_IMPOSTOR_MARGIN = _float("FACE_MIN_IMPOSTOR_MARGIN", _T["margin"])

# --- Subject selection -------------------------------------------------------
# A colleague wandering into frame should not block attendance, but the face
# that gets verified must still be the one presenting.  The subject is the
# LARGEST face: in a selfie that is whoever holds the phone, and a bystander
# behind them is naturally smaller.  Extra faces are tolerated only while the
# subject clearly dominates - two similarly sized faces mean the frame does not
# say who is presenting, so it is refused rather than guessed at.
#
# Allowing extra faces reopens one attack a hard refusal closed: holding a phone
# showing the claimed employee's photo close enough to become the largest face.
# Liveness is what closes that, so MAX_EXTRA_FACES above 0 belongs with
# LIVENESS_MODE=model.
MAX_EXTRA_FACES = _int("FACE_MAX_EXTRA_FACES", 2)
PRIMARY_FACE_DOMINANCE = _float("FACE_PRIMARY_DOMINANCE", 1.8)

# --- Image quality gates -----------------------------------------------------
MAX_IMAGE_DIMENSION = _int("FACE_MAX_IMAGE_DIMENSION", 1600)
MIN_FACE_PIXELS = _int("FACE_MIN_FACE_PIXELS", 80)
# Per-engine, and for a substantive reason: detectors return different crops so
# the same photo measures differently, and ArcFace tolerates blur that dlib
# could not.  Four photos a threshold of 40 refused embed cleanly here - 0.13 to
# 0.22 from their own identity against 0.75 to 0.81 from the nearest other
# person - so this low floor is a sanity check against catastrophic blur, not a
# tuned value.  No blur level present in tests/images broke an embedding.
MIN_BLUR_VARIANCE = _float("FACE_MIN_BLUR_VARIANCE", _T["blur"])
MIN_BRIGHTNESS = _float("FACE_MIN_BRIGHTNESS", 40.0)
MAX_BRIGHTNESS = _float("FACE_MAX_BRIGHTNESS", 225.0)
# Off by default on /compare-fr: that endpoint predates the gates, and
# tightening it silently would change production behaviour without anyone
# asking for it.
LEGACY_QUALITY_GATES = _bool("FACE_LEGACY_QUALITY_GATES", False)

# --- Liveness / presentation-attack detection --------------------------------
# "off"   - not assessed at all
# "model" - run the MiniFASNet weights in LIVENESS_MODEL_DIR
#
# Measured on tests/images against tests/images/spoof (69 live faces, 9 screen
# photos): live scores ran 0.551-1.000, spoofs 0.000-0.0001, ROC AUC 1.0000.
# On the strength of that it defaults to on - see tools/measure_liveness.py to
# re-measure after any change to the crop, the detector or the weights.
LIVENESS_MODE = os.getenv("FACE_LIVENESS_MODE", "model").strip().lower()
LIVENESS_MODEL_DIR = os.getenv(
    "FACE_LIVENESS_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "liveness"),
)
# Chosen from the measured gap, not from the EER: spoofs scored at most 0.0001
# while the hardest live face scored 0.551, so this sits well clear of both,
# with the headroom deliberately on the genuine side.  A refused live employee
# retakes a photo; an accepted spoof records attendance that never happened, so
# the two errors are not worth the same - but with spoofs this far from the
# boundary there is no need to crowd the live faces to catch them.
LIVENESS_MIN_SCORE = _float("FACE_LIVENESS_MIN_SCORE", 0.35)
# When the mode asks for the model but it cannot run, fail closed rather than
# waving the request through: having asked for liveness and silently not got it
# is the worst of the three outcomes.
LIVENESS_REQUIRED = _bool("FACE_LIVENESS_REQUIRED", True)

# --- Storage -----------------------------------------------------------------
DB_PATH = os.getenv(
    "FACE_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "faces.db"),
)
MAX_TEMPLATES_PER_EMPLOYEE = _int("FACE_MAX_TEMPLATES", 5)
LOG_VERIFICATIONS = _bool("FACE_LOG_VERIFICATIONS", True)

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
    # Cosine distance over 512 dims, range 0-2.  Measured on the det_2.5g
    # pipeline: genuine 0.016-0.540, impostor 0.646-1.20, fully separated with a
    # gap of 0.105.  The thresholds sit INSIDE that gap with headroom on both
    # sides rather than hugging either edge.
    #
    # They moved when the detector changed.  Under det_10g the closest impostor
    # was 0.677 and a review ceiling of 0.64 left 0.037 of room; under det_2.5g
    # it is 0.646 and that same 0.64 left only 0.006.  Still correct on 15
    # identities, but the closest impostor keeps falling as the roster grows, and
    # once it drops under the ceiling an impostor lands in review - accepted with
    # a flag - instead of rejected.  Hence 0.61.
    "arcface-onnx": {
        # Measured on tests/new-images: 30 identities, 10 photos each, 298
        # usable photos giving 1,332 genuine and 42,921 impostor pairs.
        # Genuine ran 0.053-0.435, impostor 0.543-1.182, gap +0.108, and the
        # threshold sweep is clean at 0% both ways anywhere from 0.45 to 0.50.
        #
        # These are attendance selfies, which is the actual use case - the older
        # tests/images fixture included deliberately extreme expressions that no
        # one produces at a clock-in, and calibrating to those pushed the
        # thresholds 0.09 too high.
        #
        # The bands sit INSIDE the measured gap: accept above the worst genuine
        # pair with room, review below the closest impostor pair with room.  On
        # this data the review band is empty, which is the point - it is
        # headroom for faces harder than anything measured yet.
        "accept": 0.47,   # 0.035 above the worst genuine pair
        "review": 0.52,   # 0.023 below the closest impostor pair
        "margin": 0.10,
        "blur": 10.0,
        # /compare-fr is being retired and has no 1:N cross-check behind it, so
        # it gets the strictest line rather than the most permissive: at 0.47 the
        # sweep shows no impostor pair accepted at all.
        "legacy": 0.47,
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

# --- ArcFace detector and recogniser ----------------------------------------
# det_2.5g replaced det_10g after measuring both on tests/images: detection went
# 237ms -> 67ms (3.5x, and detection was 44% of a 486ms request) while
# recognition held - d-prime 8.09 against 8.20, ROC AUC, EER and rank-1
# unchanged, worst-case margin slightly better at +0.222.  The recogniser is
# the same w600k_r50 in both packs, so only the detector actually changed.
#
# The acceptance threshold moved 0.5 -> 0.45, and for a reason no accuracy
# metric would have caught.  det_2.5g scores a small bystander face lower than
# det_10g did: at 0.5 it stopped seeing the second person in
# tests/images/g5.jpg altogether, which silently disables the held-photo
# defence, because bystander evidence needs the bystander detected.  At 0.45 it
# is found again and no extra faces appear anywhere across the 69 photos.
#
# That measurement rests on one photo with a genuine second person, so treat
# 0.45 as provisional until more such photos exist.
ARCFACE_DET_MODEL = os.getenv("FACE_ARCFACE_DET", "det_2.5g.onnx")
ARCFACE_REC_MODEL = os.getenv("FACE_ARCFACE_REC", "w600k_r50.onnx")
ARCFACE_DET_SIZE = _int("FACE_ARCFACE_DET_SIZE", 640)
ARCFACE_MIN_DET_SCORE = _float("FACE_ARCFACE_MIN_DET_SCORE", 0.45)

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
# photos) under the det_2.5g pipeline: live scores ran 0.536-1.000, spoofs all
# 0.0000, ROC AUC 1.0000.  The live floor moved down slightly from 0.551 when
# the detector changed - the crop MiniFASNet sees comes from the detector's box
# - which is why this gets re-measured after any detector change.
# On the strength of that it defaults to on - see tools/measure_liveness.py to
# re-measure after any change to the crop, the detector or the weights.
LIVENESS_MODE = os.getenv("FACE_LIVENESS_MODE", "model").strip().lower()
LIVENESS_MODEL_DIR = os.getenv(
    "FACE_LIVENESS_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "liveness"),
)
# Chosen from the measured gap, not from the EER: spoofs scored 0.0000 while the
# hardest live face scored 0.536, so this sits well clear of both,
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

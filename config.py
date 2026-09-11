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

# Distance scales belong to a VECTOR SPACE, not to a backend module, so that is
# what these are keyed by.  Two pipelines that produce different embeddings need
# different thresholds even when the same Python module drives them - which is
# exactly what happens when FACE_ARCFACE_REC points at a quantised file.
#
# Adding an engine means adding an entry here plus a module under engines/.
# Adding a model variant means an entry here plus a line in _KNOWN_SPACES below.
# A pairing with no entry refuses to start rather than borrowing another space's
# numbers, because distances from a new model are on their own scale.
_ENGINE_THRESHOLDS = {
    # det_2.5g + w600k_r50, both FP32 - the buffalo_m pairing, and the default.
    #
    # Measured on tests/new-images: 30 identities, 10 photos each, 298 usable
    # photos giving 1,332 genuine and 42,921 impostor pairs.  Genuine ran
    # 0.053-0.435, impostor 0.543-1.182, gap +0.108, and the threshold sweep is
    # clean at 0% both ways anywhere from 0.45 to 0.50.
    #
    # These are attendance selfies, which is the actual use case - the older
    # tests/images fixture included deliberately extreme expressions that no one
    # produces at a clock-in, and calibrating to those pushed the thresholds
    # 0.09 too high.
    #
    # The bands sit INSIDE the measured gap: accept above the worst genuine pair
    # with room, review below the closest impostor pair with room.  On this data
    # the review band is empty, which is the point - it is headroom for faces
    # harder than anything measured yet.
    "insightface-buffalo-m": {
        "accept": 0.47,   # 0.035 above the worst genuine pair
        "review": 0.52,   # 0.023 below the closest impostor pair
        "margin": 0.10,
        "blur": 10.0,
        # /compare-fr is being retired and has no 1:N cross-check behind it, so
        # it gets the strictest line rather than the most permissive: at 0.47 the
        # sweep shows no impostor pair accepted at all.
        "legacy": 0.47,
    },
    # The same pair with the recogniser quantised to INT8: 2.43x faster end to
    # end (337ms -> 138ms) at the cost of separation, gap 0.108 -> 0.092.  Both
    # bands shift up together because the worst genuine pair moves to 0.471,
    # past the FP32 accept line.  Measured at these values on tests/new-images:
    # 268 genuine attempts accepted with none flagged, 7,772 impostor attempts
    # all rejected.
    #
    # Not the default.  What it spends is gap, and gap is a minimum over ~N^2/2
    # pairs, so it shrinks on its own as the roster grows.  Latency can also be
    # bought with hardware; separation cannot be bought back.
    "insightface-buffalo-m-rec-int8": {
        "accept": 0.50,
        "review": 0.54,
        "margin": 0.10,
        "blur": 10.0,
        "legacy": 0.50,
    },
    # A recogniser-free double for the offline suites, and the smallest complete
    # example of the backend contract.  Must match engines/stub_backend.py's
    # ENGINE_ID; engine.py asserts that they agree.
    "stub-v1": {
        "accept": 0.55,
        "review": 0.64,
        "margin": 0.10,
        "blur": 10.0,
        "legacy": 0.60,
    },
}

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

# --- which vector space are these embeddings in? -----------------------------
# ENGINE_ID names the space, and store.py refuses to compare templates across
# ids.  That promise only holds if the id follows the models, and it used to
# not: it was a hard-coded constant in engines/arcface_onnx.py while the model
# files came from env vars.  Pointing FACE_ARCFACE_REC at a quantised file
# therefore produced different embeddings under the SAME id, and the store
# compared them against FP32 templates without complaint.
#
# Measured on tests/new-images, that mix gave a separation gap of +0.082,
# against +0.108 for FP32 throughout and +0.092 for INT8 throughout: worse than
# either consistent choice, and still positive.  So nothing errors and nothing
# fails - attendance keeps working, quietly closer to the edge.  A crash
# announces itself; this does not.  That is the failure this derivation exists
# to prevent.
#
# The detector counts as much as the recogniser.  It supplies the five landmarks
# the alignment is fitted to, so different landmarks mean a different crop and a
# different embedding: quantising only the detector moved the gap 0.108 -> 0.084,
# more than quantising only the recogniser did.
_KNOWN_SPACES = {
    ("det_10g.onnx", "w600k_r50.onnx"): "insightface-buffalo-l",
    ("det_2.5g.onnx", "w600k_r50.onnx"): "insightface-buffalo-m",
    ("det_2.5g.onnx", "w600k_r50_int8.onnx"): "insightface-buffalo-m-rec-int8",
    ("det_2.5g_int8.onnx", "w600k_r50.onnx"): "insightface-buffalo-m-det-int8",
    ("det_2.5g_int8.onnx", "w600k_r50_int8.onnx"): "insightface-buffalo-m-int8",
}
_DEFAULT_DET_SIZE = 640


def _vector_space_id(det, rec, det_size):
    """Name the space that this detector/recogniser pairing embeds into.

    An unfamiliar pairing gets a derived name rather than a familiar one, so it
    is isolated from every stored template.  Failing towards isolation is the
    safe direction: a template that cannot be read costs a re-enrol, while a
    template read in the wrong space costs a wrong answer nobody sees.
    """
    name = _KNOWN_SPACES.get((det, rec))
    if name is None:
        strip = lambda f: f[:-5] if f.endswith(".onnx") else f  # noqa: E731
        name = f"insightface-{strip(det)}+{strip(rec)}"
    # The letterbox size feeds the detector, so it moves the landmarks too - a
    # different one is a different space even with identical weights.
    if det_size != _DEFAULT_DET_SIZE:
        name = f"{name}-det{det_size}"
    return name


# The stub owns its id outright; engine.py checks that the backend agrees with
# whatever is decided here, so the two can never drift apart unnoticed.
if FACE_ENGINE == "arcface-onnx":
    ENGINE_ID = _vector_space_id(
        ARCFACE_DET_MODEL, ARCFACE_REC_MODEL, ARCFACE_DET_SIZE
    )
elif FACE_ENGINE == "stub":
    ENGINE_ID = "stub-v1"
else:
    raise RuntimeError(
        f"FACE_ENGINE={FACE_ENGINE!r} has no vector-space id. Add one here and "
        f"a module under engines/; see engine.py for the backend contract."
    )

if ENGINE_ID not in _ENGINE_THRESHOLDS:
    raise RuntimeError(
        f"no thresholds for vector space {ENGINE_ID!r} (FACE_ENGINE="
        f"{FACE_ENGINE!r}, detector={ARCFACE_DET_MODEL!r}, recogniser="
        f"{ARCFACE_REC_MODEL!r}, det_size={ARCFACE_DET_SIZE}). Distances from a "
        f"new pairing are on their own scale and cannot borrow another space's "
        f"numbers, so run tests/calibrate.py against it and add an entry to "
        f"config._ENGINE_THRESHOLDS. Known spaces: {sorted(_ENGINE_THRESHOLDS)}"
    )
_T = _ENGINE_THRESHOLDS[ENGINE_ID]

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

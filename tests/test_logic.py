"""Exercise the storage and decision logic against the stub backend.

FACE_ENGINE=stub gives a backend that satisfies engine.py's contract without a
model: the test chooses the detected boxes and the vectors they embed to, so
probes can be placed at exactly known distances.  Everything else - SQLite
round-trip, index building, subject selection, threshold bands, the 1:N
cross-check - is the real code.

Liveness has its own coverage in test_api.py and tools/measure_liveness.py.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["FACE_ENGINE"] = "stub"
os.environ["FACE_LIVENESS_MODE"] = "off"
os.environ["FACE_DB_PATH"] = DB

import config           # noqa: E402
import engine           # noqa: E402
import matcher          # noqa: E402
from engines.stub_backend import STATE  # noqa: E402
from store import TemplateStore  # noqa: E402

DIM = engine.EMBEDDING_DIM

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(("  ok   " if condition else "  FAIL ") + name + ("  " + detail if detail else ""))


def vec(seed, jitter=0.0):
    """A deterministic unit vector; `jitter` moves it a controlled distance."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=DIM).astype(np.float32)
    base /= np.linalg.norm(base)
    if jitter:
        off = np.random.default_rng(seed + 9999).normal(size=DIM).astype(np.float32)
        off /= np.linalg.norm(off)
        base = base + off * jitter
    return base.astype(np.float32)


def b64_image(width=400, height=400, colour=(120, 110, 100), noisy=True):
    """A base64 JPEG with enough texture to clear the blur gate."""
    import base64, io
    from PIL import Image
    arr = np.full((height, width, 3), colour, dtype=np.uint8)
    if noisy:
        arr = (arr + np.random.default_rng(1).integers(-40, 40, arr.shape)).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


IMG = b64_image()

print("\n== engine: quality gates and multi-face refusal ==")

STATE["vector"] = vec(1)
STATE["boxes"] = [(10, 210, 210, 10)]
res = engine.embed_base64(IMG)
check("single face embeds", res.embedding.shape == (DIM,) and res.faces_found == 1)
check("quality metrics populated", res.quality()["blur_variance"] > 0 and res.quality()["face_pixels"] == 200)

# A bystander is tolerated while the subject clearly dominates: 200x200 against
# 110x140 is a 2.6x area ratio, past the 1.8x the policy requires.
STATE["boxes"] = [(10, 210, 210, 10), (10, 380, 120, 240)]
res2 = engine.embed_base64(IMG)
check("bystander tolerated when subject dominates", len(res2.others) == 1, str(len(res2.others)))
check("subject is the largest face", res2.face_pixels == 200, str(res2.face_pixels))
check("extra faces reported in quality", res2.quality()["extra_faces"] == 1)

# Two similarly sized faces: the frame does not say who is presenting.
STATE["boxes"] = [(10, 210, 210, 10), (10, 420, 210, 220)]
try:
    engine.embed_base64(IMG)
    check("similar-sized second face refused", False, "no error raised")
except engine.FaceError as e:
    check("similar-sized second face refused", e.reason == "ambiguous_subject", e.reason)

# Opting out entirely restores the old fail-closed behaviour.
config.MAX_EXTRA_FACES = 0
STATE["boxes"] = [(10, 210, 210, 10), (10, 380, 120, 240)]
try:
    engine.embed_base64(IMG)
    check("bystander refused when not opted in", False, "no error raised")
except engine.FaceError as e:
    check("bystander refused when not opted in", e.reason == "multiple_faces", e.reason)
config.MAX_EXTRA_FACES = 2

# More bystanders than allowed.
STATE["boxes"] = [(10, 210, 210, 10), (10, 380, 120, 240),
                  (250, 380, 340, 300), (250, 200, 330, 130)]
try:
    engine.embed_base64(IMG)
    check("too many faces refused", False, "no error raised")
except engine.FaceError as e:
    check("too many faces refused", e.reason == "too_many_faces", e.reason)

STATE["boxes"] = [(10, 50, 50, 10)]  # 40px face
try:
    engine.embed_base64(IMG)
    check("tiny face refused when gated", False, "no error raised")
except engine.FaceError as e:
    check("tiny face refused when gated", e.reason == "face_too_small", e.reason)

r = engine.embed_base64(IMG, quality_gates=False)
check("tiny face allowed when ungated (legacy path)", r.face_pixels == 40)

STATE["boxes"] = [(10, 210, 210, 10)]
flat = b64_image(noisy=False)  # perfectly flat -> zero laplacian variance
try:
    engine.embed_base64(flat)
    check("blurry image refused", False, "no error raised")
except engine.FaceError as e:
    check("blurry image refused", e.reason == "low_quality_blur", e.reason)

try:
    engine.decode_base64_image("not base64 at all!!")
    check("garbage base64 refused", False)
except engine.FaceError as e:
    check("garbage base64 refused", e.reason == "invalid_image")

print("\n== store: round-trip and index ==")

store = TemplateStore(DB)
BUD, AND, CIT = "EMP-BUDI", "EMP-ANDI", "EMP-CITRA"

store.enroll("T1", BUD, [(vec(1), {"blur_variance": 90.0, "face_pixels": 200})])
store.enroll("T1", AND, [(vec(2), {"blur_variance": 90.0, "face_pixels": 200})])
store.enroll("T1", CIT, [(vec(3), {}), (vec(3, 0.2), {})])

idx = store.index("T1")
check("index row count", idx.size == 4, f"size={idx.size}")
check("index matrix shape", idx.matrix.shape == (4, DIM), str(idx.matrix.shape))
check("embedding survives round-trip", np.allclose(idx.matrix[0], vec(1), atol=1e-6))
check("template_count per employee", store.template_count("T1", CIT) == 2)
check("tenants isolated", store.index("T2").size == 0)

store.enroll("T1", BUD, [(vec(1, 0.1), {})])  # replace=True
check("re-enroll replaces", store.template_count("T1", BUD) == 1)

for i in range(8):
    store.enroll("T1", "EMP-CAP", [(vec(50 + i), {})], replace=False)
check(
    "template cap enforced",
    store.template_count("T1", "EMP-CAP") == config.MAX_TEMPLATES_PER_EMPLOYEE,
    str(store.template_count("T1", "EMP-CAP")),
)

print("\n== matcher: threshold band ==")

store2 = TemplateStore(os.path.join(tempfile.mkdtemp(), "m.db"))
store2.enroll("T", BUD, [(vec(1), {})])
store2.enroll("T", AND, [(vec(2), {})])
index = store2.index("T")

# genuine: probe very close to Budi's template
d = matcher.verify(index, BUD, vec(1, 0.30))
check("close genuine accepted", d.decision == "accept", f"{d.decision} dist={d.distance:.3f}")
check("accepted match flag", d.match and not d.requires_review)

# borderline: inside the review band.  Derived from config rather than
# hardcoded, so retuning the thresholds does not break the test.
target_d = (config.ACCEPT_MAX_DISTANCE + config.REVIEW_MAX_DISTANCE) / 2
direction = np.random.default_rng(7).normal(size=DIM).astype(np.float32)
direction /= np.linalg.norm(direction)
d = matcher.verify(index, BUD, vec(1) + direction * target_d)
check("borderline goes to review", d.decision == "review", f"{d.decision} dist={d.distance:.3f}")
check("review still counts as match", d.match and d.requires_review)
check("review reason reported", "borderline_distance" in d.reasons, str(d.reasons))

d = matcher.verify(index, BUD, vec(1) + direction * (config.REVIEW_MAX_DISTANCE + 0.4))
check("far probe rejected", d.decision == "reject", f"{d.decision} dist={d.distance:.3f}")
check("rejected match flag false", not d.match)
check("reject reason reported", "below_threshold" in d.reasons, str(d.reasons))

d = matcher.verify(index, "EMP-NOBODY", vec(1))
check("unknown employee -> not_enrolled", d.reasons == ["not_enrolled"], str(d.reasons))

print("\n== matcher: 1:N impostor cross-check (the buddy-punching case) ==")

# Andi's face passes the 1:1 bar against Budi's template, but is nearer his own.
store3 = TemplateStore(os.path.join(tempfile.mkdtemp(), "x.db"))
budi = vec(1)
andi = budi + direction * 0.42          # a lookalike: inside the accept band
store3.enroll("T", BUD, [(budi, {})])
store3.enroll("T", AND, [(andi, {})])
index3 = store3.index("T")

probe_andi = andi + direction * 0.01    # Andi taking attendance as Budi
d = matcher.verify(index3, BUD, probe_andi)
check(
    "1:1 alone would have accepted",
    engine.distance(budi, probe_andi) <= config.ACCEPT_MAX_DISTANCE,
    f"dist to Budi={engine.distance(budi, probe_andi):.3f}",
)
check("cross-check rejects the impostor", d.decision == "reject", f"{d.decision}")
check("identity_mismatch flagged", "identity_mismatch" in d.reasons, str(d.reasons))
check("margin is negative", d.margin is not None and d.margin < 0, f"margin={d.margin:.3f}")
check("runner-up hidden from response", "runner_up_id" not in d.public_data())
check("runner-up kept for internal log", d.runner_up_id == AND)

# Andi taking his own attendance still works.
d = matcher.verify(index3, AND, probe_andi)
check("genuine employee still accepted", d.decision == "accept", f"{d.decision} dist={d.distance:.3f}")

# Two employees so close that the claim is not safe on its own -> review.
store4 = TemplateStore(os.path.join(tempfile.mkdtemp(), "y.db"))
store4.enroll("T", BUD, [(budi, {})])
store4.enroll("T", AND, [(budi + direction * 0.40, {})])
d = matcher.verify(store4.index("T"), BUD, budi + direction * 0.19)
check("thin margin downgrades to review", d.decision == "review", f"{d.decision} margin={d.margin:.3f}")
check("low_margin flagged", "low_margin" in d.reasons, str(d.reasons))

# With only one employee enrolled there is no runner-up to compare against.
store5 = TemplateStore(os.path.join(tempfile.mkdtemp(), "z.db"))
store5.enroll("T", BUD, [(budi, {})])
d = matcher.verify(store5.index("T"), BUD, budi + direction * 0.2)
check("single-employee tenant works", d.decision == "accept" and d.margin is None)

print("\n== store: logging and deletion ==")

store3.log_verification(
    tenant_id="T", employee_id=BUD, source="verify", decision="reject",
    distance=0.46, runner_up_id=AND, runner_up_distance=0.02, margin=-0.44,
    reasons=["identity_mismatch"], quality={"blur_variance": 90.0},
)
row = store3._conn.execute(
    "SELECT * FROM verify_log ORDER BY id DESC LIMIT 1"
).fetchone()
check("verification logged", row["decision"] == "reject" and row["runner_up_id"] == AND)
check("log records engine id", row["engine_id"] == engine.ENGINE_ID)

removed = store3.delete_employee("T", AND)
check("delete removes templates", removed == 1)
check("index refreshed after delete", store3.index("T").size == 1)
d = matcher.verify(store3.index("T"), BUD, probe_andi)
check("impostor now passes once rival is deleted (expected)", d.decision == "accept")

print("\n== persistence across restart ==")
reopened = TemplateStore(DB)
check("templates survive reopen", reopened.index("T1").size >= 3, str(reopened.index("T1").size))
check("stats reports engine", reopened.stats()["engine_id"] == engine.ENGINE_ID)
check("no stale templates", reopened.stats()["stale_templates"] == 0)

print("\n== vector space isolation ==")
# The id has to follow the models, not the module.  When it was a constant,
# pointing FACE_ARCFACE_REC at a quantised file produced different embeddings
# under the same id, and store.py compared them against FP32 templates happily.
# Measured, that mix ran at a separation gap of +0.082 against +0.108 - worse
# than either consistent choice, still positive, so nothing ever raised.
space = config._vector_space_id
FP32 = ("det_2.5g.onnx", "w600k_r50.onnx", 640)
check("default pairing keeps its historical id",
      space(*FP32) == "insightface-buffalo-m", space(*FP32))
check("quantised recogniser is a different space",
      space("det_2.5g.onnx", "w600k_r50_int8.onnx", 640) != space(*FP32))
check("quantised detector is a different space too",
      space("det_2.5g_int8.onnx", "w600k_r50.onnx", 640) != space(*FP32))
check("detector letterbox size is part of the space",
      space("det_2.5g.onnx", "w600k_r50.onnx", 320) != space(*FP32))
unknown = space("det_future.onnx", "w600k_future.onnx", 640)
check("an unfamiliar pairing gets a derived id, not a familiar one",
      unknown not in (space(*FP32), "insightface-buffalo-l"), unknown)
check("the running space has thresholds",
      config.ENGINE_ID in config._ENGINE_THRESHOLDS, config.ENGINE_ID)
check("engine and config agree on the space",
      engine.ENGINE_ID == config.ENGINE_ID, engine.ENGINE_ID)

# A pairing nobody calibrated must refuse to start rather than borrow another
# space's numbers.  Run in a subprocess because config reads env at import.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_proc = subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.path.insert(0, %r); import config" % _root],
    # FACE_ENGINE is named explicitly: this suite runs on the stub, whose
    # space is fixed, so an inherited FACE_ENGINE would skip the pairing
    # check entirely and the test would pass without testing anything.
    env=dict(os.environ, FACE_ENGINE="arcface-onnx",
             FACE_ARCFACE_REC="w600k_r50_int8.onnx",
             FACE_ARCFACE_DET="det_2.5g_int8.onnx"),
    capture_output=True, text=True)
check("uncalibrated pairing refuses to start", _proc.returncode != 0,
      f"exit {_proc.returncode}")
check("and names the space it could not price",
      "no thresholds for vector space" in _proc.stderr,
      (_proc.stderr.strip()[-110:] if _proc.stderr else "(no stderr)"))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)

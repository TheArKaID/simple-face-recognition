"""Exercise the new storage / decision logic with face_recognition stubbed out.

dlib is not installable here, so the recogniser is replaced by a fake whose
"embeddings" are chosen to reproduce the situations the service must handle.
Everything else - SQLite round-trip, index building, threshold bands, the 1:N
cross-check - is the real code.
"""
import os
import sys
import types
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- stub face_recognition ---------------------------------------------------
FAKE = types.ModuleType("face_recognition")
STATE = {"boxes": [(10, 210, 210, 10)], "vector": None}


def face_locations(array, number_of_times_to_upsample=1):
    return STATE["boxes"]


def face_encodings(array, known_face_locations=None, num_jitters=1, model="large"):
    return [STATE["vector"]]


FAKE.face_locations = face_locations
FAKE.face_encodings = face_encodings
sys.modules["face_recognition"] = FAKE

DB = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["FACE_DB_PATH"] = DB

import config           # noqa: E402
import engine           # noqa: E402
import matcher          # noqa: E402
from store import TemplateStore  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(("  ok   " if condition else "  FAIL ") + name + ("  " + detail if detail else ""))


def vec(seed, jitter=0.0):
    """A deterministic 128-d vector; `jitter` moves it a controlled distance."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=engine.EMBEDDING_DIM).astype(np.float32)
    base /= np.linalg.norm(base)
    if jitter:
        off = np.random.default_rng(seed + 9999).normal(size=engine.EMBEDDING_DIM).astype(np.float32)
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
check("single face embeds", res.embedding.shape == (128,) and res.faces_found == 1)
check("quality metrics populated", res.quality()["blur_variance"] > 0 and res.quality()["face_pixels"] == 200)

STATE["boxes"] = [(10, 210, 210, 10), (10, 380, 120, 240)]
try:
    engine.embed_base64(IMG)
    check("two faces refused", False, "no error raised")
except engine.FaceError as e:
    check("two faces refused", e.reason == "multiple_faces", e.reason)

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
check("index matrix shape", idx.matrix.shape == (4, 128), str(idx.matrix.shape))
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
direction = np.random.default_rng(7).normal(size=128).astype(np.float32)
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

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)

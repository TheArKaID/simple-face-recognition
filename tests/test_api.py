"""Drive the FastAPI surface end to end.

dlib cannot be installed here, so the recogniser is stubbed; the routing,
request schemas and response bodies exercised below are the real ones.
The fake recogniser returns a vector chosen per image so identity scenarios can
be scripted precisely.
"""
import base64
import io
import os
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Most of this file tests the identity path; the liveness gate has its own
# section at the end, driven by a stub so it needs no onnxruntime or weights.
os.environ["FACE_LIVENESS_MODE"] = "off"
os.environ["FACE_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "api.db")

# --- stubs -------------------------------------------------------------------
VECTORS = {}          # image tag -> embedding
BOXES = {"n": 1}

fr = types.ModuleType("face_recognition")


def _tag_of(array):
    # The tag is painted into the first pixel row so the stub can tell images apart.
    return int(array[0, 0, 0])


def face_locations(array, number_of_times_to_upsample=1):
    boxes = [(10, 210, 210, 10)]
    if BOXES["n"] > 1:
        boxes.append((10, 380, 120, 240))
    return boxes


def face_encodings(array, known_face_locations=None, num_jitters=1, model="large"):
    # One encoding per requested box; bystanders reuse the same vector, which is
    # enough to exercise the plumbing.
    return [VECTORS[_tag_of(array)]] * len(known_face_locations or [None])


fr.face_locations = face_locations
fr.face_encodings = face_encodings
sys.modules["face_recognition"] = fr

import config  # noqa: E402
import engine  # noqa: E402

# --- helpers -----------------------------------------------------------------
DIRECTION = np.random.default_rng(7).normal(size=128).astype(np.float32)
DIRECTION /= np.linalg.norm(DIRECTION)
BASE = np.random.default_rng(1).normal(size=128).astype(np.float32)
BASE /= np.linalg.norm(BASE)


def make_image(tag, embedding):
    """A base64 JPEG tagged with `tag`, which the stub maps to `embedding`."""
    from PIL import Image
    arr = np.random.default_rng(tag).integers(60, 200, (400, 400, 3)).astype(np.uint8)
    arr[0, 0, 0] = tag                      # JPEG is lossy, so re-read after encode
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")   # lossless, tag survives exactly
    VECTORS[tag] = embedding.astype(np.float32)
    return base64.b64encode(buf.getvalue()).decode()


BUDI_PROFILE = make_image(11, BASE)
BUDI_SELFIE = make_image(12, BASE + DIRECTION * 0.12)          # genuine
ANDI_PROFILE = make_image(21, BASE + DIRECTION * 0.42)         # lookalike
ANDI_SELFIE = make_image(22, BASE + DIRECTION * 0.43)          # impostor attempt
CITRA_PROFILE = make_image(31, np.random.default_rng(99).normal(size=128).astype(np.float32) * 0.4)
STRANGER = make_image(41, -BASE)

from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402

client = TestClient(main.app)

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(("  ok   " if condition else "  FAIL ") + name + ("  " + detail if detail else ""))


print("\n== /health ==")
r = client.get("/health")
check("health 200", r.status_code == 200)
check("health reports thresholds", r.json()["data"]["thresholds"]["cross_check_enabled"] is True)

print("\n== /enroll ==")
r = client.post("/enroll", json={"tenant_id": "PT-ABC", "employee_id": "EMP-BUDI",
                                 "images": [BUDI_PROFILE]})
check("enroll 200", r.status_code == 200, str(r.status_code))
check("templates stored", r.json()["data"]["templates_stored"] == 1, r.text[:120])

r = client.post("/enroll", json={"tenant_id": "PT-ABC", "employee_id": "EMP-ANDI",
                                 "images": [ANDI_PROFILE]})
check("second employee enrolled", r.json()["data"]["templates_stored"] == 1)
client.post("/enroll", json={"tenant_id": "PT-ABC", "employee_id": "EMP-CITRA",
                             "images": [CITRA_PROFILE]})

r = client.post("/enroll", json={"tenant_id": "PT-ABC", "employee_id": "EMP-X",
                                 "images": ["garbage!!"]})
check("unusable enrol image -> 422", r.status_code == 422, str(r.status_code))
check("reason reported", r.json()["reason"] == "invalid_image", r.text[:140])

r = client.post("/enroll", json={"employee_id": "EMP-Y"})
check("missing images -> 422 validation", r.status_code == 422)

r = client.get("/enroll/PT-ABC/EMP-BUDI")
check("enrollment status enrolled", r.json()["data"]["enrolled"] is True)
r = client.get("/enroll/PT-ABC/EMP-NOBODY")
check("enrollment status not enrolled", r.json()["data"]["enrolled"] is False)

print("\n== /verify: genuine ==")
r = client.post("/verify", json={"tenant_id": "PT-ABC", "employee_id": "EMP-BUDI",
                                 "image": BUDI_SELFIE})
body = r.json()["data"]
check("genuine 200", r.status_code == 200)
check("genuine accepted", body["decision"] == "accept", str(body))
check("match true", body["match"] is True)
check("candidates checked", body["candidates_checked"] == 3, str(body["candidates_checked"]))
check("runner-up identity withheld", "runner_up_id" not in body)
check("runner-up score present", body["runner_up_distance"] is not None)
check("quality returned", "blur_variance" in body["quality"])

print("\n== /verify: the buddy-punching case ==")
r = client.post("/verify", json={"tenant_id": "PT-ABC", "employee_id": "EMP-BUDI",
                                 "image": ANDI_SELFIE})
body = r.json()["data"]
check("impostor 200 (a decision, not an error)", r.status_code == 200)
check("impostor rejected", body["decision"] == "reject", str(body))
check("match false", body["match"] is False)
check("identity_mismatch reason", "identity_mismatch" in body["reasons"], str(body["reasons"]))
check("negative margin", body["margin"] < 0, str(body["margin"]))
check(
    "1:1 alone would have passed the old 0.6 tolerance",
    body["distance"] <= 0.6,
    f"distance={body['distance']}",
)

r = client.post("/verify", json={"tenant_id": "PT-ABC", "employee_id": "EMP-ANDI",
                                 "image": ANDI_SELFIE})
check("Andi's own attendance still accepted", r.json()["data"]["decision"] == "accept", r.text[:160])

print("\n== /verify: strangers, errors, tenants ==")
r = client.post("/verify", json={"tenant_id": "PT-ABC", "employee_id": "EMP-BUDI",
                                 "image": STRANGER})
check("stranger rejected", r.json()["data"]["decision"] == "reject", r.text[:160])
check("below_threshold reason", "below_threshold" in r.json()["data"]["reasons"])

r = client.post("/verify", json={"tenant_id": "PT-ABC", "employee_id": "EMP-NOBODY",
                                 "image": BUDI_SELFIE})
check("not enrolled -> 409", r.status_code == 409, str(r.status_code))
check("not_enrolled reason", r.json()["reason"] == "not_enrolled", r.text[:140])

r = client.post("/verify", json={"tenant_id": "PT-ABC", "employee_id": "EMP-BUDI",
                                 "image": "not-an-image"})
check("bad image -> 422", r.status_code == 422)
check("bad image reason", r.json()["reason"] == "invalid_image")

# A bystander in frame no longer blocks attendance: the subject is the largest
# face, and the stub's second box is small enough to stay a bystander.
BOXES["n"] = 2
r = client.post("/verify", json={"tenant_id": "PT-ABC", "employee_id": "EMP-BUDI",
                                 "image": BUDI_SELFIE})
body = r.json().get("data", {})
check("bystander in frame still verifies", r.status_code == 200, r.text[:160])
check("extra face reported", body.get("quality", {}).get("extra_faces") == 1, r.text[:200])
check("extra_faces_present flagged", "extra_faces_present" in body.get("reasons", []),
      str(body.get("reasons")))

# With bystanders switched off, the original fail-closed error comes back.
import config as _cfg
_cfg.MAX_EXTRA_FACES = 0
r = client.post("/verify", json={"tenant_id": "PT-ABC", "employee_id": "EMP-BUDI",
                                 "image": BUDI_SELFIE})
check("opt-out restores 422", r.status_code == 422, str(r.status_code))
check("multiple_faces reason", r.json().get("reason") == "multiple_faces", r.text[:140])
_cfg.MAX_EXTRA_FACES = 2
BOXES["n"] = 1

r = client.post("/verify", json={"tenant_id": "PT-OTHER", "employee_id": "EMP-BUDI",
                                 "image": BUDI_SELFIE})
check("other tenant does not see the template", r.status_code == 409, str(r.status_code))

print("\n== /verify: lazy enrolment for migration ==")
r = client.post("/verify", json={"tenant_id": "PT-ABC", "employee_id": "EMP-NEW",
                                 "image": BUDI_SELFIE, "reference_image": BUDI_PROFILE})
check("lazy enrol then verify", r.status_code == 200 and r.json()["data"]["decision"] in ("accept", "review", "reject"), r.text[:160])
check("employee now enrolled", client.get("/enroll/PT-ABC/EMP-NEW").json()["data"]["enrolled"] is True)

print("\n== DELETE /enroll ==")
r = client.delete("/enroll/PT-ABC/EMP-NEW")
check("delete 200", r.status_code == 200 and r.json()["data"]["templates_removed"] == 1, r.text[:140])
check("gone afterwards", client.get("/enroll/PT-ABC/EMP-NEW").json()["data"]["enrolled"] is False)

print("\n== /compare-fr: legacy contract preserved ==")
r = client.post("/compare-fr", json={"reference_image": BUDI_PROFILE, "target_image": BUDI_SELFIE})
body = r.json()
check("legacy 200", r.status_code == 200)
check("legacy envelope unchanged", body["status"] == "success" and set(["match", "distance"]) <= set(body["data"]))
check("legacy match true", body["data"]["match"] is True, str(body["data"]))
check("legacy tolerance now 0.5", body["data"]["tolerance"] == 0.5, str(body["data"]["tolerance"]))

r = client.post("/compare-fr", json={"reference_image": BUDI_PROFILE, "target_image": ANDI_SELFIE})
check(
    "lookalike still passes legacy 1:1 (why /verify matters)",
    r.json()["data"]["match"] is True,
    f"distance={r.json()['data']['distance']:.3f}",
)

r = client.post("/compare-fr", json={"reference_image": BUDI_PROFILE, "target_image": STRANGER})
check("legacy stranger rejected", r.json()["data"]["match"] is False)

r = client.post("/compare-fr", json={"reference_image": "bad", "target_image": BUDI_SELFIE})
check("legacy error stays HTTP 200", r.status_code == 200, str(r.status_code))
check("legacy error envelope unchanged", r.json()["status"] == "error" and "errors" in r.json())

# The legacy endpoint follows the same policy, since production still calls it.
BOXES["n"] = 2
r = client.post("/compare-fr", json={"reference_image": BUDI_PROFILE, "target_image": BUDI_SELFIE})
body = r.json()["data"]
check("legacy tolerates a bystander", body["match"] is True, r.text[:200])
check("legacy reports the extra face", body["reason"] == "extra_faces_present", str(body["reason"]))
check("legacy counts the extra face", body["quality"]["extra_faces"] == 1)

import config as _c
_c.MAX_EXTRA_FACES = 0
r = client.post("/compare-fr", json={"reference_image": BUDI_PROFILE, "target_image": BUDI_SELFIE})
check("legacy opt-out refuses multi-face",
      r.json()["status"] == "error" and r.json()["reason"] == "multiple_faces", r.text[:140])
_c.MAX_EXTRA_FACES = 2
BOXES["n"] = 1

print("\n== calibration log ==")
rows = main.store._conn.execute("SELECT source, decision, COUNT(*) n FROM verify_log GROUP BY 1,2").fetchall()
summary = {(r["source"], r["decision"]): r["n"] for r in rows}
check("verify attempts logged", sum(v for (s, _), v in summary.items() if s == "verify") >= 8, str(summary))
check("legacy attempts logged", sum(v for (s, _), v in summary.items() if s == "legacy") >= 3, str(summary))
row = main.store._conn.execute(
    "SELECT * FROM verify_log WHERE reasons LIKE '%identity_mismatch%' LIMIT 1"
).fetchone()
check("impostor row keeps runner-up for investigation", row is not None and row["runner_up_id"] == "EMP-ANDI",
      "" if row is None else str(row["runner_up_id"]))


print("")
print("== /verify: liveness gate ==")
# The real model is measured by tools/measure_liveness.py.  Here the point is
# the gate's behaviour: what the API does with a live verdict, a spoof verdict,
# and a model that has been asked for but cannot run.
import liveness as _lv

_saved_check, _saved_available = _lv.check, _lv.available
_cfg.LIVENESS_MODE = "model"
try:
    _lv.available = lambda: True

    _lv.check = lambda image, bbox: _lv.LivenessResult(0.97, True, "stub")
    r = client.post("/verify", json={"tenant_id": "PT-ABC", "employee_id": "EMP-BUDI",
                                     "image": BUDI_SELFIE})
    body = r.json().get("data", {})
    check("live face passes the gate", r.status_code == 200 and body.get("match") is True,
          r.text[:160])
    check("liveness score returned", body.get("liveness", {}).get("score") == 0.97,
          str(body.get("liveness")))

    _lv.check = lambda image, bbox: _lv.LivenessResult(0.02, False, "stub")
    r = client.post("/verify", json={"tenant_id": "PT-ABC", "employee_id": "EMP-BUDI",
                                     "image": BUDI_SELFIE})
    check("spoof refused with 422", r.status_code == 422, str(r.status_code))
    check("spoof_suspected reason", r.json().get("reason") == "spoof_suspected", r.text[:160])

    # Identity must not be consulted once liveness fails - a photo of the right
    # person cannot pass on the strength of being the right person.
    r = client.post("/verify", json={"tenant_id": "PT-ABC", "employee_id": "EMP-NOBODY",
                                     "image": BUDI_SELFIE})
    check("liveness runs before identity", r.json().get("reason") == "spoof_suspected",
          r.text[:160])

    _lv.check = lambda image, bbox: None
    _lv.available = lambda: False
    r = client.post("/verify", json={"tenant_id": "PT-ABC", "employee_id": "EMP-BUDI",
                                     "image": BUDI_SELFIE})
    check("unavailable model fails closed with 503", r.status_code == 503, str(r.status_code))
    check("liveness_unavailable reason", r.json().get("reason") == "liveness_unavailable",
          r.text[:160])

    _cfg.LIVENESS_REQUIRED = False
    r = client.post("/verify", json={"tenant_id": "PT-ABC", "employee_id": "EMP-BUDI",
                                     "image": BUDI_SELFIE})
    check("LIVENESS_REQUIRED=false passes unassessed",
          r.status_code == 200 and r.json()["data"]["liveness"] is None, r.text[:160])
    _cfg.LIVENESS_REQUIRED = True
finally:
    _lv.check, _lv.available = _saved_check, _saved_available
    _cfg.LIVENESS_MODE = "off"


print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
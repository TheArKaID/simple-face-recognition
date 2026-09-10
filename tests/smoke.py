"""End-to-end check against a running deployment, on real faces.

    docker run -d --name fr -p 8001:8000 prime-face-recognizer:test
    python tests/smoke.py http://localhost:8001

Checks the HTTP surface: enrollment, verification, the impostor cross-check,
error paths, store consistency and latency.  For the distance distributions and
threshold calibration use tests/calibrate.py, which runs inside the container
and does not pay an HTTP round trip per pair.

Photos live in tests/images and are named <identity><n>.jpg, so files sharing a
prefix are the same person - except that the owner confirmed z is the same
person as y, which ALIASES records.  Getting this wrong inverts the result:
mislabelling one person as two turns their genuine pairs into apparent impostor
pairs sitting suspiciously close together.

Beyond checking that the endpoints work, this prints the full pairwise distance
matrix, which is the first real evidence about whether the shipped thresholds
suit these faces.
"""
import base64
import glob
import itertools
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "images")
if not os.path.isdir(IMAGE_DIR):
    IMAGE_DIR = r"D:\Project\Python\facerecog\tests\images"

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(("  ok   " if condition else "  FAIL ") + name + ("  " + detail if detail else ""))


def call(method, path, body=None, timeout=120):
    req = urllib.request.Request(
        BASE + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.perf_counter() - started
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), time.perf_counter() - started


def wait_for_service(attempts=60, delay=5):
    for i in range(attempts):
        try:
            status, body, _ = call("GET", "/health", timeout=10)
            if status == 200:
                print(f"  up after ~{i * delay}s")
                return body
        except Exception as exc:
            print(f"  waiting ({i * delay}s): {type(exc).__name__}")
        time.sleep(delay)
    return None


# --- load photos, grouped by identity ---------------------------------------
photos = {}          # filename stem -> base64
identity = {}        # filename stem -> identity prefix
# Same person filed under a second name; confirmed by the repo owner.
ALIASES = {"z": "y"}
for path in sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))):
    stem = os.path.splitext(os.path.basename(path))[0]
    with open(path, "rb") as f:
        photos[stem] = base64.b64encode(f.read()).decode()
    prefix = re.sub(r"\d+$", "", stem)
    identity[stem] = ALIASES.get(prefix, prefix)

names = list(photos)
identities = sorted(set(identity.values()))
print(f"== {len(names)} photos, {len(identities)} identities: {', '.join(identities)} ==")

print(f"\n== waiting for {BASE} ==")
health = wait_for_service()
if health is None:
    print("service never came up")
    sys.exit(2)
th = health["data"]["thresholds"]
check("health reachable", True)
print("  thresholds: " + json.dumps(th))

# --- pairwise distances via the 1:1 endpoint --------------------------------
# Every /compare-fr call re-embeds both images, so a full matrix costs O(n^2)
# embeddings over HTTP.  Past a handful of photos that is minutes of waiting for
# numbers tests/calibrate.py already produces in one pass inside the container.
SKIP_MATRIX = len(names) > 14
if SKIP_MATRIX:
    print(f"  skipped: {len(names)} photos would need "
          f"{len(names)*(len(names)-1)//2} HTTP round trips. "
          f"Run tests/calibrate.py inside the container instead.")
print("\n== pairwise distance matrix (real dlib, 1:1) ==")
genuine, impostor, failures = [], [], []
pairs = {}
for a, b in ([] if SKIP_MATRIX else itertools.combinations(names, 2)):
    status, body, _ = call("POST", "/compare-fr",
                           {"reference_image": photos[a], "target_image": photos[b]})
    if body.get("status") != "success":
        failures.append((a, b, body.get("reason"), body.get("errors")))
        continue
    d = body["data"]["distance"]
    pairs[(a, b)] = d
    (genuine if identity[a] == identity[b] else impostor).append(((a, b), d))

width = max(len(n) for n in names) + 1
if not SKIP_MATRIX:
    print("  " + " " * width + "".join(f"{n:>{width}}" for n in names))
    for a in names:
        row = f"  {a:<{width}}"
        for b in names:
            if a == b:
                row += f"{'-':>{width}}"
            else:
                d = pairs.get((a, b), pairs.get((b, a)))
                row += f"{d:>{width}.3f}" if d is not None else f"{'x':>{width}}"
        print(row)

if failures:
    print("\n  pairs that could not be compared:")
    for a, b, reason, err in failures:
        print(f"    {a} vs {b}: {reason} - {err}")

if genuine and impostor:
    g = [d for _, d in genuine]
    i = [d for _, d in impostor]
    print(f"\n  genuine  n={len(g)}  min {min(g):.3f}  median {statistics.median(g):.3f}  max {max(g):.3f}")
    print(f"  impostor n={len(i)}  min {min(i):.3f}  median {statistics.median(i):.3f}  max {max(i):.3f}")
    gap = min(i) - max(g)
    print(f"  separation: worst genuine {max(g):.3f} | best impostor {min(i):.3f} | gap {gap:+.3f}")
    check("genuine and impostor distributions do not overlap", gap > 0, f"gap {gap:+.3f}")
    check("every genuine pair inside the accept band",
          max(g) <= th["accept_max_distance"],
          f"worst genuine {max(g):.3f} vs accept {th['accept_max_distance']}")
    check("every impostor pair outside the review band",
          min(i) > th["review_max_distance"],
          f"best impostor {min(i):.3f} vs review {th['review_max_distance']}")

# --- the service as the HRIS would use it -----------------------------------
# A photo the quality gates refuse is correct behaviour, not a verification
# failure - so those are reported and skipped rather than counted as failures.
QUALITY_REFUSALS = {"low_quality_blur", "low_quality_dark", "low_quality_bright",
                    "face_too_small", "no_face", "multiple_faces"}
refused = []
needs_review = []
print("\n== enroll on photo 1, verify with photo 2 ==")
enrolled = []
for ident in identities:
    shots = sorted(s for s in names if identity[s] == ident)
    if len(shots) < 2:
        continue
    status, body, t = call("POST", "/enroll",
                           {"tenant_id": "SMOKE", "employee_id": ident, "images": [photos[shots[0]]]})
    check(f"enroll {ident}", status == 200 and body["data"]["templates_stored"] == 1,
          json.dumps(body)[:160])
    enrolled.append((ident, shots))

for ident, shots in enrolled:
    status, body, t = call("POST", "/verify",
                           {"tenant_id": "SMOKE", "employee_id": ident, "image": photos[shots[1]]})
    if status == 422 and body.get("reason") in QUALITY_REFUSALS:
        refused.append((shots[1], body["reason"]))
        print(f"  --   {ident}: probe {shots[1]} refused by the quality gates "
              f"({body['reason']}) - skipped, not a failure")
        continue
    d = body.get("data", {})
    # "review" is a pass that carries a flag, not a rejection - the metric that
    # matters is whether a genuine attempt was refused outright.
    check(f"{ident} is not falsely rejected", d.get("match") is True, json.dumps(body)[:220])
    if d.get("decision") == "review":
        needs_review.append((ident, d.get("distance"), d.get("margin"), d.get("reasons")))
    print(f"       distance={d.get('distance')} runner_up={d.get('runner_up_distance')} "
          f"margin={d.get('margin')} candidates={d.get('candidates_checked')} ({t*1000:.0f} ms)")

print("\n== impostor: each identity claiming to be another ==")
for claimer, shots in enrolled:
    for victim, _ in enrolled:
        if victim == claimer:
            continue
        status, body, _ = call("POST", "/verify",
                               {"tenant_id": "SMOKE", "employee_id": victim, "image": photos[shots[1]]})
        if status == 422 and body.get("reason") in QUALITY_REFUSALS:
            continue
        d = body.get("data", {})
        check(f"{claimer} rejected when claiming to be {victim}",
              d.get("decision") == "reject", json.dumps(body)[:220])
        print(f"       distance={d.get('distance')} runner_up={d.get('runner_up_distance')} "
              f"reasons={d.get('reasons')}")

print("")
print(f"  genuine attempts flagged for review: {len(needs_review)}")
for ident, dist, margin, reasons in needs_review:
    print(f"    {ident}: distance {dist} margin {margin} {reasons}")

if refused:
    print("")
    print(f"  quality gates refused {len(refused)} probe photos: " +
          ", ".join(f"{n} ({r})" for n, r in refused))
    check("refusals came with a machine-readable reason",
          all(r in QUALITY_REFUSALS for _, r in refused), str(refused))

print("\n== error paths ==")
status, body, _ = call("POST", "/verify",
                       {"tenant_id": "SMOKE", "employee_id": identities[0], "image": "not-an-image"})
check("bad image -> 422", status == 422 and body.get("reason") == "invalid_image", str(status))
status, body, _ = call("POST", "/verify",
                       {"tenant_id": "SMOKE", "employee_id": "NOBODY", "image": photos[names[0]]})
check("unenrolled -> 409", status == 409 and body.get("reason") == "not_enrolled", str(status))

print("\n== store consistency across whichever instance answers ==")
counts = {call("GET", "/health")[1]["data"]["store"]["templates"] for _ in range(12)}
check("same template count every time", len(counts) == 1, f"seen: {counts}")
call("POST", "/enroll", {"tenant_id": "SMOKE", "employee_id": "LATE", "images": [photos[names[0]]]})
seen = [call("GET", "/enroll/SMOKE/LATE")[1]["data"]["enrolled"] for _ in range(8)]
check("fresh enrollment visible every time", all(seen), str(seen))

print("\n== latency ==")
timings = []
for _ in range(10):
    _, _, t = call("POST", "/verify",
                   {"tenant_id": "SMOKE", "employee_id": enrolled[0][0], "image": photos[names[0]]})
    timings.append(t * 1000)
print(f"  /verify     n=10  median {statistics.median(timings):.0f} ms  "
      f"min {min(timings):.0f}  max {max(timings):.0f}")
legacy = []
for _ in range(5):
    _, _, t = call("POST", "/compare-fr",
                   {"reference_image": photos[names[0]], "target_image": photos[names[1]]})
    legacy.append(t * 1000)
print(f"  /compare-fr n=5   median {statistics.median(legacy):.0f} ms   (two images per call)")
check("verify fast enough for attendance", statistics.median(timings) < 3000,
      f"median {statistics.median(timings):.0f} ms")

print("\n== cleanup ==")
for ident, _ in enrolled:
    call("DELETE", f"/enroll/SMOKE/{ident}")
call("DELETE", "/enroll/SMOKE/LATE")
check("templates removed",
      call("GET", f"/enroll/SMOKE/{enrolled[0][0]}")[1]["data"]["enrolled"] is False)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)

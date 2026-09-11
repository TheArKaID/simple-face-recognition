"""Do four replicas matching faces at the same time step on each other?

docker-stack.yml runs `replicas: 4` behind the ingress mesh, all sharing one
SQLite file on one volume. tests/test_replicas.py already covers the store logic
across processes with a stubbed recogniser; this drives the real thing over HTTP:
real ONNX inference, real liveness, real concurrent writes to verify_log.

    python tests/concurrency.py http://localhost:8010 http://localhost:8011 ...

What it is looking for, in order of how much it would matter:

  * any 5xx at all - a correct verification that fails because something
    incidental (the calibration log, say) could not complete
  * "database is locked" surfacing to a caller
  * decisions that disagree between replicas for the same input, which would
    mean one of them is serving a stale template index
  * latency under load, against the same request run alone
"""
import base64
import glob
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

BASES = sys.argv[1:] or ["http://localhost:8000"]
IMAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
TENANT = "CONC"
WORKERS = int(os.getenv("CONC_WORKERS", 8))
REQUESTS = int(os.getenv("CONC_REQUESTS", 64))

PASS, FAIL = [], []
_print_lock = threading.Lock()


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    with _print_lock:
        print(("  ok   " if condition else "  FAIL ") + name + ("  " + detail if detail else ""))


def call(base, method, path, body=None, timeout=120):
    req = urllib.request.Request(
        base + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.perf_counter() - started
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read())
        except Exception:
            payload = {}
        return e.code, payload, time.perf_counter() - started
    except Exception as e:
        return 0, {"transport_error": f"{type(e).__name__}: {e}"}, time.perf_counter() - started


def wait_for(base, attempts=60, delay=5):
    for i in range(attempts):
        status, _, _ = call(base, "GET", "/health", timeout=10)
        if status == 200:
            return True
        time.sleep(delay)
    return False


print(f"== {len(BASES)} instances, {REQUESTS} requests, {WORKERS} concurrent ==")
for base in BASES:
    if not wait_for(base):
        print(f"{base} never came up")
        sys.exit(2)
check("every instance answers /health", True, f"{len(BASES)} up")

# Confirm they really do share one store - otherwise the rest proves nothing.
stores = set()
for base in BASES:
    _, body, _ = call(base, "GET", "/health")
    stores.add(body["data"]["store"]["templates"])
check("instances report the same template count", len(stores) == 1, f"seen {stores}")

# --- enrol through one instance, verify through all of them -----------------
photos = {}
for path in sorted(glob.glob(os.path.join(IMAGES, "*.jpg"))):
    stem = os.path.splitext(os.path.basename(path))[0]
    with open(path, "rb") as fh:
        photos[stem] = base64.b64encode(fh.read()).decode()

identities = {}
for stem in photos:
    ident = "y" if stem[0] in "yz" else stem.rstrip("0123456789")
    identities.setdefault(ident, []).append(stem)
identities = {k: sorted(v) for k, v in identities.items() if len(v) >= 2}

enrolled = []
for ident, shots in sorted(identities.items()):
    for shot in shots[:-1]:
        status, body, _ = call(BASES[0], "POST", "/enroll",
                               {"tenant_id": TENANT, "employee_id": ident,
                                "images": [photos[shot]]})
        if status == 200:
            enrolled.append((ident, shots[-1]))
            break
check("enrolled a roster through one instance", len(enrolled) >= 5, f"{len(enrolled)} identities")

# Every instance must see that roster, or one of them is serving a stale index.
seen = set()
for base in BASES:
    _, body, _ = call(base, "GET", f"/enroll/{TENANT}/{enrolled[0][0]}")
    seen.add(body["data"]["enrolled"])
check("every instance sees the new enrolment", seen == {True}, str(seen))

# --- baseline: one request at a time -----------------------------------------
ident, probe = enrolled[0]
solo = []
for _ in range(5):
    _, _, elapsed = call(BASES[0], "POST", "/verify",
                         {"tenant_id": TENANT, "employee_id": ident, "image": photos[probe]})
    solo.append(elapsed * 1000)
solo_median = statistics.median(solo)
print(f"\n  unloaded /verify median: {solo_median:.0f} ms")

# --- the actual load ---------------------------------------------------------
jobs = [random.choice(enrolled) for _ in range(REQUESTS)]
results = []


def one(index):
    ident, probe = jobs[index]
    base = BASES[index % len(BASES)]
    status, body, elapsed = call(base, "POST", "/verify",
                                 {"tenant_id": TENANT, "employee_id": ident,
                                  "image": photos[probe]})
    decision = (body.get("data") or {}).get("decision")
    return {
        "base": base,
        "ident": ident,
        "probe": probe,
        "status": status,
        "decision": decision,
        "reason": body.get("reason"),
        "errors": body.get("errors"),
        "transport": (body or {}).get("transport_error"),
        "ms": elapsed * 1000,
    }


started = time.perf_counter()
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    results = list(pool.map(one, range(len(jobs))))
wall = time.perf_counter() - started

print(f"\n== {len(results)} requests in {wall:.1f}s "
      f"({len(results)/wall:.1f}/s across {len(BASES)} instances) ==")

statuses = Counter(r["status"] for r in results)
print(f"  status codes: {dict(statuses)}")
server_errors = [r for r in results if r["status"] >= 500 or r["status"] == 0]
check("no 5xx and no dropped connections", not server_errors,
      json.dumps(server_errors[:3], default=str)[:300])

locked = [r for r in results
          if "locked" in json.dumps(r.get("errors") or "").lower()
          or "locked" in (r.get("transport") or "").lower()]
check("no 'database is locked' reached a caller", not locked,
      json.dumps(locked[:2], default=str)[:200])

decisions = Counter(r["decision"] for r in results)
print(f"  decisions: {dict(decisions)}")
check("every genuine attempt was accepted",
      set(decisions) <= {"accept"}, str(dict(decisions)))

# The same probe sent to different instances must decide the same way; a
# divergence means one instance is matching against a different index.
by_probe = {}
for r in results:
    by_probe.setdefault((r["ident"], r["probe"]), set()).add(r["decision"])
divergent = {k: v for k, v in by_probe.items() if len(v) > 1}
check("instances agree on identical input", not divergent, str(divergent))

times = sorted(r["ms"] for r in results)
p50 = statistics.median(times)
p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
print(f"\n  latency under load: p50 {p50:.0f} ms   p95 {p95:.0f} ms   max {times[-1]:.0f} ms")
print(f"  unloaded p50 was {solo_median:.0f} ms -> {p50/solo_median:.1f}x")
per_base = {}
for r in results:
    per_base.setdefault(r["base"], []).append(r["ms"])
for base, ms in sorted(per_base.items()):
    print(f"    {base}: n={len(ms)} p50 {statistics.median(ms):.0f} ms")

# Queueing is expected and fine; a blowup past it means thread oversubscription
# or lock convoy rather than honest queueing.
#
# The budget is the queue depth each instance actually sees - concurrent callers
# spread over the instances - times the unloaded cost of one request, with 1.5x
# of slack.  An earlier version used `len(BASES) * 3`, which had the scaling
# backwards: it granted MORE latency budget the more instances were sharing the
# load, and ignored the concurrency level entirely.  Against a single instance
# at 8 concurrent callers it demanded 3x where honest serialisation alone costs
# 8x, so it failed a system that was behaving correctly.
queue_depth = WORKERS / len(BASES)
budget = solo_median * queue_depth * 1.5
check("latency degrades proportionally, not pathologically",
      p50 < budget,
      f"p50 {p50:.0f} ms vs budget {budget:.0f} ms "
      f"({solo_median:.0f} ms unloaded x {queue_depth:.1f} queued x 1.5)")

print("\n== cleanup ==")
for ident, _ in enrolled:
    call(BASES[0], "DELETE", f"/enroll/{TENANT}/{ident}")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)

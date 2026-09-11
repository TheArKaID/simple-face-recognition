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

# There is no closed-form budget here, and two attempts at one were both wrong.
#
# The first scaled the allowance by `len(BASES) * 3`, which had it backwards:
# more instances sharing the load bought MORE latency allowance, and the
# concurrency level was ignored entirely.  The second used the per-instance
# queue depth, WORKERS / len(BASES), which assumes each instance has its own
# CPU.  Measured on one host with four replicas, that is false: every replica
# saturated its 2-CPU limit at ~200% while the machine served 8 CPU-equivalents
# total, so 4x the replicas returned 2.3x the throughput (1.5 -> 3.4 req/s) and
# a single request was slower than its unloaded cost even at queue depth 1.
# Replicas on one machine do not have independent capacity, so no per-instance
# queueing model can hold.
#
# What is left is the operational question rather than a ratio: does a clock-in
# finish fast enough to be usable, and does anything come back broken?  The
# correctness checks above carry the pathology signal - a lock convoy shows up
# as 'database is locked' or a 5xx, not as a slow percentile.  Scaling is
# reported for comparison across runs and deliberately not asserted, because a
# laptop under other load cannot support that claim.
ceiling = float(os.getenv('CONC_P95_CEILING_MS', 10000))
print(f"  scaling: {len(times) / wall:.1f} req/s across {len(BASES)} instance(s), p50 {p50 / solo_median:.1f}x unloaded")
check("p95 latency within the operational ceiling",
      p95 < ceiling,
      f"p95 {p95:.0f} ms vs ceiling {ceiling:.0f} ms "
      f"(set CONC_P95_CEILING_MS to match your clock-in budget)")

print("\n== cleanup ==")
for ident, _ in enrolled:
    call(BASES[0], "DELETE", f"/enroll/{TENANT}/{ident}")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)

"""Reproduce the Swarm deployment: several replicas sharing one template store.

docker-stack.yml runs `replicas: 4`.  Each replica is a separate process with
its own in-memory index cache, all reading and writing one SQLite file on a
shared volume.  The failure mode this guards against is subtle: a replica whose
cache is already warm will happily keep serving a stale index and never notice
that another replica enrolled somebody, so the 1:N cross-check silently stops
seeing part of the workforce.

Real subprocesses are used rather than threads, because threads would share the
cache and hide exactly the bug under test.
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Vectors are supplied directly here, so no recogniser is needed - the stub
# backend satisfies engine.py's contract without weights or onnxruntime.
os.environ["FACE_ENGINE"] = "stub"
os.environ["FACE_LIVENESS_MODE"] = "off"


# --------------------------------------------------------------------------
# Worker: one process = one replica, kept alive so its cache stays warm.
# --------------------------------------------------------------------------
def run_worker(db_path):
    import numpy as np
    import engine
    import matcher
    from store import TemplateStore

    store = TemplateStore(db_path)

    def vec(seed, offset=0.0):
        dim = engine.EMBEDDING_DIM
        rng = np.random.default_rng(seed)
        v = rng.normal(size=dim).astype(np.float32)
        v /= np.linalg.norm(v)
        if offset:
            d = np.random.default_rng(7).normal(size=dim).astype(np.float32)
            d /= np.linalg.norm(d)
            v = v + d * offset
        return v.astype(np.float32)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        cmd = json.loads(line)
        op = cmd["op"]
        try:
            if op == "enroll":
                n = store.enroll(
                    cmd["tenant"], cmd["employee"], [(vec(cmd["seed"], cmd.get("offset", 0.0)), {})]
                )
                out = {"ok": True, "templates": n}
            elif op == "delete":
                out = {"ok": True, "removed": store.delete_employee(cmd["tenant"], cmd["employee"])}
            elif op == "verify":
                idx = store.index(cmd["tenant"])
                d = matcher.verify(idx, cmd["employee"], vec(cmd["seed"], cmd.get("offset", 0.0)))
                out = {
                    "ok": True,
                    "decision": d.decision,
                    "reasons": d.reasons,
                    "candidates": d.candidates_checked,
                    "runner_up_id": d.runner_up_id,
                    "index_version": idx.version,
                }
            elif op == "index_size":
                out = {"ok": True, "size": store.index(cmd["tenant"]).size}
            elif op == "stop":
                break
            else:
                out = {"ok": False, "error": f"unknown op {op}"}
        except Exception as exc:  # surface, don't crash the replica
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if len(sys.argv) > 2 and sys.argv[1] == "--worker":
    run_worker(sys.argv[2])
    sys.exit(0)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
class Replica:
    def __init__(self, name, db_path):
        self.name = name
        self.proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--worker", db_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )

    def send(self, **cmd):
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def stop(self):
        try:
            self.proc.stdin.write(json.dumps({"op": "stop"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(("  ok   " if condition else "  FAIL ") + name + ("  " + detail if detail else ""))


def main():
    db = os.path.join(tempfile.mkdtemp(), "swarm.db")
    replicas = [Replica(f"r{i}", db) for i in range(4)]
    r0, r1, r2, r3 = replicas
    T = "PT-ABC"

    try:
        print("\n== a template enrolled on one replica is visible on the others ==")
        r0.send(op="enroll", tenant=T, employee="EMP-BUDI", seed=1)
        for r in (r1, r2, r3):
            res = r.send(op="verify", tenant=T, employee="EMP-BUDI", seed=1, offset=0.15)
            check(f"{r.name} sees Budi enrolled by r0", res["decision"] == "accept", str(res))

        print("\n== a replica with a WARM cache still sees later writes ==")
        # r1's cache is warm from the verify above.  This is the regression test:
        # without cross-process invalidation r1 keeps serving the old index.
        before = r1.send(op="index_size", tenant=T)["size"]
        r0.send(op="enroll", tenant=T, employee="EMP-ANDI", seed=1, offset=0.42)
        after = r1.send(op="index_size", tenant=T)["size"]
        check("warm replica picks up the new template", after == before + 1, f"{before} -> {after}")

        print("\n== the cross-check works across replicas ==")
        # Andi taking attendance as Budi, judged by a replica that never wrote
        # either template.
        res = r2.send(op="verify", tenant=T, employee="EMP-BUDI", seed=1, offset=0.43)
        check("impostor rejected by a different replica", res["decision"] == "reject", str(res))
        check("identity_mismatch flagged", "identity_mismatch" in res["reasons"], str(res["reasons"]))
        check("runner-up correctly identified", res["runner_up_id"] == "EMP-ANDI", str(res["runner_up_id"]))

        res = r3.send(op="verify", tenant=T, employee="EMP-ANDI", seed=1, offset=0.43)
        check("Andi's own attendance accepted elsewhere", res["decision"] == "accept", str(res))

        print("\n== deletion propagates too ==")
        r3.send(op="verify", tenant=T, employee="EMP-BUDI", seed=1, offset=0.15)  # warm r3
        r0.send(op="delete", tenant=T, employee="EMP-ANDI")
        res = r3.send(op="verify", tenant=T, employee="EMP-ANDI", seed=1, offset=0.43)
        check("warm replica sees the deletion", "not_enrolled" in res["reasons"], str(res))

        print("\n== concurrent writes from all four replicas ==")
        # Fire an enrol at every replica before reading any reply, so the writes
        # genuinely overlap on the shared file.
        for i, r in enumerate(replicas):
            r.proc.stdin.write(json.dumps(
                {"op": "enroll", "tenant": T, "employee": f"EMP-C{i}", "seed": 100 + i}
            ) + "\n")
            r.proc.stdin.flush()
        results = [json.loads(r.proc.stdout.readline()) for r in replicas]
        check("no write failed under contention", all(x["ok"] for x in results), str(results))

        sizes = {r.name: r.send(op="index_size", tenant=T)["size"] for r in replicas}
        check("all replicas converge on the same index", len(set(sizes.values())) == 1, str(sizes))
        check("index holds every template", set(sizes.values()) == {5}, str(sizes))

        print("\n== a fresh replica joining later sees everything ==")
        newcomer = Replica("r4", db)
        try:
            res = newcomer.send(op="verify", tenant=T, employee="EMP-BUDI", seed=1, offset=0.15)
            check("newly started replica works immediately", res["decision"] == "accept", str(res))
            check("newcomer sees full index", res["candidates"] == 5, str(res["candidates"]))
        finally:
            newcomer.stop()
    finally:
        for r in replicas:
            r.stop()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

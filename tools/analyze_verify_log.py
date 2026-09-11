"""What production traffic says about the thresholds.

This is step 6 - recalibration from real attendance data rather than a photo
fixture - and it has to start by admitting what the log cannot do.

`verify_log` records the employee_id that was CLAIMED, never whether the claim
was true.  There is no ground-truth column, so false-reject and false-accept
rates cannot be computed from it: a rejected attempt might be an impostor caught
or an honest employee in bad light, and nothing here can tell those apart.  Any
tool that prints an FRR from this table is making it up.

Two things it can do honestly.

**The impostor side survives on one weak assumption.**  `runner_up_distance` is
the distance to the nearest enrolled employee who is NOT the one claimed, so on
an honest clock-in it samples "how close does a real face get to somebody else".
But on an IMPOSTOR attempt the runner-up is the attacker's own identity, which
makes it a genuine distance wearing impostor clothing - and an earlier version of
this tool pooled both and reported a closest near-miss of 0.070, which was
simply someone matching themselves.

So the impostor side is taken only from attempts where attendance was RECORDED.
On those the claimed employee out-scored the runner-up by at least the configured
margin, which is strong evidence the runner-up is a different person.  The
assumption is that a recorded attendance was genuine; that is the same assumption
the service already acts on, so nothing new is being risked.  The minimum over
those rows is the closest real-world near-miss on record, and unlike a fixture it
grows more informative with every clock-in.

**The genuine side is visible but censored.**  `distance` on accepted rows is a
genuine-side sample only if you assume accepted attempts were really genuine,
and it is truncated: anything past the accept threshold was rejected and appears
in the log as a rejection indistinguishable from an impostor.  So the genuine
distribution can be described up to the threshold and no further, which means it
can justify moving the accept line DOWN but never up.

    docker exec <container> python /app/tools/analyze_verify_log.py

FACE_DB_PATH selects the database; reads only.  ANALYZE_DAYS limits the window
(default: everything).  Rows are partitioned by engine_id AND tenant_id:
distances from different vector spaces are not comparable, and "the nearest
other employee" only means something within one roster.
"""
import collections
import json
import os
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import engine  # noqa: E402

DAYS = os.getenv("ANALYZE_DAYS")
MIN_ROWS = int(os.getenv("ANALYZE_MIN_ROWS", "200"))


def rows_for(conn, source):
    sql = ("SELECT engine_id, tenant_id, employee_id, decision, distance, "
           "runner_up_distance, margin, reasons, quality, created_at "
           "FROM verify_log WHERE source = ?")
    args = [source]
    if DAYS:
        sql += " AND created_at >= datetime('now', ?)"
        args.append(f"-{int(DAYS)} days")
    return conn.execute(sql + " ORDER BY created_at", args).fetchall()


def describe(name, values, indent="    "):
    if not len(values):
        print(f"{indent}{name:<22} no data")
        return
    v = np.sort(np.asarray(values, dtype=float))
    print(f"{indent}{name:<22} n={len(v):<6d} min {v[0]:.3f}  p05 "
          f"{np.percentile(v, 5):.3f}  p50 {np.percentile(v, 50):.3f}  "
          f"p95 {np.percentile(v, 95):.3f}  max {v[-1]:.3f}")


def monthly(pairs):
    """(month, values) buckets, so a closing gap is visible as a trend."""
    buckets = collections.OrderedDict()
    for when, value in pairs:
        buckets.setdefault(str(when)[:7], []).append(value)
    return buckets


def analyse(engine_id, tenant_id, rows, findings):
    print("")
    current = engine_id == engine.ENGINE_ID
    tag = "" if current else "  (NOT the running engine - shown separately)"
    print(f"== {tenant_id or 'unknown tenant'} / {engine_id or 'unknown space'}: "
          f"{len(rows)} attempts{tag} ==")
    if not current:
        print("  These distances were produced by a different model pairing and")
        print("  are on their own scale. They cannot be pooled with the rows")
        print("  above or compared against the running thresholds.")

    decisions = collections.Counter(r["decision"] for r in rows)
    for decision, n in decisions.most_common():
        print(f"    {decision:<12}{n:>7}  {100.0 * n / len(rows):5.1f}%")

    # Why things were rejected.  A quality reason is an operations problem - a
    # camera, a light, a kiosk position - not an identity one, and the two get
    # confused because both show up as a rejected clock-in.
    reasons = collections.Counter()
    for r in rows:
        for reason in json.loads(r["reasons"] or "[]"):
            reasons[reason] += 1
    if reasons:
        print("")
        print("  reasons recorded")
        quality_codes = {"no_face", "low_confidence", "face_too_small",
                         "low_quality_blur", "low_quality_dark",
                         "low_quality_bright", "invalid_image"}
        for reason, n in reasons.most_common():
            kind = "operations" if reason in quality_codes else ""
            print(f"    {reason:<28}{n:>7}  {kind}")
        ops = sum(n for reason, n in reasons.items() if reason in quality_codes)
        if ops > 0.05 * len(rows):
            findings.append(
                f"{ops} of {len(rows)} attempts failed on image quality "
                f"({100.0 * ops / len(rows):.1f}%). That is a camera, lighting or "
                f"kiosk-placement problem, not a recognition one, and no "
                f"threshold change will fix it."
            )
        for reason in ("spoof_suspected", "liveness_unavailable"):
            if reasons.get(reason):
                findings.append(
                    f"{reasons[reason]} attempt(s) recorded {reason}."
                    + (" Each one is either a presentation attack or a false "
                       "alarm; read the rows before assuming either."
                       if reason == "spoof_suspected" else
                       " The service refused to decide, which is correct, but "
                       "it means attendance could not be recorded.")
                )

    # Restricted to recorded attendance - see the module docstring.  On a
    # rejected attempt the runner-up may well BE the person presenting, which
    # makes the number a genuine distance, not an impostor one.
    runner_up = [r["runner_up_distance"] for r in rows
                 if r["runner_up_distance"] is not None
                 and r["decision"] in ("accept", "review")]
    accepted = [r["distance"] for r in rows
                if r["decision"] == "accept" and r["distance"] is not None]
    # Only attempts that RECORDED attendance.  A rejected impostor has a large
    # negative margin by design, and pooling those with genuine ones makes the
    # worst margin look alarming when it is the mechanism working.
    margins = [r["margin"] for r in rows
               if r["margin"] is not None
               and r["decision"] in ("accept", "review")]

    print("")
    print("  distance distributions")
    describe("nearest other, if rec.", runner_up)
    describe("claimed, if accepted", accepted)
    describe("margin, if recorded", margins)

    if not current:
        return

    if runner_up:
        # A runner-up at essentially zero does not mean two people look alike;
        # it means one person is enrolled twice under two employee ids, and the
        # "impostor" distance is a person against themselves.  Reading that as
        # headroom would be nonsense, so it is called out and excluded.
        duplicates = [d for d in runner_up if d < 0.05]
        if duplicates:
            findings.append(
                f"{len(duplicates)} attempt(s) found a DIFFERENT employee id at a "
                f"distance under 0.05 - closer than any two people can be. That is "
                f"one person enrolled twice under two ids, not a lookalike. Find "
                f"them with tools/roster_audit.py and delete the duplicate before "
                f"reading anything else here."
            )
            runner_up = [d for d in runner_up if d >= 0.05]
            print("")
            print(f"  excluded {len(duplicates)} runner-up(s) under 0.05 as "
                  f"duplicate enrolments, not lookalikes")
    if runner_up:
        closest = float(min(runner_up))
        headroom = closest - config.ACCEPT_MAX_DISTANCE
        print("")
        print(f"  closest real near-miss {closest:.3f} vs accept line "
              f"{config.ACCEPT_MAX_DISTANCE:.3f} -> headroom {headroom:+.3f}")
        print("    Taken from recorded attendance only: there the claimed")
        print("    employee beat the runner-up by a margin, so the runner-up is")
        print("    almost certainly someone else. On a REJECTED attempt it may")
        print("    be the person presenting, and counting those put this figure")
        print("    at 0.070 in an earlier version - somebody matching themselves.")
        if headroom <= 0:
            findings.append(
                f"live traffic has produced a non-matching employee at {closest:.3f}, "
                f"at or inside the accept threshold {config.ACCEPT_MAX_DISTANCE}. A "
                f"1:1 comparison would have accepted them. The cross-check margin "
                f"is what stopped it; lower the accept threshold."
            )
        elif headroom < 0.03:
            findings.append(
                f"only {headroom:.3f} between the accept threshold and the closest "
                f"non-matching employee seen in production ({closest:.3f})."
            )

    if margins:
        worst = float(min(margins))
        print(f"  narrowest recorded attendance won by {worst:+.3f}"
              f"  (configured minimum {config.MIN_IMPOSTOR_MARGIN})")
        if worst < config.MIN_IMPOSTOR_MARGIN:
            findings.append(
                f"attendance was recorded on a margin of {worst:+.3f}, under the "
                f"configured minimum {config.MIN_IMPOSTOR_MARGIN}. The claimed "
                f"employee barely out-scored somebody else."
            )

    # A closing gap is a trend, not a level.  One month of rows says where you
    # are; several say which way you are going, which is the question headcount
    # growth actually raises.
    buckets = monthly([(r["created_at"], r["runner_up_distance"]) for r in rows
                       if r["runner_up_distance"] is not None])
    if len(buckets) > 1:
        print("")
        print("  closest non-matching employee, by month")
        for month, values in buckets.items():
            print(f"    {month}   n={len(values):<6d} closest {min(values):.3f}"
                  f"   median {np.percentile(values, 50):.3f}")
        first, last = list(buckets.values())[0], list(buckets.values())[-1]
        drift = min(last) - min(first)
        if drift < -0.02:
            findings.append(
                f"the closest non-matching employee has moved {drift:+.3f} from the "
                f"first month on record to the last. Separation is narrowing as the "
                f"roster grows, which is expected - re-run tests/calibrate.py "
                f"before it reaches the accept threshold, not after."
            )

    # Employees who keep failing.  Usually their enrolment photos are poor, not
    # their faces - which is a re-enrol, not a threshold change.
    per_employee = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if not r["employee_id"]:
            continue
        counts = per_employee[r["employee_id"]]
        counts[0] += 1
        if r["decision"] != "accept":
            counts[1] += 1
    struggling = sorted(
        ((fails / total, emp, total, fails)
         for emp, (total, fails) in per_employee.items()
         if total >= 5 and fails),
        reverse=True,
    )[:10]
    if struggling:
        print("")
        print("  employees whose attempts do not land on accept")
        print(f"    {'employee':<16}{'attempts':>9}{'not accepted':>14}{'rate':>8}")
        for rate, emp, total, fails in struggling:
            print(f"    {emp:<16}{total:>9}{fails:>14}{100.0 * rate:>7.0f}%")
        worst_rate, worst_emp, total, fails = struggling[0]
        if worst_rate >= 0.25:
            findings.append(
                f"employee {worst_emp} failed {fails} of {total} attempts "
                f"({100.0 * worst_rate:.0f}%). Before touching thresholds, check "
                f"their enrolment photos - one bad template drags every attempt, "
                f"and tools/roster_audit.py will show whether theirs are duplicates "
                f"or a single pose."
            )

    print("")
    if len(rows) < MIN_ROWS:
        print(f"  {len(rows)} attempts is not enough to retune on. A threshold is a")
        print(f"  decision about the tail of a distribution, and the tail is what")
        print(f"  a small sample is worst at. Wait for {MIN_ROWS}+, then re-run.")
    elif runner_up and accepted:
        floor = float(max(accepted))
        ceiling = float(min(runner_up))
        print("  what this log supports")
        print(f"    accepted attempts reach {floor:.3f}; the nearest other person")
        print(f"    got to {ceiling:.3f}. Any accept threshold between them keeps")
        print(f"    every attempt that was accepted and still refuses every")
        print(f"    near-miss on record.")
        if ceiling > floor:
            print(f"    -> usable range {floor:.3f} .. {ceiling:.3f}")
            print("    Pick from the LOW end. The genuine side is censored here:")
            print("    attempts beyond the old threshold were rejected and are")
            print("    indistinguishable from impostors in this table, so this")
            print("    range can justify tightening but never loosening.")
        else:
            print("    -> no usable range: accepted attempts already reach past")
            print("       the closest near-miss. Recalibrate from photos with")
            print("       tests/calibrate.py; this log cannot separate them.")


db = config.DB_PATH
print(f"  database   {db}")
print(f"  engine     {engine.ENGINE_ID}")
print(f"  thresholds accept {config.ACCEPT_MAX_DISTANCE}  "
      f"review {config.REVIEW_MAX_DISTANCE}  margin {config.MIN_IMPOSTOR_MARGIN}")
print(f"  window     {'last ' + DAYS + ' days' if DAYS else 'all rows'}")

if not os.path.exists(db):
    sys.exit(f"  no database at {db}")

conn = sqlite3.connect("file:" + db.replace("?", "%3f") + "?mode=ro", uri=True)
conn.row_factory = sqlite3.Row

findings = []
verify_rows = rows_for(conn, "verify")
legacy_rows = rows_for(conn, "legacy")

if not verify_rows and not legacy_rows:
    print("")
    print("== no traffic on record ==")
    print("  verify_log is empty. The thresholds in use were calibrated on photo")
    print("  fixtures, and nothing has tested them against real clock-ins yet.")
    print("  Point the HRIS at /verify, then re-run this.")
    sys.exit(0)

# Partitioned by tenant as well as vector space.  A roster is per-tenant, so
# "the nearest other employee" only means anything within one - and pooling a
# production tenant with a test one silently poisons the statistics: the first
# run of this tool reported a closest near-miss of 0.000, which was a smoke-test
# tenant that enrols the same photo under several employee ids.
groups = collections.defaultdict(list)
for r in verify_rows:
    groups[(r["engine_id"], r["tenant_id"])].append(r)
order = sorted(groups, key=lambda k: (k[0] != engine.ENGINE_ID, -len(groups[k])))
for engine_id, tenant_id in order:
    analyse(engine_id, tenant_id, groups[(engine_id, tenant_id)], findings)

if legacy_rows:
    print("")
    print(f"== /compare-fr: {len(legacy_rows)} attempts ==")
    decisions = collections.Counter(r["decision"] for r in legacy_rows)
    for decision, n in decisions.most_common():
        print(f"    {decision:<12}{n:>7}  {100.0 * n / len(legacy_rows):5.1f}%")
    print("  These rows carry no employee_id and no runner-up, because the")
    print("  endpoint compares two photos with no roster behind it. They record")
    print("  distances but cannot say who was claimed, so they are not usable")
    print("  for calibration - only as a count of traffic still to migrate.")
    findings.append(
        f"{len(legacy_rows)} attempt(s) still went through /compare-fr, which runs "
        f"no liveness check and no 1:N cross-check. Those clock-ins were not "
        f"protected by anything measured in this repo."
    )

conn.close()

print("")
if findings:
    print(f"== {len(findings)} finding(s) ==")
    for i, finding in enumerate(findings, 1):
        print(f"  {i}. {finding}")
    sys.exit(1)
print("== no findings ==")
print("  Nothing in the log contradicts the thresholds in use. That is not the")
print("  same as confirming them: this table has no ground truth, so it can")
print("  reveal a threshold that is too loose but never prove one is right.")

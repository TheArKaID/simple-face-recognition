"""How would a new accept/review line have judged traffic already on record?

Run this against the CURRENTLY DEPLOYED database, with the NEW threshold
values about to go live, BEFORE deploying the code that enforces them. It
answers the one question that matters before any threshold change reaches
real employees: how many of their own past clock-ins would land in a
different band under the new line.

Every verify_log row keeps the raw `distance` regardless of which threshold
was active when it was decided, so this needs no re-verification, no image
files, and works read-only against a live database - it just re-bands each
recorded distance against the numbers you are about to switch to.

    python tools/threshold_impact.py /path/to/faces.db --accept 0.47 --review 0.52

Only stdlib (sqlite3, argparse) - runs inside the OLD container too, so this
check can run before the new code is even built.

Restricted to rows where source='verify' and the OLD decision was accept or
review - attendance that WAS recorded.  A rejected attempt is not analysed:
many rejections are the system working as intended (an impostor probe, a
cross-check catch), and a stricter line cannot turn a rejection into something
worse.  Only a previously-recorded attendance moving to review or reject is a
real user-facing regression, and that is exactly what this counts.
"""
import argparse
import os
import sqlite3
import sys

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("db")
ap.add_argument("--accept", type=float, required=True)
ap.add_argument("--review", type=float, required=True)
ap.add_argument("--engine-id", default=None,
                help="restrict to one vector space; default: whichever has the most rows")
ap.add_argument("--days", type=int, default=None,
                help="only look at the last N days (default: everything on record)")
args = ap.parse_args()

if args.review < args.accept:
    sys.exit(f"--review ({args.review}) must be >= --accept ({args.accept})")

if not os.path.exists(args.db):
    sys.exit(f"no database at {args.db}")
try:
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("SELECT 1 FROM verify_log LIMIT 1")
except sqlite3.OperationalError as exc:
    sys.exit(f"cannot read {args.db}: {exc}")

engine_id = args.engine_id
if engine_id is None:
    row = conn.execute(
        "SELECT engine_id, COUNT(*) AS n FROM verify_log "
        "WHERE source = 'verify' AND distance IS NOT NULL "
        "GROUP BY engine_id ORDER BY n DESC LIMIT 1"
    ).fetchone()
    if not row:
        sys.exit(f"no verify_log rows with a recorded distance in {args.db}")
    engine_id = row["engine_id"]
    print(f"  no --engine-id given; using the most common on record: "
          f"{engine_id!r} ({row['n']} rows)")

sql = ("SELECT employee_id, distance, decision, created_at FROM verify_log "
       "WHERE source = 'verify' AND engine_id = ? "
       "AND decision IN ('accept', 'review') AND distance IS NOT NULL")
params = [engine_id]
if args.days:
    sql += " AND created_at >= datetime('now', ?)"
    params.append(f"-{args.days} days")
rows = conn.execute(sql + " ORDER BY distance", params).fetchall()

if not rows:
    sys.exit(f"no recorded accept/review attempts for engine_id={engine_id!r}"
             + (f" in the last {args.days} days" if args.days else ""))


def band(distance):
    if distance <= args.accept:
        return "accept"
    if distance <= args.review:
        return "review"
    return "reject"


changed = [(r["employee_id"], r["distance"], r["decision"], band(r["distance"]),
            r["created_at"]) for r in rows if band(r["distance"]) != r["decision"]]

print(f"  engine       {engine_id}")
print(f"  new line     accept <= {args.accept}   review <= {args.review}")
print(f"  checked      {len(rows)} recorded accept/review attempts")
print(f"  would move   {len(changed)}  "
      f"({100.0 * len(changed) / len(rows):.1f}%)")

if changed:
    to_reject = [c for c in changed if c[3] == "reject"]
    to_review = [c for c in changed if c[3] == "review"]
    print("")
    print(f"  {len(to_reject)} would move all the way to REJECT - those clock-ins")
    print(f"  would fail outright under the new line, on a photo that used to work.")
    print(f"  {len(to_review)} would move from accept to review - still recorded,")
    print(f"  but now flagged, and closer to the new line than intended.")

    affected = sorted({c[0] for c in changed})
    print("")
    print(f"  {len(affected)} distinct employee(s) affected: "
          f"{', '.join(str(e) for e in affected[:15])}"
          f"{' ...' if len(affected) > 15 else ''}")

    print("")
    print(f"  {'employee':<16}{'distance':>10}  {'was':<8}-> new         when")
    for emp, d, old, new, when in changed[:30]:
        print(f"  {str(emp):<16}{d:>10.4f}  {old:<8}-> {new:<8}  {str(when)[:19]}")
    if len(changed) > 30:
        print(f"  ... and {len(changed) - 30} more")

    print("")
    print("  Before deploying: check whether the employees listed are borderline")
    print("  cases worth re-enrolling with clearer photos first, or a false alarm")
    print("  from one bad lighting day. tools/roster_audit.py shows their current")
    print("  templates; a re-enrol before the threshold changes costs nothing and")
    print("  may keep them out of this list entirely.")
else:
    print("")
    print("  Every previously recorded accept/review attempt stays in the same")
    print("  band under the new line. This is the number that matters most before")
    print("  deploying a threshold change to a live roster.")

conn.close()

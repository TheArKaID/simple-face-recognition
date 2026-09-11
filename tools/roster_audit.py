"""How close are the real enrolled employees to each other, right now?

Everything else in this repo measures separation on folders of photos.  That
answers a question about a fixture.  This answers the question about the
workforce: it reads the live template store and reports how much room is left
between the hardest genuine pair and the closest impostor pair among the people
actually enrolled.

The distinction matters because the separation gap is a MINIMUM over roughly
N^2/2 pairs.  Thirty employees give 42,921 impostor pairs; a hundred give about
half a million.  Every added employee is another chance that two people land
closer together than anyone measured, and a minimum can only fall.  So the gap
shrinks on its own as the company grows, without a line of code changing, and
nothing in the service notices.  This tool is what notices.

Run it after each enrolment round, or on a schedule:

    docker exec <container> python /app/tools/roster_audit.py

Exit status is 0 when there is room left, 1 when there is not, so it can drive
an alert.  FACE_DB_PATH selects the database; the audit only reads.

Two things it deliberately does not do.  It does not extrapolate to a future
headcount - the shrink rate depends on who joins, not on arithmetic.  And it
does not touch the thresholds: if the roster has outgrown them, that is a
calibration decision for a person, taken with tests/calibrate.py.
"""
import os
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import engine  # noqa: E402
from store import TemplateStore  # noqa: E402

WARN_HEADROOM = float(os.getenv("AUDIT_MIN_HEADROOM", "0.03"))


def read_only(db_path):
    """A read-only connection, so an audit can never damage the roster."""
    return sqlite3.connect("file:" + db_path.replace("?", "%3f") + "?mode=ro",
                           uri=True)


def tenants(db_path):
    """Tenant ids holding templates in the running engine's vector space."""
    try:
        conn = read_only(db_path)
    except sqlite3.OperationalError as exc:
        sys.exit(f"  cannot open {db_path} read-only: {exc}")
    try:
        rows = conn.execute(
            "SELECT DISTINCT tenant_id FROM face_template WHERE engine_id = ? "
            "ORDER BY tenant_id",
            (engine.ENGINE_ID,),
        ).fetchall()
        return [r[0] for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def audit_tenant(store, tenant_id):
    """Report one tenant's roster; returns the findings it produced."""
    index = store.index(tenant_id)
    ids, mat = index.employee_ids, index.matrix
    roster = sorted(set(ids.tolist()))
    print("")
    print(f"== tenant {tenant_id}: {len(roster)} employees, {len(ids)} templates ==")

    if len(roster) < 2:
        print("  Only one employee enrolled, so impostor distance is undefined.")
        print("  The cross-check has nothing to compare against yet and identity")
        print("  rests entirely on the 1:1 threshold until a second person exists.")
        return []

    # Full pairwise matrix.  A roster is a few hundred templates at most, so the
    # square is cheap, and exact beats sampling when the answer is a minimum.
    dist = np.stack([engine.distances(mat, mat[i]) for i in range(len(mat))])
    np.fill_diagonal(dist, np.inf)
    same = ids[:, None] == ids[None, :]

    genuine = dist[same & np.isfinite(dist)]
    impostor = dist[~same]
    closest = float(impostor.min())
    worst_genuine = float(genuine.max()) if len(genuine) else None
    findings = []

    print("  distances among enrolled templates")
    if len(genuine):
        print(f"    same person   n={len(genuine):<7d} min {genuine.min():.3f}"
              f"  p50 {pct(genuine, 50):.3f}  max {genuine.max():.3f}")
    else:
        print("    same person   none - every employee holds a single template.")
    print(f"    other people  n={len(impostor):<7d} min {impostor.min():.3f}"
          f"  p50 {pct(impostor, 50):.3f}  max {impostor.max():.3f}")

    if worst_genuine is not None:
        gap = closest - worst_genuine
        print("")
        print(f"  worst genuine {worst_genuine:.3f} | closest impostor {closest:.3f}"
              f" | gap {gap:+.3f}")
        if gap <= 0:
            findings.append(
                f"genuine and impostor distances now OVERLAP (gap {gap:+.3f}): two "
                f"different people sit closer than one person's own two photos, so "
                f"no 1:1 threshold separates them. The cross-check margin is the "
                f"only thing still deciding."
            )

    # Headroom is what this audit is really for.  The accept line has to stay
    # below every impostor pair; this says by how much it still does.
    headroom = closest - config.ACCEPT_MAX_DISTANCE
    print(f"  accept line {config.ACCEPT_MAX_DISTANCE:.3f}"
          f" | headroom to closest impostor {headroom:+.3f}")
    if headroom <= 0:
        findings.append(
            f"the accept threshold {config.ACCEPT_MAX_DISTANCE:.3f} is at or above "
            f"the closest impostor pair {closest:.3f}. A 1:1 comparison would "
            f"accept those two people as each other. Recalibrate."
        )
    elif headroom < WARN_HEADROOM:
        findings.append(
            f"only {headroom:.3f} left between the accept threshold and the "
            f"closest impostor pair. One more similar-looking hire can close it."
        )

    flat = int(np.argmin(np.where(same, np.inf, dist)))
    a, b = np.unravel_index(flat, dist.shape)
    print(f"  closest pair: {ids[a]} vs {ids[b]} at {closest:.3f}")

    # Named, because "the roster is getting tight" is not actionable and
    # "24 and 21 are getting tight" is.
    print("")
    print("  per-employee margin (nearest other person minus own nearest template)")
    print(f"    {'employee':<16}{'templates':>10}{'own':>8}{'nearest':>9}"
          f"{'margin':>9}  closest to")
    rows = []
    for emp in roster:
        mine = ids == emp
        n = int(mine.sum())
        cross = dist[mine][:, ~mine]
        d_other = float(cross.min())
        col = int(np.unravel_index(int(cross.argmin()), cross.shape)[1])
        nearest = str(ids[~mine][col])
        if n > 1:
            block = dist[mine][:, mine]
            d_self = float(block[np.isfinite(block)].min())
            margin = d_other - d_self
        else:
            d_self, margin = None, None
        rows.append((float("inf") if margin is None else margin,
                     emp, n, d_self, d_other, nearest))
    rows.sort()
    for margin, emp, n, d_self, d_other, nearest in rows:
        own = f"{d_self:.3f}" if d_self is not None else "  -  "
        shown = "   -  " if margin == float("inf") else f"{margin:+.3f}"
        print(f"    {emp:<16}{n:>10}{own:>8}{d_other:>9.3f}{shown:>9}  {nearest}")
        if margin != float("inf") and margin < config.MIN_IMPOSTOR_MARGIN:
            findings.append(
                f"employee {emp} has a margin of {margin:+.3f}, under the "
                f"configured minimum {config.MIN_IMPOSTOR_MARGIN}. Their probes "
                f"get downgraded to review or rejected. Re-enrol them with "
                f"clearer photos, and check they are not the same person as "
                f"{nearest} enrolled twice."
            )

    # A template that duplicates another wastes one of the per-employee slots:
    # it adds a second copy of one pose under one light, so it cannot widen the
    # range of appearances the employee is recognised across.
    dupes = [(emp, d_self) for _, emp, n, d_self, _, _ in rows
             if d_self is not None and d_self < 0.01]
    if dupes:
        findings.append(
            f"{len(dupes)} employee(s) hold near-identical templates "
            f"({', '.join(e for e, _ in dupes[:6])}): "
            f"{'; '.join(f'{e} at {d:.3f}' for e, d in dupes[:3])}. Each copy "
            f"spends one of the {config.MAX_TEMPLATES_PER_EMPLOYEE} slots without "
            f"adding a pose or a lighting condition. Re-enrol from photos taken "
            f"at different times."
        )

    singles = [emp for _, emp, n, _, _, _ in rows if n == 1]
    if singles:
        shown = ", ".join(singles[:8]) + (" ..." if len(singles) > 8 else "")
        print("")
        print(f"  {len(singles)} employee(s) hold a single template: {shown}")
        print("    Margin cannot be computed for them, and one photo carries one")
        print("    pose under one light. Measured on 30 identities, moving from")
        print("    one template to several took the worst genuine distance from")
        print("    0.368 to 0.326 and the worst margin from 0.274 to 0.288.")

    return findings


def log_summary(db_path):
    """What the service has actually been deciding, from verify_log."""
    conn = read_only(db_path)
    try:
        rows = conn.execute(
            "SELECT decision, COUNT(*) FROM verify_log GROUP BY decision"
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return []

    counts = {r[0]: r[1] for r in rows}
    total = sum(counts.values())
    print("")
    print("== decisions on record ==")
    if not total:
        print("  verify_log is empty. Nothing has called /verify yet, so none of")
        print("  the calibration above has met live traffic.")
        conn.close()
        return []

    for decision, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {decision:<24}{n:>8}  {100.0 * n / total:5.1f}%")

    findings = []
    reviews = counts.get("review", 0)
    if reviews:
        findings.append(
            f"{reviews} verification(s) landed in the review band. On every "
            f"dataset measured so far that band was empty, so these are faces "
            f"harder than anything the thresholds were calibrated on. Worth "
            f"reading before the next recalibration."
        )
        tight = conn.execute(
            "SELECT employee_id, distance, runner_up_id, margin, created_at "
            "FROM verify_log WHERE decision = 'review' "
            "ORDER BY margin ASC LIMIT 5"
        ).fetchall()
        print("")
        print("  tightest review decisions")
        for emp, d, rival, margin, when in tight:
            shown = "  -  " if margin is None else f"{margin:+.3f}"
            dist_shown = "  -  " if d is None else f"{d:.3f}"
            print(f"    {str(emp):<16} d={dist_shown}  runner-up "
                  f"{str(rival):<16} margin {shown}  {str(when)[:19]}")
    conn.close()
    return findings


db = config.DB_PATH
print(f"  database   {db}")
print(f"  engine     {engine.ENGINE_ID}  ({engine.EMBEDDING_DIM}-d)")
print(f"  thresholds accept {config.ACCEPT_MAX_DISTANCE}  "
      f"review {config.REVIEW_MAX_DISTANCE}  margin {config.MIN_IMPOSTOR_MARGIN}")

if not os.path.exists(db):
    sys.exit(f"  no database at {db} - nobody is enrolled yet.")

store = TemplateStore(db)
names = tenants(db)
all_findings = []

if not names:
    print("")
    print("== nothing enrolled ==")
    print(f"  No templates in the {engine.ENGINE_ID} vector space.")
    print("  Until employees are enrolled and the HRIS calls /verify, the 1:N")
    print("  cross-check protects nobody: traffic on /compare-fr compares two")
    print("  photos with no roster behind it and no liveness check.")
    stale = store.stats().get("stale_templates", 0)
    if stale:
        print("")
        print(f"  {stale} template(s) exist under a DIFFERENT engine id and are")
        print("  invisible here. That is the isolation working, not a fault - but")
        print("  those employees need re-enrolling before they can clock in.")
else:
    for tenant_id in names:
        all_findings += audit_tenant(store, tenant_id)

all_findings += log_summary(db)

print("")
if all_findings:
    print(f"== {len(all_findings)} finding(s) ==")
    for i, finding in enumerate(all_findings, 1):
        print(f"  {i}. {finding}")
    sys.exit(1)

# An empty roster has no findings, but saying "no findings" about it would read
# as a clean bill of health for a system that is protecting nobody.
if not names:
    print("== nothing to audit ==")
    print("  An empty roster cannot fail this audit, which is not the same as")
    print("  passing it. Enrol employees, point the HRIS at /verify, then re-run.")
    sys.exit(0)

print("== no findings ==")
print("  There is room between the accept threshold and the closest impostor")
print("  pair, and every employee clears the configured margin. Re-run after")
print("  each enrolment round: the gap narrows as the roster grows.")

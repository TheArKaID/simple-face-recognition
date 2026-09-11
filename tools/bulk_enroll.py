"""Enrol every employee from a directory of photos, one folder per person.

    photos/
      1042/  a.jpg  b.jpg  c.jpg ...     <- folder name IS the employee_id
      1043/  a.jpg  b.jpg ...

    python tools/bulk_enroll.py photos/
    python tools/bulk_enroll.py photos/ --base http://10.0.0.5:8000 --tenant acme

This talks to a running service over HTTP rather than writing the database
directly, and that is the important design choice.  Writing the file directly
would mean loading the ONNX models outside the container, in a second place, and
staying in step with whatever vector space the service is in - templates are
stored against `engine_id`, so a mismatch does not fail, it just makes every
template invisible.  Going through POST /enroll means one engine, one set of
quality gates, one `index_version` bump that tells every replica to reload, and
no models needed here at all.  This file is pure standard library: it runs with
the plain `python3` on the server.

Copying a pre-built .db into place is the other tempting shortcut and it has
sharper edges: SQLite in WAL mode is three files, so copying just faces.db can
silently lose the newest writes, and dropping a database in also discards
whatever enrolments and verify_log rows are already there.  It works exactly
once, on an empty deployment.  This tool works every time, including for the
one new hire next month.

Photos are NOT committed to this repository - they are biometric data.  Put
them on the server and point this at them.

Options:
  --base URL        service address (default http://localhost:8000)
  --tenant NAME     tenant_id (default "default")
  --max N           photos per employee (default: the server's limit, 5)
  --add             keep existing templates instead of replacing them
  --skip-enrolled   leave employees who already have templates alone
  --dry-run         report what would be sent, send nothing
  --timeout SEC     per-request timeout (default 180)
"""
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def call(base, method, path, body=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"message": raw.decode("utf-8", "replace")[:200]}
    except Exception as exc:  # transport, DNS, timeout
        return 0, {"message": f"{type(exc).__name__}: {exc}"}


def spread(items, limit):
    """Pick `limit` photos spread across the set rather than the first few.

    The server keeps the newest N templates and discards the rest, so sending
    everything just throws away the embedding work.  Which N to keep matters:
    filenames here sort chronologically, and photos taken minutes apart carry
    the same pose under the same light.  Spreading the choice across the whole
    set buys the variety that multi-photo enrolment is for - measured on 30
    employees, several templates instead of one moved the worst genuine distance
    from 0.368 to 0.326.
    """
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    last = len(items) - 1
    picked = sorted({round(i * last / (limit - 1)) for i in range(limit)})
    return [items[i] for i in picked]


def main():
    ap = argparse.ArgumentParser(add_help=True, description=__doc__.split("\n")[0])
    ap.add_argument("directory")
    ap.add_argument("--base", default=os.getenv("FACE_BASE", "http://localhost:8000"))
    ap.add_argument("--tenant", default=os.getenv("FACE_TENANT", "default"))
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--add", action="store_true")
    ap.add_argument("--skip-enrolled", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout", type=float, default=180)
    args = ap.parse_args()

    if not os.path.isdir(args.directory):
        sys.exit(f"  not a directory: {args.directory}")

    status, health = call(args.base, "GET", "/health", timeout=args.timeout)
    if status != 200:
        sys.exit(f"  service not reachable at {args.base}: "
                 f"{health.get('message', status)}")
    store = health["data"]["store"]
    limit = args.max or health["data"].get("max_templates_per_employee") or 5
    print(f"  service    {args.base}")
    print(f"  engine     {store['engine_id']}")
    print(f"  tenant     {args.tenant}")
    print(f"  photos/emp up to {limit}, spread across each folder")
    print(f"  mode       {'ADD to' if args.add else 'REPLACE'} existing templates"
          f"{', skipping already-enrolled' if args.skip_enrolled else ''}"
          f"{'  (DRY RUN)' if args.dry_run else ''}")

    folders = sorted(
        d for d in os.listdir(args.directory)
        if os.path.isdir(os.path.join(args.directory, d))
    )
    if not folders:
        sys.exit(f"  no employee folders in {args.directory} - expected one "
                 f"directory per employee, named with the employee_id")

    print("")
    print(f"  {'employee':<16}{'sent':>6}{'stored':>8}{'skipped':>9}  notes")
    ok, empty, failed, skipped = [], [], [], []

    for employee_id in folders:
        folder = os.path.join(args.directory, employee_id)
        photos = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(SUFFIXES)
        )
        if not photos:
            print(f"  {employee_id:<16}{0:>6}{'-':>8}{'-':>9}  no image files")
            empty.append(employee_id)
            continue

        if args.skip_enrolled:
            st, body = call(args.base, "GET",
                            f"/enroll/{args.tenant}/{employee_id}",
                            timeout=args.timeout)
            if st == 200 and body["data"]["enrolled"]:
                held = body["data"]["templates_stored"]
                print(f"  {employee_id:<16}{0:>6}{held:>8}{'-':>9}  already enrolled")
                skipped.append(employee_id)
                continue

        chosen = spread(photos, limit)
        if args.dry_run:
            names = ", ".join(os.path.basename(p) for p in chosen)
            print(f"  {employee_id:<16}{len(chosen):>6}{'-':>8}"
                  f"{len(photos) - len(chosen):>9}  would send: {names}")
            continue

        images, unreadable = [], []
        for path in chosen:
            try:
                with open(path, "rb") as fh:
                    images.append(base64.b64encode(fh.read()).decode())
            except OSError as exc:
                unreadable.append(f"{os.path.basename(path)}: {exc.strerror}")
        if not images:
            print(f"  {employee_id:<16}{0:>6}{'-':>8}{'-':>9}  "
                  f"unreadable: {'; '.join(unreadable)}")
            failed.append(employee_id)
            continue

        st, body = call(args.base, "POST", "/enroll", {
            "tenant_id": args.tenant,
            "employee_id": employee_id,
            "images": images,
            "replace": not args.add,
        }, timeout=args.timeout)

        if st != 200:
            reason = body.get("reason") or body.get("message") or st
            print(f"  {employee_id:<16}{len(images):>6}{0:>8}{'-':>9}  "
                  f"REFUSED {st}: {reason}")
            failed.append(f"{employee_id} ({reason})")
            continue

        data = body["data"]
        stored = data["templates_stored"]
        # A 200 with rejected photos still stored the rest.  Print the reasons:
        # they are almost always fixable at the source (a blurry or dark photo),
        # and silently dropping them is how an employee ends up on one template.
        notes = []
        for item in data.get("rejected", []):
            notes.append(f"{os.path.basename(chosen[item['index']])}={item['reason']}")
        notes += unreadable
        print(f"  {employee_id:<16}{len(images):>6}{stored:>8}"
              f"{len(photos) - len(chosen):>9}  {'; '.join(notes)}")
        (ok if stored else failed).append(employee_id)

    print("")
    if args.dry_run:
        print(f"  dry run: {len(folders)} folder(s) inspected, nothing sent")
        return 0

    print(f"  enrolled {len(ok)}  |  skipped {len(skipped)}  |  "
          f"no photos {len(empty)}  |  failed {len(failed)}")
    for label, names in (("no image files", empty), ("failed", failed)):
        if names:
            print(f"    {label}: {', '.join(str(n) for n in names[:12])}"
                  f"{' ...' if len(names) > 12 else ''}")

    if ok or skipped:
        print("")
        print("  Next: run tools/roster_audit.py. Separation between employees is")
        print("  a minimum over every pair, so it is at its widest right now and")
        print("  narrows with each new hire - the audit names the closest pair and")
        print("  flags anyone enrolled twice or on a single template.")
    return 1 if (failed or empty) else 0


if __name__ == "__main__":
    sys.exit(main())

"""Back up the live template store, safely, while the service keeps running.

    python tools/backup_db.py
    python tools/backup_db.py --out /data/backup-20260914.db

Uses sqlite3.Connection.backup() rather than copying the file. A plain file
copy of a WAL-mode database can catch the main file mid-write with its -wal
and -shm companions in an inconsistent state - three files have to be copied
together to mean anything, and even then a copy taken while something is
writing is not guaranteed consistent. The backup API is SQLite's own answer to
that: it copies page-by-page through a live connection and is safe to run
against a database still being written to, which this one always is.

No sqlite3 CLI is installed in this image - only the sqlite3 module Python
already ships with - so this uses that rather than shelling out to a binary
that is not there.

Restore with the same idea in reverse:

    python -c "import sqlite3; s=sqlite3.connect('backup.db'); \\
        d=sqlite3.connect('/data/faces.db'); s.backup(d)"

or simply stop the service and copy the backup file over faces.db (plus
deleting any stale faces.db-wal/-shm beside it) if a straight file swap is
acceptable for that restore.
"""
import argparse
import datetime
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--db", default=config.DB_PATH, help="source database (default: FACE_DB_PATH)")
ap.add_argument("--out", default=None,
                help="destination path (default: alongside --db, dated)")
args = ap.parse_args()

if not os.path.exists(args.db):
    sys.exit(f"no database at {args.db}")

out = args.out or os.path.join(
    os.path.dirname(args.db) or ".",
    f"backup-{datetime.date.today():%Y%m%d}.db",
)
if os.path.exists(out):
    sys.exit(f"{out} already exists - remove it or pass a different --out")

src = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
dst = sqlite3.connect(out)
try:
    src.backup(dst)
finally:
    dst.close()
    src.close()

size_mb = os.path.getsize(out) / (1024 * 1024)
print(f"  backed up {args.db} -> {out}  ({size_mb:.1f} MB)")

# The backup is a plain, non-WAL file - a straight sanity read proves it opened
# cleanly rather than assuming the copy succeeded because no exception fired.
check = sqlite3.connect(f"file:{out}?mode=ro", uri=True)
counts = check.execute(
    "SELECT (SELECT COUNT(*) FROM face_template), (SELECT COUNT(*) FROM verify_log)"
).fetchone()
check.close()
print(f"  verified readable: {counts[0]} templates, {counts[1]} verify_log rows")

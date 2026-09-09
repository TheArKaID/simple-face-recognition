"""Template storage and the in-memory index used for the 1:N cross-check.

SQLite is the system of record; each tenant's templates are also held as one
contiguous float32 matrix so a verification is a single vectorised pass over
every enrolled employee instead of a query per candidate.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Sequence, Tuple

import numpy as np

import config
import engine

_SCHEMA = """
CREATE TABLE IF NOT EXISTS face_template (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id   TEXT NOT NULL,
    employee_id TEXT NOT NULL,
    engine_id   TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    blur        REAL,
    face_pixels INTEGER,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_template_tenant ON face_template(tenant_id);
CREATE INDEX IF NOT EXISTS idx_template_employee ON face_template(tenant_id, employee_id);

CREATE TABLE IF NOT EXISTS verify_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id          TEXT,
    employee_id        TEXT,
    source             TEXT NOT NULL,
    decision           TEXT NOT NULL,
    distance           REAL,
    runner_up_id       TEXT,
    runner_up_distance REAL,
    margin             REAL,
    reasons            TEXT,
    quality            TEXT,
    engine_id          TEXT,
    created_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_log_created ON verify_log(created_at);
CREATE INDEX IF NOT EXISTS idx_log_employee ON verify_log(tenant_id, employee_id);

-- Bumped on every template write.  Several replicas share one database file, so
-- a process cannot rely on its own writes to know when its cached index went
-- stale; it checks this counter instead, which costs one tiny read per request.
CREATE TABLE IF NOT EXISTS index_version (
    tenant_id TEXT PRIMARY KEY,
    version   INTEGER NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TenantIndex:
    """All templates for one tenant, flattened for vectorised comparison."""

    employee_ids: np.ndarray  # shape (N,), unicode dtype for vectorised masking
    matrix: np.ndarray        # shape (N, dim), dtype float32
    version: int = 0          # index_version this snapshot was built from

    @property
    def size(self) -> int:
        return int(self.matrix.shape[0])


class TemplateStore:
    def __init__(self, path: str = None):
        self.path = path or config.DB_PATH
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL lets the replicas read while one of them writes; busy_timeout
        # makes the rare concurrent write wait instead of failing outright.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._index_cache: Dict[str, TenantIndex] = {}

    # --- writes --------------------------------------------------------------

    def _bump_version(self, cursor, tenant_id: str) -> None:
        cursor.execute(
            "INSERT INTO index_version (tenant_id, version) VALUES (?, 1) "
            "ON CONFLICT(tenant_id) DO UPDATE SET version = version + 1",
            (tenant_id,),
        )

    def _current_version(self, tenant_id: str) -> int:
        row = self._conn.execute(
            "SELECT version FROM index_version WHERE tenant_id = ?", (tenant_id,)
        ).fetchone()
        return int(row["version"]) if row else 0

    def enroll(
        self,
        tenant_id: str,
        employee_id: str,
        templates: Sequence[Tuple[np.ndarray, dict]],
        replace: bool = True,
    ) -> int:
        """Store templates for one employee, returning how many are now held."""
        with self._lock:
            cur = self._conn.cursor()
            if replace:
                cur.execute(
                    "DELETE FROM face_template WHERE tenant_id = ? AND employee_id = ?",
                    (tenant_id, employee_id),
                )
            for embedding, quality in templates:
                cur.execute(
                    "INSERT INTO face_template "
                    "(tenant_id, employee_id, engine_id, embedding, blur, face_pixels, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        tenant_id,
                        employee_id,
                        engine.ENGINE_ID,
                        np.asarray(embedding, dtype=np.float32).tobytes(),
                        quality.get("blur_variance"),
                        quality.get("face_pixels"),
                        _now(),
                    ),
                )
            # Keep only the newest N templates if the caller kept appending.
            cur.execute(
                "DELETE FROM face_template WHERE id IN ("
                "  SELECT id FROM face_template WHERE tenant_id = ? AND employee_id = ?"
                "  ORDER BY id DESC LIMIT -1 OFFSET ?)",
                (tenant_id, employee_id, config.MAX_TEMPLATES_PER_EMPLOYEE),
            )
            self._bump_version(cur, tenant_id)
            self._conn.commit()
            self._index_cache.pop(tenant_id, None)
            return self.template_count(tenant_id, employee_id)

    def delete_employee(self, tenant_id: str, employee_id: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM face_template WHERE tenant_id = ? AND employee_id = ?",
                (tenant_id, employee_id),
            )
            removed = cur.rowcount
            self._bump_version(cur, tenant_id)
            self._conn.commit()
            self._index_cache.pop(tenant_id, None)
            return removed

    def log_verification(self, **fields) -> None:
        if not config.LOG_VERIFICATIONS:
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO verify_log "
                "(tenant_id, employee_id, source, decision, distance, runner_up_id, "
                " runner_up_distance, margin, reasons, quality, engine_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fields.get("tenant_id"),
                    fields.get("employee_id"),
                    fields.get("source", "verify"),
                    fields.get("decision", "error"),
                    fields.get("distance"),
                    fields.get("runner_up_id"),
                    fields.get("runner_up_distance"),
                    fields.get("margin"),
                    json.dumps(fields.get("reasons") or []),
                    json.dumps(fields.get("quality") or {}),
                    engine.ENGINE_ID,
                    _now(),
                ),
            )
            self._conn.commit()

    # --- reads ---------------------------------------------------------------

    def template_count(self, tenant_id: str, employee_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM face_template "
                "WHERE tenant_id = ? AND employee_id = ? AND engine_id = ?",
                (tenant_id, employee_id, engine.ENGINE_ID),
            ).fetchone()
            return int(row["n"])

    def index(self, tenant_id: str) -> TenantIndex:
        """Every current-engine template for a tenant, as one matrix."""
        with self._lock:
            version = self._current_version(tenant_id)
            cached = self._index_cache.get(tenant_id)
            if cached is not None and cached.version == version:
                return cached

            rows = self._conn.execute(
                "SELECT employee_id, embedding FROM face_template "
                "WHERE tenant_id = ? AND engine_id = ? ORDER BY id",
                (tenant_id, engine.ENGINE_ID),
            ).fetchall()

            if rows:
                blob = b"".join(bytes(r["embedding"]) for r in rows)
                matrix = np.frombuffer(blob, dtype=np.float32).reshape(
                    len(rows), engine.EMBEDDING_DIM
                )
                employee_ids = np.array([r["employee_id"] for r in rows])
            else:
                matrix = np.empty((0, engine.EMBEDDING_DIM), dtype=np.float32)
                employee_ids = np.empty(0, dtype="<U64")

            index = TenantIndex(
                employee_ids=employee_ids, matrix=matrix, version=version
            )
            self._index_cache[tenant_id] = index
            return index

    def stats(self) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS templates, COUNT(DISTINCT employee_id) AS employees, "
                "COUNT(DISTINCT tenant_id) AS tenants FROM face_template WHERE engine_id = ?",
                (engine.ENGINE_ID,),
            ).fetchone()
            stale = self._conn.execute(
                "SELECT COUNT(*) AS n FROM face_template WHERE engine_id != ?",
                (engine.ENGINE_ID,),
            ).fetchone()
            return {
                "templates": int(row["templates"]),
                "employees": int(row["employees"]),
                "tenants": int(row["tenants"]),
                "stale_templates": int(stale["n"]),
                "engine_id": engine.ENGINE_ID,
            }

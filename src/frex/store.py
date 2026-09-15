"""SQLite-backed state: the memory that makes runs resumable.

The CSV files in the archive root are a human-readable projection of this database
(see :mod:`frex.report`); this store is the source of truth.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .models import FileRecord, FileStatus, Identity, normalize

SCHEMA_VERSION = "1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    content_hash TEXT PRIMARY KEY,
    rel_path TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    identity_id TEXT,
    destination TEXT,
    similarity REAL,
    faces_found INTEGER NOT NULL DEFAULT 0,
    note TEXT,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS files_status_idx ON files(status);
CREATE INDEX IF NOT EXISTS files_identity_idx ON files(identity_id);

CREATE TABLE IF NOT EXISTS identities (
    identity_id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    display_name TEXT,
    centroid BLOB NOT NULL,
    observation_count INTEGER NOT NULL DEFAULT 0,
    best_quality REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS identity_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_id TEXT NOT NULL REFERENCES identities(identity_id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    quality REAL NOT NULL,
    source_hash TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS identity_embeddings_idx ON identity_embeddings(identity_id);

CREATE TABLE IF NOT EXISTS move_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


def _to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).reshape(-1).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


class Store:
    """Thin, explicit data-access layer. One instance per archive root."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.set_meta("schema_version", SCHEMA_VERSION)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN")
        try:
            yield self.conn
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # ---------------------------------------------------------------- meta

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    # --------------------------------------------------------------- files

    def upsert_file(self, record: FileRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO files(content_hash, rel_path, size, mtime, kind, status,
                              identity_id, destination, similarity, faces_found, note, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(content_hash) DO UPDATE SET
                rel_path=excluded.rel_path,
                size=excluded.size,
                mtime=excluded.mtime,
                kind=excluded.kind,
                status=excluded.status,
                identity_id=excluded.identity_id,
                destination=excluded.destination,
                similarity=excluded.similarity,
                faces_found=excluded.faces_found,
                note=excluded.note,
                updated_at=excluded.updated_at
            """,
            (
                record.content_hash,
                record.rel_path,
                record.size,
                record.mtime,
                record.kind,
                record.status.value,
                record.identity_id,
                record.destination,
                record.similarity,
                record.faces_found,
                record.note,
                time.time(),
            ),
        )

    def touch_file_path(self, content_hash: str, rel_path: str) -> None:
        self.conn.execute(
            "UPDATE files SET rel_path = ?, updated_at = ? WHERE content_hash = ?",
            (rel_path, time.time(), content_hash),
        )

    def get_file(self, content_hash: str) -> FileRecord | None:
        row = self.conn.execute(
            "SELECT * FROM files WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return None if row is None else _row_to_file(row)

    def iter_files(self, statuses: Sequence[FileStatus] | None = None) -> list[FileRecord]:
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            rows = self.conn.execute(
                f"SELECT * FROM files WHERE status IN ({placeholders}) ORDER BY rel_path",
                [status.value for status in statuses],
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM files ORDER BY rel_path").fetchall()
        return [_row_to_file(row) for row in rows]

    def mark_missing(self, known_hashes: set[str]) -> int:
        """Flag rows whose file disappeared from the archive since the last run."""
        rows = self.conn.execute("SELECT content_hash, status FROM files").fetchall()
        missing = [
            row["content_hash"]
            for row in rows
            if row["content_hash"] not in known_hashes and row["status"] != FileStatus.MISSING.value
        ]
        for content_hash in missing:
            self.conn.execute(
                "UPDATE files SET status = ?, updated_at = ? WHERE content_hash = ?",
                (FileStatus.MISSING.value, time.time(), content_hash),
            )
        return len(missing)

    def status_counts(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT status, COUNT(*) AS n FROM files GROUP BY status")
        return {str(row["status"]): int(row["n"]) for row in rows}

    def rewrite_destination(self, identity_id: str, destination: str) -> None:
        self.conn.execute(
            "UPDATE files SET destination = ?, updated_at = ? WHERE identity_id = ?",
            (destination, time.time(), identity_id),
        )

    # ---------------------------------------------------------- identities

    def create_identity(self, centroid: np.ndarray, quality: float, slug: str | None = None) -> str:
        identity_id = uuid.uuid4().hex[:12]
        slug = slug or self._next_slug()
        now = time.time()
        self.conn.execute(
            """
            INSERT INTO identities(identity_id, slug, display_name, centroid,
                                   observation_count, best_quality, created_at, updated_at)
            VALUES(?, ?, NULL, ?, 0, ?, ?, ?)
            """,
            (identity_id, slug, _to_blob(normalize(centroid)), quality, now, now),
        )
        return identity_id

    def _next_slug(self) -> str:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM identities").fetchone()
        index = int(row["n"]) + 1
        while True:
            slug = f"person_{index:04d}"
            exists = self.conn.execute(
                "SELECT 1 FROM identities WHERE slug = ?", (slug,)
            ).fetchone()
            if exists is None:
                return slug
            index += 1

    def get_identity(self, identity_id: str) -> Identity | None:
        row = self.conn.execute(
            "SELECT * FROM identities WHERE identity_id = ?", (identity_id,)
        ).fetchone()
        return None if row is None else _row_to_identity(row)

    def list_identities(self) -> list[Identity]:
        rows = self.conn.execute("SELECT * FROM identities ORDER BY slug").fetchall()
        return [_row_to_identity(row) for row in rows]

    def set_display_name(self, identity_id: str, display_name: str | None) -> None:
        self.conn.execute(
            "UPDATE identities SET display_name = ?, updated_at = ? WHERE identity_id = ?",
            (display_name, time.time(), identity_id),
        )

    def add_embeddings(
        self,
        identity_id: str,
        embeddings: Sequence[tuple[np.ndarray, float]],
        source_hash: str | None,
        keep: int,
    ) -> None:
        """Append observations, keep only the ``keep`` best ones, refresh the centroid.

        Keeping the best-quality embeddings rather than the most recent ones keeps a
        blurry frame from dragging the centroid away from the subject.
        """
        now = time.time()
        self.conn.executemany(
            """
            INSERT INTO identity_embeddings
                (identity_id, embedding, quality, source_hash, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            [
                (identity_id, _to_blob(normalize(vec)), float(quality), source_hash, now)
                for vec, quality in embeddings
            ],
        )
        self.conn.execute(
            """
            DELETE FROM identity_embeddings
            WHERE identity_id = ? AND id NOT IN (
                SELECT id FROM identity_embeddings
                WHERE identity_id = ?
                ORDER BY quality DESC, id DESC
                LIMIT ?
            )
            """,
            (identity_id, identity_id, keep),
        )
        self._refresh_centroid(identity_id)

    def _refresh_centroid(self, identity_id: str) -> None:
        rows = self.conn.execute(
            "SELECT embedding, quality FROM identity_embeddings WHERE identity_id = ?",
            (identity_id,),
        ).fetchall()
        if not rows:
            return
        vectors = np.stack([_from_blob(row["embedding"]) for row in rows])
        weights = np.array([max(float(row["quality"]), 1e-3) for row in rows], dtype=np.float32)
        centroid = normalize((vectors * weights[:, None]).sum(axis=0) / weights.sum())
        self.conn.execute(
            """
            UPDATE identities
            SET centroid = ?, observation_count = ?, best_quality = ?, updated_at = ?
            WHERE identity_id = ?
            """,
            (
                _to_blob(centroid),
                len(rows),
                float(weights.max()),
                time.time(),
                identity_id,
            ),
        )

    def identity_embeddings(self, identity_id: str) -> np.ndarray:
        rows = self.conn.execute(
            "SELECT embedding FROM identity_embeddings WHERE identity_id = ?",
            (identity_id,),
        ).fetchall()
        if not rows:
            return np.zeros((0, 512), dtype=np.float32)
        return np.stack([_from_blob(row["embedding"]) for row in rows])

    # -------------------------------------------------------- move journal

    def journal_begin(self, content_hash: str, src: str, dst: str) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO move_journal(content_hash, src, dst, state, created_at)
            VALUES(?, ?, ?, 'STARTED', ?)
            """,
            (content_hash, src, dst, time.time()),
        )
        return int(cursor.lastrowid or 0)

    def journal_finish(self, journal_id: int, state: str = "DONE") -> None:
        self.conn.execute("UPDATE move_journal SET state = ? WHERE id = ?", (state, journal_id))

    def pending_moves(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM move_journal WHERE state = 'STARTED' ORDER BY id"
        ).fetchall()


def _row_to_file(row: sqlite3.Row) -> FileRecord:
    return FileRecord(
        content_hash=str(row["content_hash"]),
        rel_path=str(row["rel_path"]),
        size=int(row["size"]),
        mtime=float(row["mtime"]),
        kind=str(row["kind"]),
        status=FileStatus(str(row["status"])),
        identity_id=row["identity_id"],
        destination=row["destination"],
        similarity=row["similarity"],
        faces_found=int(row["faces_found"]),
        note=row["note"],
    )


def _row_to_identity(row: sqlite3.Row) -> Identity:
    return Identity(
        identity_id=str(row["identity_id"]),
        slug=str(row["slug"]),
        display_name=row["display_name"],
        centroid=_from_blob(row["centroid"]),
        observation_count=int(row["observation_count"]),
        best_quality=float(row["best_quality"]),
    )

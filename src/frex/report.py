"""Human-readable CSV projections of the state database.

``face_index.csv`` is the file the user asked for: first column the media file,
second column its destination folder (or ``E`` when no subject was recognized).
``folder_names.csv`` is the rename contract: the user writes the folder name they
want in ``desired_name`` and the next run applies it.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from .config import Layout
from .models import CSV_STATUS, FileStatus
from .store import Store

INDEX_HEADER = [
    "file",
    "destination",
    "status",
    "identity_id",
    "similarity",
    "faces_found",
    "note",
]
NAMES_HEADER = ["identity_id", "current_folder", "desired_name", "files", "observations"]


def _atomic_write_rows(path: Path, header: list[str], rows: list[list[str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp, path)


def export_index_csv(store: Store, layout: Layout) -> Path:
    rows = []
    for record in store.iter_files():
        destination = record.destination or ""
        marker = CSV_STATUS[record.status]
        if record.status in (FileStatus.NO_FACE, FileStatus.FAILED, FileStatus.UNCERTAIN):
            destination = marker
        rows.append(
            [
                record.rel_path,
                destination,
                record.status.value,
                record.identity_id or "",
                "" if record.similarity is None else f"{record.similarity:.3f}",
                str(record.faces_found),
                record.note or "",
            ]
        )
    _atomic_write_rows(layout.index_csv, INDEX_HEADER, rows)
    return layout.index_csv


def export_names_csv(store: Store, layout: Layout) -> Path:
    """Rewrite the rename sheet, preserving whatever the user typed in it."""
    existing = read_names_csv(layout)
    counts = {
        identity.identity_id: len(
            [f for f in store.iter_files() if f.identity_id == identity.identity_id]
        )
        for identity in store.list_identities()
    }
    rows = []
    for identity in store.list_identities():
        desired = existing.get(identity.identity_id, identity.display_name or "")
        rows.append(
            [
                identity.identity_id,
                identity.folder_name,
                desired,
                str(counts.get(identity.identity_id, 0)),
                str(identity.observation_count),
            ]
        )
    _atomic_write_rows(layout.names_csv, NAMES_HEADER, rows)
    return layout.names_csv


def read_names_csv(layout: Layout) -> dict[str, str]:
    """Return ``identity_id -> desired_name`` for non-empty user entries."""
    if not layout.names_csv.exists():
        return {}
    mapping: dict[str, str] = {}
    with layout.names_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            identity_id = (row.get("identity_id") or "").strip()
            desired = (row.get("desired_name") or "").strip()
            if identity_id and desired:
                mapping[identity_id] = desired
    return mapping

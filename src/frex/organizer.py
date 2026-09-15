"""Folder ownership, renames and crash-safe moves.

A folder is bound to a subject by the hidden ``.identity`` marker it contains, never
by its name: the user is free to rename folders between runs, by hand or through
``folder_names.csv``, and the binding survives.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import IDENTITY_MARKER, STATE_DIRNAME, Layout
from .models import Identity
from .report import read_names_csv
from .store import Store

_SAFE_NAME = re.compile(r"[^\w\-. ]+", re.UNICODE)


def sanitize_folder_name(name: str) -> str:
    cleaned = _SAFE_NAME.sub("_", name).strip(" .")
    return cleaned or "unnamed"


def read_marker(folder: Path) -> str | None:
    marker = folder / IDENTITY_MARKER
    if not marker.is_file():
        return None
    return marker.read_text(encoding="utf-8").strip() or None


def write_marker(folder: Path, identity_id: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / IDENTITY_MARKER).write_text(identity_id, encoding="utf-8")


def scan_identity_folders(layout: Layout) -> dict[str, Path]:
    """Map ``identity_id -> folder`` by reading the markers under the archive root."""
    found: dict[str, Path] = {}
    for entry in sorted(layout.root.iterdir()):
        if not entry.is_dir() or entry.name == STATE_DIRNAME:
            continue
        identity_id = read_marker(entry)
        if identity_id:
            found[identity_id] = entry
    return found


def unique_folder(root: Path, name: str, identity_id: str) -> Path:
    """Pick a free folder name, unless the colliding folder is already ours."""
    candidate = root / name
    if not candidate.exists() or read_marker(candidate) == identity_id:
        return candidate
    for suffix in range(2, 1000):
        alternative = root / f"{name} ({suffix})"
        if not alternative.exists() or read_marker(alternative) == identity_id:
            return alternative
    return root / f"{name}_{identity_id}"


@dataclass
class RenameEvent:
    identity_id: str
    old_name: str
    new_name: str
    source: str


def sync_folders(store: Store, layout: Layout) -> list[RenameEvent]:
    """Reconcile folders with the database before any file is processed.

    Handles both directions: folders renamed by hand on disk (adopted as the new
    display name) and names requested through ``folder_names.csv`` (applied to disk).
    """
    events: list[RenameEvent] = []
    on_disk = scan_identity_folders(layout)
    desired_names = read_names_csv(layout)

    for identity in store.list_identities():
        folder = on_disk.get(identity.identity_id)
        if folder is not None and folder.name != identity.folder_name:
            store.set_display_name(identity.identity_id, folder.name)
            events.append(
                RenameEvent(identity.identity_id, identity.folder_name, folder.name, "disk")
            )
            identity = _reload(store, identity.identity_id, identity)

        desired = desired_names.get(identity.identity_id)
        if desired:
            target_name = sanitize_folder_name(desired)
            if target_name != identity.folder_name:
                folder = on_disk.get(identity.identity_id)
                target = unique_folder(layout.root, target_name, identity.identity_id)
                if folder is not None and folder.exists():
                    os.replace(folder, target)
                else:
                    target.mkdir(parents=True, exist_ok=True)
                write_marker(target, identity.identity_id)
                store.set_display_name(identity.identity_id, target.name)
                events.append(
                    RenameEvent(identity.identity_id, identity.folder_name, target.name, "csv")
                )
                on_disk[identity.identity_id] = target
                identity = _reload(store, identity.identity_id, identity)

        store.rewrite_destination(identity.identity_id, identity.folder_name)

    _refresh_file_paths(store, layout)
    return events


def _reload(store: Store, identity_id: str, fallback: Identity) -> Identity:
    return store.get_identity(identity_id) or fallback


def _refresh_file_paths(store: Store, layout: Layout) -> None:
    """Keep ``rel_path`` in sync after folders moved under the files."""
    folders = scan_identity_folders(layout)
    for record in store.iter_files():
        if not record.identity_id:
            continue
        folder = folders.get(record.identity_id)
        if folder is None:
            continue
        expected = f"{folder.name}/{Path(record.rel_path).name}"
        if record.rel_path != expected and (layout.root / expected).exists():
            store.touch_file_path(record.content_hash, expected)


def ensure_identity_folder(store: Store, layout: Layout, identity: Identity) -> Path:
    existing = scan_identity_folders(layout).get(identity.identity_id)
    if existing is not None:
        return existing
    folder = unique_folder(layout.root, sanitize_folder_name(identity.folder_name),
                           identity.identity_id)
    write_marker(folder, identity.identity_id)
    if folder.name != identity.folder_name:
        store.set_display_name(identity.identity_id, folder.name)
    return folder


def unique_destination(folder: Path, filename: str) -> Path:
    target = folder / filename
    if not target.exists():
        return target
    stem, suffix = Path(filename).stem, Path(filename).suffix
    for index in range(2, 10000):
        candidate = folder / f"{stem}__{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise OSError(f"cannot find a free name for {filename} in {folder}")


def move_into(store: Store, layout: Layout, source: Path, folder: Path, content_hash: str) -> Path:
    """Move a file into its subject folder, journalling the operation first."""
    if source.parent == folder:
        return source
    folder.mkdir(parents=True, exist_ok=True)
    target = unique_destination(folder, source.name)
    journal_id = store.journal_begin(
        content_hash,
        source.relative_to(layout.root).as_posix(),
        target.relative_to(layout.root).as_posix(),
    )
    shutil.move(str(source), str(target))
    store.journal_finish(journal_id)
    return target


def recover_pending_moves(store: Store, layout: Layout) -> int:
    """Finish or roll forward moves interrupted by a crash in a previous run."""
    recovered = 0
    for row in store.pending_moves():
        source = layout.root / str(row["src"])
        target = layout.root / str(row["dst"])
        if target.exists() and not source.exists():
            store.journal_finish(int(row["id"]))
        elif source.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            store.journal_finish(int(row["id"]))
        else:
            store.journal_finish(int(row["id"]), state="UNRESOLVED")
        recovered += 1
    return recovered

"""Run orchestration: scan, analyse, move, export."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from .config import Layout, Settings
from .faces import FaceExtractor, save_thumbnail
from .hashing import content_hash
from .identities import IdentityIndex, cluster_observations, decide
from .media import MediaFile, discover
from .models import FaceObservation, FileRecord, FileStatus
from .organizer import (
    ensure_identity_folder,
    move_into,
    recover_pending_moves,
    scan_identity_folders,
    sync_folders,
)
from .report import export_index_csv, export_names_csv
from .store import Store

logger = logging.getLogger("frex")

T = TypeVar("T")
R = TypeVar("R")

REPROCESS_STATUSES = {FileStatus.UNCERTAIN, FileStatus.NO_FACE, FileStatus.FAILED}


@dataclass
class RunStats:
    scanned: int = 0
    new_files: int = 0
    missing: int = 0
    analysed: int = 0
    relocated: int = 0
    new_identities: int = 0
    matched: int = 0
    no_face: int = 0
    uncertain: int = 0
    failed: int = 0
    renames: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"scanned={self.scanned} new={self.new_files} analysed={self.analysed} "
            f"matched={self.matched} new_identities={self.new_identities} "
            f"relocated={self.relocated} no_face={self.no_face} "
            f"uncertain={self.uncertain} failed={self.failed} missing={self.missing}"
        )


def _prefetch(
    items: Iterable[T], worker: Callable[[T], R], workers: int
) -> Iterator[tuple[T, R | None, BaseException | None]]:
    """Run ``worker`` a few items ahead of the consumer.

    Decoding video releases the GIL, so overlapping it with the database work of the
    previous file is a cheap throughput win without a process pool.
    """
    iterator = iter(items)
    if workers <= 1:
        for item in iterator:
            try:
                yield item, worker(item), None
            except BaseException as exc:  # noqa: BLE001 - reported per file
                yield item, None, exc
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: list[tuple[T, Future[R]]] = []

        def fill() -> None:
            while len(pending) < workers + 1:
                try:
                    item = next(iterator)
                except StopIteration:
                    return
                pending.append((item, pool.submit(worker, item)))

        fill()
        while pending:
            item, future = pending.pop(0)
            try:
                yield item, future.result(), None
            except BaseException as exc:  # noqa: BLE001 - reported per file
                yield item, None, exc
            fill()


def scan(store: Store, layout: Layout, settings: Settings, stats: RunStats) -> list[MediaFile]:
    """Refresh the index and return the files that still need analysis."""
    pending: list[MediaFile] = []
    seen: set[str] = set()
    folders = scan_identity_folders(layout)

    for media in discover(layout, settings):
        stats.scanned += 1
        digest = content_hash(media.path)
        media.content_hash = digest
        seen.add(digest)
        record = store.get_file(digest)

        if record is None:
            store.upsert_file(
                FileRecord(
                    content_hash=digest,
                    rel_path=media.rel_path,
                    size=media.size,
                    mtime=media.mtime,
                    kind=media.kind,
                )
            )
            stats.new_files += 1
            pending.append(media)
            continue

        if record.rel_path != media.rel_path:
            store.touch_file_path(digest, media.rel_path)

        if record.status is FileStatus.DONE and record.identity_id:
            folder = folders.get(record.identity_id)
            if folder is not None and media.path.parent != folder:
                # Known subject, file back at the root: relocate from memory alone.
                target = move_into(store, layout, media.path, folder, digest)
                store.touch_file_path(digest, target.relative_to(layout.root).as_posix())
                stats.relocated += 1
            continue

        if record.status is FileStatus.MISSING:
            store.upsert_file(
                FileRecord(
                    content_hash=digest,
                    rel_path=media.rel_path,
                    size=media.size,
                    mtime=media.mtime,
                    kind=media.kind,
                    status=FileStatus.PENDING,
                )
            )
            pending.append(media)
            continue

        if record.status is FileStatus.PENDING or (
            settings.retry_failed and record.status in REPROCESS_STATUSES
        ):
            pending.append(media)

    stats.missing = store.mark_missing(seen)
    return pending


def _register(
    store: Store,
    index: IdentityIndex,
    layout: Layout,
    settings: Settings,
    media: MediaFile,
    digest: str,
    observations: list[FaceObservation],
    stats: RunStats,
    dry_run: bool,
) -> FileRecord:
    clusters = cluster_observations(observations, settings.cluster_threshold)
    min_observations = settings.min_cluster_observations if media.kind == "video" else 1
    result = decide(clusters, index, settings, min_observations=min_observations)

    record = FileRecord(
        content_hash=digest,
        rel_path=media.rel_path,
        size=media.size,
        mtime=media.mtime,
        kind=media.kind,
        status=result.status,
        similarity=result.similarity or None,
        faces_found=len(observations),
        note=result.note,
    )

    if result.status is not FileStatus.DONE or result.cluster is None:
        if result.status is FileStatus.NO_FACE:
            stats.no_face += 1
        elif result.status is FileStatus.UNCERTAIN:
            stats.uncertain += 1
        record.identity_id = result.identity_id
        if not dry_run:
            store.upsert_file(record)
        return record

    identity_id = result.identity_id
    if result.is_new or identity_id is None:
        if dry_run:
            record.destination = "<new identity>"
            stats.new_identities += 1
            return record
        identity_id = store.create_identity(result.cluster.centroid, result.cluster.quality)
        stats.new_identities += 1
    else:
        stats.matched += 1

    if dry_run:
        identity = store.get_identity(identity_id)
        record.identity_id = identity_id
        record.destination = identity.folder_name if identity else ""
        return record

    store.add_embeddings(
        identity_id,
        [(obs.embedding, obs.quality) for obs in _best_observations(result.cluster.observations)],
        digest,
        settings.embeddings_per_identity,
    )
    index.refresh(identity_id)

    identity = store.get_identity(identity_id)
    assert identity is not None
    folder = ensure_identity_folder(store, layout, identity)
    _update_thumbnail(layout, identity_id, result.cluster.best, identity.best_quality)

    target = move_into(store, layout, media.path, folder, digest)
    record.identity_id = identity_id
    record.destination = folder.name
    record.rel_path = target.relative_to(layout.root).as_posix()
    store.upsert_file(record)
    return record


def _best_observations(observations: list[FaceObservation], keep: int = 8
                       ) -> list[FaceObservation]:
    return sorted(observations, key=lambda o: o.quality, reverse=True)[:keep]


def _update_thumbnail(layout: Layout, identity_id: str, best: FaceObservation,
                      stored_quality: float) -> None:
    """Keep one preview crop per subject, overwritten whenever a sharper one shows up."""
    if best.crop is None or best.crop.size == 0:
        return
    path = layout.thumbnails_dir / f"{identity_id}.jpg"
    if path.exists() and best.quality < stored_quality:
        return
    try:
        save_thumbnail(best.crop, path)
    except Exception as exc:  # noqa: BLE001 - a preview is never worth failing a run
        logger.debug("thumbnail failed for %s: %s", identity_id, exc)


def run(
    root: Path,
    settings: Settings,
    extractor: FaceExtractor,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    store: Store | None = None,
) -> RunStats:
    layout = Layout(root=root.resolve())
    layout.ensure()
    owns_store = store is None
    store = store or Store(layout.db_path)
    stats = RunStats()

    try:
        store.set_meta("root", str(layout.root))
        recovered = recover_pending_moves(store, layout)
        if recovered:
            logger.info("recovered %d interrupted move(s)", recovered)

        for event in sync_folders(store, layout):
            stats.renames.append(f"{event.old_name} -> {event.new_name} ({event.source})")
            logger.info("folder renamed: %s -> %s [%s]", event.old_name, event.new_name,
                        event.source)

        pending = scan(store, layout, settings, stats)
        if limit is not None:
            pending = pending[:limit]
        logger.info("%d file(s) to analyse", len(pending))

        index = IdentityIndex(store)
        for media, observations, error in _prefetch(pending, extractor.extract, settings.workers):
            digest = media.content_hash
            if error is not None or observations is None:
                stats.failed += 1
                logger.warning("analysis failed for %s: %s", media.rel_path, error)
                store.upsert_file(
                    FileRecord(
                        content_hash=digest,
                        rel_path=media.rel_path,
                        size=media.size,
                        mtime=media.mtime,
                        kind=media.kind,
                        status=FileStatus.FAILED,
                        note=str(error),
                    )
                )
                continue
            stats.analysed += 1
            record = _register(store, index, layout, settings, media, digest, observations,
                               stats, dry_run)
            logger.info("%s -> %s [%s]", media.rel_path, record.destination or "-",
                        record.status.value)

        export_index_csv(store, layout)
        export_names_csv(store, layout)
        return stats
    finally:
        if owns_store:
            store.close()

"""Discovery of media files and frame sampling.

Decoding, not inference, is the bottleneck on a multi-terabyte archive, so videos
are sampled every ``sample_interval`` seconds (optionally keyframes only) instead of
being decoded frame by frame.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import IMAGE_EXTENSIONS, STATE_DIRNAME, VIDEO_EXTENSIONS, Layout, Settings


@dataclass
class MediaFile:
    path: Path
    rel_path: str
    kind: str
    size: int
    mtime: float
    content_hash: str = ""


def classify(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    return None


def discover(layout: Layout, settings: Settings) -> Iterator[MediaFile]:
    """Walk the archive recursively, including the identity folders.

    Already-sorted files must be visited too: that is how the tool notices that a
    file was moved or that a folder was renamed by hand.
    """
    ignored = {STATE_DIRNAME, *settings.ignore_dirs}
    stack = [layout.root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except (PermissionError, FileNotFoundError):
            continue
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name in ignored or entry.name.startswith("."):
                    continue
                stack.append(entry)
                continue
            kind = classify(entry)
            if kind is None:
                continue
            try:
                stat = entry.stat()
            except FileNotFoundError:
                continue
            yield MediaFile(
                path=entry,
                rel_path=entry.relative_to(layout.root).as_posix(),
                kind=kind,
                size=stat.st_size,
                mtime=stat.st_mtime,
            )


def read_image(path: Path) -> np.ndarray | None:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return image


def iter_video_frames(path: Path, settings: Settings) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_index, timestamp_seconds, bgr_frame)`` samples from a video."""
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            return
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if settings.keyframes_only:
            stream.codec_context.skip_frame = "NONKEY"
        time_base = float(stream.time_base) if stream.time_base else 0.0
        emitted = 0
        next_timestamp = 0.0
        for index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.pts) * time_base if frame.pts is not None else index / 30.0
            if not settings.keyframes_only and timestamp + 1e-6 < next_timestamp:
                continue
            next_timestamp = timestamp + settings.sample_interval
            yield index, timestamp, frame.to_ndarray(format="bgr24")
            emitted += 1
            if emitted >= settings.max_frames_per_video:
                return

"""Content identity for media files.

Files are keyed by content, never by path: folders get renamed and files get moved
between runs, and a rename must not look like "a new file plus a deleted one".
Hashing a terabyte of video in full would dominate the runtime, so a large file is
identified by its size plus the head and tail chunks, which is collision-free in
practice for camera footage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK_SIZE = 8 * 1024 * 1024
FULL_HASH_LIMIT = 2 * CHUNK_SIZE


def content_hash(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    size = path.stat().st_size
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(size).encode())
    with path.open("rb") as handle:
        if size <= max(FULL_HASH_LIMIT, 2 * chunk_size):
            while block := handle.read(1024 * 1024):
                digest.update(block)
        else:
            digest.update(handle.read(chunk_size))
            handle.seek(-chunk_size, 2)
            digest.update(handle.read(chunk_size))
    return digest.hexdigest()

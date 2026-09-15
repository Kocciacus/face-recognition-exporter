"""Core value types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class FileStatus(str, Enum):
    """Lifecycle of a media file.

    ``NO_FACE`` is exported as ``E`` in the human-readable CSV, as requested.
    """

    PENDING = "PENDING"
    DONE = "DONE"
    NO_FACE = "NO_FACE"
    UNCERTAIN = "UNCERTAIN"
    FAILED = "FAILED"
    MISSING = "MISSING"


CSV_STATUS = {
    FileStatus.PENDING: "",
    FileStatus.DONE: "",
    FileStatus.NO_FACE: "E",
    FileStatus.UNCERTAIN: "U",
    FileStatus.FAILED: "F",
    FileStatus.MISSING: "M",
}


@dataclass
class FaceObservation:
    """A single detected face in a single sampled frame."""

    embedding: np.ndarray
    quality: float
    det_score: float
    bbox: tuple[int, int, int, int]
    frame_index: int
    timestamp: float
    crop: np.ndarray | None = None


@dataclass
class FaceCluster:
    """Observations of one subject within one media file."""

    observations: list[FaceObservation]

    @property
    def size(self) -> int:
        return len(self.observations)

    @property
    def best(self) -> FaceObservation:
        return max(self.observations, key=lambda o: o.quality)

    @property
    def quality(self) -> float:
        return self.best.quality

    @property
    def centroid(self) -> np.ndarray:
        stacked = np.stack([o.embedding for o in self.observations])
        weights = np.array([max(o.quality, 1e-3) for o in self.observations], dtype=np.float32)
        mean = (stacked * weights[:, None]).sum(axis=0) / weights.sum()
        return normalize(mean)

    @property
    def prominence(self) -> float:
        """Heuristic ranking of subjects inside a file: how present and how sharp."""
        return float(np.log1p(self.size) * (0.5 + self.quality))


@dataclass
class FileRecord:
    """Row of the ``files`` table."""

    content_hash: str
    rel_path: str
    size: int
    mtime: float
    kind: str
    status: FileStatus = FileStatus.PENDING
    identity_id: str | None = None
    destination: str | None = None
    similarity: float | None = None
    faces_found: int = 0
    note: str | None = None


@dataclass
class Identity:
    """Row of the ``identities`` table: a subject and its folder."""

    identity_id: str
    slug: str
    display_name: str | None
    centroid: np.ndarray
    observation_count: int
    best_quality: float

    @property
    def folder_name(self) -> str:
        return self.display_name or self.slug


@dataclass
class MatchResult:
    """Outcome of matching a file's clusters against the known identities."""

    identity_id: str | None
    similarity: float
    cluster: FaceCluster | None
    status: FileStatus
    is_new: bool = False
    note: str | None = None


def normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize, tolerating the degenerate zero vector."""
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec
    return vec / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))

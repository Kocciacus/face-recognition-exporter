from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from frex.config import Settings
from frex.media import MediaFile
from frex.models import FaceObservation, normalize

DIM = 512


def subject_vector(name: str) -> np.ndarray:
    """Deterministic, well-separated unit vector standing in for an ArcFace embedding."""
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    return normalize(rng.normal(size=DIM).astype(np.float32))


def observation(name: str, quality: float = 0.8, noise: float = 0.03,
                seed: int = 0) -> FaceObservation:
    rng = np.random.default_rng((abs(hash(name)) + seed) % (2**32))
    embedding = normalize(subject_vector(name) + noise * rng.normal(size=DIM).astype(np.float32))
    return FaceObservation(
        embedding=embedding,
        quality=quality,
        det_score=0.9,
        bbox=(0, 0, 100, 100),
        frame_index=seed,
        timestamp=float(seed),
        crop=None,
    )


@dataclass
class FakeExtractor:
    """Returns the subjects encoded in each file name: ``alice_bob_clip.mp4``."""

    calls: list[str] = field(default_factory=list)
    fail_on: set[str] = field(default_factory=set)
    forbidden: set[str] = field(default_factory=set)

    def extract(self, media: MediaFile) -> list[FaceObservation]:
        name = media.path.name
        self.calls.append(name)
        assert name not in self.forbidden, f"{name} should not have been analysed again"
        if name in self.fail_on:
            raise OSError("boom")
        subjects = Path(name).stem.split("_")[0].split("+")
        observations: list[FaceObservation] = []
        for position, subject in enumerate(subjects):
            if subject == "nobody":
                continue
            for repeat in range(4):
                observations.append(
                    observation(
                        subject,
                        quality=0.9 - 0.1 * position,
                        seed=repeat + 10 * position,
                    )
                )
        return observations


@pytest.fixture
def settings() -> Settings:
    return Settings(workers=1, min_cluster_observations=1)


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    root.mkdir()
    return root


def make_media(root: Path, name: str, payload: bytes | None = None) -> Path:
    path = root / name
    path.write_bytes(payload or name.encode() * 32)
    return path

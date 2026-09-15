"""Configuration and on-disk layout of a managed archive folder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

STATE_DIRNAME = ".face_index"
DB_FILENAME = "state.sqlite"
INDEX_CSV_NAME = "face_index.csv"
NAMES_CSV_NAME = "folder_names.csv"
THUMBNAILS_DIRNAME = "thumbnails"
IDENTITY_MARKER = ".identity"

VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mts", ".m2ts", ".mpg", ".mpeg", ".wmv", ".3gp"}
)
IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
)


@dataclass(frozen=True)
class Layout:
    """Absolute paths of every file the tool owns inside an archive root."""

    root: Path

    @property
    def state_dir(self) -> Path:
        return self.root / STATE_DIRNAME

    @property
    def db_path(self) -> Path:
        return self.state_dir / DB_FILENAME

    @property
    def thumbnails_dir(self) -> Path:
        return self.state_dir / THUMBNAILS_DIRNAME

    @property
    def index_csv(self) -> Path:
        return self.root / INDEX_CSV_NAME

    @property
    def names_csv(self) -> Path:
        return self.root / NAMES_CSV_NAME

    def ensure(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    """Tunables for sampling, recognition and clustering.

    Thresholds are cosine similarities between L2-normalized ArcFace embeddings.
    """

    sample_interval: float = 2.0
    max_frames_per_video: int = 120
    keyframes_only: bool = False
    detection_size: int = 640

    match_threshold: float = 0.50
    review_threshold: float = 0.38
    cluster_threshold: float = 0.55

    min_face_pixels: int = 60
    min_det_score: float = 0.60
    min_cluster_observations: int = 2
    min_new_identity_quality: float = 0.45

    embeddings_per_identity: int = 32
    retry_failed: bool = False
    workers: int = 2
    device: str = "auto"
    model_name: str = "buffalo_l"
    ignore_dirs: tuple[str, ...] = field(default_factory=tuple)

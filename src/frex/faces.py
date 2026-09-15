"""Face detection and embedding.

The pipeline only depends on the :class:`FaceExtractor` protocol, so the heavy
InsightFace stack stays an optional dependency and the tests can drive the whole
flow with a stub extractor.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .config import Settings
from .media import MediaFile, iter_video_frames, read_image
from .models import FaceObservation, normalize

if TYPE_CHECKING:
    from insightface.app import FaceAnalysis


class FaceExtractor(Protocol):
    def extract(self, media: MediaFile) -> list[FaceObservation]: ...


def sharpness(crop: np.ndarray) -> float:
    """Variance of the Laplacian, squashed to ``[0, 1]``: rejects motion blur."""
    import cv2

    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(min(1.0, variance / 400.0))


def frontality(keypoints: np.ndarray | None) -> float:
    """Rough yaw penalty from the 5-point landmarks (eyes, nose, mouth corners)."""
    if keypoints is None or len(keypoints) < 3:
        return 0.5
    left_eye, right_eye, nose = keypoints[0], keypoints[1], keypoints[2]
    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    if eye_distance < 1e-3:
        return 0.0
    center = (left_eye + right_eye) / 2.0
    offset = abs(float(nose[0] - center[0])) / eye_distance
    return float(max(0.0, 1.0 - 2.0 * offset))


def quality_score(bbox: tuple[int, int, int, int], det_score: float, crop: np.ndarray,
                  keypoints: np.ndarray | None, settings: Settings) -> float:
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    size_score = min(1.0, min(width, height) / (4.0 * settings.min_face_pixels))
    return float(
        0.35 * size_score
        + 0.25 * sharpness(crop)
        + 0.20 * frontality(keypoints)
        + 0.20 * min(1.0, det_score)
    )


class InsightFaceExtractor:
    """ArcFace embeddings (``buffalo_l``) over sampled frames."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._app: FaceAnalysis | None = None

    @property
    def app(self) -> FaceAnalysis:
        if self._app is None:
            from insightface.app import FaceAnalysis

            providers = _providers(self.settings.device)
            app = FaceAnalysis(name=self.settings.model_name, providers=providers)
            app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1,
                        det_size=(self.settings.detection_size, self.settings.detection_size))
            self._app = app
        return self._app

    def extract(self, media: MediaFile) -> list[FaceObservation]:
        if media.kind == "image":
            image = read_image(media.path)
            if image is None:
                raise OSError(f"unreadable image: {media.path}")
            return self._detect(image, frame_index=0, timestamp=0.0)
        observations: list[FaceObservation] = []
        for index, timestamp, frame in iter_video_frames(media.path, self.settings):
            observations.extend(self._detect(frame, index, timestamp))
        return observations

    def _detect(self, frame: np.ndarray, frame_index: int, timestamp: float
                ) -> list[FaceObservation]:
        settings = self.settings
        results: list[FaceObservation] = []
        for face in self.app.get(frame):
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            if min(x2 - x1, y2 - y1) < settings.min_face_pixels:
                continue
            if float(face.det_score) < settings.min_det_score:
                continue
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            keypoints = getattr(face, "kps", None)
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = face.embedding
            results.append(
                FaceObservation(
                    embedding=normalize(np.asarray(embedding, dtype=np.float32)),
                    quality=quality_score((x1, y1, x2, y2), float(face.det_score), crop,
                                          keypoints, settings),
                    det_score=float(face.det_score),
                    bbox=(x1, y1, x2, y2),
                    frame_index=frame_index,
                    timestamp=timestamp,
                    crop=crop.copy() if crop.size else None,
                )
            )
        return results


def _providers(device: str) -> list[str]:
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        import onnxruntime
    except ImportError:
        return ["CPUExecutionProvider"]
    available = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def save_thumbnail(crop: np.ndarray, path: Path) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)

"""Within-file clustering and cross-run identity matching."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config import Settings
from .models import (
    FaceCluster,
    FaceObservation,
    FileStatus,
    MatchResult,
    normalize,
)
from .store import Store

TOP_K_NEIGHBOURS = 3


def cluster_observations(
    observations: Sequence[FaceObservation], threshold: float
) -> list[FaceCluster]:
    """Greedy agglomeration of the faces seen in one file into subjects.

    Videos are processed one file at a time and hold a handful of people, so the
    O(n·k) greedy pass is both sufficient and far cheaper than a full linkage.
    """
    clusters: list[FaceCluster] = []
    centroids: list[np.ndarray] = []
    for observation in sorted(observations, key=lambda o: o.quality, reverse=True):
        best_index, best_similarity = -1, -1.0
        for index, centroid in enumerate(centroids):
            similarity = float(np.dot(centroid, observation.embedding))
            if similarity > best_similarity:
                best_index, best_similarity = index, similarity
        if best_index >= 0 and best_similarity >= threshold:
            clusters[best_index].observations.append(observation)
            centroids[best_index] = clusters[best_index].centroid
        else:
            clusters.append(FaceCluster(observations=[observation]))
            centroids.append(observation.embedding)
    return sorted(clusters, key=lambda c: c.prominence, reverse=True)


class IdentityIndex:
    """In-memory mirror of the known subjects, kept in sync with the store."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self._centroids: dict[str, np.ndarray] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self.reload()

    def reload(self) -> None:
        self._centroids = {}
        self._embeddings = {}
        for identity in self.store.list_identities():
            self._centroids[identity.identity_id] = normalize(identity.centroid)
            self._embeddings[identity.identity_id] = self.store.identity_embeddings(
                identity.identity_id
            )

    def refresh(self, identity_id: str) -> None:
        identity = self.store.get_identity(identity_id)
        if identity is None:
            return
        self._centroids[identity_id] = normalize(identity.centroid)
        self._embeddings[identity_id] = self.store.identity_embeddings(identity_id)

    def __len__(self) -> int:
        return len(self._centroids)

    def score(self, embedding: np.ndarray, identity_id: str) -> float:
        """Similarity of one embedding to a subject.

        The centroid alone is brittle for a subject seen under very different
        conditions, so it is blended with the mean of the nearest stored samples.
        """
        centroid_similarity = float(np.dot(self._centroids[identity_id], embedding))
        samples = self._embeddings.get(identity_id)
        if samples is None or len(samples) == 0:
            return centroid_similarity
        similarities = samples @ embedding
        top = np.sort(similarities)[-min(TOP_K_NEIGHBOURS, len(similarities)) :]
        return float(max(centroid_similarity, float(top.mean())))

    def best_match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        best_id, best_score = None, -1.0
        for identity_id in self._centroids:
            score = self.score(embedding, identity_id)
            if score > best_score:
                best_id, best_score = identity_id, score
        return best_id, best_score


def decide(
    clusters: Sequence[FaceCluster],
    index: IdentityIndex,
    settings: Settings,
    min_observations: int = 1,
) -> MatchResult:
    """Pick the destination subject for one file.

    Multi-subject rule: if any subject in the file is already known, the file goes to
    that subject's folder (the strongest match wins) and the other subjects are
    ignored. Only when nobody in the file is known does the most prominent subject
    get a brand new folder.
    """
    usable = [
        cluster
        for cluster in clusters
        if cluster.size >= min_observations or cluster.quality >= settings.min_new_identity_quality
    ]
    if not usable:
        return MatchResult(None, 0.0, None, FileStatus.NO_FACE, note="no usable face")

    scored = []
    for cluster in usable:
        identity_id, similarity = index.best_match(cluster.centroid)
        scored.append((cluster, identity_id, similarity))

    confident = [item for item in scored if item[1] and item[2] >= settings.match_threshold]
    if confident:
        cluster, identity_id, similarity = max(
            confident, key=lambda item: (item[2], item[0].prominence)
        )
        note = "multi-subject: matched known identity" if len(usable) > 1 else None
        return MatchResult(identity_id, similarity, cluster, FileStatus.DONE, note=note)

    borderline = [item for item in scored if item[1] and item[2] >= settings.review_threshold]
    if borderline:
        cluster, identity_id, similarity = max(borderline, key=lambda item: item[2])
        return MatchResult(
            identity_id,
            similarity,
            cluster,
            FileStatus.UNCERTAIN,
            note=f"ambiguous match with {identity_id} ({similarity:.2f})",
        )

    dominant = usable[0]
    if dominant.quality < settings.min_new_identity_quality:
        return MatchResult(
            None,
            0.0,
            dominant,
            FileStatus.UNCERTAIN,
            note=f"face quality too low for a new identity ({dominant.quality:.2f})",
        )
    return MatchResult(None, 0.0, dominant, FileStatus.DONE, is_new=True)

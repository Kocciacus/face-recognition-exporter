from __future__ import annotations

from pathlib import Path

import numpy as np

from conftest import observation, subject_vector
from frex.config import Layout, Settings
from frex.hashing import content_hash
from frex.identities import IdentityIndex, cluster_observations, decide
from frex.models import FaceCluster, FileStatus, normalize
from frex.organizer import sanitize_folder_name, unique_destination
from frex.store import Store


def test_hash_follows_content_not_name(tmp_path: Path) -> None:
    first = tmp_path / "a.mp4"
    first.write_bytes(b"payload" * 1000)
    digest = content_hash(first)
    renamed = tmp_path / "b.mp4"
    first.rename(renamed)

    assert content_hash(renamed) == digest

    other = tmp_path / "c.mp4"
    other.write_bytes(b"different" * 1000)
    assert content_hash(other) != digest


def test_clustering_separates_two_subjects() -> None:
    observations = [observation("alice", seed=i) for i in range(5)]
    observations += [observation("bob", seed=i) for i in range(3)]

    clusters = cluster_observations(observations, threshold=0.55)

    assert len(clusters) == 2
    assert [cluster.size for cluster in clusters] == [5, 3]


def test_borderline_similarity_is_flagged_for_review(tmp_path: Path) -> None:
    settings = Settings()
    with Store(Layout(root=tmp_path).db_path) as store:
        base = subject_vector("alice")
        identity_id = store.create_identity(base, 0.9)
        store.add_embeddings(identity_id, [(base, 0.9)], None, 8)
        index = IdentityIndex(store)

        drifted = observation("alice", seed=1)
        drifted.embedding = normalize(0.44 * base + 0.9 * subject_vector("stranger"))
        cluster = FaceCluster(observations=[drifted])

        result = decide([cluster], index, settings)

        assert result.status is FileStatus.UNCERTAIN
        assert settings.review_threshold <= result.similarity < settings.match_threshold


def test_store_keeps_only_the_best_embeddings(tmp_path: Path) -> None:
    with Store(Layout(root=tmp_path).db_path) as store:
        identity_id = store.create_identity(subject_vector("alice"), 0.5)
        store.add_embeddings(
            identity_id,
            [(subject_vector("alice"), quality / 10) for quality in range(10)],
            None,
            keep=3,
        )

        stored = store.identity_embeddings(identity_id)
        identity = store.get_identity(identity_id)

        assert len(stored) == 3
        assert identity is not None
        assert np.isclose(identity.best_quality, 0.9)


def test_folder_name_sanitizing_and_collisions(tmp_path: Path) -> None:
    assert sanitize_folder_name("Nonna / Maria*") == "Nonna _ Maria_"
    assert sanitize_folder_name("   ") == "unnamed"

    (tmp_path / "clip.mp4").write_bytes(b"x")
    assert unique_destination(tmp_path, "clip.mp4").name == "clip__2.mp4"

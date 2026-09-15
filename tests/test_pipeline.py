from __future__ import annotations

import csv
import shutil
from pathlib import Path

from conftest import FakeExtractor, make_media
from frex.config import IDENTITY_MARKER, Layout, Settings
from frex.models import FileStatus
from frex.organizer import scan_identity_folders
from frex.pipeline import run
from frex.store import Store


def read_index(root: Path) -> dict[str, dict[str, str]]:
    with (root / "face_index.csv").open(newline="", encoding="utf-8") as handle:
        return {row["file"]: row for row in csv.DictReader(handle)}


def folder_of(root: Path, filename: str) -> Path:
    matches = [p for p in root.rglob(filename) if p.is_file()]
    assert len(matches) == 1, f"{filename} not found exactly once: {matches}"
    return matches[0].parent


def test_first_run_creates_one_folder_per_subject(archive: Path, settings: Settings) -> None:
    make_media(archive, "alice_001.mp4")
    make_media(archive, "alice_002.mp4")
    make_media(archive, "bob_001.mp4")

    stats = run(archive, settings, FakeExtractor())

    assert stats.new_identities == 2
    assert stats.matched == 1
    assert folder_of(archive, "alice_001.mp4") == folder_of(archive, "alice_002.mp4")
    assert folder_of(archive, "bob_001.mp4") != folder_of(archive, "alice_001.mp4")

    index = read_index(archive)
    assert index["person_0001/alice_001.mp4"]["destination"] == "person_0001"
    assert (archive / "person_0001" / IDENTITY_MARKER).is_file()


def test_no_face_is_marked_e_and_file_stays(archive: Path, settings: Settings) -> None:
    make_media(archive, "nobody_001.mp4")

    run(archive, settings, FakeExtractor())

    assert (archive / "nobody_001.mp4").exists()
    row = read_index(archive)["nobody_001.mp4"]
    assert row["destination"] == "E"
    assert row["status"] == FileStatus.NO_FACE.value


def test_second_run_reuses_identities_from_previous_run(archive: Path,
                                                        settings: Settings) -> None:
    make_media(archive, "alice_001.mp4")
    run(archive, settings, FakeExtractor())

    make_media(archive, "alice_003.mp4")
    extractor = FakeExtractor(forbidden={"alice_001.mp4"})
    stats = run(archive, settings, extractor)

    assert stats.new_identities == 0
    assert stats.matched == 1
    assert folder_of(archive, "alice_003.mp4").name == "person_0001"


def test_known_file_returned_to_root_is_relocated_without_analysis(
    archive: Path, settings: Settings
) -> None:
    make_media(archive, "alice_001.mp4")
    run(archive, settings, FakeExtractor())
    shutil.move(str(archive / "person_0001" / "alice_001.mp4"), str(archive / "alice_001.mp4"))

    stats = run(archive, settings, FakeExtractor(forbidden={"alice_001.mp4"}))

    assert stats.relocated == 1
    assert folder_of(archive, "alice_001.mp4").name == "person_0001"


def test_multi_subject_goes_to_the_known_subject_folder(archive: Path,
                                                        settings: Settings) -> None:
    make_media(archive, "bob_001.mp4")
    run(archive, settings, FakeExtractor())
    bob_folder = folder_of(archive, "bob_001.mp4").name

    # carol is dominant in the file but unknown; bob is already known and wins.
    make_media(archive, "carol+bob_002.mp4")
    stats = run(archive, settings, FakeExtractor())

    assert stats.new_identities == 0
    assert folder_of(archive, "carol+bob_002.mp4").name == bob_folder
    row = read_index(archive)[f"{bob_folder}/carol+bob_002.mp4"]
    assert "multi-subject" in row["note"]


def test_multi_subject_with_nobody_known_creates_one_folder(archive: Path,
                                                            settings: Settings) -> None:
    make_media(archive, "dave+erin_001.mp4")

    stats = run(archive, settings, FakeExtractor())

    assert stats.new_identities == 1
    assert folder_of(archive, "dave+erin_001.mp4").name == "person_0001"


def test_rename_via_csv_is_applied_and_destinations_follow(archive: Path,
                                                           settings: Settings) -> None:
    make_media(archive, "alice_001.mp4")
    run(archive, settings, FakeExtractor())

    names = archive / "folder_names.csv"
    rows = list(csv.DictReader(names.open(newline="", encoding="utf-8")))
    rows[0]["desired_name"] = "Alice Rossi"
    with names.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    make_media(archive, "alice_004.mp4")
    run(archive, settings, FakeExtractor())

    assert (archive / "Alice Rossi").is_dir()
    assert not (archive / "person_0001").exists()
    assert folder_of(archive, "alice_001.mp4").name == "Alice Rossi"
    assert folder_of(archive, "alice_004.mp4").name == "Alice Rossi"
    index = read_index(archive)
    assert index["Alice Rossi/alice_001.mp4"]["destination"] == "Alice Rossi"


def test_manual_folder_rename_is_adopted(archive: Path, settings: Settings) -> None:
    make_media(archive, "alice_001.mp4")
    run(archive, settings, FakeExtractor())
    (archive / "person_0001").rename(archive / "Nonna")

    make_media(archive, "alice_005.mp4")
    stats = run(archive, settings, FakeExtractor())

    assert [event for event in stats.renames if "Nonna" in event]
    assert folder_of(archive, "alice_005.mp4").name == "Nonna"
    assert len(scan_identity_folders(Layout(root=archive))) == 1


def test_renaming_a_file_does_not_reprocess_it(archive: Path, settings: Settings) -> None:
    make_media(archive, "alice_001.mp4")
    run(archive, settings, FakeExtractor())
    sorted_file = archive / "person_0001" / "alice_001.mp4"
    sorted_file.rename(sorted_file.with_name("holidays.mp4"))

    stats = run(archive, settings, FakeExtractor(forbidden={"holidays.mp4"}))

    assert stats.analysed == 0
    assert read_index(archive)["person_0001/holidays.mp4"]["status"] == FileStatus.DONE.value


def test_deleted_file_is_marked_missing(archive: Path, settings: Settings) -> None:
    make_media(archive, "alice_001.mp4")
    run(archive, settings, FakeExtractor())
    (archive / "person_0001" / "alice_001.mp4").unlink()

    run(archive, settings, FakeExtractor())

    assert read_index(archive)["person_0001/alice_001.mp4"]["status"] == FileStatus.MISSING.value


def test_failed_analysis_is_recorded_and_retried_on_demand(archive: Path,
                                                           settings: Settings) -> None:
    make_media(archive, "alice_001.mp4")

    run(archive, settings, FakeExtractor(fail_on={"alice_001.mp4"}))
    assert read_index(archive)["alice_001.mp4"]["status"] == FileStatus.FAILED.value

    run(archive, settings, FakeExtractor())
    assert read_index(archive)["alice_001.mp4"]["status"] == FileStatus.FAILED.value

    settings.retry_failed = True
    run(archive, settings, FakeExtractor())
    assert folder_of(archive, "alice_001.mp4").name == "person_0001"


def test_dry_run_moves_nothing(archive: Path, settings: Settings) -> None:
    make_media(archive, "alice_001.mp4")

    run(archive, settings, FakeExtractor(), dry_run=True)

    assert (archive / "alice_001.mp4").exists()
    assert not (archive / "person_0001").exists()


def test_interrupted_move_is_recovered(archive: Path, settings: Settings) -> None:
    make_media(archive, "alice_001.mp4")
    run(archive, settings, FakeExtractor())

    layout = Layout(root=archive)
    with Store(layout.db_path) as store:
        store.journal_begin("deadbeef", "person_0001/alice_001.mp4", "person_0002/alice_001.mp4")
    (archive / "person_0002").mkdir()
    shutil.move(str(archive / "person_0001" / "alice_001.mp4"),
                str(archive / "person_0002" / "alice_001.mp4"))

    run(archive, settings, FakeExtractor())

    with Store(layout.db_path) as store:
        assert not store.pending_moves()

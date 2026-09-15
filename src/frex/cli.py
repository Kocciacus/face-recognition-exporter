"""Command line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Layout, Settings
from .models import FileStatus
from .organizer import scan_identity_folders, sync_folders
from .pipeline import run
from .report import export_index_csv, export_names_csv
from .store import Store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frex",
        description="Organize photos and videos into per-person folders, resumably.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="scan the folder and sort its media")
    run_parser.add_argument("folder", type=Path)
    run_parser.add_argument("--sample-interval", type=float, default=Settings.sample_interval,
                            help="seconds between sampled video frames")
    run_parser.add_argument("--max-frames", type=int, default=Settings.max_frames_per_video)
    run_parser.add_argument("--keyframes-only", action="store_true",
                            help="decode only keyframes: fastest, slightly less accurate")
    run_parser.add_argument("--match-threshold", type=float, default=Settings.match_threshold)
    run_parser.add_argument("--review-threshold", type=float, default=Settings.review_threshold)
    run_parser.add_argument("--cluster-threshold", type=float, default=Settings.cluster_threshold)
    run_parser.add_argument("--workers", type=int, default=Settings.workers)
    run_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run_parser.add_argument("--model", default=Settings.model_name)
    run_parser.add_argument("--limit", type=int, default=None, help="stop after N files")
    run_parser.add_argument("--retry-failed", action="store_true",
                            help="re-analyse files marked E, U or F")
    run_parser.add_argument("--dry-run", action="store_true", help="decide but do not move")

    names_parser = subparsers.add_parser(
        "apply-names", help="apply folder_names.csv without analysing anything"
    )
    names_parser.add_argument("folder", type=Path)

    status_parser = subparsers.add_parser("status", help="print the state of the archive")
    status_parser.add_argument("folder", type=Path)

    export_parser = subparsers.add_parser("export", help="regenerate the CSV files")
    export_parser.add_argument("folder", type=Path)
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings(
        sample_interval=args.sample_interval,
        max_frames_per_video=args.max_frames,
        keyframes_only=args.keyframes_only,
        match_threshold=args.match_threshold,
        review_threshold=args.review_threshold,
        cluster_threshold=args.cluster_threshold,
        retry_failed=args.retry_failed,
        workers=max(1, args.workers),
        device=args.device,
        model_name=args.model,
    )


def _require_folder(folder: Path) -> Path:
    resolved = folder.expanduser().resolve()
    if not resolved.is_dir():
        raise SystemExit(f"not a directory: {resolved}")
    return resolved


def cmd_run(args: argparse.Namespace) -> int:
    from .faces import InsightFaceExtractor

    root = _require_folder(args.folder)
    settings = settings_from_args(args)
    stats = run(root, settings, InsightFaceExtractor(settings), dry_run=args.dry_run,
                limit=args.limit)
    print(stats.summary())
    for rename in stats.renames:
        print(f"renamed: {rename}")
    return 0


def cmd_apply_names(args: argparse.Namespace) -> int:
    root = _require_folder(args.folder)
    layout = Layout(root=root)
    layout.ensure()
    with Store(layout.db_path) as store:
        events = sync_folders(store, layout)
        export_index_csv(store, layout)
        export_names_csv(store, layout)
    for event in events:
        print(f"renamed: {event.old_name} -> {event.new_name} ({event.source})")
    if not events:
        print("no folder rename requested")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    root = _require_folder(args.folder)
    layout = Layout(root=root)
    if not layout.db_path.exists():
        print("no state yet: this folder has never been processed")
        return 0
    with Store(layout.db_path) as store:
        counts = store.status_counts()
        total = sum(counts.values())
        print(f"files indexed: {total}")
        for status in FileStatus:
            if counts.get(status.value):
                print(f"  {status.value:<9} {counts[status.value]}")
        folders = scan_identity_folders(layout)
        print(f"identities: {len(store.list_identities())}")
        for identity in store.list_identities():
            folder = folders.get(identity.identity_id)
            location = folder.name if folder else "<missing folder>"
            print(
                f"  {identity.identity_id}  {location:<30} "
                f"observations={identity.observation_count}"
            )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    root = _require_folder(args.folder)
    layout = Layout(root=root)
    layout.ensure()
    with Store(layout.db_path) as store:
        print(export_index_csv(store, layout))
        print(export_names_csv(store, layout))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    handlers = {
        "run": cmd_run,
        "apply-names": cmd_apply_names,
        "status": cmd_status,
        "export": cmd_export,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

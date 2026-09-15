# face-recognition-exporter (`frex`)

Sorts a large photo/video archive into one folder per person, and remembers what it
learned so it can be stopped and restarted as many times as needed.

Given a folder, `frex`:

1. indexes every photo and video into `face_index.csv` (first column the file, second
   column its destination folder);
2. analyses each file with ArcFace face embeddings (InsightFace `buffalo_l`);
3. moves the file into the folder of the recognized subject, creating a new folder for
   a subject never seen before — including in previous runs;
4. writes `E` in the destination column when no subject could be recognized.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[recognition]"     # CPU
pip install -e ".[gpu]"             # NVIDIA GPU (onnxruntime-gpu)
```

The InsightFace models (~300 MB) download automatically on the first run into
`~/.insightface`.

## Usage

```bash
frex run /media/archive                 # analyse and sort
frex run /media/archive --dry-run       # decide without moving anything
frex run /media/archive --limit 50      # process 50 files and stop
frex status /media/archive              # what is indexed, which people exist
frex apply-names /media/archive         # apply folder renames only
frex export /media/archive              # regenerate the CSV files
```

Interrupt it at any time (`Ctrl+C`, reboot, power loss): state is committed per file,
and a move interrupted halfway is completed on the next run.

## Files the tool owns

| Path | Role |
| --- | --- |
| `face_index.csv` | the index requested: file, destination, status, identity, similarity |
| `folder_names.csv` | rename sheet: write the name you want in `desired_name` |
| `.face_index/state.sqlite` | source of truth: files, identities, embeddings, move journal |
| `.face_index/thumbnails/` | one preview crop per person, replaced by better shots |
| `<person folder>/.identity` | hidden marker binding the folder to a subject |

`face_index.csv` is regenerated from the database at the end of every run; edit
`folder_names.csv`, not the index.

### Status values

| Status | CSV destination | Meaning |
| --- | --- | --- |
| `DONE` | folder name | recognized and moved |
| `NO_FACE` | `E` | no usable face found |
| `UNCERTAIN` | `U` | similarity in the grey zone, or face too poor for a new folder — left in place for review |
| `FAILED` | `F` | file unreadable or decoder error |
| `MISSING` | `M` | indexed previously, no longer on disk |

Re-analyse the `E`/`U`/`F` files with `frex run <folder> --retry-failed`.

## Memory across runs

Recognition state is **face embeddings**, not snapshots: each subject keeps a
quality-weighted centroid plus its best 32 sample vectors, so a subject stays
recognizable across runs, lighting and years. Thumbnails are only a human-readable
preview.

Files are keyed by **content hash** (size + head/tail chunks), so renaming a file or
a folder never looks like a delete plus an add. Hashing a full terabyte would dominate
the runtime; head/tail hashing is collision-free in practice for camera footage.

## Renaming folders

Folders are bound to a subject by the hidden `.identity` file, so you can rename them
however you like:

* **Rename on disk** — the next run adopts the new name and rewrites the destinations.
* **Rename through `folder_names.csv`** — write the name in `desired_name`; the next
  run (or `frex apply-names`) renames the folder on disk.

Either way the identity, its embeddings, and the `face_index.csv` destinations follow.

## Several people in one file

If at least one subject in the file is already known, the file goes to that subject's
folder (strongest match wins) and the other subjects are ignored. Only when nobody in
the file is known does the most prominent subject get a new folder.

## Tuning

| Option | Default | Effect |
| --- | --- | --- |
| `--sample-interval` | `2.0` | seconds between sampled video frames; the main speed/accuracy lever |
| `--keyframes-only` | off | decode keyframes only — fastest, slightly less accurate |
| `--max-frames` | `120` | cap per video, keeps long videos from dominating a run |
| `--match-threshold` | `0.50` | cosine similarity required to reuse an existing folder |
| `--review-threshold` | `0.38` | below the match threshold but above this → `UNCERTAIN` |
| `--cluster-threshold` | `0.55` | grouping of faces into subjects within one file |
| `--workers` | `2` | decode threads running ahead of the recognizer |
| `--device` | `auto` | `cpu`, `cuda`, or auto-detect |

Raise `--match-threshold` if different people end up in the same folder; lower it if
one person gets several folders.

## Throughput

Decoding dominates, not the neural network. With the defaults, expect roughly one to
two days for 1 TB on a single NVIDIA GPU, and one to two weeks on CPU only. Reducing
the sample rate (`--sample-interval 5`, or `--keyframes-only`) is the fastest way to
cut it down.

## Development

```bash
pip install -e ".[dev]"
pytest          # the whole pipeline runs against a stub extractor, no models needed
ruff check .
mypy
```

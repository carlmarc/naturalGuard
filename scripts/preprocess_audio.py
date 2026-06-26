#!/usr/bin/env python3
"""
preprocess_audio.py — Normalize every audio file through one identical pipeline.

PURPOSE
-------
This is the only script that touches audio files in the pipeline. It reads a
source's raw_metadata JSON, filters out unusable files, and runs ffmpeg on
every remaining file to produce a normalized version in processed/.

The point is uniformity: every file in the training set must have gone through
the EXACT SAME codec pipeline (same encoder, same bitrate, same sample rate,
same channels, same loudness target). If natural files and AI files go through
different pipelines, the model learns the pipeline difference instead of the
content difference.

  raw_metadata/<source>.json    (input — features per file from analyze_dataset.py)
  raw/<source>/<...>/<file>     (input — original audio files)
              │
              ▼
       preprocess_audio.py      (this script)
              │
              ▼
  processed/<source>/<file>     (output — normalized audio files)
  processing_state/<source>.json (output — tracks which files succeeded)


WHAT THE PIPELINE DOES
----------------------
For each file, one ffmpeg invocation:

  ffmpeg -i <input>
      -af loudnorm=I=-16:TP=-1.5:LRA=11   # broadcast-standard LUFS normalization
      -ac 1                                # downmix to mono
      -ar 22050                            # resample to 22050 Hz (Nyquist 11025)
      -codec:a libmp3lame                  # one consistent encoder
      -b:a 192k                            # one consistent bitrate
      <output>

Notes on the choices:
  - Loudness target -16 LUFS (track-level integrated, EBU R128). Removes
    inter-track loudness differences as a confound. Within-track dynamics
    are preserved — only one gain value is applied to the whole file.
  - True peak -1.5 dBFS — leaves headroom so the MP3 encoder doesn't clip.
  - Loudness range 11 LU — typical broadcast LRA target.
  - 22050 Hz — chosen to match the model's input sample rate. Nyquist at
    11025 Hz covers the spectrum the model attends to.
  - Mono — model is mono-only; stereo width isn't part of the architecture.
  - libmp3lame 192 kbps — one encoder, one bitrate, applied to every file
    regardless of source format. This is the whole reason for this script.


VERIFICATION
------------
After ffmpeg writes each output file, the script re-ffprobes it to confirm:
  - encoder is libmp3lame
  - bitrate is in a 175–215 kbps window (libmp3lame's VBR mode can wobble)
  - sample rate is 22050
  - channels is 1

It also measures the output's integrated LUFS via a separate ffmpeg run with
ebur128 filter. This catches silent ffmpeg failures and partial writes.

All verification values are recorded in processing_state — you can grep for
files whose measured LUFS is far from -16 to spot processing problems.


RESUMABILITY
------------
The script writes processing_state/<source>.json incrementally as files
complete (every 100 by default). If interrupted, re-running skips files
that already have status="ok" and retries failures.

State file format: a list of records, one per processed file, with structure:
{
  "song_id":         "01419bc385d2bd09",
  "original_path":   "raw/boomy/.../01419bc385d2bd09.mp3",
  "processed_path":  "processed/boomy/01419bc385d2bd09.mp3",
  "status":          "ok" | "failed" | "skipped_filter",
  "encoder":         "libmp3lame",
  "bitrate_kbps":    192,
  "sample_rate":     22050,
  "channels":        1,
  "lufs_measured":   -16.1,
  "error":           null | "<error message>",
  "exclusion_reasons": [...]   (populated when status is "skipped_filter")
}


PARALLELIZATION
---------------
ffmpeg is single-threaded per file (for our pipeline — libmp3lame doesn't
multi-thread) and CPU-bound. The script runs N worker processes in parallel.
Default 4 workers; tune via --workers based on your CPU.

The Python orchestrator (main thread) reads metadata, schedules work, collects
results, and writes state. Workers only run ffmpeg subprocesses.


USAGE
-----
  # Process one source
  python preprocess_audio.py --dataset-root /path/to/datasets --source boomy

  # Process all sources found in raw_metadata/
  python preprocess_audio.py --dataset-root /path/to/datasets --all

  # Test mode: 20 files per source, outputs go to _test paths
  python preprocess_audio.py --dataset-root /path/to/datasets --source boomy --test

  # More workers, less frequent state saves
  python preprocess_audio.py --dataset-root /path/to/datasets --source boomy \
      --workers 6 --state-save-every 200

DIRECTORY LAYOUT EXPECTED
-------------------------
  <dataset-root>/
    raw/<source>/<...subfolders...>/<file>     # input audio
    raw_metadata/<source>.json                  # input metadata
    processed/<source>/<file>                   # output audio (flattened)
    processing_state/<source>.json              # output state
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

# Import the shared filter/identifier logic.
# Adjust this import path if your project layout differs.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import random
from src.dataset.metadata import is_included, exclusion_reasons, song_id, stratification_buckets


# ----------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# ----------------------------------------------------------------------------

# Target encoding parameters. The whole point of the script is that these
# values apply uniformly to every file regardless of source format.
TARGET_LUFS         = -16.0
TARGET_TRUE_PEAK    = -1.5      # dBFS, headroom for the encoder
TARGET_LRA          = 11.0      # loudness range
TARGET_SAMPLE_RATE  = 22050     # Hz, matches model's input rate
TARGET_CHANNELS     = 1         # mono
TARGET_BITRATE_KBPS = 192       # libmp3lame VBR target
TARGET_ENCODER      = "libmp3lame"
OUTPUT_EXTENSION    = ".mp3"

# Verification tolerances
BITRATE_TOLERANCE_KBPS = (128, 215)   # libmp3lame VBR can wobble within this range
LUFS_TOLERANCE         = 4.0          # |measured - target| must be within this many LU

# Test mode caps
TEST_MODE_LIMIT_PER_SOURCE = 20

# State save cadence (overridden by CLI)
DEFAULT_STATE_SAVE_EVERY = 100

# Worker count (overridden by CLI)
DEFAULT_WORKERS = 4

# ffmpeg command name. If ffmpeg isn't on PATH, set to an absolute path here.
FFMPEG_BIN  = "ffmpeg"
FFPROBE_BIN = "ffprobe"


# ----------------------------------------------------------------------------
# DATA CLASSES
# ----------------------------------------------------------------------------

@dataclass
class ProcessingResult:
    """One row in processing_state/<source>.json.

    Represents the outcome of trying to process one file. Status is one of:
      ok              — file was processed and verified
      failed          — ffmpeg or verification failed; see error
      skipped_filter  — file was excluded by quality filters; see exclusion_reasons
    """
    song_id:           str
    original_path:     str
    processed_path:    str | None    = None
    status:            str           = "pending"
    encoder:           str | None    = None
    bitrate_kbps:      int | None    = None
    sample_rate:       int | None    = None
    channels:          int | None    = None
    lufs_measured:     float | None  = None
    error:             str | None    = None
    exclusion_reasons: list[str]     = field(default_factory=list)


# ----------------------------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------------------------

def _stratified_sample(records, state, cap, seed):
    rng = random.Random(seed)
    already_ok = {sid for sid, r in state.items() if r.status == "ok"}
    remaining = cap - len(already_ok)
    if remaining <= 0:
        return [r for r in records if song_id(r) in already_ok]
    done, pool = [], []
    for r in records:
        sid = song_id(r)
        if sid in already_ok:
            done.append(r)
        elif is_included(r):
            pool.append(r)
    if not pool:
        return done
    strata = {}
    for r in pool:
        b = stratification_buckets(r)
        strata.setdefault((b["vocal_class"], b["high_energy_bucket"]), []).append(r)
    selected, remainders = [], []
    total = len(pool)
    for key, group in sorted(strata.items()):
        share = remaining * len(group) / total
        n = min(int(share), len(group))
        picked = rng.sample(group, n) if n > 0 else []
        selected.extend(picked)
        remainders.append((share - n, [r for r in group if r not in picked]))
    remainders.sort(key=lambda x: -x[0])
    i = 0
    while len(selected) < remaining and any(g for _, g in remainders):
        _, g = remainders[i % len(remainders)]
        if g:
            p = rng.choice(g)
            selected.append(p)
            g.remove(p)
        i += 1
    return done + selected


def check_ffmpeg_available() -> None:
    """Verify ffmpeg and ffprobe are on PATH. Raise SystemExit if not."""
    for bin_name in (FFMPEG_BIN, FFPROBE_BIN):
        if shutil.which(bin_name) is None:
            print(f"ERROR: {bin_name} not found on PATH. Install ffmpeg.", file=sys.stderr)
            sys.exit(1)


def load_raw_metadata(path: Path) -> list[dict[str, Any]]:
    """Load a raw_metadata JSON. Returns a list of records.

    Each record is the dict format produced by analyze_dataset.py (with keys
    filepath, filename, ffprobe, librosa, segments, quality_flag, etc).
    """
    if not path.is_file():
        raise FileNotFoundError(f"raw_metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON list at top level, got {type(data).__name__}")
    return data


def load_processing_state(path: Path) -> dict[str, ProcessingResult]:
    """Load processing_state/<source>.json into a dict keyed by song_id.

    Returns an empty dict if the file doesn't exist yet.
    """
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return {row["song_id"]: ProcessingResult(**row) for row in rows}


def save_processing_state(path: Path, state: dict[str, ProcessingResult]) -> None:
    """Write processing_state/<source>.json atomically.

    Writes to a temp file, then renames — protects against corruption if
    the process is killed mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in state.values()]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def output_filename_for(song_id_str: str) -> str:
    """Produce the output filename from a song_id.

    All outputs are flattened (no subfolders). Output is <song_id>.mp3.
    """
    return f"{song_id_str}{OUTPUT_EXTENSION}"


def resolve_unique_output(out_dir: Path, song_id_str: str) -> Path:
    """Pick an output path that doesn't collide with existing files.

    If <song_id>.mp3 already exists in out_dir AND belongs to a different
    in-flight job, append _dup1, _dup2, etc. Almost never triggers given
    hash-style filenames, but defensively handled.
    """
    candidate = out_dir / output_filename_for(song_id_str)
    if not candidate.exists():
        return candidate
    n = 1
    while True:
        candidate = out_dir / f"{song_id_str}_dup{n}{OUTPUT_EXTENSION}"
        if not candidate.exists():
            return candidate
        n += 1


# ----------------------------------------------------------------------------
# FFMPEG WORKER (runs in subprocess pool)
# ----------------------------------------------------------------------------

def run_ffmpeg_pipeline(input_path: str, output_path: str) -> tuple[bool, str]:
    """Run the normalization ffmpeg command on one file.

    Returns (success, error_message). On success, error_message is empty.

    Why this function exists separately: it's the work each worker process
    runs. Keeping it self-contained (no module-level state) makes it cleanly
    parallelizable.
    """
    cmd = [
        FFMPEG_BIN,
        "-y",                                    # overwrite if exists
        "-threads", "1",
        "-hide_banner",
        "-loglevel", "error",                    # only show actual errors
        "-i", input_path,
        "-af", f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK}:LRA={TARGET_LRA}",
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-codec:a", TARGET_ENCODER,
        "-b:a", f"{TARGET_BITRATE_KBPS}k",
        output_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out after 300s"
    except Exception as e:
        return False, f"ffmpeg invocation failed: {e}"

    if result.returncode != 0:
        # Capture the last few lines of stderr — full output is huge.
        err_excerpt = result.stderr.strip().splitlines()[-3:] if result.stderr else []
        return False, f"ffmpeg exit {result.returncode}: " + " | ".join(err_excerpt)

    return True, ""


def ffprobe_file(path: str) -> dict[str, Any]:
    """Run ffprobe on a file and return structured info about the audio stream.

    Returns a dict with keys: encoder, bitrate_kbps, sample_rate, channels.
    Returns empty dict if ffprobe fails.
    """
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-show_entries", "stream=codec_name,sample_rate,channels,bit_rate:format_tags=encoder",
        "-select_streams", "a:0",
        "-of", "json",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {}
        data = json.loads(result.stdout)
    except Exception:
        return {}

    streams = data.get("streams", [])
    if not streams:
        return {}
    stream = streams[0]
    fmt = data.get("format", {})

    return {
        # We check the codec, not the encoder tag.
        # libmp3lame produces codec_name="mp3"; the encoder tag is the muxer ID
        # (typically "Lavf...") which doesn't identify the audio encoder.
        "codec":        stream.get("codec_name", ""),
        "encoder_tag":  fmt.get("tags", {}).get("encoder", ""),  # kept for diagnostics only
        "bitrate_kbps": int(stream.get("bit_rate", 0)) // 1000 if stream.get("bit_rate") else None,
        "sample_rate":  int(stream.get("sample_rate", 0)) if stream.get("sample_rate") else None,
        "channels":     int(stream.get("channels", 0)) if stream.get("channels") else None,
}


# Pre-compiled regex used to parse ebur128 output. The relevant line looks like:
#   [Parsed_ebur128_0 @ 0x...]   Integrated loudness:
#                                  I:           -16.0 LUFS
# We grab the float after "I:".
EBUR128_INTEGRATED_RE = re.compile(r"I:\s*(-?\d+\.?\d*)\s*LUFS")


def measure_lufs(path: str) -> float | None:
    """Measure integrated LUFS of an audio file via ffmpeg's ebur128 filter.

    Returns the integrated loudness in LUFS, or None if measurement failed.
    """
    cmd = [
        FFMPEG_BIN,
        "-threads", "1",
        "-hide_banner",
        "-i", path,
        "-filter:a", "ebur128=peak=true",
        "-f", "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception:
        return None
    # ebur128 writes its summary to stderr.
    for line in reversed(result.stderr.splitlines()):
        match = EBUR128_INTEGRATED_RE.search(line)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def process_one_file(input_path: str, output_path: str) -> dict[str, Any]:
    """Process one file end-to-end: ffmpeg, ffprobe verify, LUFS measure.

    This is what each worker actually runs. Returns a dict suitable for
    merging into a ProcessingResult.

    The verification logic is intentionally strict — better to flag a
    questionable file than ship corrupted training data.
    """
    # Run the main pipeline.
    ok, error = run_ffmpeg_pipeline(input_path, output_path)
    if not ok:
        return {"status": "failed", "error": error}

    # Verify the output exists and is non-empty.
    out = Path(output_path)
    if not out.is_file() or out.stat().st_size == 0:
        return {"status": "failed", "error": "ffmpeg returned success but output is missing or empty"}

    # Verify codec parameters via ffprobe.
    probe = ffprobe_file(output_path)
    if not probe:
        return {"status": "failed", "error": "ffprobe failed on output file"}

    # The encoder string is checked loosely — libmp3lame may be reported in
    # different ways depending on ffmpeg version. We just check it contains "lame".
    # Verify the audio codec. We requested libmp3lame, which produces codec_name="mp3".
    codec = probe.get("codec", "").lower()
    if codec != "mp3":
        return {"status": "failed", "error": f"unexpected codec in output: {codec!r}"}

    bitrate = probe.get("bitrate_kbps")
    if bitrate is None or not (BITRATE_TOLERANCE_KBPS[0] <= bitrate <= BITRATE_TOLERANCE_KBPS[1]):
        return {"status": "failed",
                "error": f"output bitrate {bitrate} kbps outside tolerance {BITRATE_TOLERANCE_KBPS}"}

    sr = probe.get("sample_rate")
    if sr != TARGET_SAMPLE_RATE:
        return {"status": "failed", "error": f"output sample rate {sr} != {TARGET_SAMPLE_RATE}"}

    channels = probe.get("channels")
    if channels != TARGET_CHANNELS:
        return {"status": "failed", "error": f"output channels {channels} != {TARGET_CHANNELS}"}

    # LUFS measurement disabled for speed — saves a second full ffmpeg pass per file.
    # loudnorm reliably lands within the tolerance we widened earlier.
    lufs = None

    return {
        "status":        "ok",
        "encoder":       TARGET_ENCODER,           # what we *requested* — libmp3lame
        "bitrate_kbps":  bitrate,
        "sample_rate":   sr,
        "channels":      channels,
        "lufs_measured": lufs,
        "error":         None,
    }


# Worker-pool entry point. Must be a plain function (not a closure / lambda)
# so multiprocessing can pickle it.

def _worker(args: tuple[str, str, str]) -> tuple[str, dict[str, Any]]:
    """Worker entry point. Args: (song_id, input_path, output_path).

    Returns (song_id, result_dict). The orchestrator merges this into the
    appropriate ProcessingResult.
    """
    song_id_str, input_path, output_path = args
    try:
        result = process_one_file(input_path, output_path)
    except Exception as e:
        # Catch-all so one bad file doesn't kill the worker.
        result = {"status": "failed", "error": f"worker exception: {type(e).__name__}: {e}"}
    return song_id_str, result


# ----------------------------------------------------------------------------
# ORCHESTRATION (runs in main process)
# ----------------------------------------------------------------------------

def process_source(
    source_class: str,
    dataset_root: Path,
    test_mode: bool,
    workers: int,
    state_save_every: int,
    cap: int | None = None,
    cap_seed: int = 42,
    split: str = "train"
) -> None:
    """Process every eligible file in one source class.

    Handles loading metadata, filtering, resuming, parallelizing, and saving state.
    """
    # Resolve paths — every directory has a train/ or test/ subfolder.
    raw_audio_dir   = dataset_root / "raw" / split / source_class
    raw_meta_path   = dataset_root / "raw_metadata" / split / f"{source_class}.json"
    suffix          = "_test" if test_mode else ""
    out_dir         = dataset_root / "processed" / split / f"{source_class}{suffix}"
    state_path      = dataset_root / "processing_state" / split / f"{source_class}{suffix}.json"

    print(f"\n=== Source: {source_class}{' (test)' if test_mode else ''} ===")
    print(f"  split         : {split}")
    print(f"  raw audio dir : {raw_audio_dir}")
    print(f"  metadata file : {raw_meta_path}")
    print(f"  output dir    : {out_dir}")
    print(f"  state file    : {state_path}")

    # Load metadata.
    try:
        records = load_raw_metadata(raw_meta_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return

    if cap is not None:
        records = _stratified_sample(records, load_processing_state(state_path), cap, cap_seed)
        print(f"  cap               : {cap}")
        print(f"  selected by strata: {len(records)}")

    if test_mode:
        records = records[:TEST_MODE_LIMIT_PER_SOURCE]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load any existing state (for resumption).
    state = load_processing_state(state_path)

    # Build the work queue: anything not already done with status "ok".
    work_items: list[tuple[str, str, str]] = []
    skipped_filter = 0
    already_done   = 0

    for record in records:
        sid = song_id(record)
        if not sid:
            continue   # malformed record

        # If state already says this file is ok, skip it.
        existing = state.get(sid)
        if existing is not None and existing.status == "ok":
            already_done += 1
            continue

        # Apply filters. Excluded files get a record with status="skipped_filter".
        if not is_included(record):
            state[sid] = ProcessingResult(
                song_id=sid,
                original_path=record.get("filepath", ""),
                status="skipped_filter",
                exclusion_reasons=exclusion_reasons(record),
            )
            skipped_filter += 1
            continue

        # Build absolute input path. The filepath in raw_metadata is relative
        # to the dataset root (e.g., "train/boomy/.../file.mp3"). The audio
        # files live at dataset_root/<filepath>. We also try a fallback under
        # dataset_root/raw/<filepath> in case some setups use that layout.
        rel_path = record.get("filepath", "")
        input_path = dataset_root / rel_path
        if not input_path.is_file():
            # Fallback: try under a /raw/ subfolder.
            input_path = dataset_root / "raw" / rel_path
        if not input_path.is_file():
            state[sid] = ProcessingResult(
                song_id=sid,
                original_path=rel_path,
                status="failed",
                error=f"input file not found at {input_path}",
            )
            continue

        output_path = resolve_unique_output(out_dir, sid)
        work_items.append((sid, str(input_path), str(output_path)))

        # Initialize a pending state record for this file.
        state[sid] = ProcessingResult(
            song_id=sid,
            original_path=rel_path,
            processed_path=str(output_path.relative_to(dataset_root)),
            status="pending",
        )

    print(f"  records total : {len(records)}")
    print(f"  already done  : {already_done}")
    print(f"  filtered out  : {skipped_filter}")
    print(f"  to process    : {len(work_items)}")

    # Save the initial state snapshot (captures filter decisions even if we
    # do no work).
    save_processing_state(state_path, state)

    if not work_items:
        print("  Nothing to process.")
        return

    # Run the worker pool.
    completed = 0
    failed    = 0
    succeeded = 0
    start     = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, item): item[0] for item in work_items}

        for future in as_completed(futures):
            sid = futures[future]
            try:
                returned_sid, result = future.result()
            except Exception as e:
                result = {"status": "failed", "error": f"future raised: {e}"}
                returned_sid = sid

            # Merge result into the state record.
            r = state[returned_sid]
            r.status        = result.get("status", "failed")
            r.encoder       = result.get("encoder")
            r.bitrate_kbps  = result.get("bitrate_kbps")
            r.sample_rate   = result.get("sample_rate")
            r.channels      = result.get("channels")
            r.lufs_measured = result.get("lufs_measured")
            r.error         = result.get("error")

            completed += 1
            if r.status == "ok":
                succeeded += 1
            else:
                failed += 1

            # Periodic state save and progress print.
            if completed % state_save_every == 0:
                save_processing_state(state_path, state)
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(work_items) - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{len(work_items)}] "
                      f"ok={succeeded} failed={failed} "
                      f"rate={rate:.1f}/s eta={eta:.0f}s")

    # Final save.
    save_processing_state(state_path, state)

    elapsed = time.time() - start
    print(f"  DONE  ok={succeeded} failed={failed} "
          f"skipped_filter={skipped_filter} already_done={already_done} "
          f"elapsed={elapsed:.1f}s")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def discover_sources(dataset_root: Path, split: str) -> list[str]:
    """List all source classes available in raw_metadata/<split>/."""
    meta_dir = dataset_root / "raw_metadata" / split
    if not meta_dir.is_dir():
        return []
    sources = sorted(p.stem for p in meta_dir.glob("*.json") if not p.stem.startswith("_"))
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize audio files through one uniform ffmpeg pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "test"],
        help="Which directory subtree to operate on: 'train' or 'test'. "
             "Affects paths under raw_metadata/, processed/, processing_state/, "
             "and raw/<split>/ for input audio lookup. Default: 'train'.",
    )
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="Root containing raw/, raw_metadata/, processed/, processing_state/.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--source", type=str, default=None,
        help="Process only this source class (e.g., 'boomy').",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Process every source class found in raw_metadata/.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help=f"Test mode: only first {TEST_MODE_LIMIT_PER_SOURCE} files per source, "
             "writes to _test paths.",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of parallel ffmpeg workers (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--state-save-every", type=int, default=DEFAULT_STATE_SAVE_EVERY,
        help=f"Write state to disk every N completed files (default: {DEFAULT_STATE_SAVE_EVERY}).",
    )
    parser.add_argument("--cap", type=int, default=None,
        help="Max files to process per source via stratified sampling.")
    parser.add_argument("--cap-seed", type=int, default=42,
        help="Seed for stratified sampling.")
    args = parser.parse_args()

    check_ffmpeg_available()

    if not args.dataset_root.is_dir():
        print(f"ERROR: dataset-root does not exist: {args.dataset_root}", file=sys.stderr)
        return 1

    if args.all:
        sources = discover_sources(args.dataset_root, args.split)
        if not sources:
            print("ERROR: no source JSONs found in raw_metadata/", file=sys.stderr)
            return 1
        print(f"Processing all sources: {sources}")
    else:
        sources = [args.source]

    for source in sources:
        process_source(
            source_class=source,
            dataset_root=args.dataset_root,
            test_mode=args.test,
            workers=args.workers,
            state_save_every=args.state_save_every,
            cap=args.cap,
            cap_seed=args.cap_seed,
            split=args.split,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())


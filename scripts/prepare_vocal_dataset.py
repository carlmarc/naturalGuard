#!/usr/bin/env python3
"""
prepare_vocal_dataset.py — Ingest a vocal-source dataset into one MP3 per speaker.

PURPOSE
-------
For Phase 2 supervised fine-tuning, we construct synthetic mixed examples with
the label [human=1, ai_class=1] by combining a clean human vocal with an AI
instrumental. This script prepares the vocal side.

For each speaker in the source dataset:
  1. Concatenate all their clips into one long audio track.
  2. Strip silences longer than ~0.5s (common in singing/speech material).
  3. Encode the result through the same MP3 pipeline as everywhere else:
     -16 LUFS, 22050 Hz mono, 192 kbps libmp3lame.
  4. Write outputs as <output-name>_processed/{train_pool,test_pool}/<speaker>.mp3.
  5. Write a manifest JSON describing speakers, their pool, and the processed
     file's duration. build_mixed_examples.py consumes this later.

By producing one MP3 per speaker (instead of one MP3 per clip), the downstream
mixing logic is uniform across vocal source datasets that have very different
clip shapes (LibriSpeech: many short utterances per speaker; NUS-48E: a few
long sung songs per speaker). The mixing builder picks a random 7-second
window anywhere in a speaker's MP3, and doesn't need to care about the
underlying clip structure.


WHY A SPEAKER-LEVEL SPLIT
-------------------------
If we split utterances randomly, the same speaker appears in both train and
test, and the model might memorise speaker characteristics (formants, accent,
timbre). By splitting at the SPEAKER level - e.g. 80% of speakers go to the
train pool, 20% to the test pool - train and test mixed examples are built
from disjoint speaker sets and the model has to generalise across speakers.


WHY PRE-PROCESS THE VOCALS NOW (NOT AT MIXING TIME)
---------------------------------------------------
By running the vocal through the same ffmpeg pipeline now, the vocal and the
AI instrumental have the same codec history when they enter the tensor-domain
mixing operation. The final mixed file then has uniform encoding history
throughout - the model can't use codec-signature differences as a class
shortcut.

Cost: one generation of 192k MP3 compression on the vocal source. At 192k
this is mostly inaudible.


SUPPORTED FORMATS
-----------------
  --format librispeech
      Structure: <root>/<speaker_id>/<chapter_id>/<file>.flac
      Speaker grouping: top-level directory name (numeric string)

  --format nus-48e
      Structure: <root>/<speaker>/{read,sing}/<file>.wav
      Speaker grouping: top-level directory (initials like "ADIZ", "JLEE")
      Only the sing/ subset is used; the read/ portion overlaps with
      LibriSpeech's spoken material.

  Add more formats by extending FORMAT_DISPATCH below.


PIPELINE PER SPEAKER
--------------------
  1. Enumerate all valid clips for the speaker
  2. ffmpeg concat demuxer: glue clips into one long PCM
  3. ffmpeg silenceremove filter: strip silent regions >SILENCE_DURATION_SEC
     at <SILENCE_THRESHOLD_DB
  4. ffmpeg loudnorm + libmp3lame: produce final 192k MP3 at -16 LUFS
  5. Probe output duration; if shorter than MIN_USABLE_DURATION_SEC, drop
     the speaker (not enough material for sampling 7-second windows)


USAGE
-----
  # LibriSpeech dev-clean
  python prepare_vocal_dataset.py \\
      --format librispeech \\
      --source-dir /path/to/music_detection/ingredients/LibriSpeech/dev-clean \\
      --output-root /path/to/music_detection/ingredients \\
      --output-name librispeech_devclean

  # NUS-48E (sung portion only)
  python prepare_vocal_dataset.py \\
      --format nus-48e \\
      --source-dir /path/to/music_detection/ingredients/NUS-48E \\
      --output-root /path/to/music_detection/ingredients \\
      --output-name nus_48e_sing


OUTPUT LAYOUT
-------------
  <output-root>/
    <output-name>_processed/
      train_pool/<speaker>.mp3
      test_pool/<speaker>.mp3
    <output-name>_split.json     ← manifest


RESUMABILITY
------------
If the manifest already exists, re-running:
  - Re-reads the existing speaker split (same pool assignment).
  - Skips speakers whose output MP3 already exists.
  - Re-processes anyone whose output is missing.

To force a fresh split, delete the manifest first.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable


# ----------------------------------------------------------------------------
# THRESHOLDS
# ----------------------------------------------------------------------------

# After concatenation and silence-removal, a speaker's MP3 must have at least
# this many seconds of usable content. Below this, we can't reliably sample
# 7-second windows for mixed-example construction.
MIN_USABLE_DURATION_SEC = 30.0

# Silence removal — strip any silence region of at least this duration
# at or below this level. Conservative defaults that preserve breath sounds
# and short pauses while removing long gaps (e.g. between sung phrases).
SILENCE_DURATION_SEC = 0.5
SILENCE_THRESHOLD_DB = -40.0


# ----------------------------------------------------------------------------
# ENCODING PARAMETERS (must match preprocess_audio.py for codec-history uniformity)
# ----------------------------------------------------------------------------

TARGET_LUFS         = -16.0
TARGET_TRUE_PEAK    = -1.5      # dBFS, headroom for the encoder
TARGET_LRA          = 11.0
TARGET_SAMPLE_RATE  = 22050
TARGET_CHANNELS     = 1
TARGET_BITRATE_KBPS = 192
TARGET_ENCODER      = "libmp3lame"
OUTPUT_EXTENSION    = ".mp3"

DEFAULT_WORKERS = 4

FFMPEG_BIN  = "ffmpeg"
FFPROBE_BIN = "ffprobe"


# ----------------------------------------------------------------------------
# FORMAT-SPECIFIC FILE ENUMERATION
# ----------------------------------------------------------------------------

def enumerate_librispeech(source_dir: Path) -> dict[str, list[Path]]:
    """Walk a LibriSpeech subset and group files by speaker.

    Structure: <source_dir>/<speaker_id>/<chapter_id>/<file>.flac
    Returns: {speaker_id: [absolute Path, absolute Path, ...]}
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"LibriSpeech source dir not found: {source_dir}")
    by_speaker: dict[str, list[Path]] = {}
    for speaker_dir in sorted(source_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        flacs = sorted(speaker_dir.rglob("*.flac"))
        if flacs:
            by_speaker[speaker_id] = flacs
    return by_speaker


def enumerate_nus_48e(source_dir: Path) -> dict[str, list[Path]]:
    """Walk NUS-48E and return only the sung portion, grouped by speaker.

    Structure: <source_dir>/<SPEAKER>/{read,sing}/<file>.wav
    Speaker grouping: top-level directory (initials like "ADIZ").
    Only files under <SPEAKER>/sing/ are returned; read/ is skipped because
    it overlaps in role with LibriSpeech.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"NUS-48E source dir not found: {source_dir}")
    by_speaker: dict[str, list[Path]] = {}
    for speaker_dir in sorted(source_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        sing_dir = speaker_dir / "sing"
        if not sing_dir.is_dir():
            continue
        wavs = sorted(sing_dir.rglob("*.wav"))
        if wavs:
            by_speaker[speaker_id] = wavs
    return by_speaker


FORMAT_DISPATCH: dict[str, Callable[[Path], dict[str, list[Path]]]] = {
    "librispeech": enumerate_librispeech,
    "nus-48e":     enumerate_nus_48e,
}


# ----------------------------------------------------------------------------
# AUDIO PROBE / ENCODE
# ----------------------------------------------------------------------------

def check_ffmpeg_available() -> None:
    """Raise SystemExit if ffmpeg or ffprobe is missing."""
    for bin_name in (FFMPEG_BIN, FFPROBE_BIN):
        if shutil.which(bin_name) is None:
            print(f"ERROR: {bin_name} not found on PATH. Install ffmpeg.",
                  file=sys.stderr)
            sys.exit(1)


def probe_duration(audio_path: Path) -> float | None:
    """Return audio duration in seconds via ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            [FFPROBE_BIN, "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(audio_path)],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return None


def concat_and_process_speaker(
    speaker_id: str,
    clip_paths: list[Path],
    output_path: Path,
) -> tuple[bool, str, float | None]:
    """Concatenate clips, strip silence, encode to standard MP3.

    Pipeline:
      1. Create an ffmpeg concat-demuxer listfile.
      2. One ffmpeg call: concat → silenceremove → loudnorm → libmp3lame.

    Returns (success, error_message, output_duration_sec).
    The duration is measured by ffprobe on the final output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg's concat demuxer needs a text file with `file '<path>'` lines.
    # Paths must be ffmpeg-safe — we write a temp file with single quotes
    # around each path and escape any single quote characters in the path.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as listfile:
        for clip_path in clip_paths:
            # Escape single quotes in the path itself (rare but possible).
            escaped = str(clip_path).replace("'", r"'\''")
            listfile.write(f"file '{escaped}'\n")
        listfile_path = Path(listfile.name)

    try:
        cmd = [
            FFMPEG_BIN, "-y",
            "-threads", "1",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", str(listfile_path),
            "-af", (
                # Silence removal — strip any silence longer than
                # SILENCE_DURATION_SEC at or below SILENCE_THRESHOLD_DB.
                # stop_periods=-1 means strip every silent region, not just leading/trailing.
                f"silenceremove="
                f"stop_periods=-1:"
                f"stop_duration={SILENCE_DURATION_SEC}:"
                f"stop_threshold={SILENCE_THRESHOLD_DB}dB,"
                # Loudness normalisation.
                f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK}:LRA={TARGET_LRA}"
            ),
            "-ac", str(TARGET_CHANNELS),
            "-ar", str(TARGET_SAMPLE_RATE),
            "-codec:a", TARGET_ENCODER,
            "-b:a", f"{TARGET_BITRATE_KBPS}k",
            str(output_path),
        ]
        try:
            # Long timeout — concatenating a speaker's full LibriSpeech material
            # can take a minute for hundreds of clips.
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return False, "ffmpeg timed out after 600s", None
        except Exception as e:
            return False, f"ffmpeg invocation failed: {e}", None
    finally:
        listfile_path.unlink(missing_ok=True)

    if result.returncode != 0:
        err_excerpt = result.stderr.strip().splitlines()[-3:] if result.stderr else []
        return False, f"ffmpeg exit {result.returncode}: " + " | ".join(err_excerpt), None

    if not output_path.is_file() or output_path.stat().st_size == 0:
        return False, "ffmpeg reported success but output is missing or empty", None

    duration = probe_duration(output_path)
    if duration is None:
        # Output exists but we couldn't measure duration; still call it success
        # but with unknown duration. Downstream filter will treat as 0.
        return True, "", None

    return True, "", duration


def _worker(args: tuple[str, list[str], str]) -> tuple[str, bool, str, float | None]:
    """Worker entry point for ProcessPoolExecutor.

    Args: (speaker_id, [clip_path_str, ...], output_path_str).
    Returns: (speaker_id, success, error_message, output_duration_sec).
    """
    speaker_id, clip_strs, output_str = args
    try:
        clip_paths = [Path(p) for p in clip_strs]
        ok, err, dur = concat_and_process_speaker(
            speaker_id, clip_paths, Path(output_str),
        )
    except Exception as e:
        ok, err, dur = False, f"worker exception: {type(e).__name__}: {e}", None
    return speaker_id, ok, err, dur


# ----------------------------------------------------------------------------
# CORE: enumerate, split, prepare work items
# ----------------------------------------------------------------------------

def build_manifest_skeleton(
    fmt: str,
    source_dir: Path,
    output_root: Path,
    output_name: str,
    test_ratio: float,
    seed: int,
) -> tuple[dict[str, Any], list[tuple[str, list[str], str]]]:
    """Enumerate speakers, do the train/test split, build work items.

    Returns:
      manifest    — dict to be written as JSON (without "summary" yet)
      work_items  — list of (speaker_id, [clip_paths], output_path) to process
    """
    if fmt not in FORMAT_DISPATCH:
        raise ValueError(f"Unknown format {fmt!r}. Supported: {list(FORMAT_DISPATCH)}")

    print(f"Enumerating files (format={fmt})...")
    speakers = FORMAT_DISPATCH[fmt](source_dir)
    if not speakers:
        raise RuntimeError(f"No speakers found under {source_dir}.")
    total_clips = sum(len(cs) for cs in speakers.values())
    print(f"  found {total_clips} clips across {len(speakers)} speakers")

    if len(speakers) < 2:
        raise RuntimeError(
            f"Only {len(speakers)} speaker(s) found. "
            "Cannot create a meaningful train/test split."
        )

    # Deterministic speaker-level train/test split.
    sorted_speakers = sorted(speakers.keys())
    rng = random.Random(seed)
    shuffled = list(sorted_speakers)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_ratio)))
    test_speakers = set(shuffled[:n_test])

    speakers_out: dict[str, dict[str, Any]] = {}
    work_items: list[tuple[str, list[str], str]] = []

    for sp in sorted_speakers:
        pool = "test" if sp in test_speakers else "train"
        pool_dir_name = "test_pool" if pool == "test" else "train_pool"
        clips = speakers[sp]

        processed_rel = (
            Path(f"{output_name}_processed") / pool_dir_name / f"{sp}{OUTPUT_EXTENSION}"
        )
        processed_abs = output_root / processed_rel

        speakers_out[sp] = {
            "pool":          pool,
            "n_source_clips": len(clips),
            "source_paths":   [str(c.relative_to(source_dir)) for c in clips],
            "processed_path": str(processed_rel),
            # duration_sec is filled in after encoding completes
            "duration_sec":   None,
            "status":         "pending",
            "error":          None,
        }
        work_items.append((sp, [str(c) for c in clips], str(processed_abs)))

    manifest = {
        "format":      fmt,
        "source_dir":  str(source_dir.resolve()),
        "output_root": str(output_root.resolve()),
        "output_name": output_name,
        "split_seed":  seed,
        "test_ratio":  test_ratio,
        "thresholds": {
            "min_usable_duration_sec": MIN_USABLE_DURATION_SEC,
            "silence_duration_sec":    SILENCE_DURATION_SEC,
            "silence_threshold_db":    SILENCE_THRESHOLD_DB,
        },
        "encoding": {
            "lufs":         TARGET_LUFS,
            "sample_rate":  TARGET_SAMPLE_RATE,
            "channels":     TARGET_CHANNELS,
            "bitrate_kbps": TARGET_BITRATE_KBPS,
            "encoder":      TARGET_ENCODER,
        },
        "speakers": speakers_out,
    }
    return manifest, work_items


def load_existing_manifest(manifest_path: Path) -> dict[str, Any] | None:
    """Load an existing manifest if present, else return None."""
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest_atomic(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest JSON atomically (temp + rename)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    tmp.replace(manifest_path)


def filter_work_items_for_resume(
    work_items: list[tuple[str, list[str], str]],
) -> tuple[list[tuple[str, list[str], str]], int]:
    """Drop work items whose output file already exists and is non-empty."""
    remaining: list[tuple[str, list[str], str]] = []
    skipped = 0
    for item in work_items:
        _, _, output_str = item
        p = Path(output_str)
        if p.is_file() and p.stat().st_size > 0:
            skipped += 1
        else:
            remaining.append(item)
    return remaining, skipped


def summarise(manifest: dict[str, Any]) -> dict[str, Any]:
    """Compute the summary block of the manifest."""
    train = [info for info in manifest["speakers"].values() if info["pool"] == "train"]
    test  = [info for info in manifest["speakers"].values() if info["pool"] == "test"]
    train_ok = [s for s in train if s["status"] == "ok"]
    test_ok  = [s for s in test  if s["status"] == "ok"]
    sec_train = sum(s["duration_sec"] or 0.0 for s in train_ok)
    sec_test  = sum(s["duration_sec"] or 0.0 for s in test_ok)
    n_failed = sum(1 for s in manifest["speakers"].values() if s["status"] == "failed")
    n_too_short = sum(1 for s in manifest["speakers"].values() if s["status"] == "too_short")
    return {
        "speakers_total":     len(manifest["speakers"]),
        "speakers_ok":        len(train_ok) + len(test_ok),
        "speakers_failed":    n_failed,
        "speakers_too_short": n_too_short,
        "speakers_train_ok":  len(train_ok),
        "speakers_test_ok":   len(test_ok),
        "hours_train":        round(sec_train / 3600.0, 2),
        "hours_test":         round(sec_test / 3600.0, 2),
    }


def run_encoding(
    work_items: list[tuple[str, list[str], str]],
    manifest_path: Path,
    manifest: dict[str, Any],
    workers: int,
) -> None:
    """Run the encoding worker pool. Updates manifest in place.

    Status transitions per speaker:
      pending  → ok           (processed, duration >= MIN_USABLE_DURATION_SEC)
      pending  → too_short    (processed, duration < MIN_USABLE_DURATION_SEC)
      pending  → failed       (ffmpeg failed; error message in manifest)
    """
    if not work_items:
        print("  Nothing to encode.")
        return

    print(f"Encoding {len(work_items)} speakers with {workers} workers...")
    completed = 0
    failed    = 0
    too_short = 0
    save_every = max(1, min(20, len(work_items) // 10))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, item): item[0] for item in work_items}
        for future in as_completed(futures):
            speaker_id = futures[future]
            try:
                _, ok, err, dur = future.result()
            except Exception as e:
                ok, err, dur = False, f"future raised: {e}", None

            info = manifest["speakers"][speaker_id]
            if ok:
                info["duration_sec"] = round(dur, 2) if dur is not None else None
                if dur is None or dur < MIN_USABLE_DURATION_SEC:
                    info["status"] = "too_short"
                    info["error"]  = (
                        f"after silence removal, only {dur or 0.0:.1f}s usable "
                        f"(<{MIN_USABLE_DURATION_SEC}s)"
                    )
                    too_short += 1
                else:
                    info["status"] = "ok"
                    info["error"]  = None
            else:
                info["status"] = "failed"
                info["error"]  = err
                failed += 1

            completed += 1
            if completed % save_every == 0:
                write_manifest_atomic(manifest_path, manifest)
                print(f"  [{completed}/{len(work_items)}] "
                      f"ok={completed - failed - too_short} "
                      f"failed={failed} too_short={too_short}")

    write_manifest_atomic(manifest_path, manifest)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Concatenate, silence-strip, and encode vocal-source data per speaker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--format", required=True, choices=sorted(FORMAT_DISPATCH.keys()),
        help="Which dataset format to ingest.",
    )
    parser.add_argument(
        "--source-dir", type=Path, required=True,
        help="Root of the source dataset on disk.",
    )
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="Root for processed outputs (e.g. .../music_detection/ingredients).",
    )
    parser.add_argument(
        "--output-name", type=str, required=True,
        help="Name prefix for processed dir and manifest (e.g. 'librispeech_devclean').",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.2,
        help="Fraction of speakers in the test pool (default: 0.2).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the speaker shuffle (default: 42).",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"ffmpeg parallel workers (default: {DEFAULT_WORKERS}).",
    )
    args = parser.parse_args()

    check_ffmpeg_available()

    if not (0.0 < args.test_ratio < 1.0):
        print(f"ERROR: --test-ratio must be in (0, 1), got {args.test_ratio}",
              file=sys.stderr)
        return 1

    manifest_path = args.output_root / f"{args.output_name}_split.json"
    existing = load_existing_manifest(manifest_path)

    if existing is not None:
        if existing.get("format") != args.format:
            print(f"ERROR: existing manifest is for format={existing.get('format')!r} "
                  f"but you ran with --format {args.format!r}. "
                  f"Delete the manifest to start fresh.", file=sys.stderr)
            return 1
        print(f"Found existing manifest at {manifest_path} - resuming.")
        print(f"  using seed={existing['split_seed']}, "
              f"test_ratio={existing['test_ratio']}")
        manifest = existing
        # Reconstruct work_items from the manifest.
        work_items: list[tuple[str, list[str], str]] = []
        source_dir = Path(manifest["source_dir"])
        for sp, info in manifest["speakers"].items():
            clip_paths = [str(source_dir / p) for p in info["source_paths"]]
            output_abs = args.output_root / info["processed_path"]
            work_items.append((sp, clip_paths, str(output_abs)))
    else:
        manifest, work_items = build_manifest_skeleton(
            fmt=args.format,
            source_dir=args.source_dir,
            output_root=args.output_root,
            output_name=args.output_name,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        write_manifest_atomic(manifest_path, manifest)

    work_items, n_skipped = filter_work_items_for_resume(work_items)
    print(f"  already processed (skipped): {n_skipped}")
    print(f"  to process:                  {len(work_items)}")

    run_encoding(
        work_items=work_items,
        manifest_path=manifest_path,
        manifest=manifest,
        workers=args.workers,
    )

    manifest["summary"] = summarise(manifest)
    write_manifest_atomic(manifest_path, manifest)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, value in manifest["summary"].items():
        print(f"  {key:<22}: {value}")
    print()
    print(f"Manifest written to: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
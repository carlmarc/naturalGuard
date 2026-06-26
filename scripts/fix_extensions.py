#!/usr/bin/env python3
"""
fix_extensions.py — Rename audio files whose extension doesn't match their actual codec.

PURPOSE
-------
Some files in your dataset have misleading extensions — most commonly, PCM audio
wrapped in an MP3 container and named .mp3. These files work in ffmpeg-based
pipelines (because ffmpeg trusts the file content, not the extension), but they
cause confusion for any tool that does trust extensions.

This script:
  1. Reads each raw_metadata/<source>.json
  2. For each record, checks whether ffprobe.codec matches what ffprobe.ext implies
  3. On mismatch:
       - renames the audio file on disk to use the correct extension
       - updates filepath / filename / ffprobe.ext in the JSON record
  4. Writes back each updated JSON atomically (temp file + rename)

The pipeline still works either way — preprocess_audio.py uses ffprobe internally
and ignores extensions. This script is purely about making the dataset clean and
honest on disk.


CODEC → EXTENSION MAPPING
-------------------------
  pcm_s16le, pcm_s24le, pcm_s32le, pcm_f32le, pcm_u8  → .wav
  flac                                                 → .flac
  mp3                                                  → .mp3
  aac                                                  → .m4a
  vorbis                                               → .ogg
  opus                                                 → .opus

Codecs not in this mapping are reported but left untouched (no rename).
That keeps the script conservative — better to skip an unfamiliar codec than
to invent a wrong extension.


USAGE
-----
  # Preview what would change (no actual modifications)
  python fix_extensions.py --dataset-root /path/to/datasets --dry-run

  # Actually apply changes to a single source
  python fix_extensions.py --dataset-root /path/to/datasets --source melodia

  # Apply to every source
  python fix_extensions.py --dataset-root /path/to/datasets --all

  # Apply to a few specific sources
  python fix_extensions.py --dataset-root /path/to/datasets --source melodia natural


SAFETY
------
  - Dry-run mode shows everything that would happen without modifying anything.
  - JSONs are written atomically (write to .tmp then rename) so an interrupted
    run cannot corrupt your metadata.
  - If the target rename path already exists, the file is SKIPPED, not overwritten.
    A warning is printed; you'll need to investigate manually.
  - If the source audio file is missing, the record is reported but no JSON
    change is made — we don't want stale metadata pointing to renamed files
    that may or may not be the same content.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------------------
# CODEC → EXTENSION MAPPING
# ----------------------------------------------------------------------------
# These are the codecs we know how to map to a correct extension. Any codec
# not listed here is reported but left untouched — we'd rather skip than
# invent the wrong extension.

CODEC_TO_EXTENSION: dict[str, str] = {
    # PCM family — all become .wav
    "pcm_s16le": ".wav",
    "pcm_s24le": ".wav",
    "pcm_s32le": ".wav",
    "pcm_f32le": ".wav",
    "pcm_f64le": ".wav",
    "pcm_u8":    ".wav",
    "pcm_s16be": ".wav",
    "pcm_s24be": ".wav",
    # Lossless compressed
    "flac": ".flac",
    "alac": ".m4a",
    # Lossy compressed
    "mp3":    ".mp3",
    "aac":    ".m4a",
    "vorbis": ".ogg",
    "opus":   ".opus",
}


# ----------------------------------------------------------------------------
# CORE: decide what extension a record should have
# ----------------------------------------------------------------------------

def expected_extension(codec: str) -> str | None:
    """Return the correct extension for a given codec, or None if unknown.

    Returns include the leading dot, e.g. ".wav". Returns None if we don't
    have a mapping for this codec — in which case we'll leave the file alone.
    """
    return CODEC_TO_EXTENSION.get(codec.lower())


def needs_rename(record: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    """Decide whether this record's audio file needs renaming.

    Returns (needs_rename, current_ext, target_ext).

      needs_rename  True iff current_ext differs from the expected extension.
      current_ext   The extension currently in the record (e.g. ".mp3"). May be empty.
      target_ext    The correct extension based on the codec. None if codec is unknown.
    """
    ffprobe  = record.get("ffprobe", {})
    codec    = ffprobe.get("codec", "")
    current  = ffprobe.get("ext", "")

    target = expected_extension(codec)
    if target is None:
        # Unknown codec — leave alone.
        return False, current, None

    # Normalize comparison: both lowercase, both leading-dot.
    current_norm = current.lower()
    if not current_norm.startswith("."):
        current_norm = "." + current_norm if current_norm else ""

    return (current_norm != target), current_norm, target


# ----------------------------------------------------------------------------
# UTILITY: atomic JSON write
# ----------------------------------------------------------------------------

def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON to path atomically: temp file + rename.

    If the process is interrupted mid-write, the original file is untouched
    and only a .tmp file is left behind (safe to delete).
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


# ----------------------------------------------------------------------------
# UTILITY: resolve audio path
# ----------------------------------------------------------------------------

def resolve_audio_path(dataset_root: Path, rel_path: str) -> Path | None:
    """Find the absolute path of an audio file given its raw_metadata-relative path.

    Tries dataset_root / rel_path first (matches the user's layout), then
    dataset_root / "raw" / rel_path as a fallback. Returns None if neither exists.
    """
    candidate = dataset_root / rel_path
    if candidate.is_file():
        return candidate
    candidate = dataset_root / "raw" / rel_path
    if candidate.is_file():
        return candidate
    return None


# ----------------------------------------------------------------------------
# CORE: process one source
# ----------------------------------------------------------------------------

def process_source(
    source_class: str,
    dataset_root: Path,
    dry_run: bool,
) -> dict[str, int]:
    """Examine one source's raw_metadata and rename mismatched files.

    Returns a stats dict with counts:
      total          number of records in the JSON
      mismatched     number that have the wrong extension (and would be renamed)
      renamed        number actually renamed (= mismatched in real run, 0 in dry-run)
      missing_audio  number where the audio file couldn't be located
      collision      number skipped because rename target already exists
      unknown_codec  number left untouched because codec wasn't in our mapping
    """
    meta_path = dataset_root / "raw_metadata" / f"{source_class}.json"
    if not meta_path.is_file():
        print(f"  ERROR: {meta_path} not found", file=sys.stderr)
        return {}

    print(f"\n=== Source: {source_class}{' (dry-run)' if dry_run else ''} ===")
    print(f"  metadata: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    stats: Counter[str] = Counter()
    stats["total"] = len(records)

    # Track codec / current-ext combinations for reporting (informational).
    codec_ext_combos: Counter[tuple[str, str]] = Counter()

    # Track examples per (current_ext, target_ext) for first-few logging.
    examples: dict[tuple[str, str], list[str]] = {}

    json_modified = False

    for record in records:
        ffprobe = record.get("ffprobe", {})
        codec   = ffprobe.get("codec", "?")
        ext     = ffprobe.get("ext", "?")
        codec_ext_combos[(ext, codec)] += 1

        change, current_ext, target_ext = needs_rename(record)

        if target_ext is None and codec != "?":
            stats["unknown_codec"] += 1
            continue

        if not change:
            continue

        stats["mismatched"] += 1

        # Build current and new paths from the record.
        rel_path = record.get("filepath", "")
        if not rel_path:
            continue

        current_abs = resolve_audio_path(dataset_root, rel_path)
        if current_abs is None:
            stats["missing_audio"] += 1
            continue

        new_abs = current_abs.with_suffix(target_ext)
        new_rel = str(Path(rel_path).with_suffix(target_ext))
        new_filename = Path(record.get("filename", "")).with_suffix(target_ext).name

        # Collision check — refuse to overwrite an existing file.
        if new_abs.exists() and new_abs.resolve() != current_abs.resolve():
            stats["collision"] += 1
            print(f"  COLLISION: would rename to existing file, skipping: {new_abs}",
                  file=sys.stderr)
            continue

        # Record an example for the first few of each (current → target) pair.
        key = (current_ext, target_ext)
        if key not in examples:
            examples[key] = []
        if len(examples[key]) < 2:
            examples[key].append(f"{current_abs.name} → {new_abs.name}")

        if dry_run:
            stats["would_rename"] += 1
            continue

        # Real run — rename on disk and update the record.
        try:
            current_abs.rename(new_abs)
        except OSError as e:
            stats["rename_error"] += 1
            print(f"  RENAME ERROR for {current_abs}: {e}", file=sys.stderr)
            continue

        record["filepath"]        = new_rel
        record["filename"]        = new_filename
        record["ffprobe"]["ext"]  = target_ext
        json_modified = True
        stats["renamed"] += 1

    # Save JSON if it was modified.
    if json_modified:
        write_json_atomic(meta_path, records)
        print(f"  JSON updated: {meta_path}")

    # Report per-source summary.
    print(f"  records total       : {stats['total']}")

    if codec_ext_combos:
        print(f"  codec / extension combinations seen:")
        for (ext, codec), count in sorted(codec_ext_combos.items(), key=lambda x: -x[1]):
            target = expected_extension(codec)
            tag = ""
            if target is None:
                tag = "  (unknown codec — left alone)"
            elif ext.lower().lstrip(".") != target.lstrip("."):
                tag = f"  ← MISMATCH (should be {target})"
            print(f"    {count:>6}  ext={ext!r:<10} codec={codec!r:<14}{tag}")

    print(f"  mismatched          : {stats['mismatched']}")
    if dry_run:
        print(f"  would rename        : {stats.get('would_rename', 0)}")
    else:
        print(f"  renamed             : {stats.get('renamed', 0)}")
    print(f"  missing audio files : {stats.get('missing_audio', 0)}")
    print(f"  collisions skipped  : {stats.get('collision', 0)}")
    print(f"  rename errors       : {stats.get('rename_error', 0)}")
    print(f"  unknown codecs      : {stats.get('unknown_codec', 0)}")

    if examples:
        print(f"  example renames:")
        for (curr, targ), names in examples.items():
            print(f"    {curr} → {targ}:")
            for name in names:
                print(f"      {name}")

    return dict(stats)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def discover_sources(dataset_root: Path) -> list[str]:
    """List all source classes available in raw_metadata/."""
    meta_dir = dataset_root / "raw_metadata"
    if not meta_dir.is_dir():
        return []
    return sorted(p.stem for p in meta_dir.glob("*.json") if not p.stem.startswith("_"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rename audio files whose extension doesn't match their codec.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="Root containing raw_metadata/ and the audio file tree.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--source", nargs="+", default=None,
        help="One or more source classes to process (e.g., --source melodia natural).",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Process every source class found in raw_metadata/.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be renamed without changing anything.",
    )
    args = parser.parse_args()

    if not args.dataset_root.is_dir():
        print(f"ERROR: dataset-root does not exist: {args.dataset_root}", file=sys.stderr)
        return 1

    if args.all:
        sources = discover_sources(args.dataset_root)
        if not sources:
            print("ERROR: no source JSONs found in raw_metadata/", file=sys.stderr)
            return 1
    else:
        sources = args.source

    print(f"{'DRY-RUN' if args.dry_run else 'LIVE RUN'}: processing {len(sources)} source(s): {sources}")

    grand_total: Counter[str] = Counter()
    for source in sources:
        stats = process_source(source, args.dataset_root, args.dry_run)
        for key, value in stats.items():
            grand_total[key] += value

    print()
    print("=" * 60)
    print("GRAND TOTAL")
    print("=" * 60)
    for key in ("total", "mismatched", "would_rename", "renamed",
                "missing_audio", "collision", "rename_error", "unknown_codec"):
        if key in grand_total:
            print(f"  {key:<20}: {grand_total[key]}")

    if args.dry_run and grand_total.get("would_rename", 0) > 0:
        print()
        print("This was a DRY RUN — no files were modified.")
        print("Re-run without --dry-run to apply the changes.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
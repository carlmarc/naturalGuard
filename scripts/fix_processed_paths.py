#!/usr/bin/env python3
"""
fix_processed_paths.py — One-time fix for stale processed_path values.

CONTEXT
-------
The processing_state/train/<source>.json files were written by an earlier
version of preprocess_audio.py (before --split). At that time, the script
wrote outputs to processed/<source>/<file>.mp3 (flat layout) and recorded
that path in the state JSON.

After restructuring to processed/train/<source>/, the files on disk moved,
but the JSON records still have the old paths. Scripts that read these
state files have to know to insert the missing 'train/' segment.

This script does a one-time fix: walk each processing_state/<split>/*.json,
update every record's processed_path field to insert the missing split
segment, write the file back atomically. After running, downstream scripts
can use processed_path directly with no fix-up.

The script is idempotent — running it twice has the same effect as running
it once. It only modifies paths that need fixing (e.g., 'processed/boomy/...'
becomes 'processed/train/boomy/...'); paths that already include train/ or
test/ are left alone.


USAGE
-----
  python fix_processed_paths.py \\
      --dataset-root /home/cma/Samsung990_2T/datasets/music_detection

  # Dry run — shows what would change without writing.
  python fix_processed_paths.py \\
      --dataset-root /home/cma/Samsung990_2T/datasets/music_detection \\
      --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fix_split(state_dir: Path, split: str, dry_run: bool) -> tuple[int, int]:
    """Fix all JSON files under state_dir/split/.

    Returns (n_files_processed, n_records_fixed).
    """
    split_dir = state_dir / split
    if not split_dir.is_dir():
        return 0, 0

    n_files = 0
    n_fixed_total = 0

    for json_path in sorted(split_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            print(f"  WARN: {json_path.name} is not a list — skipping", file=sys.stderr)
            continue

        n_fixed = 0
        for r in records:
            old_path = r.get("processed_path")
            if not old_path or not isinstance(old_path, str):
                continue

            # Skip if already correct (contains '/train/' or '/test/' after 'processed').
            if "/train/" in old_path or "/test/" in old_path:
                continue
            # Skip if not starting with the expected prefix.
            if not old_path.startswith("processed/"):
                continue

            # Insert the split segment after 'processed/'.
            # 'processed/boomy/abc.mp3' → 'processed/train/boomy/abc.mp3'
            new_path = old_path.replace("processed/", f"processed/{split}/", 1)
            r["processed_path"] = new_path
            n_fixed += 1

        n_files += 1
        n_fixed_total += n_fixed
        if n_fixed > 0:
            print(f"  {json_path.name}: fixed {n_fixed} of {len(records)} records")
            if not dry_run:
                # Atomic write
                tmp = json_path.with_suffix(json_path.suffix + ".tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2, ensure_ascii=False)
                tmp.replace(json_path)
        else:
            print(f"  {json_path.name}: no changes needed ({len(records)} records)")

    return n_files, n_fixed_total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Insert missing 'train/' or 'test/' segment into processed_path "
                    "in processing_state JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="Root containing processing_state/{train,test}/*.json files.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing.",
    )
    args = parser.parse_args()

    state_root = args.dataset_root / "processing_state"
    if not state_root.is_dir():
        print(f"ERROR: {state_root} not found", file=sys.stderr)
        return 1

    if args.dry_run:
        print("=== DRY RUN — no files will be written ===")

    grand_files = 0
    grand_fixed = 0
    for split in ("train", "test"):
        print(f"\n--- {split} ---")
        n_files, n_fixed = fix_split(state_root, split, args.dry_run)
        grand_files += n_files
        grand_fixed += n_fixed

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  files scanned : {grand_files}")
    print(f"  records fixed : {grand_fixed}")
    if args.dry_run and grand_fixed > 0:
        print()
        print("  Re-run without --dry-run to apply changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
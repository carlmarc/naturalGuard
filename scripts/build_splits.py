#!/usr/bin/env python3
"""
scripts/build_splits.py — Build training manifests from processed audio files.

PURPOSE
-------
This script bridges the preprocessing pipeline and training code. It reads
processing_state records (what was successfully normalized), joins them with
raw_metadata (stratification features), and produces JSON manifests that
src/detector/dataset.py consumes directly.

Three manifests are produced:

  naturalguard_pretrain.json  — Phase 1: all natural tracks, for NT-Xent
                                 contrastive pretraining. No pos_weight
                                 (NT-Xent is self-supervised, not binary).

  sourceid_train.json         — Phase 2 train set (90% per source).
                                 Includes natural + all SourceID AI sources
                                 + mixed examples from mixed.json.
                                 Includes top-level pos_weight per class.

  sourceid_val.json           — Phase 2 val set (10% per source).
                                 Same structure; pos_weight from train (not
                                 recomputed on val, which would be leakage).

WHY PRECOMPUTED MANIFESTS (NOT RUNTIME SPLITTING)
--------------------------------------------------
Precomputing manifests guarantees:
  - Deterministic, auditable splits (inspect before training)
  - pos_weight consistent across runs (computed once, stored in manifest)
  - Dataset code stays simple (open manifest, iterate entries)
  - Re-running build_mixed_examples.py then re-running this script adds
    mixed examples without disturbing the pure-source splits

SOURCES INCLUDED
----------------
SourceID classes (8): human, suno, boomy, mubert, mureka, musicgen,
stable_audio, elevenlabsmusic.

  - natural    → human=1, all others=0; uncapped
  - AI sources → their own class=1; capped at AI_TRACK_CAP (default 2500)

Explicitly excluded (per CLAUDE.md spec):
  - melodia:      AI, but its fingerprint is not a SourceID class. It belongs
                  in NaturalGuard's AI-pool at evaluation time, not in
                  SourceID training.
  - ldm2:         Excluded entirely (16 kHz source, empty high-band — would
                  be a trivial frequency-coverage shortcut rather than a
                  creative-agency signal).
  - melodia_test: Held-out test subset of melodia; never used for training.

USAGE
-----
  python scripts/build_splits.py \\
      --dataset-root /home/user/datasets/music_detection \\
      --output-dir data/splits

  # Override defaults:
  python scripts/build_splits.py \\
      --dataset-root /path/to/data \\
      --output-dir data/splits \\
      --val-fraction 0.10 \\
      --ai-cap 2500 \\
      --seed 42
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Bring src/ onto the path so we can import src.dataset.metadata without
# requiring the package to be installed.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset.metadata import song_id as _song_id_fn, stratification_buckets


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Canonical SourceID class order. Used for label dicts and pos_weight dicts.
# The MultilabelHead in model.py must use the same ordering when it converts
# these dicts to tensors.
SOURCEID_CLASSES: list[str] = [
    "human",
    "suno",
    "boomy",
    "mubert",
    "mureka",
    "musicgen",
    "stable_audio",
    "elevenlabsmusic",
]

# AI source names in processing_state that map to SourceID label classes.
SOURCEID_AI_SOURCES: list[str] = [
    "suno",
    "boomy",
    "mubert",
    "mureka",
    "musicgen",
    "stable_audio",
    "elevenlabsmusic",
]

DEFAULT_AI_CAP = 2500
DEFAULT_VAL_FRACTION = 0.10
DEFAULT_SEED = 42

# Sources present on disk that are intentionally excluded from all manifests.
# Documented here (rather than silently skipped) so the exclusion is visible
# when reading the code.
_EXCLUDED_SOURCES: frozenset[str] = frozenset({"melodia", "ldm2", "melodia_test"})


# ---------------------------------------------------------------------------
# PATH NORMALIZATION
# ---------------------------------------------------------------------------


def _resolve_processed_path(raw_path: str, dataset_root: Path) -> str:
    """Return the correct relative processed_path for use in manifests.

    The preprocessing script wrote paths as 'processed/<source>/...' but the
    canonical on-disk layout is 'processed/train/<source>/...'. Patch the
    path transparently if the as-is form doesn't resolve.

    We correct this here rather than rewriting the processing_state files
    because those files are append-only logs of what was processed. Rewriting
    them would risk corruption if a preprocessing run is in progress.
    """
    if (dataset_root / raw_path).exists():
        return raw_path

    # Insert 'train/' after 'processed/' when it's missing.
    parts = Path(raw_path).parts
    if len(parts) >= 2 and parts[0] == "processed" and parts[1] != "train":
        fixed = str(Path("processed") / "train" / Path(*parts[1:]))
        if (dataset_root / fixed).exists():
            return fixed

    # Return as-is and let the caller decide how to handle a missing file.
    return raw_path


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------


def _load_source_records(source: str, dataset_root: Path) -> list[dict[str, Any]]:
    """Load and join processing_state + raw_metadata records for one source.

    Returns one dict per usable processed file with fields:
      song_id, source_class, processed_path, duration_sec,
      vocal_class, duration_bucket, high_energy_bucket.

    Joining is by song_id (filename stem). Records in processing_state that
    have no matching raw_metadata entry get safe default bucket values and
    duration_sec=None. In practice this should never happen for the current
    dataset (verified at dev time).

    Only records with status == "ok" and empty exclusion_reasons are returned.
    Quality filtering was performed by preprocess_audio.py; we trust its output.
    """
    ps_path = dataset_root / "processing_state" / "train" / f"{source}.json"
    rm_path = dataset_root / "raw_metadata" / "train" / f"{source}.json"

    with open(ps_path) as f:
        ps_records: list[dict] = json.load(f)

    with open(rm_path) as f:
        rm_records: list[dict] = json.load(f)

    # Build lookup: song_id → raw_metadata record.
    rm_by_id: dict[str, dict] = {_song_id_fn(r): r for r in rm_records}

    results: list[dict[str, Any]] = []
    for rec in ps_records:
        if rec.get("status") != "ok":
            continue
        if rec.get("exclusion_reasons"):
            continue

        sid = rec["song_id"]
        raw = rm_by_id.get(sid)
        processed_path = _resolve_processed_path(rec["processed_path"], dataset_root)

        if raw is not None:
            buckets = stratification_buckets(raw)
            duration_sec: float | None = raw.get("ffprobe", {}).get("duration")
        else:
            # Safe defaults — treat as worst-case bucket so the record
            # doesn't inflate the most advantaged stratum.
            buckets = {
                "vocal_class": "none",
                "duration_bucket": "<30s",
                "high_energy_bucket": "low",
            }
            duration_sec = None

        results.append(
            {
                "song_id": sid,
                "source_class": source,
                "processed_path": processed_path,
                "duration_sec": duration_sec,
                "vocal_class": buckets["vocal_class"],
                "duration_bucket": buckets["duration_bucket"],
                "high_energy_bucket": buckets["high_energy_bucket"],
            }
        )

    return results


def _load_mixed_records(dataset_root: Path) -> list[dict[str, Any]]:
    """Load mixed example records from processing_state/train/mixed.json.

    Mixed examples are [human=1, ai_class=1] synthetic files built by
    build_mixed_examples.py. They arrive here already in manifest-entry
    format — labels are pre-computed two-hot vectors, processed_path is
    dataset-root-relative, and duration is fixed at 7 s (one window).

    Returns [] if mixed.json doesn't exist yet (script is re-entrant).

    Bucket values for mixed entries:
      vocal_class       = "voiced"  — every mixed example has a human vocal
      duration_bucket   = "<30s"    — 7 s window is always the shortest bucket
      high_energy_bucket = "mid"    — not analyzed per-file; default used here
                                      because mixed examples are excluded from
                                      the stratification cap that protects
                                      against the high-band energy confound
    """
    mixed_path = dataset_root / "processing_state" / "train" / "mixed.json"
    if not mixed_path.exists():
        return []

    with open(mixed_path) as f:
        raw_records: list[dict] = json.load(f)

    results: list[dict[str, Any]] = []
    for rec in raw_records:
        if rec.get("status") != "ok":
            continue
        results.append({
            "song_id":            rec["song_id"],
            "source_class":       "mixed",
            "processed_path":     rec["processed_path"],
            "duration_sec":       7.0,   # fixed — every mixed example is one 7-second window
            "labels":             rec["labels"],
            "vocal_class":        "voiced",
            "duration_bucket":    "<30s",
            "high_energy_bucket": "mid",
        })

    return results


# ---------------------------------------------------------------------------
# STRATIFIED CAP
# ---------------------------------------------------------------------------


def _cap_stratified(
    records: list[dict[str, Any]],
    cap: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Return at most `cap` records with proportional allocation per stratum.

    Strata are (vocal_class, high_energy_bucket). Within each stratum, records
    are shuffled before selection so the chosen subset is representative.

    If len(records) <= cap, returns all records unchanged.

    Why proportional stratification?
    A flat shuffle-and-slice preserves the overall distribution on average but
    can skew small strata. Explicit proportional allocation guarantees that each
    (vocal_class, high_energy_bucket) cell contributes the right fraction of
    the capped pool, preventing the cap from inadvertently shifting confound
    distributions relative to the uncapped natural set.
    """
    if len(records) <= cap:
        return records

    # ---- Group into strata ----

    # Stratum key: (vocal_class, high_energy_bucket) — the two confound axes
    # that stratification_buckets() in metadata.py exposes.
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = (r["vocal_class"], r["high_energy_bucket"])
        buckets[key].append(r)

    total = len(records)  # denominator for fractional shares

    # ---- Largest-Remainder allocation ----
    # Each stratum's ideal share: cap × (stratum_size / total).
    # Floor-division leaves `remainder` slots unassigned; award them one-by-one
    # to the strata with the largest fractional parts.  This is the standard
    # Hamilton/Largest-Remainder method — guarantees sum(allocs) == cap exactly.

    raw_allocs = {k: cap * len(v) / total for k, v in buckets.items()}
    floor_allocs = {k: int(v) for k, v in raw_allocs.items()}  # step 1: floor each share
    remainder = cap - sum(floor_allocs.values())                # slots left after flooring
    # Rank strata by descending fractional part (raw - floor); highest gets first extra slot.
    sorted_keys = sorted(
        buckets.keys(), key=lambda k: -(raw_allocs[k] - floor_allocs[k])
    )
    allocs = dict(floor_allocs)
    for k in sorted_keys[:remainder]:   # award one extra slot per stratum, largest-remainder first
        allocs[k] += 1

    # ---- Sample within each stratum ----

    selected: list[dict[str, Any]] = []
    for k, v in buckets.items():
        shuffled = list(v)
        rng.shuffle(shuffled)           # random subset, not first-N (which could be ordered by batch)
        selected.extend(shuffled[: allocs[k]])

    return selected


# ---------------------------------------------------------------------------
# TRAIN / VAL SPLIT
# ---------------------------------------------------------------------------


def _train_val_split(
    records: list[dict[str, Any]],
    val_fraction: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into train/val stratified by (vocal_class, high_energy_bucket).

    Within each stratum, records are shuffled then split. Stratifying is
    important for small strata (e.g. mureka 'none/low' has ~98 tracks — a
    plain random split could put zero in val).

    val size per stratum = max(1, round(n * val_fraction)) when n > 1,
    else 0 (a single-track stratum goes entirely to train).

    All windows from one song land in the same split because the split is at
    the track level and windows are sampled at runtime by the Dataset class
    (CLAUDE.md rule 7: "Splits are by song_id, never by segment").
    """
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = (r["vocal_class"], r["high_energy_bucket"])
        buckets[key].append(r)

    train_out: list[dict[str, Any]] = []
    val_out: list[dict[str, Any]] = []

    for v in buckets.values():
        shuffled = list(v)
        rng.shuffle(shuffled)
        # Single-track stratum: can't split without leaving train empty.
        # Send it entirely to train; val gets nothing for this cell.
        # round() rather than int() so a 10-track stratum yields 1 val track
        # (int(10 * 0.10) = 1 too, but round(3 * 0.10) = 0 whereas we want 0
        # there — max(1, ...) only kicks in when n > 1).
        n_val = max(1, round(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
        val_out.extend(shuffled[:n_val])
        train_out.extend(shuffled[n_val:])

    return train_out, val_out


# ---------------------------------------------------------------------------
# MANIFEST ENTRY BUILDER
# ---------------------------------------------------------------------------


def _make_entry(record: dict[str, Any]) -> dict[str, Any]:
    """Build one manifest entry dict from a loaded source record.

    Labels are a dict mapping each SourceID class name to 0 or 1. For natural
    audio (source_class == "natural"), human=1 and all AI classes=0. For an AI
    source, human=0 and the matching class=1.

    Mixed content examples ([human=1, ai_class=1]) are generated separately by
    build_mixed_examples.py and will be inserted into the manifests in a later
    run of this script.
    """
    src = record["source_class"]

    # ---- Build one-hot label dict ----
    # Start all-zero; flip exactly one bit for pure-source tracks.
    # Mixed content ([human=1, ai_class=1]) is handled separately — those
    # records arrive from _load_mixed_records() with labels already set and
    # never go through _make_entry().
    labels: dict[str, int] = {cls: 0 for cls in SOURCEID_CLASSES}
    if src == "natural":
        labels["human"] = 1         # natural audio → creative agency is human
    elif src in labels:
        labels[src] = 1             # AI source → flip its own class bit
    # src not in SOURCEID_CLASSES (e.g. melodia passed by mistake) → all-zero
    # labels; callers should not pass such records.

    return {
        "song_id": record["song_id"],
        "source_class": src,
        "processed_path": record["processed_path"],
        "duration_sec": record["duration_sec"],
        "labels": labels,
        "vocal_class": record["vocal_class"],
        "duration_bucket": record["duration_bucket"],
        "high_energy_bucket": record["high_energy_bucket"],
    }


# ---------------------------------------------------------------------------
# POS_WEIGHT
# ---------------------------------------------------------------------------


def _compute_pos_weight(entries: list[dict[str, Any]]) -> dict[str, float]:
    """Compute BCEWithLogitsLoss pos_weight for each SourceID class from train entries.

    pos_weight[c] = negatives_for_c / positives_for_c

    This compensates for the natural:AI track imbalance. With ~9k natural tracks
    and ~2.5k per AI source, human positives outnumber each AI class positives
    ~3.5x; the AI sources are the positives that need upweighting.

    Why track-level counts?
    windows_per_track_per_epoch is applied uniformly to every track, so the
    ratio of window counts equals the ratio of track counts. Computing
    pos_weight from tracks avoids hardcoding the windows_per_track constant here.

    Called on the TRAIN set only. Storing the result in both train and val
    manifests means the training loop never recomputes it and val metrics use
    the same weight (consistent evaluation).
    """
    # ---- Count positives and negatives per class ----
    # Each entry contributes independently to every class (multilabel, not
    # multiclass), so a mixed [human=1, suno=1] entry increments pos for both
    # human and suno and neg for the other six classes.
    pos: dict[str, int] = {c: 0 for c in SOURCEID_CLASSES}
    neg: dict[str, int] = {c: 0 for c in SOURCEID_CLASSES}
    for entry in entries:
        for cls, val in entry["labels"].items():
            if val == 1:
                pos[cls] += 1
            else:
                neg[cls] += 1

    # ---- Compute per-class weight ----
    # pos_weight[c] = neg[c] / pos[c]  →  BCE loss treats each positive as
    # if it were neg/pos negatives, balancing the gradient contribution
    # regardless of how skewed the natural:AI track ratio is.
    weights: dict[str, float] = {}
    for cls in SOURCEID_CLASSES:
        if pos[cls] == 0:
            # Class has no positive examples in train — weight 1.0 is a safe
            # no-op; training will never update on this class.
            weights[cls] = 1.0
        else:
            weights[cls] = neg[cls] / pos[cls]
    return weights


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------


def _print_split_stats(
    source: str,
    n_train: int,
    n_val: int,
    total_before_split: int,
) -> None:
    pct = 100.0 * n_val / (n_train + n_val) if (n_train + n_val) else 0
    print(f"  {source:20s}  total={total_before_split:5d}  train={n_train:5d}  val={n_val:4d}  ({pct:.1f}% val)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, build splits, write manifests, print summary."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Root of the music_detection dataset tree (contains processed/, processing_state/, raw_metadata/).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write manifests into (created if absent).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help=f"Fraction of each source to hold out for validation (default {DEFAULT_VAL_FRACTION}).",
    )
    parser.add_argument(
        "--ai-cap",
        type=int,
        default=DEFAULT_AI_CAP,
        help=f"Maximum tracks per AI source, applied before splitting (default {DEFAULT_AI_CAP}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed for reproducible cap sampling and splits (default {DEFAULT_SEED}).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)   # seeded once; all sub-calls draw from this instance
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load records for all sources ----

    print("Loading sources ...")
    natural_records = _load_source_records("natural", args.dataset_root)
    print(f"  natural: {len(natural_records)} usable tracks")

    ai_records: dict[str, list[dict[str, Any]]] = {}
    for src in SOURCEID_AI_SOURCES:
        records = _load_source_records(src, args.dataset_root)
        # Cap each AI source to DEFAULT_AI_CAP (2500) tracks, preserving the
        # vocal_class × high_energy_bucket distribution.  Natural is uncapped
        # (~9 k tracks) because the pos_weight mechanism handles the imbalance.
        capped = _cap_stratified(records, args.ai_cap, rng)
        ai_records[src] = capped
        cap_note = f" (capped from {len(records)})" if len(capped) < len(records) else ""
        print(f"  {src}: {len(capped)} usable tracks{cap_note}")

    # Mixed examples are not capped — build_mixed_examples.py already controls
    # their volume, and they don't need stratification (all are 7-second voiced clips).
    mixed_records = _load_mixed_records(args.dataset_root)
    print(f"  mixed: {len(mixed_records)} examples")

    # ---- Write naturalguard_pretrain.json ----
    # Natural-only. NT-Xent is self-supervised — no labels needed for the loss
    # function, but entries include labels for consistency with the SourceID
    # manifests (useful for debugging and future use).

    ng_entries = [_make_entry(r) for r in natural_records]
    ng_manifest: dict[str, Any] = {"entries": ng_entries}

    ng_path = args.output_dir / "naturalguard_pretrain.json"
    with open(ng_path, "w") as f:
        json.dump(ng_manifest, f, indent=2)
    print(f"\nnaturalguard_pretrain.json: {len(ng_entries)} entries")

    # ---- Compute stratified train/val split per source ----
    # Each source is split independently so every source appears in both train
    # and val at the specified fraction.  Results are then concatenated into
    # the combined sourceid_train / sourceid_val manifests.

    print("\nTrain/val split (stratified by vocal_class × high_energy_bucket):")
    all_train: list[dict[str, Any]] = []
    all_val: list[dict[str, Any]] = []

    nat_train, nat_val = _train_val_split(natural_records, args.val_fraction, rng)
    all_train.extend(_make_entry(r) for r in nat_train)
    all_val.extend(_make_entry(r) for r in nat_val)
    _print_split_stats("natural", len(nat_train), len(nat_val), len(natural_records))

    for src in SOURCEID_AI_SOURCES:
        records = ai_records[src]
        trn, val = _train_val_split(records, args.val_fraction, rng)
        all_train.extend(_make_entry(r) for r in trn)
        all_val.extend(_make_entry(r) for r in val)
        _print_split_stats(src, len(trn), len(val), len(records))

    # Mixed examples are already in entry format (labels are pre-computed two-hot).
    # Split the same 90/10 as pure sources. No _make_entry() call needed.
    if mixed_records:
        mix_trn, mix_val = _train_val_split(mixed_records, args.val_fraction, rng)
        all_train.extend(mix_trn)
        all_val.extend(mix_val)
        _print_split_stats("mixed", len(mix_trn), len(mix_val), len(mixed_records))

    # ---- Compute pos_weight and write sourceid manifests ----
    # pos_weight is computed on the train set only; same values are stored in
    # the val manifest so the training loop never has to recompute them.

    pos_weight = _compute_pos_weight(all_train)

    train_manifest: dict[str, Any] = {
        "class_names": SOURCEID_CLASSES,  # class order that MultilabelHead must match
        "pos_weight": pos_weight,
        "entries": all_train,
    }
    val_manifest: dict[str, Any] = {
        "class_names": SOURCEID_CLASSES,
        "pos_weight": pos_weight,   # from train — not recomputed on val
        "entries": all_val,
    }

    train_path = args.output_dir / "sourceid_train.json"
    val_path = args.output_dir / "sourceid_val.json"

    with open(train_path, "w") as f:
        json.dump(train_manifest, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val_manifest, f, indent=2)

    # ---- Print summary ----

    print(f"\nsourceid_train.json: {len(all_train):6d} entries")
    print(f"sourceid_val.json:   {len(all_val):6d} entries")
    total_sourceid = len(all_train) + len(all_val)
    print(f"sourceid total:      {total_sourceid:6d} entries")

    print("\npos_weight (train):")
    max_w = max(pos_weight.values())
    for cls in SOURCEID_CLASSES:
        # Bar width proportional to weight; max 40 chars.  AI classes will be
        # ~3.5× wider than human because human positives outnumber AI positives.
        bar = "█" * int(40 * pos_weight[cls] / max(max_w, 1))
        print(f"  {cls:20s}  {pos_weight[cls]:6.3f}  {bar}")


if __name__ == "__main__":
    main()

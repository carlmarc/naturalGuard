"""
src/dataset/metadata.py — Helper functions for working with raw_metadata records.

PURPOSE
-------
The raw_metadata/*.json files (produced by analyze_dataset.py) are the single
source of truth for per-file features in this project. There is no reformatted
intermediate stage. Any pipeline script that needs to filter records or assign
stratification buckets imports the functions in this module and applies them
to raw_metadata records on the fly.

The functions are pure: same input → same output, no side effects, no I/O.


WHY THIS MODULE EXISTS
----------------------
Three concerns need shared logic across scripts:

  1. Filtering — which files are too low-quality, too short, or too silent to
                 train on. Used by preprocess_audio.py (to skip them) and by
                 build_splits.py (to ignore them when sampling).

  2. Bucketing — discrete labels for stratified sampling (vocal_class,
                 duration_bucket, high_energy_bucket). Used by build_splits.py
                 to ensure confound-prone features are distributed identically
                 across classes.

  3. Identifiers — derive song_id from a record's filename. Used everywhere
                   to track files across pipeline stages and to split
                   train/val/test by song_id (never by segment).

Putting these in one module means all scripts use identical logic. Changing
a threshold here propagates everywhere automatically.


THE FEATURES ARE NOT MODEL INPUT
--------------------------------
The features stored in raw_metadata (MFCCs, rolloff, energy ratios, etc.) are
never fed to the model. They drive sampling and filtering decisions only.
The model receives audio waveforms; everything else here is metadata used to
construct a training set that doesn't contain shortcut features.


THRESHOLD CONSTANTS
-------------------
All filter thresholds and bucket boundaries are constants at the top of this
file. Change them here, downstream behavior updates automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# ----------------------------------------------------------------------------
# QUALITY FILTER THRESHOLDS
# ----------------------------------------------------------------------------
# Hard filters (file is excluded entirely):

MIN_BITRATE_KBPS    = 128       # Below this, source quality is too low to use.
MIN_DURATION_SEC    = 10.0      # Need enough length for at least one 7s window + slack.
MIN_RMS_DB          = -40.0     # Below this, the file is effectively silent overall.
MAX_SILENCE_RATIO   = 0.3       # More than 30% silent segments — file is mostly empty.

# Soft filters (flagged but still excluded, as a precaution):

MIN_RMS_DB_STD      = 2.0       # Suspiciously flat loudness over the whole track.
MIN_CREST_FACTOR_DB = 5.0       # Heavily limited / clipped.
MAX_CREST_FACTOR_DB = 25.0      # Extreme dynamics or measurement artifact.


# ----------------------------------------------------------------------------
# BUCKET BOUNDARIES (for stratified sampling)
# ----------------------------------------------------------------------------
# Duration buckets — prevents track-length confound between classes.
# (Natural mostly 3-minute songs vs Suno mostly 2-minute songs could otherwise
#  become a class shortcut.)

DURATION_BUCKETS = [
    ("<30s",     0.0,    30.0),
    ("30-60s",   30.0,   60.0),
    ("60-180s",  60.0,   180.0),
    ("180-300s", 180.0,  300.0),
    (">300s",    300.0,  float("inf")),
]

# High-band energy buckets — addresses the most important confound in this
# dataset. Natural has mean high_energy_ratio ~0.328; AI generators are at
# 0.23–0.30. Without stratification on this axis, the model trivially learns
# "high band rich → natural" as a shortcut.
# Boundaries are chosen so each bucket holds a meaningful slice of each class.

HIGH_ENERGY_BUCKETS = [
    ("low",  0.00, 0.27),
    ("mid",  0.27, 0.31),
    ("high", 0.31, 1.01),    # upper 1.01 so a value of exactly 1.0 lands in "high"
]


# ----------------------------------------------------------------------------
# IDENTIFIERS
# ----------------------------------------------------------------------------

def song_id(record: dict[str, Any]) -> str:
    """Derive a song_id from a record's filename.

    For hash-named files (boomy uses content hashes), the song_id IS the hash.
    For other naming schemes, it's whatever's left after stripping the
    extension. Used to split train/val/test by song — all windows from one
    song must land in the same split.
    """
    filename = record.get("filename", "")
    return Path(filename).stem


# ----------------------------------------------------------------------------
# QUALITY FILTERING
# ----------------------------------------------------------------------------

def is_included(record: dict[str, Any]) -> bool:
    """Return True if this record passes all quality filters.

    Use this from any script that needs to decide whether to use a file.
    The detailed reasons for exclusion are available via exclusion_reasons().
    """
    return not exclusion_reasons(record)


def exclusion_reasons(record: dict[str, Any]) -> list[str]:
    """Return a list of human-readable reasons this record fails quality filters.

    An empty list means the record passes everything (is_included returns True).
    A non-empty list means the record is excluded; each string explains one
    failed check.

    Both hard and soft failures appear here — there is no distinction at the
    inclusion level, both exclude the file. The distinction is informational
    only, useful for reporting (e.g., "how many files did we lose to soft
    filters?").
    """
    reasons: list[str] = []

    # Upstream quality flag from analyze_dataset.py. If it already said no,
    # we say no too.
    upstream = record.get("quality_flag", "unknown")
    if upstream != "ok":
        reasons.append(f"upstream_quality_flag={upstream}")

    ffprobe  = record.get("ffprobe",  {})
    features = record.get("librosa",  {})    # source JSON uses "librosa"
    segments = record.get("segments", {})

    # --- Hard filters ---

    bitrate = ffprobe.get("bitrate_kbps")
    if bitrate is None or bitrate < MIN_BITRATE_KBPS:
        reasons.append(f"bitrate_kbps={bitrate} below floor {MIN_BITRATE_KBPS}")

    duration = ffprobe.get("duration")
    if duration is None or duration < MIN_DURATION_SEC:
        reasons.append(f"duration={duration} below floor {MIN_DURATION_SEC}")

    rms = features.get("rms_db")
    if rms is None or rms < MIN_RMS_DB:
        reasons.append(f"rms_db={rms} below floor {MIN_RMS_DB}")

    silence_ratio = segments.get("silence_ratio")
    if silence_ratio is not None and silence_ratio > MAX_SILENCE_RATIO:
        reasons.append(f"silence_ratio={silence_ratio} above ceiling {MAX_SILENCE_RATIO}")

    if features.get("high_freq_low_flag") is True:
        reasons.append("high_freq_low_flag=true")

    # --- Soft filters ---

    rms_std = features.get("rms_db_std")
    if rms_std is not None and rms_std < MIN_RMS_DB_STD:
        reasons.append(f"rms_db_std={rms_std} suspiciously flat (<{MIN_RMS_DB_STD})")

    crest = features.get("crest_factor_db")
    if crest is not None:
        if crest < MIN_CREST_FACTOR_DB:
            reasons.append(f"crest_factor_db={crest} below {MIN_CREST_FACTOR_DB} (likely clipping)")
        elif crest > MAX_CREST_FACTOR_DB:
            reasons.append(f"crest_factor_db={crest} above {MAX_CREST_FACTOR_DB} (extreme dynamics)")

    return reasons


# ----------------------------------------------------------------------------
# STRATIFICATION BUCKETS
# ----------------------------------------------------------------------------

def vocal_class(record: dict[str, Any]) -> str:
    """Return "voiced" if vocals are present, "none" otherwise.

    Based on the rough track-level vocal detection from analyze_dataset.py.
    Used by build_splits.py to balance vocal/no-vocal counts across classes.
    Without this balancing, MusicGen's 30% vocal rate (vs natural's 74%)
    would let the model learn "no vocals → AI" as a shortcut.

    Missing/None has_vocals_rough is treated as "none" (safer default).
    """
    features = record.get("librosa", {})
    return "voiced" if features.get("has_vocals_rough") is True else "none"


def duration_bucket(record: dict[str, Any]) -> str:
    """Return the named duration bucket for this record.

    Buckets are <30s, 30-60s, 60-180s, 180-300s, >300s. Used to prevent
    track-length distributions from differing systematically across classes.
    """
    ffprobe = record.get("ffprobe", {})
    duration = ffprobe.get("duration")
    if duration is None:
        return "<30s"    # safest default — treat missing as worst-case
    for name, lo, hi in DURATION_BUCKETS:
        if lo <= duration < hi:
            return name
    return ">300s"


def high_energy_bucket(record: dict[str, Any]) -> str:
    """Return the named high-band energy bucket for this record.

    The most important stratification axis. Buckets are low/mid/high based on
    features.high_energy_ratio. Boundaries chosen so natural music (mean ~0.328)
    sits mostly in "high" and AI generators (means 0.23–0.30) split across
    "low" and "mid". build_splits.py samples proportionally from each bucket
    per class so the model cannot use high-band energy as a class shortcut.
    """
    features = record.get("librosa", {})
    ratio = features.get("high_energy_ratio")
    if ratio is None:
        return "low"    # safest default — treat missing as worst-case
    for name, lo, hi in HIGH_ENERGY_BUCKETS:
        if lo <= ratio < hi:
            return name
    return "high"


# ----------------------------------------------------------------------------
# CONVENIENCE: combine all buckets into one dict
# ----------------------------------------------------------------------------

def stratification_buckets(record: dict[str, Any]) -> dict[str, str]:
    """Return all stratification bucket labels for a record at once.

    Returns a dict with keys: vocal_class, duration_bucket, high_energy_bucket.
    Useful when build_splits.py is grouping records by their full bucket
    combination for stratified sampling.
    """
    return {
        "vocal_class":        vocal_class(record),
        "duration_bucket":    duration_bucket(record),
        "high_energy_bucket": high_energy_bucket(record),
    }
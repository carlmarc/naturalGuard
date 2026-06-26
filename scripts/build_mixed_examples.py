#!/usr/bin/env python3
"""
scripts/build_mixed_examples.py — Generate synthetic mixed (vocal-over-AI) examples.

PURPOSE
-------
Creates [human=1, ai_class=1] training examples by mixing a human vocal track
with an AI-generated instrumental. Phase 2 (SourceID supervised training)
needs these because real-world music often features vocals. Without mixed
examples, the model could learn "no vocals → AI" as a spurious shortcut
rather than detecting creative agency.

PIPELINE PER EXAMPLE
--------------------
1.  Pick random speaker from the correct vocal pool (train/test)
2.  Sample a 7-second window from the speaker's preprocessed MP3
3.  Pick a random AI source + random track + random 7-second window
4.  Apply pedalboard vocal chain: HPF → shelf EQ → de-ess → optional reverb
5.  Apply random vocal gain (±6 dB)
6.  Mix: mixed = vocal_factor * vocal_after_gain + instrumental
    (vocal_factor = 10 ** (VOCAL_PRE_SUM_DB / 20); default −6 dB)
7.  Apply pedalboard glue compression + makeup gain
8.  Write temp float32 WAV → ffmpeg-encode to project-standard MP3

Both ingredients have already been through preprocess_audio.py once
(−16 LUFS, 22050 Hz mono, 192 kbps), so codec history is symmetric.

OUTPUTS
-------
  processed/{train,test}/mixed/<vocal_dataset>_over_<ai_source>/<idx:05d>.mp3
  processing_state/{train,test}/mixed.json

USAGE
-----
  python scripts/build_mixed_examples.py \\
    --dataset-root /home/user/datasets/music_detection \\
    --train-count 1500 --test-count 300 \\
    --workers 4 --seed 42

REQUIREMENTS
------------
  torchaudio >= 2.0  (audio loading — never librosa)
  pedalboard >= 0.9  (DSP chain)
  soundfile          (temp float32 WAV write)
  numpy
  ffmpeg             (final MP3 encode — same params as preprocess_audio.py)
"""

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torchaudio
from pedalboard import (
    Compressor,
    Gain,
    HighpassFilter,
    HighShelfFilter,
    LowShelfFilter,
    Pedalboard,
    Reverb,
)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

WINDOW_SEC = 7
SR = 22050
WINDOW_SAMPLES = WINDOW_SEC * SR  # 154350 samples

# AI sources eligible for SourceID. melodia is excluded per spec
# (its fingerprint is not a SourceID class).
SOURCEID_AI_SOURCES: list[str] = [
    "suno", "boomy", "mubert", "mureka",
    "musicgen", "stable_audio", "elevenlabsmusic",
]

# Canonical SourceID class order — must match build_splits.py.
SOURCEID_CLASSES: list[str] = [
    "human", "suno", "boomy", "mubert", "mureka",
    "musicgen", "stable_audio", "elevenlabsmusic",
]

VOCAL_DATASETS: list[str] = ["librispeech_devclean", "nus_48e_sing"]

# Baseline vocal attenuation applied before summing with the instrumental.
# Both ingredients arrive at -16 LUFS, but a vocal at -16 LUFS concentrates
# energy in the perceptually-loud mid range, making it sound too prominent
# when summed at equal amplitude. Real productions typically seat the vocal
# 3–9 dB below the integrated instrumental level. -6 dB as the midpoint
# of that range, combined with the ±6 dB random vocal gain, yields an
# effective vocal-vs-instrumental balance ranging from -12 dB to 0 dB.
# The final ffmpeg loudnorm re-normalises absolute file level to -16 LUFS,
# so only the internal balance shifts — not the output loudness.
VOCAL_PRE_SUM_DB: float = -6.0

# A window is considered silent if max(|x|) is below this value.
# Pre-processed files are at -16 LUFS, so genuine audio should be well above.
VOCAL_SILENCE_THRESHOLD = 0.01
MAX_WINDOW_RETRIES = 15

# Reverb preset parameters. Tight = small room; loose = large hall.
_REVERB_PRESETS: dict[str, dict] = {
    "tight": {"room_size": 0.15, "damping": 0.80, "width": 1.0},
    "loose": {"room_size": 0.55, "damping": 0.30, "width": 1.0},
}
# Sampling weights for [tight, loose, none]. ~70% reverb, ~30% dry.
_REVERB_WEIGHTS = [35, 35, 30]

# Write manifest to disk every N completed examples (crash recovery).
_FLUSH_EVERY = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WORK ITEM
# ---------------------------------------------------------------------------


@dataclass
class WorkItem:
    """Everything a worker needs to generate one mixed example.

    All paths are absolute strings so the worker doesn't need dataset_root.
    The per-example seed ensures full reproducibility: same WorkItem always
    produces the same MP3.
    """

    idx: int            # per-combination index (also the output filename stem)
    split: str          # 'train' or 'test'
    combo_id: str       # e.g. 'librispeech_devclean_over_boomy'
    song_id: str        # unique manifest identifier

    vocal_dataset: str  # 'librispeech_devclean' or 'nus_48e_sing'
    vocal_speaker: str  # speaker key in the split JSON
    vocal_mp3_abs: str  # absolute path to speaker's preprocessed MP3

    ai_source: str      # SourceID class name
    ai_song_id: str     # song_id from processing_state
    ai_mp3_abs: str     # absolute path to AI track's preprocessed MP3

    output_mp3_abs: str     # absolute path for the output MP3
    output_mp3_rel: str     # dataset-root-relative path (stored in manifest)

    seed: int           # per-example RNG seed


# ---------------------------------------------------------------------------
# STABLE SEED
# ---------------------------------------------------------------------------


def _stable_seed(base: int, key: str) -> int:
    """Derive a deterministic per-example seed from a base seed and a string key.

    Python's built-in hash() is not stable across interpreter restarts (PYTHONHASHSEED).
    Using MD5 here gives the same seed for the same (base, key) regardless of
    environment, ensuring reproducibility when the script is re-run.
    """
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) & 0x7FFFFFFF
    return (base ^ h) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# FFMPEG ENCODE
# ---------------------------------------------------------------------------


def _ffmpeg_encode(input_wav: str, output_mp3: str, timeout_sec: int = 60) -> None:
    """Write output_mp3 from input_wav using the project-standard ffmpeg pipeline.

    Parameters are identical to preprocess_audio.py:
      loudnorm I=-16 TP=-1.5 LRA=11 | mono | 22050 Hz | libmp3lame 192 kbps

    The final loudnorm re-normalises the mix to -16 LUFS, compensating for
    any level shift introduced by the vocal gain and compression steps.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", input_wav,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ac", "1",
        "-ar", str(SR),
        "-codec:a", "libmp3lame",
        "-b:a", "192k",
        output_mp3,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=timeout_sec)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}): "
            f"{result.stderr.decode(errors='replace')[:400]}"
        )


# ---------------------------------------------------------------------------
# AUDIO WINDOW SAMPLING
# ---------------------------------------------------------------------------


def _sample_window(
    mp3_abs: str,
    rng: np.random.Generator,
    silence_check: bool = False,
) -> np.ndarray:
    """Sample a WINDOW_SEC window from an MP3 file. Returns [1, WINDOW_SAMPLES] float32.

    Uses torchaudio offset-loading for memory efficiency: only the target
    WINDOW_SAMPLES are decoded from disk. Falls back to full load + slice
    if the MP3 backend doesn't support seeking (some older encoders).

    Why silence_check?
    Vocal tracks can contain extended silence pauses (e.g. a speaker pause
    between paragraphs). A silent vocal window would effectively create a
    pure-instrumental mix labelled as [human=1, ai=1] — misleading training
    signal. The check avoids this by retrying with a different start offset.
    """
    # Get total frames via metadata (avoids decoding the full file).
    try:
        info = torchaudio.info(mp3_abs)
        total_frames = info.num_frames
    except Exception:
        total_frames = None

    arr = None
    for attempt in range(MAX_WINDOW_RETRIES):
        if total_frames is not None and total_frames > WINDOW_SAMPLES:
            start = int(rng.integers(0, total_frames - WINDOW_SAMPLES))
        else:
            start = 0

        try:
            wav, _ = torchaudio.load(mp3_abs, frame_offset=start, num_frames=WINDOW_SAMPLES)
            arr = wav.numpy()  # [1, N]
        except Exception:
            # Offset decoding failed — fall back to full load.
            wav, _ = torchaudio.load(mp3_abs)
            n = wav.shape[-1]
            max_start = max(0, n - WINDOW_SAMPLES)
            start = int(rng.integers(0, max_start + 1))
            arr = wav.numpy()[:, start: start + WINDOW_SAMPLES]

        # Zero-pad if the file is shorter than one window.
        if arr.shape[-1] < WINDOW_SAMPLES:
            arr = np.pad(arr, ((0, 0), (0, WINDOW_SAMPLES - arr.shape[-1])))

        if silence_check and float(np.abs(arr).max()) < VOCAL_SILENCE_THRESHOLD:
            if attempt < MAX_WINDOW_RETRIES - 1:
                continue  # try another start position
        break

    return arr.astype(np.float32)  # ensure float32 for pedalboard


# ---------------------------------------------------------------------------
# VOCAL CHAIN
# ---------------------------------------------------------------------------


def _apply_vocal_chain(
    vocal: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply HPF, shelf EQ, de-ess, and optional reverb to the vocal window.

    Chain design rationale:
    - HighpassFilter (80 Hz): removes sub-bass rumble and handling noise.
      Vocals rarely have meaningful content below 80 Hz; the AI instrumentals
      have their own low-end, so overlapping bass would muddy the mix.
    - HighShelfFilter (±2 dB, 3.5 kHz): simulates a presence boost or cut
      that a mixing engineer commonly applies to help the vocal cut through
      or sit back in the mix.
    - LowShelfFilter (−1 to −2 dB, 8 kHz): gentle de-essing / air
      reduction, as specified.
    - Reverb (70% chance, tight or loose preset): simulates room treatment
      a mixing engineer might add to dry vocal before placing it in a mix.
      Wet level is kept subtle (−20 to −10 dB) so the vocal remains dry-ish.

    Returns (processed_audio_[1, WINDOW_SAMPLES], params_for_manifest).
    """
    shelf_db  = float(rng.uniform(-2.0,  2.0))
    deess_db  = float(rng.uniform(-2.0, -1.0))
    wet_db    = float(rng.uniform(-20.0, -10.0))

    total_w = sum(_REVERB_WEIGHTS)
    preset = str(rng.choice(
        ["tight", "loose", "none"],
        p=[w / total_w for w in _REVERB_WEIGHTS],
    ))

    plugins: list = [
        HighpassFilter(cutoff_frequency_hz=80.0),
        HighShelfFilter(cutoff_frequency_hz=3500.0, gain_db=shelf_db),
        LowShelfFilter(cutoff_frequency_hz=8000.0, gain_db=deess_db),
    ]

    if preset != "none":
        wet_linear = float(10 ** (wet_db / 20.0))
        p = _REVERB_PRESETS[preset]
        plugins.append(Reverb(
            room_size=p["room_size"],
            damping=p["damping"],
            width=p["width"],
            dry_level=1.0,
            wet_level=wet_linear,
        ))

    processed = Pedalboard(plugins)(vocal, SR)

    params: dict[str, Any] = {
        "vocal_shelf_db":  shelf_db,
        "deess_db":        deess_db,
        "reverb_preset":   preset,
        "reverb_wet_db":   wet_db if preset != "none" else None,
    }
    return processed.astype(np.float32), params


# ---------------------------------------------------------------------------
# GLUE COMPRESSION
# ---------------------------------------------------------------------------


def _apply_glue_compression(
    mixed: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply gentle glue compression + makeup gain to the mixed signal.

    A 2:1 compressor set just below the mix's peak pulls the loudest
    transients back slightly, giving the mix a more cohesive sound before
    the final loudnorm step. The makeup gain restores the level so loudnorm
    has a well-levelled signal to work from.

    Formula (from CLAUDE.md):
      peak_dbfs  = 20 * log10(max|mixed| + ε)
      target     ~ U(2.0, 4.0) dB  — how far below peak to place threshold
      threshold  = peak_dbfs − target
      Compressor(threshold, ratio=2, attack=10ms, release=100ms)
      Gain(target dB)                — makeup
    """
    peak_dbfs = 20.0 * math.log10(float(np.abs(mixed).max()) + 1e-9)
    target    = float(rng.uniform(2.0, 4.0))
    threshold = peak_dbfs - target

    board = Pedalboard([
        Compressor(threshold_db=threshold, ratio=2.0, attack_ms=10.0, release_ms=100.0),
        Gain(gain_db=target),
    ])
    compressed = board(mixed, SR)

    params: dict[str, float] = {
        "compression_target_db":    target,
        "compression_threshold_db": threshold,
    }
    return compressed.astype(np.float32), params


# ---------------------------------------------------------------------------
# WORKER  (module-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------


def _worker(item: WorkItem) -> dict[str, Any]:
    """Generate one mixed example. Returns a manifest record dict.

    Designed to run inside a ProcessPoolExecutor worker process. The function
    is module-level so it is picklable. All dependencies (torchaudio,
    pedalboard, numpy, soundfile) are imported at module level and available
    in the worker process.

    On success: record['status'] == 'ok'
    On failure: record['status'] == 'failed', record['error'] contains the
                exception message. The caller decides whether to abort or skip.
    """
    rng = np.random.default_rng(item.seed)

    try:
        # Ensure output directory exists (mkdir is atomic on Linux).
        Path(item.output_mp3_abs).parent.mkdir(parents=True, exist_ok=True)

        # ── 1–2: Sample vocal window ─────────────────────────────────────
        vocal = _sample_window(item.vocal_mp3_abs, rng, silence_check=True)

        # ── 3: Sample AI instrumental window ─────────────────────────────
        instrumental = _sample_window(item.ai_mp3_abs, rng, silence_check=False)

        # ── 4: Vocal chain ────────────────────────────────────────────────
        vocal_proc, vocal_params = _apply_vocal_chain(vocal, rng)

        # ── 5: Vocal gain ─────────────────────────────────────────────────
        vocal_gain_db = float(rng.uniform(-6.0, 6.0))
        vocal_proc = vocal_proc * float(10.0 ** (vocal_gain_db / 20.0))

        # ── 6: Mix ────────────────────────────────────────────────────────
        # Apply VOCAL_PRE_SUM_DB baseline attenuation so the vocal sits
        # below the instrumental level on average, matching real-world
        # mix practice. The ±6 dB random gain from step 5 moves the
        # balance within [-12, 0] dB relative to the instrumental.
        vocal_factor = float(10.0 ** (VOCAL_PRE_SUM_DB / 20.0))
        mixed = vocal_factor * vocal_proc + instrumental

        # ── 7: Glue compression ───────────────────────────────────────────
        mixed_comp, comp_params = _apply_glue_compression(mixed, rng)

        # ── 8: Temp float32 WAV → ffmpeg MP3 ─────────────────────────────
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="mixbld_") as tmp:
            tmp_wav = tmp.name
        try:
            sf.write(tmp_wav, mixed_comp.T, SR, subtype="FLOAT")
            _ffmpeg_encode(tmp_wav, item.output_mp3_abs)
        finally:
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass

        # Build labels dict: human=1 and the AI source=1.
        labels = {cls: 0 for cls in SOURCEID_CLASSES}
        labels["human"] = 1
        labels[item.ai_source] = 1

        mix_params: dict[str, Any] = {
            "vocal_gain_db":      vocal_gain_db,
            "vocal_pre_sum_db":   VOCAL_PRE_SUM_DB,
            **vocal_params,
            **comp_params,
        }

        return {
            "song_id":              item.song_id,
            "status":               "ok",
            "idx":                  item.idx,
            "combo":                item.combo_id,
            "vocal_dataset":        item.vocal_dataset,
            "vocal_speaker":        item.vocal_speaker,
            "instrumental_source":  item.ai_source,
            "instrumental_song_id": item.ai_song_id,
            "processed_path":       item.output_mp3_rel,
            "labels":               labels,
            "mix_params":           mix_params,
            "error":                None,
        }

    except Exception as exc:
        return {
            "song_id":              item.song_id,
            "status":               "failed",
            "idx":                  item.idx,
            "combo":                item.combo_id,
            "vocal_dataset":        item.vocal_dataset,
            "vocal_speaker":        item.vocal_speaker,
            "instrumental_source":  item.ai_source,
            "instrumental_song_id": item.ai_song_id,
            "processed_path":       item.output_mp3_rel,
            "labels":               None,
            "mix_params":           None,
            "error":                str(exc),
        }


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------


def _load_speakers(
    dataset_root: Path,
    vocal_dataset: str,
    pool: str,  # 'train' or 'test'
) -> list[dict[str, Any]]:
    """Load speaker records for one vocal dataset + pool from the split manifest.

    Returns list of dicts with keys: speaker_id, mp3_abs, duration_sec.
    Only includes speakers with status='ok'.
    """
    manifest_path = dataset_root / "ingredients" / f"{vocal_dataset}_split.json"
    if not manifest_path.exists():
        _log.warning("Vocal split manifest not found: %s", manifest_path)
        return []

    with open(manifest_path) as f:
        data = json.load(f)

    result = []
    for speaker_id, rec in data["speakers"].items():
        if rec.get("pool") != pool:
            continue
        if rec.get("status") != "ok":
            continue
        mp3_rel = rec["processed_path"]   # relative to ingredients/
        mp3_abs = dataset_root / "ingredients" / mp3_rel
        result.append(
            {
                "speaker_id":   speaker_id,
                "mp3_abs":      str(mp3_abs),
                "duration_sec": rec.get("duration_sec"),
            }
        )
    return result


def _load_ai_tracks(
    dataset_root: Path,
    ai_source: str,
    split: str,  # 'train' or 'test'
) -> list[dict[str, Any]]:
    """Load AI track records for one source + split from processing_state.

    Returns list of dicts with keys: song_id, mp3_abs.
    Returns [] if the processing_state file doesn't exist (e.g. test split).
    """
    ps_path = dataset_root / "processing_state" / split / f"{ai_source}.json"
    if not ps_path.exists():
        _log.warning("No processing_state for %s/%s — skipping.", split, ai_source)
        return []

    with open(ps_path) as f:
        records = json.load(f)

    result = []
    for rec in records:
        if rec.get("status") != "ok":
            continue
        if rec.get("exclusion_reasons"):
            continue
        mp3_abs = dataset_root / rec["processed_path"]
        result.append(
            {
                "song_id": rec["song_id"],
                "mp3_abs": str(mp3_abs),
            }
        )
    return result


# ---------------------------------------------------------------------------
# WORK PLANNING
# ---------------------------------------------------------------------------


def _plan_work(
    dataset_root: Path,
    split: str,
    total_count: int,
    seed: int,
    existing_song_ids: set[str],
) -> list[WorkItem]:
    """Create the full list of WorkItems for one split (train or test).

    Distributes `total_count` examples as evenly as possible across
    (vocal_dataset, ai_source) combinations. Skips combinations where
    either the vocal pool or AI source has no data. Skips song_ids that
    already appear in the existing manifest (resumability).

    The per-combination work is deterministically ordered so re-running the
    script after a partial failure regenerates exactly the same examples
    for any index not already in the manifest.
    """
    pool = split  # 'train' → train_pool, 'test' → test_pool

    # Build speaker lists per vocal dataset.
    speakers: dict[str, list] = {}
    for vd in VOCAL_DATASETS:
        spks = _load_speakers(dataset_root, vd, pool)
        if spks:
            speakers[vd] = spks
        else:
            _log.warning("No %s speakers for %s pool — excluding from combos.", vd, pool)

    # Build AI track lists per source.
    ai_tracks: dict[str, list] = {}
    for src in SOURCEID_AI_SOURCES:
        tracks = _load_ai_tracks(dataset_root, src, split)
        if tracks:
            ai_tracks[src] = tracks
        else:
            _log.warning("No AI tracks for %s/%s — excluding from combos.", split, src)

    # All valid (vocal_dataset, ai_source) combinations.
    combos = [
        (vd, src)
        for vd in VOCAL_DATASETS if vd in speakers
        for src in SOURCEID_AI_SOURCES if src in ai_tracks
    ]

    if not combos:
        _log.error("No valid combinations for split=%s — nothing to generate.", split)
        return []

    n_combos = len(combos)
    # Base count per combo, then distribute remainder to first combos.
    base, remainder = divmod(total_count, n_combos)
    counts = {combo: base + (1 if i < remainder else 0) for i, combo in enumerate(combos)}

    # Per-combo RNG for speaker + AI track assignment (deterministic).
    combo_rng_map = {
        combo: np.random.default_rng(_stable_seed(seed, f"{split}_{combo[0]}_{combo[1]}"))
        for combo in combos
    }

    work_items: list[WorkItem] = []
    for (vd, src), n in counts.items():
        rng = combo_rng_map[(vd, src)]
        spk_list  = speakers[vd]
        ai_list   = ai_tracks[src]
        combo_id  = f"{vd}_over_{src}"

        for idx in range(n):
            song_id = f"{combo_id}_{idx:05d}"

            # Always draw from the RNG before checking skip, so that
            # resumed runs assign the same (speaker, AI track) to every
            # index regardless of which indices were already generated.
            # If the draw happened only for non-skipped indices, index k
            # in a resumed run would get the assignment intended for
            # index k-j (where j examples were skipped before it).
            spk = spk_list[int(rng.integers(0, len(spk_list)))]
            ai  = ai_list[int(rng.integers(0, len(ai_list)))]

            if song_id in existing_song_ids:
                continue  # already generated — skip

            out_rel = f"processed/{split}/mixed/{combo_id}/{idx:05d}.mp3"
            out_abs = str(dataset_root / out_rel)

            work_items.append(WorkItem(
                idx=idx,
                split=split,
                combo_id=combo_id,
                song_id=song_id,
                vocal_dataset=vd,
                vocal_speaker=spk["speaker_id"],
                vocal_mp3_abs=spk["mp3_abs"],
                ai_source=src,
                ai_song_id=ai["song_id"],
                ai_mp3_abs=ai["mp3_abs"],
                output_mp3_abs=out_abs,
                output_mp3_rel=out_rel,
                seed=_stable_seed(seed, song_id),
            ))

    _log.info("Split=%s: %d combos, %d items to generate (%d already exist).",
              split, n_combos, len(work_items),
              total_count - len(work_items))
    return work_items


# ---------------------------------------------------------------------------
# MANIFEST I/O
# ---------------------------------------------------------------------------


def _load_existing_manifest(path: Path) -> list[dict]:
    """Load existing manifest records. Returns [] if file doesn't exist."""
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _write_manifest_atomic(records: list[dict], path: Path) -> None:
    """Write records to path atomically (temp file + rename).

    This prevents a partially-written manifest from being left on disk
    if the process is interrupted during the write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(records, f, indent=2)
    tmp_path.rename(path)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, plan work, execute in parallel, write manifests."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root", required=True, type=Path,
        help="Root of the music_detection dataset tree.",
    )
    parser.add_argument(
        "--train-count", type=int, default=1500,
        help="Total number of train mixed examples to generate (default 1500).",
    )
    parser.add_argument(
        "--test-count", type=int, default=300,
        help="Total number of test mixed examples to generate (default 300).",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel worker processes (default 4).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global RNG seed for reproducibility (default 42).",
    )
    args = parser.parse_args()

    dataset_root: Path = args.dataset_root.expanduser().resolve()

    for split, total in [("train", args.train_count), ("test", args.test_count)]:
        if total == 0:
            continue

        manifest_path = dataset_root / "processing_state" / split / "mixed.json"
        existing = _load_existing_manifest(manifest_path)
        existing_ids = {r["song_id"] for r in existing if r.get("status") == "ok"}

        work = _plan_work(
            dataset_root=dataset_root,
            split=split,
            total_count=total,
            seed=args.seed,
            existing_song_ids=existing_ids,
        )

        if not work:
            _log.info("Split=%s: nothing to do.", split)
            # Still write manifest if we have existing records.
            if existing:
                _write_manifest_atomic(existing, manifest_path)
            continue

        records: list[dict] = list(existing)
        n_ok = len(existing_ids)
        n_failed = 0

        _log.info("Split=%s: launching %d workers for %d items.", split, args.workers, len(work))

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker, item): item for item in work}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                rec = future.result()
                records.append(rec)

                if rec["status"] == "ok":
                    n_ok += 1
                else:
                    n_failed += 1
                    _log.warning("FAILED %s: %s", rec["song_id"], rec.get("error"))

                if done_count % _FLUSH_EVERY == 0:
                    _write_manifest_atomic(records, manifest_path)
                    _log.info(
                        "Split=%s: %d/%d done (%d ok, %d failed).",
                        split, done_count, len(work), n_ok, n_failed,
                    )

        _write_manifest_atomic(records, manifest_path)
        _log.info(
            "Split=%s DONE: %d ok, %d failed. Manifest → %s",
            split, n_ok, n_failed, manifest_path,
        )

        # Print one example record for quick inspection.
        sample = next((r for r in records if r.get("status") == "ok"), None)
        if sample:
            _log.info("Sample record:\n%s", json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()

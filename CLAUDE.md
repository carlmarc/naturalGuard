# CLAUDE.md — Audio AI Detector

This document is the persistent specification for this project. **Read this first in every session.** Decisions documented here override any inferences Claude Code might make from code alone.

CLAUDE.md is **human-maintained**. Do not update it autonomously. If you find a decision needs to be made that this spec doesn't cover, stop and ask.

---

## What this project does

Detect whether audio was created by a human or an AI generator. The goal is to detect **creative agency**, not acoustic properties:

- A human using synths or any electronic instrument → human=1
- Output from Suno, Boomy, Mubert, MusicGen, Stable Audio, ElevenLabs → ai=1 (and the specific generator label = 1)
- Mixed content (human vocal over AI instrumental) → human=1, ai=1
- Output is **continuous sigmoid scores**, never hard labels

Two detection systems share one model architecture:

**NaturalGuard** — One-class contrastive. Models what natural audio looks like. Detects AI by distance from the human cluster. Robust to unseen generators.

**SourceID** — Supervised multilabel. Learns per-generator fingerprints. Outputs scores for [human, suno, boomy, mubert, mureka, musicgen, stable_audio, elevenlabsmusic].

One backbone, two heads. One forward pass produces both outputs.

Melodia (98 files) is excluded from SourceID training but included in NaturalGuard's AI pool. LDM2 is excluded entirely (16 kHz, empty high-band).

---

## Project status

### Already built and verified

- `scripts/analyze_dataset.py` — extracts ffprobe + librosa features per file
- `scripts/compare_datasets.py` — flags confounds
- `scripts/fix_extensions.py` — corrects mislabeled extensions
- `scripts/preprocess_audio.py` — standard ffmpeg pipeline. Supports --split train|test, --cap N, --workers N, --cap-seed N
- `scripts/prepare_vocal_dataset.py` — ingests vocal datasets via per-speaker concatenation + silence stripping + standard MP3 encoding
- `src/dataset/metadata.py` — filter and stratification helpers

Dataset state on disk under `/home/cma/Samsung990_2T/datasets/music_detection/`:

- Training audio preprocessed to MP3 in `processed/train/<source>/`
- Test audio in `raw/test/<source>/`, not yet preprocessed
- Vocal sources: LibriSpeech (40 speakers, ~5h) + NUS-48E sung (12 speakers, ~1.7h), one MP3 per speaker

### Still to build (in dependency order)

1. `scripts/build_splits.py` — train+val manifests from train pool
2. `scripts/build_mixed_examples.py` — synthetic [human=1, ai=1] examples
3. `src/detector/features.py` — MultiScaleSTFT, PerFrequencyInstanceNorm, SpectrogramAugmenter
4. `src/detector/model.py` — BandCNN, WindowEncoder (T1), DistanceHead, MultilabelHead, optionally TrackTransformer (T2)
5. `src/detector/dataset.py` — PyTorch Datasets per phase
6. `src/detector/lightning.py` — DetectorLightning with phase dispatch
7. `train.py` — Hydra entry point
8. `src/detector/metrics.py` — AP, AUC, per-class metrics
9. `scripts/inference.py` — standalone (zero Lightning/Hydra imports)
10. `tests/test_pipeline.py` — integration smoke tests

Test data preprocessing deferred until evaluation.

---

## Directory layout

```
/home/cma/Samsung990_2T/datasets/music_detection/    (outside the repo)
ingredients/
    LibriSpeech/dev-clean/                           originals
    NUS-48E/                                         originals
    librispeech_devclean_processed/
        train_pool/<speaker>.mp3                     one MP3 per speaker
        test_pool/<speaker>.mp3
    nus_48e_sing_processed/
        train_pool/<speaker>.mp3
        test_pool/<speaker>.mp3
    librispeech_devclean_split.json
    nus_48e_sing_split.json
raw/
    train/<source>/<batch>/<files>
    test/<source>/<batch>/<files>
raw_metadata/
    train/<source>.json                              from analyze_dataset.py
    test/<source>.json                               not yet created
processed/
    train/
        <source>/<song_id>.mp3                       normalized audio
        mixed/<vocal_dataset>_over_<ai>/...mp3       to build
    test/<source>/<song_id>.mp3                      to build
processing_state/
    train/<source>.json
    test/<source>.json                               to build
```

Repo layout:

```
audio_ai_detector/
    CLAUDE.md                          this file
    README.md
    pyproject.toml
    configs/                           Hydra
        config.yaml
        model/{default.yaml, small.yaml}
        data/{default.yaml, tiny.yaml}
        training/{phase1.yaml, phase2.yaml, phase3.yaml}
    src/
        detector/
            __init__.py
            features.py
            model.py
            dataset.py
            lightning.py
            metrics.py
            dataset/metadata.py        already built
    scripts/
        analyze_dataset.py             already built
        compare_datasets.py            already built
        fix_extensions.py              already built
        preprocess_audio.py            already built
        prepare_vocal_dataset.py       already built
        build_splits.py                TODO
        build_mixed_examples.py        TODO
        inference.py                   TODO
    train.py                           TODO
    tests/test_pipeline.py             TODO
    data/{splits/, checkpoints/}       not committed
    logs/                              tensorboard
```

---

## Non-negotiable rules

1. **Nothing hardcoded.** All parameters via configs. Override on the command line with Hydra.

2. **Spectrogram-domain augmentation only** during training. The sole exception is MP3 re-encoding simulation in Phase 2 (waveform-level because MP3 artifacts are in the waveform).

3. **All audio is 192 kbps libmp3lame at -16 LUFS, 22050 Hz mono** before the model sees it. Training data normalized once by preprocess_audio.py. Inference normalizes on the fly through the same ffmpeg command.

4. **Audio loading uses torchaudio at sr=22050.** Never librosa during training or inference. Librosa is only for offline analysis in analyze_dataset.py.

5. **Augmentation uses torchaudio only.** Use pedalboard for waveform-level DSP if needed (e.g., reverb in mixed example construction).

6. **inference.py has zero imports from train.py, Lightning, or Hydra.** It loads a checkpoint, builds the model from saved config, runs forward pass.

7. **Splits are by song_id, never by segment.** All windows from one song land in the same split. Train/test boundary is structural (directories). Train/val split is by song_id within train.

8. **Code documents the why.** Every non-obvious decision in the code has a docstring or comment explaining the reasoning.

---

## Architecture

### Per-window pipeline (T1 — within-window)

```
waveform [B, 1, N=154350 samples at 22050 Hz]
    |
    v MultiScaleSTFT
    |   Three STFTs in parallel: n_fft = 256, 1024, 4096, same hop_length.
    |   Same hop -> same T_frames -> stack as channels.
    |   Result: [B, 3, F_max=2049, T_frames]
    |
    v PerFrequencyInstanceNorm
    |   Normalize each frequency bin across the time axis of THIS window.
    |   Removes per-track EQ profile. Strips absolute level per bin.
    |   Preserves time-varying patterns.
    |
    v SpectrogramAugmenter (training only)
    |   Time/freq masking, gain, noise, freq tilt. Pure torch ops.
    |
    v Optional 3-band split (model.use_bands)
    |   Slice frequency: Low (20-500 Hz), Mid (300-4 kHz), High (3 k-11025 Hz).
    |   Each band -> its own BandCNN.
    |
    v BandCNN per band (or one CNN if use_bands=false)
    |   Conv2d -> Conv2d -> AdaptiveAvgPool -> Conv1d(stride=4) -> [B, T_patches, D]
    |
    v Concatenate band patches -> [B, ~150 patches, D]
    |
    v WindowTransformer (T1)
    |   CLS token + sinusoidal PE + TransformerEncoder (default 4 layers).
    |   Output CLS embedding: [B, D]
    |
    v Two heads:
    +-- DistanceHead -> L2 distance to EMA centroid (NaturalGuard)
    +-- MultilabelHead -> per-class logits [B, n_classes] (SourceID)
```

### Across-window pipeline (T2 — temporal evolution, optional)

```
window embeddings e1...eN from one track
    |
    v consistency_features() — adjacent cosine-similarity stats
    |
    v TrackTransformer (T2)
    |   Small encoder over [B, N_windows, D] with src_key_padding_mask.
    |
    v TrackHead
    |   MLP over (mean_emb + consistency_features + track_emb) -> scores
```

### Two heads, one model

**DistanceHead (NaturalGuard)** — L2 distance from CLS embedding to EMA centroid of natural audio. Non-trainable buffer, saved in checkpoint.

**MultilabelHead (SourceID)** — Linear(D, 256) -> GELU -> Linear(256, n_classes). Independent sigmoid per class. Can fire any combination, including [human=1, suno=1] for mixed content.

### Key config knobs

```
model:
  use_bands: true
  use_temporal_evolution: false
  embed_dim: 256                  # 256 on RTX 3060; try 512 if VRAM allows
  t1_num_layers: 4
  t2_num_layers: 2

training:
  precision: 16-mixed             # mandatory for RTX 3060
  windows_per_track_per_epoch: 5
```

---

## Training phases

| Phase | Trigger | Dataset | Loss | What trains |
|---|---|---|---|---|
| 1 | phase=1 | ContrastiveWindowDataset (natural only) | NT-Xent | Backbone, EMA centroid |
| 2 | phase=2 | AudioWindowDataset (all labels) | BCEWithLogitsLoss + pos_weight | Backbone + MultilabelHead |
| 3 | phase=3 | AudioTrackDataset | BCE on track labels | T2 + TrackHead only (backbone frozen) |

### Phase 1 — Contrastive pretraining

All ~9,729 natural files. For each window: augment twice -> NT-Xent loss with the other augmentation as positive, others in batch as negatives. EMA-update centroid on natural embeddings.

### Phase 2 — Supervised fine-tuning

All labeled data. Stratified per source by (vocal_class, high_energy_bucket). BCEWithLogitsLoss with pos_weight. Centroid keeps updating on natural examples in batches. SupCon optional — start without.

### Phase 3 — Temporal evolution (optional)

Backbone frozen. T2 and TrackHead train on full-track examples with track-level labels.

---

## Window sampling at training time

**Important:** Cap is at the *track* level (2500 per AI source) but training visits *windows*. Track durations vary wildly (MusicGen ~30s, Suno ~193s) so equal track counts produce unequal window counts.

**Solution:** Cap tracks, sample windows at runtime.

- Track-level cap: 2500 per AI source. Natural uncapped (~9,729).
- Dataset samples N random 7-second windows per track per epoch (default windows_per_track_per_epoch=5)
- Different windows sampled per epoch -> full coverage over training
- All classes contribute equal windows per epoch regardless of track length

Very short tracks (MusicGen at 30s) have only ~7 distinct windows. Model sees overlapping crops across epochs — acceptable for POC.

### pos_weight for class imbalance

Natural ~9,729 files; each AI generator ~2,500. With windows_per_track_per_epoch=5, that's ~48,645 vs ~12,500 windows per epoch.

BCEWithLogitsLoss(pos_weight=class_weights) compensates. Per class c:

  pos_weight[c] = (total negatives for class c) / (total positives for class c)

build_splits.py computes these once, writes them to the manifest. Training reads from manifest.

---

## Data pipeline (already built — for reference)

### MP3 normalization

Every file:

  ffmpeg -i <in> -af loudnorm=I=-16:TP=-1.5:LRA=11 -ac 1 -ar 22050 -codec:a libmp3lame -b:a 192k <out>

One pipeline, every file. The most important data decision.

### Vocal source preprocessing

Done by prepare_vocal_dataset.py. Per dataset:

1. Enumerate speakers and their clips
2. Deterministic speaker-level 80/20 split (seed=42), train_pool / test_pool
3. Per speaker: concat all clips, strip silences >0.5s at <-40 dBFS
4. Encode through standard MP3 pipeline
5. Write one MP3 per speaker

Handles wildly different clip shapes (LibriSpeech: short utterances; NUS-48E: long sung songs). Downstream sampling is uniform.

### Quality filtering and stratification

In src/dataset/metadata.py.

Hard filters: bitrate < 128 kbps, duration < 10s, rms_db < -40, silence_ratio > 0.3, high_freq_low_flag = true.

Soft filters (also exclude): rms_db_std < 2, crest_factor outside [5, 25] dB.

Stratification:
- vocal_class ("voiced"/"none") — handles MusicGen 30% vs natural 74%
- duration_bucket (<30s, 30-60s, 60-180s, 180-300s, >300s)
- high_energy_bucket ("low"/"mid"/"high") — handles natural's high-band excess

---

## Held-out evaluation

Test data structurally separated on disk: raw/test/<source>/.... Never touched during training.

build_splits.py partitions train pool into train/val by song_id (~90/10, stratified per class). Test manifest from processed/test/.

LibriSpeech and NUS-48E split at SPEAKER level. Train mixed examples use train_pool speakers; test mixed examples use test_pool. No overlap.

---

## Mixed example construction (to build)

scripts/build_mixed_examples.py creates [human=1, ai_class=1] samples.

### Pipeline per mixed example

1. Pick random speaker from correct vocal pool
2. Pick random 7s window from their MP3
3. Pick random AI instrumental from correct split, random 7s window
4. Decode both to PCM (both at 22050 mono, -16 LUFS)
5. Vocal chain via pedalboard:
   - HPF 80 Hz
   - Presence shelf ±2 dB at 3.5 kHz (random)
   - De-ess -1 to -2 dB at 8 kHz (random)
   - Reverb: random preset (tight/loose/none, 20% none), wet -20 to -10 dB
6. Random vocal gain ±6 dB
7. Sum: mixed = 0.5 * vocal + 0.5 * instrumental
8. Glue compression via pedalboard:
   - peak_dbfs = 20 * log10(max(abs(mixed)))
   - target = uniform(2.0, 4.0) dB
   - threshold = peak_dbfs - target
   - Compressor(threshold, ratio=2.0, attack=10ms, release=100ms)
   - Makeup gain +target dB
9. Encode to MP3 via standard ffmpeg pipeline
10. Write to processed/{train,test}/mixed/<vocal_dataset>_over_<ai_source>/<index>.mp3

Both ingredients have already been through the project's MP3 pipeline once, so codec history is symmetric.

### Volume

- ~1,500 train mixed examples (~200 per AI source)
- ~300 test mixed examples

### Output

Writes processing_state/{train,test}/mixed.json describing each generated file: vocal source, instrumental source, mix params, labels.

---

## Splits construction (to build)

scripts/build_splits.py produces training manifests.

### Inputs

- processing_state/train/<source>.json per source
- processing_state/train/mixed.json (after mixed examples built)
- raw_metadata/train/<source>.json for stratification

### Outputs in data/splits/

- naturalguard_pretrain.json — Phase 1: natural only
- sourceid_train.json, sourceid_val.json — Phase 2 train portion split 90/10 by song_id
- sourceid_test.json — Phase 2 held-out from processed/test/ (once available)
- tracks_train.json etc. — Phase 3 (later)

### Each entry

```
{
  "song_id": "...",
  "source_class": "boomy",
  "processed_path": "processed/train/boomy/01419b...mp3",
  "duration_sec": 122.7,
  "labels": {"human": 0, "boomy": 1, ...},
  "vocal_class": "none",
  "duration_bucket": "60-180s",
  "high_energy_bucket": "mid"
}
```

Top-level pos_weight array per class, computed once.

### Stratification

Phase 2: cap each AI at 2500 tracks, stratified by (vocal_class, high_energy_bucket). Natural uncapped. Use metadata.stratification_buckets().

Train/val: 90/10 by song_id, stratified per source class.

---

## Code conventions

### Documentation is mandatory

Every module, class, non-trivial function gets a docstring explaining **why**, not just what.

For complex code, the docstring includes:
- What it does
- Why this design
- What it removes from the signal, what it preserves
- Why it fits the project's goal

### Inline comments for readability

In addition to docstrings, use inline comments to help a first-time
reader follow the logic:

- Section markers between logical blocks (e.g. `# ---- Compute splits ----`)
- One-line comments on non-obvious code
- Domain context where it helps (audio math, PyTorch tensor shapes)
- Magic numbers labeled with origin or meaning

Don't:
- Add comments that just restate the code
- Write paragraphs inline (paragraphs belong in docstrings)
- Comment every line; comment what isn't obvious from the code itself

Goal: someone reading the file for the first time can follow the flow
without running it mentally. Clarity over coverage.

### Type hints

PEP 604 (int | None, not Optional[int]). Dataclasses where Hydra isn't enough.

### Style

- Black formatting, 100-char line limit
- Functional where natural; classes for state-carrying
- Constants at module top in UPPER_SNAKE_CASE
- Internal helpers prefixed with _

---

## What Claude Code should do

1. Read this file at the start of every session
2. Implement what's specified, in the order listed under "Still to build"
3. Write docstrings explaining reasoning
4. Pass syntax check (python -m py_compile <file>)
5. Add smoke tests in tests/test_pipeline.py
6. When uncertain or facing a decision not in this spec, stop and ask

## What Claude Code should NOT do

1. Decide architecture not in this spec — ask first
2. Add features the human didn't request
3. Refactor unrequested code
4. Add production-scale plumbing — this is a POC
5. Use librosa in model code or augmentation
6. Skip docstring explanations
7. Update CLAUDE.md autonomously

---

## Workflow notes

- Human runs scripts. Claude Code writes them.
- Configs in configs/. Override via Hydra on command line.
- TensorBoard logs in logs/. Checkpoints in data/checkpoints/. Neither committed.
- Tests are integration-style. No heavy per-function unit testing.
- Mixed-precision training (16-mixed) mandatory on RTX 3060.

## Open questions — surface, don't decide silently

- **Genre coverage in natural** — synth/electronic may be under-represented. May supplement from FMA Electronic if NaturalGuard generalizes poorly on electronic test material.
- **Phase 2 with or without SupCon** — start without; add only if MultilabelHead training is unstable.

---

## Phase plan

1. build_splits.py — manifests for Phase 1 and partial Phase 2
2. **First Phase 1 run** (natural-only contrastive) — sanity check backbone trains
3. build_mixed_examples.py — synthetic mixed examples
4. Re-run build_splits.py to include mixed examples
5. **Phase 2 run** — full supervised fine-tuning
6. inference.py + metrics.py — evaluate
7. (Later) T2 + Phase 3 — temporal
8. (Later) Test data preprocessing
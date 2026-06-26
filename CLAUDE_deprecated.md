# CLAUDE.md — Audio AI Detector

This document is the persistent specification for this project. **Read this first in every session.** Decisions documented here override any inferences Claude Code might make from code alone, because much of the reasoning lives in this spec rather than in the code.

## What this project does

Detect whether audio was created by a human or an AI generator. Specifically detect **creative agency**, not acoustic properties:

- A human using synths, drum machines, or any electronic instrument → `human=1`
- Output from Suno, Boomy, Mubert, MusicGen, Stable Audio, ElevenLabs, etc. → `ai=1` (and the specific generator label = 1 if known)
- Mixed content (real human vocal over AI instrumental) → `human=1, ai=1`
- Output is **continuous sigmoid scores**, never hard labels

Two detection systems share one model architecture:

**NaturalGuard** — One-class contrastive. Models what natural/human audio looks like in embedding space. Detects AI by distance from the human cluster. Robust to AI generators the model has never seen.

**SourceID** — Supervised multilabel. Learns per-generator fingerprints. Outputs scores for `[human, suno, boomy, mubert, mureka, musicgen, stable_audio, elevenlabsmusic]`. Provides attribution: "this looks like Suno specifically."

One backbone, two heads. At inference, one forward pass produces both outputs.

## What's already built and verified

The dataset pipeline scripts (`scripts/`) are complete. These have run on real data and produced verified outputs:

- `analyze_dataset.py` — extracts ffprobe + librosa features per audio file (run once, results in `raw_metadata/`)
- `compare_datasets.py` — flags potential confounds between sources
- `fix_extensions.py` — corrects mislabeled audio file extensions
- `preprocess_audio.py` — runs ffmpeg on every file to produce uniform 192 kbps libmp3lame at 22050 Hz mono with LUFS normalization to -16

Plus `src/dataset/metadata.py` — pure helper functions for filtering and stratification. Used by every script that needs to decide which files are eligible for training.

What's not yet built: the model code, the synthetic mixed-examples script, the splits builder, training, evaluation, and inference. Everything below describes what those should look like.

## Project layout

```
audio_ai_detector/
├── CLAUDE.md                          # this file — read first
├── README.md                          # human-facing summary
├── pyproject.toml                     # editable install + deps
├── configs/                           # Hydra configs
│   ├── config.yaml                    # default config
│   ├── model/                         # model variants
│   ├── data/                          # data variants
│   └── training/                      # phase configs
├── src/
│   └── detector/
│       ├── __init__.py
│       ├── features.py                # MultiScaleSTFT, PerFreqInstanceNorm, augmenter
│       ├── model.py                   # CNN, T1, T2, DistanceHead, MultilabelHead
│       ├── dataset.py                 # Lightning datasets, collate functions
│       ├── lightning.py               # LightningModule, loss functions
│       ├── metrics.py                 # AP, AUC, per-class metrics
│       └── dataset/
│           └── metadata.py            # (already built) filter/stratification helpers
├── scripts/                           # one-time / orchestration scripts
│   ├── analyze_dataset.py             # (already built)
│   ├── compare_datasets.py            # (already built)
│   ├── fix_extensions.py              # (already built)
│   ├── preprocess_audio.py            # (already built)
│   ├── build_mixed_examples.py        # TODO
│   ├── build_splits.py                # TODO
│   └── inference.py                   # TODO — standalone, zero Lightning imports
├── train.py                           # Hydra entry point — dispatches by phase
├── tests/                             # integration tests (no heavy unit testing)
│   └── test_pipeline.py
├── data/                              # not committed
│   ├── splits/
│   └── checkpoints/
└── logs/                              # tensorboard logs, not committed
```

Outside the repo:

```
/mnt/samsung990_2T/datasets/music_detection/
├── raw_metadata/                      # input — per-source analysis JSONs
├── processed/                         # output of preprocess_audio.py
├── processing_state/                  # per-source processing state
└── (audio file tree)                  # original audio
```

## Non-negotiable rules

These prevent specific failure modes seen in prior work. **Do not violate them without explicit human confirmation.**

1. **Nothing hardcoded.** All parameters via `configs/`. Override on the command line with Hydra. If a number appears in code that should be configurable, move it to config.

2. **Spectrogram-domain augmentation only.** No waveform augmentation during training. The sole exception is MP3 re-encoding simulation in Phase 2 fine-tuning, which has to happen at waveform level because MP3 artifacts are in the waveform.

3. **All audio is 192 kbps libmp3lame at -16 LUFS, 22050 Hz mono** before the model sees it. Training data is normalized once by `preprocess_audio.py`. Inference normalizes on the fly through the same ffmpeg command. This eliminates codec signature as a class shortcut.

4. **Audio loading uses torchaudio at sr=22050.** Never librosa during training or inference. Librosa is only for offline analysis (in `analyze_dataset.py`, which we don't rerun routinely).

5. **Augmentation uses torchaudio only.** Mixing librosa feature ops into the augmentation pipeline causes train/inference distribution mismatch. Use pedalboard for waveform-level DSP if needed (e.g., reverb in mixed example construction), torchaudio for everything in the model path.

6. **inference.py has zero imports from train.py, Lightning, or Hydra.** Inference must be runnable in a minimal environment. It loads a checkpoint, builds the model from saved config, runs forward pass. No training framework dependencies.

7. **Splits are by song_id, never by segment.** All windows from one song land in the same split (train/val/test). Splitting by segment causes train/test leakage.

8. **Code documents the why.** Every non-obvious decision in the code has a docstring or comment explaining the reasoning. The point is for a human (or Claude Code in a new session) to read a file and understand both what it does and why it does it that way. See the "code conventions" section.

## Architecture

### Per-window pipeline (T1 — within-window)

This is the core perception model. Operates on one 7-second window at a time.

```
waveform [B, 1, N=154350 samples at 22050 Hz]
  │
  ▼ MultiScaleSTFT
  │  Three STFTs in parallel with n_fft = 256, 1024, 4096, same hop_length.
  │  Same hop gives same T_frames across scales → can stack as channels.
  │  Result: [B, 3, F_max=2049, T_frames]
  │
  ▼ PerFrequencyInstanceNorm
  │  Normalize each frequency bin across the time axis of THIS window.
  │  Removes per-track EQ profile as a class shortcut.
  │  Preserves time-varying patterns; strips absolute level per bin.
  │
  ▼ SpectrogramAugmenter (training only)
  │  Time/freq masking, gain, noise, freq tilt.
  │  Pure torch ops, no librosa.
  │
  ▼ Optional 3-band split (configurable)
  │  Slice frequency axis into Low (20–500 Hz), Mid (300–4 kHz), High (3 k–11025 Hz).
  │  Each band processed by its own BandCNN.
  │  Cross-band relationships modeled later by T1's self-attention.
  │
  ▼ BandCNN per band (or one CNN over full spectrum if use_bands=false)
  │  Conv2d → Conv2d → AdaptiveAvgPool → Conv1d(stride=4) → [B, T_patches, D]
  │
  ▼ Concatenate band patch sequences → [B, ~150 patches, D]
  │
  ▼ WindowTransformer (T1)
  │  Prepend CLS token, add sinusoidal positional encoding.
  │  TransformerEncoder layers (configurable depth).
  │  Output the CLS token's final embedding: [B, D]
  │
  ▼ Two heads, both reading the same CLS embedding:
  │
  ├── DistanceHead → scalar distance to EMA centroid (NaturalGuard signal)
  └── MultilabelHead → per-class logits [B, n_classes] (SourceID signal)
```

### Across-window pipeline (T2 — temporal evolution, optional)

Only used in track-mode inference. Phase 3 of training.

```
For one track:
  Run per-window pipeline on each 7s window with 50% overlap → embeddings e1...eN
  │
  ▼ consistency_features() — adjacent cosine-similarity stats [mean, var, max, min]
  │
  ▼ TrackTransformer (T2)
  │  Small TransformerEncoder over the sequence of [B, N_windows, D].
  │  Variable-length tracks via src_key_padding_mask.
  │  Output a track-level embedding.
  │
  ▼ TrackHead
  │  MLP over (mean_emb + consistency_features + track_emb) → [human, ai, confidence]
```

### Two heads, one model

**DistanceHead (NaturalGuard)** — Computes L2 distance from CLS embedding to a stored centroid of natural audio (EMA-updated during Phase 1 and 2 from natural examples). Distance → anomaly score. Robust to unseen AI generators.

**MultilabelHead (SourceID)** — `Linear(D, 256) → GELU → Linear(256, n_classes)`. Independent sigmoid per class. The model can fire any combination of classes simultaneously, including `[human=1, suno=1]` for mixed content.

### Two transformers — different scales

**T1 (within-window)** — sees ~150 patches from one 7-second window's spectrogram. Models spatial relationships (frequency × time within one snapshot).

**T2 (across-windows)** — sees ~12-50 window embeddings across a track. Models temporal evolution. Lightweight (2 layers) because the sequence is short.

T1 is required. T2 is optional and additive.

## Training phases

| Phase | Trigger | Dataset | Loss | What trains |
|---|---|---|---|---|
| 1 | `phase=1` (default) | `ContrastiveWindowDataset` (natural only) | NT-Xent | Backbone (CNN + T1), EMA centroid update |
| 2 | `phase=2` | `AudioWindowDataset` (all labels) | BCEWithLogitsLoss + (optional SupCon) | Backbone + MultilabelHead |
| 3 | `phase=3` | `AudioTrackDataset` (full tracks) | BCE on track labels | T2 + TrackHead only (backbone frozen) |

### Phase 1 — Contrastive pretraining

**Why:** Learn what natural human audio looks like, without needing labels for AI classes.

**Data:** All ~9,729 natural files (full processed set from `processed/natural/`).

**Method:** For each window in a batch of N source windows:
1. Apply random augmentation twice → two augmented "views" of the same window
2. Compute embeddings for all 2N views
3. NT-Xent loss: each view's positive pair is the other view of the same source; all other views in the batch are negatives
4. Optimizer pulls positives together, pushes negatives apart

**Centroid update:** After each batch, EMA-update DistanceHead's centroid using natural embeddings. The centroid converges on the average natural embedding location.

**End of Phase 1:** Backbone has learned to map natural audio to a coherent cluster. The centroid represents "average natural."

### Phase 2 — Supervised fine-tuning

**Why:** Learn to distinguish AI generators specifically (SourceID), and reinforce the natural cluster boundary (NaturalGuard).

**Data:** All labeled data — natural, AI generators, mixed examples. Stratified to balance vocal/no-vocal and high-band energy.

**Method:** Standard supervised training:
- Forward pass produces CLS embedding
- MultilabelHead → per-class logits → BCEWithLogitsLoss
- pos_weight in BCE to compensate for class imbalance (natural is the majority)
- Optional SupCon on backbone (off by default — BCE alone is sufficient)
- Centroid keeps EMA-updating on natural examples that appear in batches

**End of Phase 2:** MultilabelHead outputs reliable per-generator scores. Centroid stays calibrated.

### Phase 3 — Temporal evolution (optional)

**Why:** Track-level analysis. Detect AI tracks with "too uniform" embeddings across the song (e.g., mubert's `seg_high_std=0.02` vs natural's 0.05).

**Data:** `AudioTrackDataset` — yields all windows of a track at once, with track-level labels.

**Method:** Backbone is frozen. Run backbone on every window to get embeddings e1...eN. Compute consistency features. Feed to T2 + TrackHead. BCE on track-level labels.

**End of Phase 3:** Track-level scores available at inference.

## Data pipeline (already built)

Documented for reference. Don't rebuild.

### MP3 normalization

Every file goes through the same ffmpeg command:

```
ffmpeg -i <input> \
  -af loudnorm=I=-16:TP=-1.5:LRA=11 \
  -ac 1 -ar 22050 \
  -codec:a libmp3lame -b:a 192k \
  <output>
```

One pipeline, every file, no per-file branching. This is the most important data decision — if natural and AI files go through different encoders, the model learns the encoder difference instead of the content difference.

### Quality filtering

Hard filters (file excluded entirely): bitrate < 128 kbps, duration < 10s, rms_db < -40, silence_ratio > 0.3, high_freq_low_flag = true.

Soft filters (flagged but also excluded): rms_db_std < 2 (uniform), crest_factor outside [5, 25] dB.

All logic in `src/dataset/metadata.py`. Thresholds at the top of that file.

### Stratified sampling

When building the SourceID training set, AI generators are capped at 2500 each. Sampling is stratified by `(vocal_class, high_energy_bucket)` to ensure the model can't learn obvious shortcuts:

- vocal_class — handles MusicGen's 30% vocal rate vs natural's 74%
- high_energy_bucket — handles natural's high-band energy excess (0.328) vs AI generators (0.23-0.30)

Natural is not capped — uses all ~9,729 files (or whatever passes quality filters).

### Mixed examples (TODO — build_mixed_examples.py)

Construct synthetic `[human=1, ai_class=1]` training examples. ~1500 total. For each example:

1. Pick a clean vocal segment (LibriSpeech for speech, OpenSinger or similar for singing)
2. Pick an AI instrumental (from `processed/<ai_source>/`)
3. Match both to -18 LUFS
4. Apply vocal chain: HPF 80Hz, presence shelf ±2 dB at 3.5kHz, de-ess shelf -1 to -2 dB at 8 kHz
5. Apply reverb (random tight/loose/none) with wet -20 to -10 dB
6. Apply random vocal gain ±6 dB relative to instrumental
7. Sum
8. Bus glue compression (2:1 ratio, threshold -18 dB, attack 10ms, release 100ms)
9. Final loudnorm to -16 LUFS
10. Encode through the same MP3 pipeline as everything else

Output goes to `processed/mixed_<ai_source>/`. Record fields in processing_state include vocal source, instrumental source, mix params for traceability.

Tooling: ffmpeg for loudness/codec bookending, pedalboard for the DSP in the middle.

### Splits (TODO — build_splits.py)

Reads raw_metadata and processing_state. Joins on song_id. Filters to `included` and `status=ok`. For each phase:

- Phase 1 manifest (`splits/naturalguard_pretrain.json`) — natural only
- Phase 2 manifest (`splits/sourceid_train.json`, `_val.json`, `_test.json`) — all classes, stratified, capped
- Phase 3 manifest (`splits/tracks_train.json`, ...) — full tracks with track-level labels

Split by song_id, stratified per class, 80/10/10 train/val/test.

## Configs (Hydra)

Layout:

```
configs/
├── config.yaml                # composes everything
├── model/
│   ├── default.yaml           # use_bands=true, embed_dim=256, t1_layers=4, t2_layers=2
│   └── small.yaml             # smaller for quick experiments
├── data/
│   ├── default.yaml           # paths, window length, hop, batch sizes
│   └── tiny.yaml              # subset for fast iteration
├── training/
│   ├── phase1.yaml            # contrastive pretraining
│   ├── phase2.yaml            # supervised fine-tuning
│   └── phase3.yaml            # temporal (T2)
└── inference.yaml             # inference-only config (used by inference.py)
```

Override on command line:
```bash
python train.py phase=2 model.embed_dim=512 data.batch_size=32
```

Never edit configs for an experiment — override on the command line. Configs are defaults.

## Code conventions

### Documentation is mandatory

Every module, class, and non-trivial function gets a docstring. Docstrings explain **why**, not just what. Examples in the existing scripts (`preprocess_audio.py`, `metadata.py`) show the style.

When implementing something complex (e.g., `PerFrequencyInstanceNorm`, `MultiScaleSTFT`), include in the docstring:
- What the code does
- Why this design (not just the implementation)
- What it removes from the signal and what it preserves
- Why this fits the project's goal

If a future reader can't understand why a piece of code exists by reading its docstring, the docstring is incomplete.

### Type hints

Use type hints throughout. PEP 604 union syntax (`int | None`, not `Optional[int]`). Dataclasses for structured config when Hydra isn't enough.

### Style

- Black formatting, 100-char line limit
- Functional where natural, classes only when state needs to be carried
- Constants at module top, named in `UPPER_SNAKE_CASE`
- Helper functions start with `_` if they're internal to a module

### Comments

Brief inline comments for non-obvious logic. Not "what" (the code shows that), but "why" (a comment).

```python
# Why softplus instead of relu: keeps gradient flowing for small values,
# important since our distances are mostly tiny in normalized embedding space
```

## Code that I (Claude Code) write should:

1. Be readable to the human reviewing it
2. Have a docstring explaining purpose and reasoning
3. Pass syntax check (`python -m py_compile <file>`)
4. Have at least a smoke test that exercises the basic happy path
5. Not silently drift from this spec — if a decision needs to be made that this spec doesn't cover, ask first

## What I (Claude Code) should NOT do:

1. Decide architecture details that aren't in this spec (ask the human first)
2. Add features the human didn't request
3. Refactor unrequested code
4. Add complexity that doesn't serve the POC (no production-scale plumbing)
5. Use librosa in model code or augmentation code
6. Skip the docstring explanations

## Phase plan from here

We're at the end of dataset prep. Order of next work:

1. **build_mixed_examples.py** — construct synthetic mixed examples
2. **build_splits.py** — generate phase-specific manifests
3. **src/detector/features.py** — MultiScaleSTFT, PerFrequencyInstanceNorm, SpectrogramAugmenter
4. **src/detector/model.py** — BandCNN, WindowEncoder, DistanceHead, MultilabelHead (no T2 yet)
5. **src/detector/dataset.py** — Lightning Datasets, ContrastiveWindowDataset, AudioWindowDataset
6. **src/detector/lightning.py** — LightningModule with phase dispatch, NT-Xent and BCE losses
7. **train.py** — Hydra entry point
8. **First Phase 1 run** — sanity check
9. **First Phase 2 run** — sanity check
10. (Later) T2, Phase 3, track-level model
11. **inference.py** — standalone inference script
12. **metrics.py** — evaluation metrics

Each item is a session. Each session starts by reading this file and the relevant existing code.

## Open questions to resolve as we go

- **Vocal dataset for mixed examples** — currently undecided whether to use LibriSpeech only, mix with a singing dataset, or something else. Decide before `build_mixed_examples.py`.
- **Genre coverage in natural** — synth/electronic music may be under-represented. Audit when needed.
- **Phase 2 with or without SupCon** — start without, add only if MultilabelHead training is unstable.

These are not unknowns from technical limitations; they're choices we have not yet made because we don't need to make them yet. Don't make them silently — surface to the human.
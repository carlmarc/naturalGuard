"""
src/detector/model.py — CNN + Transformer backbone with NaturalGuard and SourceID heads.

PIPELINE POSITION
-----------------
This module implements the learnable part of the per-window pipeline, taking the
front-end output from features.py and producing detection scores:

    spectrogram [B, 3, F_MAX, T]      ← post-PerFrequencyInstanceNorm / SpectrogramAugmenter
        → WindowEncoder (T1)          → cls_embedding [B, D]
        → DistanceHead                → distance [B]         (NaturalGuard score)
        → MultilabelHead              → logits [B, n_classes] (SourceID scores)

This module does NOT run the features.py pipeline. The Detector class expects a
normalized log-power spectrogram as input. Composition with features.py happens
in DetectorLightning so the STFT can optionally be offloaded to DataLoader workers.

DESIGN OVERVIEW
---------------
WindowEncoder uses a CNN front-end (BandCNN) to convert the 2D spectrogram into
a sequence of patch embeddings, then a TransformerEncoder to aggregate them into
a single CLS embedding. The CNN handles frequency×time structure; the Transformer
handles temporal dependencies. DistanceHead and MultilabelHead share the backbone
and can both be evaluated in a single forward pass.

WHY ONE BACKBONE, TWO HEADS
----------------------------
Phase 1 trains the backbone contrastively on natural audio only — the CLS
embedding learns "what natural audio looks like." Phase 2 adds the MultilabelHead
for generator identification. Sharing the backbone means both detection tasks
reinforce the same representation: the EMA centroid (NaturalGuard) and the
generator fingerprints (SourceID) both improve as the backbone matures. A single
forward pass at inference serves both tasks with no duplication of computation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Default frequency bands for use_bands=True.
# Bands overlap by design (Low/Mid share 300–500 Hz; Mid/High share 3–4 kHz).
# Overlap gives the CNN context at band boundaries; the Transformer integrates
# cross-band relationships from the concatenated patch sequence.
# Edges are in Hz; bin indices are computed at init time from sample_rate and n_fft.
_BAND_EDGES_HZ: list[tuple[float, float]] = [
    (20.0,    500.0),    # Low  — fundamentals, sub-bass, drum transients
    (300.0,  4000.0),    # Mid  — vocals, harmonics, formants, timbral content
    (3000.0, 11025.0),   # High — air, reverb tails, HF artifacts in AI generators
]

# Default number of detection classes (human + 7 AI sources).
# Must match the order of SOURCEID_CLASSES in build_splits.py.
N_CLASSES_DEFAULT: int = 8


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _hz_to_bin(hz: float, sr: int = 22050, n_fft: int = 4096) -> int:
    """Convert a frequency in Hz to the nearest STFT bin index.

    Uses bin = round(hz × n_fft / sr). The n_fft must be the largest n_fft
    used in MultiScaleSTFT (4096) because that sets the input spectrogram's
    frequency axis: F_MAX = 4096 // 2 + 1 = 2049 bins. Using a different n_fft
    here would compute wrong band boundaries against the actual frequency axis.
    """
    return round(hz * n_fft / sr)


def _sinusoidal_pe(
    seq_len: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute fixed sinusoidal positional encoding.

    Standard formulation from Vaswani et al. (2017):
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Computed on the fly (not stored as a module buffer) so it handles variable
    sequence lengths without requiring a fixed maximum-length buffer. Audio
    windows near the end of a track may be shorter than the nominal 7 seconds,
    producing a different T_frames and a different patch count.

    The dtype argument matches the PE to the input tensor dtype so there is no
    implicit promotion when adding PE to patch embeddings under autocast.

    Returns: [1, seq_len, d_model]
    """
    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)  # [L, 1]
    # Compute div_term via exponentiation in log-space for numerical stability.
    # arange(0, d_model, 2) gives ceil(d_model/2) values (handles odd d_model).
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / d_model)
    )  # [ceil(d_model/2)]

    pe = torch.zeros(1, seq_len, d_model, device=device, dtype=dtype)
    pe[0, :, 0::2] = torch.sin(position * div_term)           # even columns
    pe[0, :, 1::2] = torch.cos(position * div_term[:d_model // 2])  # odd columns
    # For even d_model (default 256): div_term has exactly d_model//2 elements
    # so the slice is a no-op. For odd d_model: div_term has one extra element
    # that we drop here (the last even column has no paired odd column).
    return pe


# ---------------------------------------------------------------------------
# BandCNN
# ---------------------------------------------------------------------------


class BandCNN(nn.Module):
    """CNN that maps one spectrogram band to a sequence of patch embeddings.

    Processes one frequency slice of the normalized log-power spectrogram and
    outputs a 1D patch sequence [B, T_patches, embed_dim], ready to be
    concatenated with the other bands' patch sequences and fed into the
    WindowEncoder's TransformerEncoder.

    ARCHITECTURE RATIONALE
    ----------------------
    Two Conv2d layers (3×3, padding=1) capture local frequency–time patterns
    within the band — e.g. a harmonic partial drifting across a few bins over
    a few frames, or a characteristic texture in the high-frequency region.
    Padding=1 preserves the spatial size so no frequency or time content is
    cut from the edges.

    AdaptiveAvgPool2d((1, None)) collapses the frequency axis to 1 while
    keeping the time axis unchanged. This makes BandCNN independent of the
    band's height: Low (≈89 bins), Mid (≈687 bins), and High (≈1492 bins)
    all pass through the same class without resizing.

    Conv1d(stride=patch_stride) then downsamples time by patch_stride and
    projects to embed_dim in one step, producing non-overlapping "patch"
    embeddings analogous to vision transformer patch projections. The kernel
    size equals the stride (non-overlapping) so each time step belongs to
    exactly one patch — no boundary ambiguity.

    WHY ADAPTIVE AVG POOL (NOT FLATTEN OR GLOBAL MAX)
    --------------------------------------------------
    The frequency bins have already been independently normalized by
    PerFrequencyInstanceNorm, so their mean is zero and variance is one.
    Average pooling over frequency is therefore unbiased: it produces the
    mean activation across equally-normalized bins, which is a reasonable
    summary of the band's response at each time step. Global max would
    amplify noise; flatten would require a fixed band height (defeating
    the purpose of the overlapping band design).

    Input:  [B, in_channels, F_band, T]
    Output: [B, T_patches, embed_dim]  where T_patches = (T - patch_stride) // patch_stride + 1

    Args:
        in_channels:  Number of input channels (default 3, from MultiScaleSTFT).
        cnn_channels: Intermediate channel count for both Conv2d layers.
        embed_dim:    Output channel count (patch embedding dimension D).
        patch_stride: Stride and kernel size for the Conv1d patch projection.
    """

    def __init__(
        self,
        in_channels: int = 3,
        cnn_channels: int = 64,
        embed_dim: int = 256,
        patch_stride: int = 4,
    ) -> None:
        super().__init__()

        # ---- 2D CNN: extract local frequency-time features ----
        # Both layers preserve spatial dimensions (padding=1, no stride).
        self.conv2d_1 = nn.Conv2d(in_channels, cnn_channels, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1)

        # Collapse frequency to 1; keep time unchanged.
        # output_size=(1, None) means: height → 1, width (T) → as-is.
        self.pool_freq = nn.AdaptiveAvgPool2d((1, None))

        # Non-overlapping 1D patch projection: kernel_size == stride.
        # Produces embed_dim-dimensional patch vectors at 1/patch_stride
        # the original temporal resolution.
        self.patch_proj = nn.Conv1d(
            cnn_channels, embed_dim, kernel_size=patch_stride, stride=patch_stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, F_band, T] spectrogram band slice.
        Returns:
            [B, T_patches, embed_dim] patch embedding sequence.
        """
        # ---- Local feature extraction over frequency × time ----
        x = torch.relu(self.conv2d_1(x))   # [B, cnn_channels, F_band, T]
        x = torch.relu(self.conv2d_2(x))   # [B, cnn_channels, F_band, T]

        # ---- Collapse frequency axis ----
        x = self.pool_freq(x)              # [B, cnn_channels, 1, T]
        x = x.squeeze(2)                   # [B, cnn_channels, T]

        # ---- Project to patch embeddings ----
        x = self.patch_proj(x)             # [B, embed_dim, T_patches]
        return x.permute(0, 2, 1)          # [B, T_patches, embed_dim]


# ---------------------------------------------------------------------------
# WindowEncoder (T1)
# ---------------------------------------------------------------------------


class WindowEncoder(nn.Module):
    """Encode a spectrogram window into a single CLS embedding via CNN + Transformer.

    Implements the T1 (within-window) pipeline:
      1. Optionally split the spectrogram into overlapping frequency bands.
      2. Run BandCNN on each band to produce patch sequences [B, T_k, D].
      3. Concatenate all patch sequences: [B, N_patches_total, D].
      4. Add sinusoidal positional encoding to the patch sequence.
      5. Prepend a learned CLS token: [B, 1 + N_patches_total, D].
      6. Run TransformerEncoder.
      7. Return the CLS token's final embedding: [B, D].

    WHY BAND SPLIT + CONCATENATE
    ----------------------------
    The three frequency bands capture qualitatively different content:
      Low  — rhythm, bass, fundamental structure
      Mid  — vocals, harmonic stack, timbral fingerprints of AI generators
      High — air, reverb tails, HF artifacts that AI synthesizers often suppress
             or artificially reproduce
    Running separate BandCNNs lets each band develop its own filter bank, tuned
    to its content's frequency scale. Concatenating before the Transformer gives
    the attention mechanism cross-band context: a 3–4 kHz signal that looks
    natural in Mid but artifactual in High will be captured by cross-band attention
    patterns, something a single CNN pooling all frequencies cannot do.

    WHY CLS TOKEN AGGREGATION
    -------------------------
    Global average-pooling of transformer output treats all patches equally,
    diluting weak but diagnostic AI artifacts. The CLS token learns to attend
    selectively to whichever patches carry the most discriminative evidence —
    analogous to how a human listener zones in on specific moments that sound "off."
    This matters because AI artifacts are often sparse: a distinctive pattern in
    a few hundred milliseconds of high-frequency content.

    WHY SINUSOIDAL PE (NOT LEARNED)
    --------------------------------
    Learned positional encodings require a fixed maximum sequence length. Window
    duration is nominally 7 seconds but can be shorter for the last window of a
    track. Sinusoidal PE is defined for any sequence length and generalizes
    to lengths outside the training distribution at no cost.

    WHY CLS DOES NOT RECEIVE PE
    ----------------------------
    The CLS token's role is to aggregate, not to represent a position in the
    audio. Adding a PE to CLS would bias the attention patterns based on
    whatever position index is assigned to it, which is arbitrary. CLS attends
    to all positionally-encoded patches and self-aggregates through the
    transformer layers without needing its own position label.

    Args:
        in_channels:     Spectrogram channel count (default 3 for MultiScaleSTFT).
        embed_dim:       Transformer model dimension D and patch embedding dimension.
        t1_num_layers:   Number of TransformerEncoder layers.
        use_bands:       If True, split into 3 overlapping frequency bands.
        num_heads:       Attention heads per TransformerEncoder layer.
        feedforward_dim: FFN hidden dimension inside each TransformerEncoder layer.
        dropout:         Dropout rate in TransformerEncoder.
        cnn_channels:    BandCNN intermediate channel count (Conv2d hidden size).
        sample_rate:     Audio sample rate; used to compute band bin indices.
        n_fft_max:       Largest n_fft in MultiScaleSTFT; sets frequency bin spacing.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        t1_num_layers: int = 4,
        use_bands: bool = True,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        cnn_channels: int = 64,
        sample_rate: int = 22050,
        n_fft_max: int = 4096,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.use_bands = use_bands

        # ---- CNN front-end ----
        if use_bands:
            # One independent BandCNN per frequency band.
            # Bands have very different heights (Low ≈89, Mid ≈687, High ≈1492 bins),
            # but BandCNN's AdaptiveAvgPool2d handles any height transparently.
            self.band_cnns = nn.ModuleList([
                BandCNN(in_channels, cnn_channels, embed_dim) for _ in _BAND_EDGES_HZ
            ])
            # Compute band bin indices at construction time (not forward time).
            # Stored as plain ints — they don't need to move with .to(device).
            self.band_bins: list[tuple[int, int]] = [
                (
                    _hz_to_bin(lo, sample_rate, n_fft_max),
                    _hz_to_bin(hi, sample_rate, n_fft_max),
                )
                for lo, hi in _BAND_EDGES_HZ
            ]
            # Clamp the high end of the last (High) band to F_MAX.
            # 11025 Hz = sr/2 → bin 2048; F_MAX = n_fft_max//2 + 1 = 2049.
            # Using F_MAX as the exclusive upper index includes the Nyquist bin (2048),
            # which can carry AI-related HF artifacts that the High band targets.
            f_max = n_fft_max // 2 + 1
            lo_last, _ = self.band_bins[-1]
            self.band_bins[-1] = (lo_last, f_max)
        else:
            # Full-spectrum mode: one BandCNN over all 2049 frequency bins.
            self.band_cnns = nn.ModuleList([
                BandCNN(in_channels, cnn_channels, embed_dim)
            ])
            self.band_bins = None  # type: ignore[assignment]  # not used when use_bands=False

        # ---- CLS token ----
        # Learned parameter initialized to zero; updated via backprop.
        # Shared across all windows (same token prepended to every sequence).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ---- Transformer ----
        # batch_first=True keeps tensor layout [B, T, D] throughout.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=t1_num_layers,
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: [B, C, F_MAX, T] normalized log-power spectrogram.
        Returns:
            [B, embed_dim] CLS token embedding.
        """
        B = spec.shape[0]

        # ---- CNN front-end: spectrogram → patch sequence ----
        if self.use_bands:
            band_patches = []
            for cnn, (lo, hi) in zip(self.band_cnns, self.band_bins):
                band_slice = spec[:, :, lo:hi, :]     # [B, C, F_band, T]
                band_patches.append(cnn(band_slice))  # [B, T_patches, D]
            # Concatenate along the patch dimension.
            # Total patches ≈ 3 × (T // 4); for T=603: ≈ 450.
            patches = torch.cat(band_patches, dim=1)   # [B, N_total, D]
        else:
            patches = self.band_cnns[0](spec)           # [B, T_patches, D]

        N_patches = patches.shape[1]

        # ---- Sinusoidal positional encoding ----
        # Added to patches before CLS prepending so CLS index (0) is unencoded.
        # Dtype matched to patches for autocast compatibility.
        pe = _sinusoidal_pe(N_patches, self.embed_dim, patches.device, patches.dtype)
        patches = patches + pe                          # [B, N_patches, D]

        # ---- Prepend CLS token ----
        # Cast CLS to patches' dtype for autocast compatibility (Parameter is float32).
        cls = self.cls_token.to(dtype=patches.dtype).expand(B, 1, self.embed_dim)
        seq = torch.cat([cls, patches], dim=1)          # [B, 1 + N_patches, D]

        # ---- Transformer: aggregate patches into CLS ----
        out = self.transformer(seq)                     # [B, 1 + N_patches, D]

        # Return only the CLS token; patch outputs carry positional detail
        # that isn't useful for the downstream detection heads.
        return out[:, 0, :]                             # [B, D]


# ---------------------------------------------------------------------------
# DistanceHead (NaturalGuard)
# ---------------------------------------------------------------------------


class DistanceHead(nn.Module):
    """NaturalGuard detection head: distance from CLS embedding to natural centroid.

    Stores a non-trainable EMA centroid of natural audio embeddings. At inference,
    returns the L2 distance from the input embedding to the centroid as a
    "how far from natural audio" score.

    WHY DISTANCE (NOT CLASSIFICATION)
    ----------------------------------
    NaturalGuard is a one-class detector: we model the natural cluster and flag
    anything far from it as AI. A classification head would require AI examples
    during training and can only detect known generators. The distance head only
    requires natural examples (Phase 1 contrastive training), so it generalizes
    to unseen generators — any AI output far enough from the natural cluster is
    detected, regardless of whether that specific generator appeared in training.

    WHY EMA (NOT BATCH STATS OR OFFLINE MEAN)
    ------------------------------------------
    Computing the true mean of all natural embeddings every epoch requires
    storing all embeddings (expensive) or a full extra pass (slow). EMA
    approximates the mean incrementally with O(D) memory. The centroid converges
    to the population mean as training progresses. Momentum=0.99 means the
    centroid incorporates roughly 100 batches of history at any point — long
    enough for stability, short enough to track distribution shifts during
    early training when the backbone is changing rapidly.

    WHY A BUFFER, NOT A PARAMETER
    ------------------------------
    The centroid is not updated by gradient descent — it is updated by explicit
    EMA calls in the training loop. Using register_buffer means:
      (a) .to(device) moves it to the right device automatically;
      (b) state_dict() saves and loads it so inference uses the trained centroid;
      (c) parameters() excludes it so optimizers ignore it.
    If it were a Parameter, the optimizer would zero-grad and update it each
    step, overwriting the EMA and destroying the learned natural cluster.

    Args:
        embed_dim: CLS embedding dimension D (must match WindowEncoder).
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        # Zero-initialized; EMA fills it during Phase 1. Stored as float32
        # so the distance computation is numerically stable regardless of
        # whether the backbone runs in float16 (mixed-precision training).
        self.register_buffer("centroid", torch.zeros(embed_dim))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute L2 distance from each embedding to the centroid.

        Args:
            embeddings: [B, D] CLS embeddings.
        Returns:
            [B] L2 distances. Higher = further from natural audio cluster.
        """
        # Cast to float32 for numerical stability: L2 norm of float16 vectors
        # overflows for large values, and the centroid is stored as float32 anyway.
        emb = embeddings.to(self.centroid.dtype)
        return torch.linalg.norm(emb - self.centroid, dim=-1)

    @torch.no_grad()
    def update_centroid_ema(
        self,
        embeddings: torch.Tensor,
        momentum: float = 0.99,
    ) -> None:
        """Update the EMA centroid with the batch mean of natural embeddings.

        Should be called in the training loop only on natural-audio batches
        (Phase 1) or on the natural-audio subset of Phase 2 batches.

        Args:
            embeddings: [B, D] CLS embeddings from natural audio.
            momentum:   EMA momentum. Higher = slower adaptation (longer memory).
                        Default 0.99 gives roughly 100-step effective window.
        """
        # Always float32: detach prevents gradient flow through EMA update,
        # which has no semantic meaning and would pollute the backbone gradients.
        batch_mean = embeddings.detach().float().mean(dim=0)   # [D]
        # In-place EMA: centroid ← momentum * centroid + (1 - momentum) * mean
        self.centroid.mul_(momentum).add_(batch_mean * (1.0 - momentum))


# ---------------------------------------------------------------------------
# MultilabelHead (SourceID)
# ---------------------------------------------------------------------------


class MultilabelHead(nn.Module):
    """SourceID detection head: per-generator logits from the CLS embedding.

    Two-layer MLP that projects the shared backbone embedding to one logit
    per detection class. Classes fire independently (multilabel), so the
    model can simultaneously output [human=1, suno=1] for mixed content.

    WHY TWO LAYERS (NOT ONE LINEAR)
    --------------------------------
    A single linear layer forces the model to find separating hyperplanes
    directly in the backbone's contrastive embedding space, which was shaped
    by Phase 1 for distance-to-natural, not generator discrimination. The
    intermediate layer lets the MLP project the shared representation into a
    space where generator fingerprints are more linearly separable, without
    disturbing the backbone embedding that DistanceHead also uses.

    WHY NO SIGMOID HERE
    -------------------
    BCEWithLogitsLoss (Phase 2) fuses sigmoid and binary cross-entropy
    numerically, avoiding log(0) when logits are very large or very small
    during early training. Calling sigmoid before BCEWithLogitsLoss is
    incorrect (it applies sigmoid twice). Inference code that needs
    probabilities in [0, 1] applies torch.sigmoid to the logits explicitly.

    Args:
        embed_dim:  CLS embedding dimension (input to the first linear layer).
        n_classes:  Number of output classes (default 8: human + 7 AI sources).
        hidden_dim: Hidden layer size. Default 256 matches the architecture spec
                    (Linear(D, 256) → GELU → Linear(256, n_classes)).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_classes: int = N_CLASSES_DEFAULT,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, embed_dim] CLS embeddings.
        Returns:
            [B, n_classes] raw logits. Sigmoid is NOT applied.
        """
        return self.layers(embeddings)


# ---------------------------------------------------------------------------
# Detector (top-level composition)
# ---------------------------------------------------------------------------


class Detector(nn.Module):
    """Full per-window detector: backbone + NaturalGuard + SourceID heads.

    Composes WindowEncoder, DistanceHead, and MultilabelHead into a single
    module. One forward pass yields outputs for both detection systems.

    Input:  Normalized log-power spectrogram [B, in_channels, F_MAX, T].
            This must be the output of PerFrequencyInstanceNorm (and optionally
            SpectrogramAugmenter during training). The Detector does NOT run
            the features.py pipeline internally.

    Output: dict with three keys:
            "cls_embedding"  [B, D]         — shared backbone representation
            "distance"       [B]            — NaturalGuard L2 distance score
            "logits"         [B, n_classes] — SourceID per-class logits

    Args:
        in_channels:    Input spectrogram channel count (default 3).
        embed_dim:      Transformer/patch embedding dimension D.
        t1_num_layers:  Number of WindowEncoder transformer layers.
        use_bands:      If True, split spectrogram into 3 frequency bands.
        n_classes:      Number of SourceID output classes.
        num_heads:      Attention heads per transformer layer.
        feedforward_dim: FFN hidden dim in each transformer layer.
        dropout:        Transformer dropout.
        cnn_channels:   BandCNN Conv2d intermediate channel count.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        t1_num_layers: int = 4,
        use_bands: bool = True,
        n_classes: int = N_CLASSES_DEFAULT,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        cnn_channels: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = WindowEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            t1_num_layers=t1_num_layers,
            use_bands=use_bands,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            cnn_channels=cnn_channels,
        )
        self.distance_head = DistanceHead(embed_dim=embed_dim)
        self.multilabel_head = MultilabelHead(embed_dim=embed_dim, n_classes=n_classes)

    def forward(self, spec_input: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            spec_input: [B, in_channels, F, T] normalized log-power spectrogram.
        Returns:
            {
              "cls_embedding": [B, D]            — shared backbone embedding,
              "distance":      [B]               — NaturalGuard distance score,
              "logits":        [B, n_classes]    — SourceID logits (no sigmoid),
            }
        """
        cls_emb = self.encoder(spec_input)      # [B, D]
        distance = self.distance_head(cls_emb)  # [B]
        logits = self.multilabel_head(cls_emb)  # [B, n_classes]

        return {
            "cls_embedding": cls_emb,
            "distance":      distance,
            "logits":        logits,
        }

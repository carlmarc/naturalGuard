"""
src/detector/features.py — Spectrogram front-end: STFT, normalization, augmentation.

PIPELINE POSITION
-----------------
These three modules form the front-end of the per-window pipeline, executed in
order before any CNN or Transformer layer sees the audio:

    waveform [B, 1, N]
        → MultiScaleSTFT         → [B, 3, F_MAX, T]     (log-power, 3 scales)
        → PerFrequencyInstanceNorm → [B, 3, F_MAX, T]   (per-bin normalized)
        → SpectrogramAugmenter   → [B, 3, F_MAX, T]     (stochastic, train only)

WHY THREE SEPARATE MODULES
---------------------------
MultiScaleSTFT is deterministic and could in principle be cached or run on CPU
in a DataLoader worker. PerFrequencyInstanceNorm is also deterministic and could
be fused with the STFT stage later. SpectrogramAugmenter is stochastic and
training-only; keeping it isolated makes it easy to disable, reconfigure, or
replace without touching the other two.

WHY LOG SPECTROGRAMS
--------------------
Raw power spectrograms span 4–5 decades of dynamic range. Log compression
brings them into a range the CNN can process uniformly without needing very
deep networks or custom activations. All downstream computations operate on
log-magnitude (natural log) values.

WHY MULTI-SCALE
---------------
Different artifacts live at different frequency resolutions:
  n_fft=256  (11.7 Hz bins) — coarse spectral shapes, rhythm-correlated energy.
  n_fft=1024  (2.9 Hz bins) — formants, harmonics, broad spectral tilt.
  n_fft=4096  (0.7 Hz bins) — fine harmonic structure, voicing, the subtle
                               artifacts AI generators leave in the harmonic
                               stack.
Stacking the three scales as channels lets the CNN learn which resolution
matters for which artifact type.

WHY PER-FREQUENCY INSTANCE NORM
--------------------------------
nn.InstanceNorm2d normalizes over the joint (F × T) spatial extent. That mixes
frequency and time statistics: a loud note at a specific frequency would raise
the normalization denominator across all frequencies, distorting the result.
We normalize only across the time axis (per frequency bin, per batch item)
so that each frequency bin's mean and variance are determined by this 7-second
window alone, removing per-track EQ profiles while preserving all time-varying
patterns within the window.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

# ε for log transform: avoids log(0) and sets the noise floor in log space.
# At -16 LUFS, 22050 Hz, the DC and near-DC bins near silence sit comfortably
# above this floor after the power spectral estimate.
LOG_EPS: float = 1e-5

# ε for per-frequency instance norm: prevents division by zero on constant bins
# (silence, DC, or near-Nyquist content band-limited to a flat floor).
NORM_EPS: float = 1e-5

# F_MAX: frequency bins produced by n_fft=4096 at any sample rate.
# Smaller n_fft outputs are zero-padded to this size so all three scales can
# be stacked as channels and share the same spatial layout in the CNN.
F_MAX: int = 2049  # = 4096 // 2 + 1


class MultiScaleSTFT(nn.Module):
    """Compute log-power spectrograms at three frequency resolutions.

    Three parallel torchaudio Spectrograms with n_fft ∈ {256, 1024, 4096} and
    identical hop_length. Using the same hop_length guarantees that all three
    scales produce the same number of time frames T, so they can be stacked as
    channels without any temporal resampling.

    Smaller n_fft outputs are zero-padded along the frequency axis to F_MAX=2049
    (the bin count for n_fft=4096). Zero-padding is harmless: the padded bins
    were never computed (sub-Nyquist for that FFT size), and PerFrequencyInstance
    Norm will normalize them uniformly to zero anyway.

    Log is applied here (natural log, not log10) to compress the 4–5 decade
    dynamic range of raw power spectrograms into a CNN-friendly range. LOG_EPS
    sets the noise floor: log(LOG_EPS) ≈ -11.5, so the floor is well separated
    from real signal values even in quiet passages.

    Output shape: [B, 3, F_MAX, T_frames]
      B          — batch size
      3          — three frequency scales
      F_MAX=2049 — frequency bins (n_fft=4096 scale, smaller padded)
      T_frames   — N // hop_length + 1 (603 for 154350 samples at hop=256)

    Args:
        hop_length: STFT hop size in samples, shared by all three scales.
        n_ffts:     Tuple of three FFT sizes (small, medium, large).
        power:      Spectrogram power (2.0 = power spectrum, 1.0 = magnitude).
    """

    def __init__(
        self,
        hop_length: int = 256,
        n_ffts: tuple[int, int, int] = (256, 1024, 4096),
        power: float = 2.0,
    ) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.n_ffts = n_ffts

        # Register each Spectrogram as a named submodule so .to(device) and
        # .state_dict() handle them correctly. Spectrogram has no learnable
        # parameters, but it contains internal window buffers.
        self.stft_small = T.Spectrogram(n_fft=n_ffts[0], hop_length=hop_length, power=power)
        self.stft_mid = T.Spectrogram(n_fft=n_ffts[1], hop_length=hop_length, power=power)
        self.stft_large = T.Spectrogram(n_fft=n_ffts[2], hop_length=hop_length, power=power)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, 1, N] float32 mono audio at 22050 Hz.
        Returns:
            [B, 3, F_MAX, T_frames] log-power spectrogram.
        """
        # Spectrogram expects [B, N] or [N]; squeeze the channel dim.
        x = waveform.squeeze(1)  # [B, N]

        # Each STFT produces [B, F_k, T] where F_k = n_fft_k // 2 + 1.
        s_small = self.stft_small(x)  # [B, 129,  T]
        s_mid = self.stft_mid(x)      # [B, 513,  T]
        s_large = self.stft_large(x)  # [B, 2049, T]

        # Log-compress: log(S + ε). Natural log, not log10.
        # +ε prevents log(0) on silent bins; sets floor at log(ε) ≈ -11.5.
        s_small = torch.log(s_small + LOG_EPS)
        s_mid = torch.log(s_mid + LOG_EPS)
        s_large = torch.log(s_large + LOG_EPS)

        # Zero-pad smaller scales along F to F_MAX so all three are stackable.
        # F.pad args go from last dim inward: (T_left, T_right, F_left, F_right).
        s_small = F.pad(s_small, (0, 0, 0, F_MAX - s_small.shape[1]))   # [B, 2049, T]
        s_mid = F.pad(s_mid, (0, 0, 0, F_MAX - s_mid.shape[1]))         # [B, 2049, T]
        # s_large is already [B, 2049, T]; no padding needed.

        # Stack along a new channel dimension: [B, 3, F_MAX, T]
        return torch.stack([s_small, s_mid, s_large], dim=1)


class PerFrequencyInstanceNorm(nn.Module):
    """Normalize each frequency bin independently across the time axis.

    For every (batch, channel, frequency) triplet, subtract the time-axis mean
    and divide by the time-axis std of that slice. This removes per-track EQ
    profiles and absolute per-bin level differences, while preserving all
    time-varying patterns within the 7-second window.

    WHY NOT nn.InstanceNorm2d
    -------------------------
    InstanceNorm2d normalizes over the joint (H × W) = (F × T) spatial extent.
    That mixes frequency and time statistics: a loud note at one frequency would
    raise the shared normalization denominator and distort all other frequency
    bins' normalized values. We need each frequency bin to have its own
    independent statistics, which requires normalizing along dim=-1 (T) only.

    WHY NO LEARNABLE SCALE/SHIFT (γ, β)
    -------------------------------------
    The intent is purely to remove confounding EQ variation from the input, not
    to add another set of per-frequency parameters. The CNN layers that follow
    have their own learnable weights; adding γ/β here would let them effectively
    re-introduce the EQ profile the normalization was supposed to remove.

    Args:
        eps: Small constant added to std to prevent division by zero on constant
             frequency bins (silence, DC, or band-limited near-Nyquist content).
    """

    def __init__(self, eps: float = NORM_EPS) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, T] log-power spectrogram.
        Returns:
            [B, C, F, T] with each (b, c, f) slice normalized to mean≈0, std≈1.
        """
        # Compute statistics along T (dim=-1) only; keepdim for broadcast.
        mean = x.mean(dim=-1, keepdim=True)              # [B, C, F, 1]
        std = x.std(dim=-1, keepdim=True, unbiased=False)  # [B, C, F, 1]
        # unbiased=False matches the population std (divides by T, not T-1).
        # For T=603 the difference is negligible, but the choice must be
        # consistent between forward() and any test that verifies std≈1.
        return (x - mean) / (std + self.eps)


class SpectrogramAugmenter(nn.Module):
    """Stochastic spectrogram augmentation, active in training mode only.

    Applies four augmentation strategies in sequence. When self.training is
    False (eval/inference), the input is returned unchanged — no randomness,
    no copies. All operations are pure PyTorch; no librosa, no waveform DSP.

    Input and output are in log-domain (natural log) after PerFrequencyInstance
    Norm, so gain operations are additive (log(S·g) = log(S) + log(g)).

    STRATEGY                WHAT IT PREVENTS
    ───────────────────────────────────────────────────────────────────────────
    Frequency masking       Model relying on artifacts at specific frequency
                            bands. AI generators may leave fingerprints in
                            narrow bands; masking prevents memorizing their
                            exact position.

    Time masking            Model relying on events at specific temporal
                            positions within the 7-second window.

    Gain jitter (±6 dB)     Learning "louder → human". The -16 LUFS normali-
                            zation has ±LRA tolerance; gain jitter covers that
                            residual variation and more.

    Gaussian noise          Over-reliance on exact log-spec values. Minor codec
                            artifact variations across re-encodes are of this
                            character.

    Frequency tilt          Learning spectral slope as a class feature. AI
                            generators and natural music differ in their average
                            high-band energy; tilt jitter prevents the model
                            from trivially using this as a shortcut even after
                            the high_energy_ratio stratification in the splits.

    Args:
        time_mask_param: Maximum number of time frames to mask per band.
        freq_mask_param: Maximum number of frequency bins to mask per band.
        gain_db:         Max gain jitter range (±this many dB, uniform).
        noise_std:       Standard deviation of additive Gaussian noise.
        max_tilt_db:     Max spectral tilt at Nyquist (±this many dB, uniform).
                         The tilt is 0 at DC and linearly ramps to ±max_tilt_db.
    """

    # Natural log of 10; needed to convert dB to natural-log units.
    # Input spectrograms are torch.log (natural log), not log10.
    _LN10: float = 2.302585092994046

    def __init__(
        self,
        time_mask_param: int = 50,
        freq_mask_param: int = 30,
        gain_db: float = 6.0,
        noise_std: float = 0.01,
        max_tilt_db: float = 3.0,
    ) -> None:
        super().__init__()
        self.gain_db = gain_db
        self.noise_std = noise_std
        self.max_tilt_db = max_tilt_db

        # iid_masks=True gives each batch item an independent mask position.
        # Without this flag, all items in the batch share the same mask
        # coordinates, which defeats the regularization purpose for batches > 1.
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param, iid_masks=True)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, T] normalized log-power spectrogram.
        Returns:
            [B, C, F, T] augmented (train mode) or identical tensor (eval mode).
        """
        if not self.training:
            # Eval/inference: return the same object, not a copy.
            return x

        B, _C, F, _T = x.shape

        # ---- Masking ----
        # torchaudio masking expects (..., freq, time) = (..., F, T) which
        # matches our layout. The masked positions are filled with 0.0, which
        # in normalized log space sits at the mean of that frequency bin (≈0),
        # so it does not introduce an outlier signal.
        x = self.freq_masking(x)
        x = self.time_masking(x)

        # ---- Gain jitter ----
        # One scalar per batch item drawn from Uniform([-gain_db, +gain_db]).
        # Convert dB to nepers (natural-log units): gain_nepers = gain_db * ln(10)/20.
        gain_db = torch.empty(B, device=x.device, dtype=x.dtype).uniform_(
            -self.gain_db, self.gain_db
        )
        gain_ln = gain_db * (self._LN10 / 20.0)  # [B] in neper units
        x = x + gain_ln.view(B, 1, 1, 1)          # broadcast over C, F, T

        # ---- Gaussian noise ----
        x = x + torch.randn_like(x) * self.noise_std

        # ---- Frequency tilt ----
        # Linear ramp from 0 (DC) to ±max_tilt_db (Nyquist), one slope per
        # batch item. Simulates random EQ curves that survive -16 LUFS norm.
        tilt_db = torch.empty(B, device=x.device, dtype=x.dtype).uniform_(
            -self.max_tilt_db, self.max_tilt_db
        )
        # freq_ramp[0]=0 (no gain at DC), freq_ramp[F-1]=1 (full tilt at Nyquist).
        freq_ramp = torch.linspace(0.0, 1.0, F, device=x.device, dtype=x.dtype)  # [F]
        tilt_ln = (
            tilt_db.view(B, 1, 1, 1) * freq_ramp.view(1, 1, F, 1) * (self._LN10 / 20.0)
        )
        x = x + tilt_ln  # broadcast over C and T

        return x

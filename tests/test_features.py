"""
tests/test_features.py — Smoke tests for src/detector/features.py.

These are integration-style shape and property checks, not unit tests for
each arithmetic operation. The goal is to catch:
  - Wrong output shape (padding, stacking, or squeez bugs)
  - NaN/Inf from log(0) or divide-by-zero in normalization
  - SpectrogramAugmenter leaking into eval mode
  - PerFrequencyInstanceNorm not normalizing across T correctly

Running:
    pytest tests/test_features.py -v

Requires torch and torchaudio to be installed. No GPU required; all tests
run on CPU.
"""

import sys
from pathlib import Path

import pytest
import torch

# Put src/ on the path so `from detector.features import ...` resolves.
# No pyproject.toml / editable install yet, so we patch sys.path directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detector.features import (  # noqa: E402
    F_MAX,
    MultiScaleSTFT,
    PerFrequencyInstanceNorm,
    SpectrogramAugmenter,
)

# ---------------------------------------------------------------------------
# CONSTANTS matching CLAUDE.md architecture
# ---------------------------------------------------------------------------

SR = 22050
WINDOW_SEC = 7
WINDOW_SAMPLES = WINDOW_SEC * SR  # 154350
# T_frames for hop_length=256, center=True: floor(154350/256) + 1 = 603
T_FRAMES = 603
BATCH = 2


# ---------------------------------------------------------------------------
# MultiScaleSTFT
# ---------------------------------------------------------------------------


class TestMultiScaleSTFT:
    """MultiScaleSTFT: shape, channel count, zero-padding, numerical sanity."""

    def setup_method(self):
        self.stft = MultiScaleSTFT()
        torch.manual_seed(0)
        self.wav = torch.randn(BATCH, 1, WINDOW_SAMPLES)

    def test_output_shape(self):
        out = self.stft(self.wav)
        assert out.shape == (BATCH, 3, F_MAX, T_FRAMES), (
            f"Expected [{BATCH}, 3, {F_MAX}, {T_FRAMES}], got {out.shape}"
        )

    def test_three_channels(self):
        out = self.stft(self.wav)
        assert out.shape[1] == 3, "Must produce exactly 3 frequency-scale channels"

    def test_frequency_dim_is_fmax(self):
        out = self.stft(self.wav)
        assert out.shape[2] == F_MAX, (
            f"Frequency dim should be F_MAX={F_MAX}, got {out.shape[2]}"
        )

    def test_no_nan_inf(self):
        out = self.stft(self.wav)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_single_item_batch(self):
        wav = torch.randn(1, 1, WINDOW_SAMPLES)
        out = self.stft(wav)
        assert out.shape == (1, 3, F_MAX, T_FRAMES)

    def test_padded_bins_are_zero(self):
        # The small STFT (n_fft=256) produces 129 bins; bins 129..2048 are padded.
        # After log-compression log(0 + EPS) fills the padded region, but the
        # zero-padding is applied before log, so the padded area = log(EPS),
        # which is a constant. Verify the padded zone is uniform (all equal).
        out = self.stft(self.wav)
        # Channel 0 (n_fft=256): bins 129..2048 are padded.
        padded_zone = out[:, 0, 129:, :]  # [B, 1920, T]
        # All values should be identical (log(EPS))
        first_val = padded_zone[:, :1, :1]
        assert torch.allclose(padded_zone, first_val.expand_as(padded_zone)), (
            "Padded frequency bins in small-STFT channel should be uniform"
        )

    def test_log_values_are_negative(self):
        # Power spectrogram values are in [0, ~1e3]; log + EPS gives negative
        # or small positive values. Values above ~7 would indicate a bug.
        out = self.stft(self.wav)
        assert out.max().item() < 20.0, (
            f"Unexpectedly large log-spec value: {out.max().item():.1f}"
        )


# ---------------------------------------------------------------------------
# PerFrequencyInstanceNorm
# ---------------------------------------------------------------------------


class TestPerFrequencyInstanceNorm:
    """PerFrequencyInstanceNorm: mean≈0 and std≈1 across T for every (b,c,f)."""

    def setup_method(self):
        self.norm = PerFrequencyInstanceNorm()
        torch.manual_seed(1)
        self.x = torch.randn(BATCH, 3, F_MAX, T_FRAMES)

    def test_output_shape_preserved(self):
        out = self.norm(self.x)
        assert out.shape == self.x.shape

    def test_time_mean_near_zero(self):
        out = self.norm(self.x)
        # mean across T (dim=-1) for every (b, c, f) slice
        mean = out.mean(dim=-1)  # [B, C, F]
        max_deviation = mean.abs().max().item()
        assert max_deviation < 1e-4, (
            f"Time-axis mean not near 0; max |mean| = {max_deviation:.2e}"
        )

    def test_time_std_near_one(self):
        out = self.norm(self.x)
        # std across T with unbiased=False (matches the forward() computation)
        std = out.std(dim=-1, unbiased=False)  # [B, C, F]
        max_deviation = (std - 1.0).abs().max().item()
        # Tolerance 0.02: at T=603 frames the sample-to-population std ratio
        # is ~1 ± 0.001, so 0.02 gives plenty of headroom.
        assert max_deviation < 0.02, (
            f"Time-axis std not near 1; max |std-1| = {max_deviation:.4f}"
        )

    def test_no_nan_inf(self):
        out = self.norm(self.x)
        assert torch.isfinite(out).all(), "Normalization produced NaN or Inf"

    def test_constant_bin_does_not_crash(self):
        # A frequency bin that is constant over time (std=0) should produce
        # 0.0 after normalization (not NaN), because of the eps guard.
        x = self.x.clone()
        x[:, :, 100, :] = 5.0  # constant at bin 100 for all B, C, T
        out = self.norm(x)
        assert torch.isfinite(out).all(), "Constant bin caused NaN"
        # After normalization the constant bin should be 0 (mean=5, std≈0 → (5-5)/(0+eps)=0)
        assert out[:, :, 100, :].abs().max().item() < 1.0, (
            "Constant frequency bin should normalize to 0"
        )


# ---------------------------------------------------------------------------
# SpectrogramAugmenter
# ---------------------------------------------------------------------------


class TestSpectrogramAugmenter:
    """SpectrogramAugmenter: no-op in eval, stochastic in train, shape stable."""

    def setup_method(self):
        self.aug = SpectrogramAugmenter()
        torch.manual_seed(2)
        self.x = torch.randn(BATCH, 3, F_MAX, T_FRAMES)

    def test_eval_mode_returns_same_tensor(self):
        self.aug.eval()
        out = self.aug(self.x)
        # Must return the identical object, not a copy.
        assert out is self.x, (
            "Eval mode must return the input tensor unchanged (same object)"
        )

    def test_eval_mode_is_deterministic(self):
        self.aug.eval()
        out1 = self.aug(self.x)
        out2 = self.aug(self.x)
        assert torch.allclose(out1, out2)

    def test_train_mode_changes_output(self):
        self.aug.train()
        x_copy = self.x.clone()
        out = self.aug(x_copy)
        assert not torch.allclose(out, self.x), (
            "Train mode must produce output different from input"
        )

    def test_output_shape_unchanged(self):
        self.aug.train()
        out = self.aug(self.x.clone())
        assert out.shape == self.x.shape

    def test_no_nan_inf_in_train_mode(self):
        self.aug.train()
        out = self.aug(self.x.clone())
        assert torch.isfinite(out).all(), "Augmented output contains NaN or Inf"

    def test_two_train_calls_differ(self):
        # Two successive forward passes in train mode should produce different
        # outputs because the mask positions and gain values are sampled fresh.
        self.aug.train()
        x1 = self.x.clone()
        x2 = self.x.clone()
        out1 = self.aug(x1)
        out2 = self.aug(x2)
        assert not torch.allclose(out1, out2), (
            "Two train-mode forward passes should differ (stochastic augmentation)"
        )


# ---------------------------------------------------------------------------
# End-to-end: MultiScaleSTFT → PerFrequencyInstanceNorm → SpectrogramAugmenter
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Verify the three modules chain without shape or numerical errors."""

    def test_full_pipeline_shape(self):
        stft = MultiScaleSTFT()
        norm = PerFrequencyInstanceNorm()
        aug = SpectrogramAugmenter()
        aug.train()

        torch.manual_seed(3)
        wav = torch.randn(BATCH, 1, WINDOW_SAMPLES)

        spec = stft(wav)
        spec = norm(spec)
        spec = aug(spec)

        assert spec.shape == (BATCH, 3, F_MAX, T_FRAMES), (
            f"End-to-end shape wrong: {spec.shape}"
        )
        assert torch.isfinite(spec).all(), "End-to-end pipeline produced NaN or Inf"

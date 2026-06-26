"""
tests/test_model.py — Smoke tests for src/detector/model.py.

These are integration-style shape and property checks. They cover:
  - Correct output shapes for each component
  - NaN/Inf absence in all outputs
  - DistanceHead centroid: buffer presence, EMA convergence, state_dict persistence
  - CLS embedding changes after EMA update (verifies no silent identity)
  - Detector full composition with use_bands=True and use_bands=False
  - state_dict round-trip preserving the centroid buffer

Running:
    pytest tests/test_model.py -v

Requires torch and torchaudio. No GPU required; all tests run on CPU.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detector.model import (  # noqa: E402
    N_CLASSES_DEFAULT,
    BandCNN,
    Detector,
    DistanceHead,
    MultilabelHead,
    WindowEncoder,
    _hz_to_bin,
)

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

BATCH = 2
EMBED_DIM = 256
F_MAX = 2049
# Use T=604 (slightly different from the exact 7-second 603 value) to verify
# that variable-length inputs work correctly throughout the pipeline.
T_FRAMES = 604


# ---------------------------------------------------------------------------
# Helper: band bin boundaries (mirrors WindowEncoder's init logic)
# ---------------------------------------------------------------------------


def test_hz_to_bin_spot_checks():
    """Verify the Hz→bin conversion against known values."""
    # sr=22050, n_fft=4096 → freq_resolution ≈ 5.38 Hz/bin
    assert _hz_to_bin(20.0)    == 4      # 20 / 5.38 ≈ 3.71 → 4
    assert _hz_to_bin(500.0)   == 93     # 500 / 5.38 ≈ 92.9 → 93
    assert _hz_to_bin(300.0)   == 56     # 300 / 5.38 ≈ 55.7 → 56
    assert _hz_to_bin(4000.0)  == 743    # 4000 / 5.38 ≈ 743.0 → 743
    assert _hz_to_bin(3000.0)  == 557    # 3000 / 5.38 ≈ 557.3 → 557
    assert _hz_to_bin(11025.0) == 2048   # Nyquist → n_fft//2 = 2048


# ---------------------------------------------------------------------------
# BandCNN
# ---------------------------------------------------------------------------


class TestBandCNN:
    """BandCNN: shape, no NaN/Inf, variable band height."""

    def setup_method(self):
        self.cnn = BandCNN(in_channels=3, cnn_channels=64, embed_dim=EMBED_DIM)
        torch.manual_seed(0)

    def test_output_shape(self):
        # Input [B, C_in, F_band, T] with F_band=50, T=600.
        # Conv1d(k=4, s=4): T_patches = (600-4)//4 + 1 = 150.
        x = torch.randn(BATCH, 3, 50, 600)
        out = self.cnn(x)
        assert out.shape == (BATCH, 150, EMBED_DIM), (
            f"Expected [{BATCH}, 150, {EMBED_DIM}], got {out.shape}"
        )

    def test_embed_dim_matches(self):
        x = torch.randn(BATCH, 3, 50, 600)
        out = self.cnn(x)
        assert out.shape[-1] == EMBED_DIM

    def test_no_nan_inf(self):
        x = torch.randn(BATCH, 3, 50, 600)
        out = self.cnn(x)
        assert torch.isfinite(out).all()

    def test_variable_band_heights(self):
        # The three bands have different heights: Low~89, Mid~687, High~1492.
        for f_band in (89, 687, 1492):
            x = torch.randn(BATCH, 3, f_band, T_FRAMES)
            out = self.cnn(x)
            # Shape is [B, T_patches, D]; T_patches depends only on T, not F_band.
            assert out.shape[0] == BATCH
            assert out.shape[2] == EMBED_DIM, (
                f"F_band={f_band}: embed_dim wrong, got {out.shape[2]}"
            )

    def test_t_patches_formula(self):
        # Non-overlapping Conv1d(k=4, s=4): T_patches = (T - 4) // 4 + 1
        for t_in in (400, 603, 604, 800):
            x = torch.randn(1, 3, 50, t_in)
            out = self.cnn(x)
            expected = (t_in - 4) // 4 + 1
            assert out.shape[1] == expected, (
                f"T={t_in}: expected T_patches={expected}, got {out.shape[1]}"
            )


# ---------------------------------------------------------------------------
# WindowEncoder
# ---------------------------------------------------------------------------


class TestWindowEncoderBands:
    """WindowEncoder with use_bands=True."""

    def setup_method(self):
        self.enc = WindowEncoder(embed_dim=EMBED_DIM, t1_num_layers=2, use_bands=True)
        torch.manual_seed(1)
        self.spec = torch.randn(BATCH, 3, F_MAX, T_FRAMES)

    def test_output_shape(self):
        out = self.enc(self.spec)
        assert out.shape == (BATCH, EMBED_DIM), (
            f"Expected [{BATCH}, {EMBED_DIM}], got {out.shape}"
        )

    def test_no_nan_inf(self):
        out = self.enc(self.spec)
        assert torch.isfinite(out).all()

    def test_batch_items_differ(self):
        # Different inputs in the batch should produce different embeddings.
        out = self.enc(self.spec)
        assert not torch.allclose(out[0], out[1]), (
            "Two different batch items produced identical CLS embeddings"
        )

    def test_three_band_cnns_exist(self):
        assert len(self.enc.band_cnns) == 3

    def test_band_bins_are_computed(self):
        # Band bin tuples should be non-trivial.
        assert self.enc.band_bins is not None
        for lo, hi in self.enc.band_bins:
            assert hi > lo > 0, f"Invalid band: lo={lo}, hi={hi}"


class TestWindowEncoderNoBands:
    """WindowEncoder with use_bands=False (full spectrum, one BandCNN)."""

    def setup_method(self):
        self.enc = WindowEncoder(embed_dim=EMBED_DIM, t1_num_layers=2, use_bands=False)
        torch.manual_seed(2)
        self.spec = torch.randn(BATCH, 3, F_MAX, T_FRAMES)

    def test_output_shape(self):
        out = self.enc(self.spec)
        assert out.shape == (BATCH, EMBED_DIM)

    def test_no_nan_inf(self):
        out = self.enc(self.spec)
        assert torch.isfinite(out).all()

    def test_one_band_cnn(self):
        assert len(self.enc.band_cnns) == 1

    def test_band_bins_is_none(self):
        assert self.enc.band_bins is None


# ---------------------------------------------------------------------------
# DistanceHead
# ---------------------------------------------------------------------------


class TestDistanceHead:
    """DistanceHead: output shape, centroid buffer, EMA convergence."""

    def setup_method(self):
        self.head = DistanceHead(embed_dim=EMBED_DIM)
        torch.manual_seed(3)
        self.emb = torch.randn(BATCH, EMBED_DIM)

    def test_output_shape(self):
        dist = self.head(self.emb)
        assert dist.shape == (BATCH,), f"Expected [{BATCH}], got {dist.shape}"

    def test_output_is_non_negative(self):
        dist = self.head(self.emb)
        # L2 distance is always ≥ 0.
        assert (dist >= 0).all()

    def test_centroid_in_state_dict(self):
        sd = self.head.state_dict()
        assert "centroid" in sd, "centroid buffer missing from state_dict"

    def test_centroid_not_in_parameters(self):
        param_names = [n for n, _ in self.head.named_parameters()]
        assert "centroid" not in param_names, (
            "centroid should be a buffer, not a learnable parameter"
        )

    def test_centroid_changes_after_ema_update(self):
        centroid_before = self.head.centroid.clone()
        self.head.update_centroid_ema(self.emb, momentum=0.9)
        assert not torch.allclose(self.head.centroid, centroid_before), (
            "Centroid unchanged after EMA update"
        )

    def test_centroid_approaches_input_mean(self):
        # With constant input x and many EMA steps, centroid → mean(x).
        # Geometric series: c_n = (1-m)*mean * sum_{k=0}^{n-1} m^k → mean as n→∞.
        target = torch.ones(EMBED_DIM) * 3.0
        emb = target.unsqueeze(0).expand(BATCH, -1)   # [B, D] all = 3.0
        head = DistanceHead(embed_dim=EMBED_DIM)
        for _ in range(300):
            head.update_centroid_ema(emb, momentum=0.9)
        # After 300 steps with m=0.9: residual = 0.9^300 ≈ 1e-14 (negligible).
        assert torch.allclose(head.centroid, target, atol=0.1), (
            f"Centroid did not converge to input mean; "
            f"max diff = {(head.centroid - target).abs().max():.4f}"
        )

    def test_no_nan_inf(self):
        dist = self.head(self.emb)
        assert torch.isfinite(dist).all()

    def test_zero_centroid_distance_equals_norm(self):
        # When centroid=0, distance should equal ||embedding||.
        expected = torch.linalg.norm(self.emb.float(), dim=-1)
        dist = self.head(self.emb)
        assert torch.allclose(dist, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# MultilabelHead
# ---------------------------------------------------------------------------


class TestMultilabelHead:
    """MultilabelHead: output shape and raw logit range."""

    def setup_method(self):
        self.head = MultilabelHead(embed_dim=EMBED_DIM, n_classes=N_CLASSES_DEFAULT)
        torch.manual_seed(4)
        self.emb = torch.randn(BATCH, EMBED_DIM)

    def test_output_shape(self):
        logits = self.head(self.emb)
        assert logits.shape == (BATCH, N_CLASSES_DEFAULT), (
            f"Expected [{BATCH}, {N_CLASSES_DEFAULT}], got {logits.shape}"
        )

    def test_no_sigmoid_applied(self):
        # Raw logits should not be constrained to [0, 1].
        # With random weights and inputs, at least some logits should be < 0.
        logits = self.head(self.emb)
        assert logits.min().item() < 0.0, (
            "All logits are non-negative; sigmoid may have been applied inside the head"
        )

    def test_no_nan_inf(self):
        logits = self.head(self.emb)
        assert torch.isfinite(logits).all()

    def test_n_classes_8(self):
        assert self.head(self.emb).shape[-1] == 8


# ---------------------------------------------------------------------------
# Detector (full composition)
# ---------------------------------------------------------------------------


class TestDetector:
    """Detector: end-to-end shape, keys, state_dict centroid persistence."""

    def setup_method(self):
        # Use t1_num_layers=2 to keep tests fast (less memory, faster forward).
        self.model = Detector(embed_dim=EMBED_DIM, t1_num_layers=2, use_bands=True)
        torch.manual_seed(5)
        self.spec = torch.randn(BATCH, 3, F_MAX, T_FRAMES)

    def test_output_keys(self):
        out = self.model(self.spec)
        assert set(out.keys()) == {"cls_embedding", "distance", "logits"}, (
            f"Unexpected output keys: {set(out.keys())}"
        )

    def test_cls_embedding_shape(self):
        out = self.model(self.spec)
        assert out["cls_embedding"].shape == (BATCH, EMBED_DIM)

    def test_distance_shape(self):
        out = self.model(self.spec)
        assert out["distance"].shape == (BATCH,)

    def test_logits_shape(self):
        out = self.model(self.spec)
        assert out["logits"].shape == (BATCH, N_CLASSES_DEFAULT)

    def test_distance_non_negative(self):
        out = self.model(self.spec)
        assert (out["distance"] >= 0).all()

    def test_no_nan_inf(self):
        out = self.model(self.spec)
        for k, v in out.items():
            assert torch.isfinite(v).all(), f"NaN/Inf in '{k}'"

    def test_no_bands_produces_same_shape(self):
        model_no_bands = Detector(embed_dim=EMBED_DIM, t1_num_layers=2, use_bands=False)
        out = model_no_bands(self.spec)
        assert out["cls_embedding"].shape == (BATCH, EMBED_DIM)
        assert out["distance"].shape      == (BATCH,)
        assert out["logits"].shape        == (BATCH, N_CLASSES_DEFAULT)

    def test_state_dict_contains_centroid(self):
        sd = self.model.state_dict()
        assert "distance_head.centroid" in sd

    def test_state_dict_round_trip_preserves_centroid(self):
        # Set a non-zero centroid via EMA update.
        out = self.model(self.spec)
        self.model.distance_head.update_centroid_ema(
            out["cls_embedding"], momentum=0.0  # momentum=0 → centroid = batch mean exactly
        )
        centroid_before = self.model.distance_head.centroid.clone()

        # Save and reload.
        state = self.model.state_dict()
        model2 = Detector(embed_dim=EMBED_DIM, t1_num_layers=2, use_bands=True)
        model2.load_state_dict(state)

        assert torch.allclose(model2.distance_head.centroid, centroid_before), (
            "Centroid not preserved after state_dict save/load"
        )

    def test_two_forward_passes_are_deterministic_in_eval(self):
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(self.spec)
            out2 = self.model(self.spec)
        assert torch.allclose(out1["cls_embedding"], out2["cls_embedding"])
        assert torch.allclose(out1["distance"],      out2["distance"])
        assert torch.allclose(out1["logits"],        out2["logits"])

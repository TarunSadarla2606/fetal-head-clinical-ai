"""Shape, dtype, and value-range checks for core src modules.

Run with:  pytest tests/
"""

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# src.data.dataset
# ---------------------------------------------------------------------------


class TestFillHollowMask:
    def test_output_shape_matches_input(self):
        from src.data.dataset import fill_hollow_mask

        ring = np.zeros((256, 384), dtype=np.uint8)
        ring[50:200, 80:320] = 255  # thick ring stand-in
        out = fill_hollow_mask(ring)
        assert out.shape == ring.shape

    def test_output_dtype_uint8(self):
        from src.data.dataset import fill_hollow_mask

        ring = np.zeros((64, 64), dtype=np.uint8)
        out = fill_hollow_mask(ring)
        assert out.dtype == np.uint8

    def test_solid_mask_contains_ring(self):
        from src.data.dataset import fill_hollow_mask

        ring = np.zeros((64, 64), dtype=np.uint8)
        ring[10, 10:50] = 255  # horizontal bar
        out = fill_hollow_mask(ring)
        # All original ring pixels must remain set
        assert np.all(out[ring > 0] > 0)


# ---------------------------------------------------------------------------
# src.evaluate — metric functions
# ---------------------------------------------------------------------------


class TestDiceCoefficient:
    def test_perfect_overlap(self):
        from src.evaluate import dice_coefficient

        mask = np.ones((64, 64), dtype=np.uint8)
        assert dice_coefficient(mask, mask) == pytest.approx(1.0)

    def test_no_overlap(self):
        from src.evaluate import dice_coefficient

        a = np.zeros((64, 64), dtype=np.uint8)
        b = np.ones((64, 64), dtype=np.uint8)
        a[:32] = 1
        b[:32] = 0
        score = dice_coefficient(a, b)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_range(self):
        from src.evaluate import dice_coefficient

        rng = np.random.default_rng(0)
        a = (rng.random((128, 128)) > 0.5).astype(np.uint8)
        b = (rng.random((128, 128)) > 0.5).astype(np.uint8)
        score = dice_coefficient(a, b)
        assert 0.0 <= score <= 1.0


class TestIoU:
    def test_perfect_overlap(self):
        from src.evaluate import iou

        mask = np.ones((64, 64), dtype=np.uint8)
        assert iou(mask, mask) == pytest.approx(1.0)

    def test_range(self):
        from src.evaluate import iou

        rng = np.random.default_rng(1)
        a = (rng.random((64, 64)) > 0.5).astype(np.uint8)
        b = (rng.random((64, 64)) > 0.5).astype(np.uint8)
        score = iou(a, b)
        assert 0.0 <= score <= 1.0


class TestEstimateHcMm:
    def test_returns_float_for_valid_mask(self):
        from src.evaluate import estimate_hc_mm

        mask = np.zeros((256, 384), dtype=np.uint8)
        mask[80:180, 120:260] = 1
        result = estimate_hc_mm(mask, pixel_spacing_mm=0.070)
        assert isinstance(result, float)
        assert result > 0

    def test_returns_none_for_empty_mask(self):
        from src.evaluate import estimate_hc_mm

        mask = np.zeros((256, 384), dtype=np.uint8)
        assert estimate_hc_mm(mask, pixel_spacing_mm=0.070) is None

    def test_scales_with_pixel_spacing(self):
        from src.evaluate import estimate_hc_mm

        mask = np.zeros((256, 384), dtype=np.uint8)
        mask[80:180, 120:260] = 1
        hc1 = estimate_hc_mm(mask, pixel_spacing_mm=0.070)
        hc2 = estimate_hc_mm(mask, pixel_spacing_mm=0.140)
        assert hc2 == pytest.approx(hc1 * 2, rel=1e-5)


class TestHadlockGa:
    def test_output_keys(self):
        from src.evaluate import hadlock_ga

        out = hadlock_ga(250.0)
        assert set(out.keys()) == {"ga_weeks", "ga_str", "trimester"}

    def test_ga_clipped_to_range(self):
        from src.evaluate import hadlock_ga

        assert 10.0 <= hadlock_ga(0.0)["ga_weeks"] <= 42.0
        assert 10.0 <= hadlock_ga(9999.0)["ga_weeks"] <= 42.0

    def test_ga_str_format(self):
        from src.evaluate import hadlock_ga

        ga_str = hadlock_ga(250.0)["ga_str"]
        assert "w" in ga_str and "d" in ga_str

    def test_trimester_labels(self):
        from src.evaluate import hadlock_ga
        # Boundaries: <14w → First; 14–28w → Second; ≥28w → Third
        # HC=80mm → 13.23w (First); HC=200mm → 21.35w (Second); HC=350mm → 40.07w (Third)
        assert hadlock_ga(80.0)["trimester"] == "First trimester (<14w)"
        assert hadlock_ga(200.0)["trimester"] == "Second trimester (14–28w)"
        assert hadlock_ga(350.0)["trimester"] == "Third trimester (≥28w)"


class TestReliabilityScore:
    def test_identical_values_give_one(self):
        from src.evaluate import reliability_score

        assert reliability_score([250.0, 250.0, 250.0]) == pytest.approx(1.0)

    def test_single_value_returns_one(self):
        from src.evaluate import reliability_score

        assert reliability_score([250.0]) == pytest.approx(1.0)

    def test_range(self):
        from src.evaluate import reliability_score

        score = reliability_score([200.0, 220.0, 260.0, 300.0])
        assert 0.0 <= score <= 1.0


class TestEvaluatePredictions:
    def _make_masks(self, n=4):
        rng = np.random.default_rng(42)
        preds = [(rng.random((64, 64)) > 0.4).astype(np.uint8) for _ in range(n)]
        gts = [(rng.random((64, 64)) > 0.4).astype(np.uint8) for _ in range(n)]
        return preds, gts

    def test_output_keys(self):
        from src.evaluate import evaluate_predictions

        preds, gts = self._make_masks()
        out = evaluate_predictions(preds, gts)
        assert {"dice_mean", "dice_std", "iou_mean", "mae_mm", "rmse_mm", "r2"} <= out.keys()

    def test_per_image_spacing_overrides_scalar(self):
        from src.evaluate import evaluate_predictions

        preds, gts = self._make_masks()
        out_scalar = evaluate_predictions(preds, gts, pixel_spacing_mm=0.070)
        out_perimg = evaluate_predictions(preds, gts, pixel_spacings=[0.070] * len(preds))
        assert out_scalar["mae_mm"] == pytest.approx(out_perimg["mae_mm"], rel=1e-5)


# ---------------------------------------------------------------------------
# src.models — architecture forward shapes
# ---------------------------------------------------------------------------


class TestResidualUNetDS:
    @pytest.fixture
    def model(self):
        from src.models.residual_unet import ResidualUNetDS

        return ResidualUNetDS(in_ch=1, base_ch=32).eval()

    def test_inference_output_shape(self, model):
        x = torch.zeros(1, 1, 256, 384)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 256, 384)

    def test_training_returns_three_tensors(self, model):
        model.train()
        x = torch.zeros(2, 1, 256, 384)
        out = model(x)
        assert isinstance(out, tuple) and len(out) == 3
        main, aux2, aux3 = out
        assert main.shape == (2, 1, 256, 384)
        assert aux2.shape == (2, 1, 256, 384)
        assert aux3.shape == (2, 1, 256, 384)

    def test_output_dtype_float32(self, model):
        x = torch.zeros(1, 1, 256, 384)
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32


class TestResidualBlock:
    def test_output_shape_same_channels(self):
        from src.models.residual_unet import ResidualBlock

        block = ResidualBlock(32, 32).eval()
        x = torch.zeros(2, 32, 64, 64)
        with torch.no_grad():
            out = block(x)
        assert out.shape == x.shape

    def test_output_shape_channel_change(self):
        from src.models.residual_unet import ResidualBlock

        block = ResidualBlock(32, 64).eval()
        x = torch.zeros(2, 32, 64, 64)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (2, 64, 64, 64)


class TestBoundaryWeightedDiceLoss:
    def test_perfect_prediction_low_loss(self):
        from src.models.residual_unet import BoundaryWeightedDiceLoss

        loss_fn = BoundaryWeightedDiceLoss()
        mask = torch.ones(2, 1, 64, 64)
        loss = loss_fn(mask, mask)
        assert loss.item() < 0.01

    def test_accepts_4d_target(self):
        from src.models.residual_unet import BoundaryWeightedDiceLoss

        loss_fn = BoundaryWeightedDiceLoss()
        pred = torch.rand(2, 1, 64, 64)
        target = (torch.rand(2, 1, 64, 64) > 0.5).float()
        # Must not raise RuntimeError (was crashing before the unsqueeze fix)
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

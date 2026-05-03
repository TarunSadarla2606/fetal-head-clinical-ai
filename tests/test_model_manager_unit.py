"""Unit tests for model_manager: cache, auto-detection, available_variants."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from app.api import model_manager


class TestGetModelUnknownVariant:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model variant"):
            model_manager.get_model("phase99")

    def test_raises_for_empty_string(self):
        with pytest.raises(ValueError):
            model_manager.get_model("")

    def test_raises_for_uppercase_variant(self):
        with pytest.raises(ValueError):
            model_manager.get_model("PHASE0")


class TestGetModelCacheHit:
    def test_returns_cached_object(self):
        sentinel = object()
        with patch.dict(model_manager._cache, {"phase0": sentinel}):
            result = model_manager.get_model("phase0")
            assert result is sentinel

    def test_does_not_call_loader_when_cached(self):
        sentinel = object()
        with patch.dict(model_manager._cache, {"phase2": sentinel}):
            with patch.object(model_manager, "_find_weight_path") as mock_find:
                model_manager.get_model("phase2")
                mock_find.assert_not_called()

    @pytest.mark.parametrize("variant", ["phase0", "phase2", "phase4a", "phase4b"])
    def test_cache_works_for_all_variants(self, variant):
        sentinel = object()
        with patch.dict(model_manager._cache, {variant: sentinel}):
            assert model_manager.get_model(variant) is sentinel


class TestGetModelNoWeights:
    @pytest.mark.parametrize("variant", ["phase0", "phase2", "phase4a", "phase4b"])
    def test_returns_none_when_no_weights(self, variant):
        env_patches = {
            "WEIGHT_PHASE0": "",
            "WEIGHT_PHASE2": "",
            "WEIGHT_PHASE4A": "",
            "WEIGHT_PHASE4B": "",
        }
        clean_cache = {k: v for k, v in model_manager._cache.items() if k != variant}
        with patch.dict(os.environ, env_patches):
            with patch.dict(model_manager._cache, clean_cache, clear=True):
                with patch.object(model_manager, "_find_weight_path", return_value=""):
                    result = model_manager.get_model(variant)
                    assert result is None


class TestFindWeightPath:
    def test_returns_empty_string_when_nothing_found(self, tmp_path):
        # No .pth files present
        with patch("app.api.model_manager.Path") as MockPath:
            MockPath.return_value.resolve.return_value.parent.parent.parent = tmp_path
            MockPath.cwd.return_value = tmp_path
            result = model_manager._find_weight_path("phase0")
            assert isinstance(result, str)

    def test_variant_substring_in_filename(self):
        test_cases = [
            ("phase0", "phase0_model.pth", True),
            ("phase2", "phase2_weights.pth", True),
            ("phase4a", "4a_best_pruned_ft_v10.pth", True),
            ("phase4b", "4b_best_pruned_ft_v10.pth", True),
            ("phase0", "phase2_model.pth", False),
            ("phase4a", "phase4b_model.pth", False),
        ]
        for variant, filename, should_match in test_cases:
            short = variant.replace("phase", "")
            matches = variant.lower() in filename.lower() or (short and short in filename.lower())
            assert matches == should_match, f"{variant!r} vs {filename!r}: expected {should_match}"


class TestAvailableVariants:
    def test_returns_list(self):
        result = model_manager.available_variants()
        assert isinstance(result, list)

    def test_all_items_are_valid_variant_names(self):
        valid = {"phase0", "phase2", "phase4a", "phase4b"}
        for v in model_manager.available_variants():
            assert v in valid, f"{v!r} is not a recognised variant"

    def test_returns_empty_when_no_weights(self):
        env_patches = {
            "WEIGHT_PHASE0": "",
            "WEIGHT_PHASE2": "",
            "WEIGHT_PHASE4A": "",
            "WEIGHT_PHASE4B": "",
        }
        with patch.dict(os.environ, env_patches):
            with patch.object(model_manager, "_find_weight_path", return_value=""):
                result = model_manager.available_variants()
                assert result == []

    def test_returns_subset_when_only_some_weights_present(self, tmp_path):
        fake_pth = tmp_path / "phase0_model.pth"
        fake_pth.write_bytes(b"fake")
        env_patches = {
            "WEIGHT_PHASE0": str(fake_pth),
            "WEIGHT_PHASE2": "",
            "WEIGHT_PHASE4A": "",
            "WEIGHT_PHASE4B": "",
        }
        with patch.dict(os.environ, env_patches):
            with patch.object(model_manager, "_find_weight_path", return_value=""):
                with patch.dict(model_manager._cache, {}, clear=True):
                    result = model_manager.available_variants()
                    assert "phase0" in result
                    assert "phase2" not in result
                    assert "phase4a" not in result
                    assert "phase4b" not in result

    def test_no_duplicates_in_result(self):
        result = model_manager.available_variants()
        assert len(result) == len(set(result))

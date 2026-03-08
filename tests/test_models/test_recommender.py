"""Tests for model recommendation logic."""

from mita.models.hardware import GPUInfo, HardwareInfo
from mita.models.recommender import recommend_models


def _make_hw(
    ram_gb: float = 16,
    vram_gb: float | None = None,
    apple_silicon: bool = False,
    unified_memory: bool = False,
) -> HardwareInfo:
    """Helper to create HardwareInfo for tests."""
    gpus = []
    if vram_gb is not None:
        vendor = "apple" if apple_silicon else "nvidia"
        gpus = [GPUInfo(name="Test GPU", vram_gb=vram_gb, vendor=vendor)]

    return HardwareInfo(
        ram_gb=ram_gb,
        cpu_cores=8,
        cpu_name="Test CPU",
        gpus=gpus,
        os="darwin" if apple_silicon else "linux",
        apple_silicon=apple_silicon,
        unified_memory=unified_memory,
    )


class TestRecommendModels:
    def test_small_machine_gets_small_models(self) -> None:
        """A 4GB machine should only get very small models."""
        hw = _make_hw(ram_gb=4)
        recs = recommend_models(hw)
        # Should not recommend 32B models
        for rec in recs:
            assert rec.model.min_vram_gb <= hw.available_vram_gb

    def test_large_machine_gets_more_models(self) -> None:
        """A 64GB machine should get more recommendations than a 4GB machine."""
        hw_small = _make_hw(ram_gb=4)
        hw_large = _make_hw(ram_gb=64)
        recs_small = recommend_models(hw_small)
        recs_large = recommend_models(hw_large)
        assert len(recs_large) > len(recs_small)

    def test_sorted_by_fit_score(self) -> None:
        """Results should be sorted by fit_score descending."""
        hw = _make_hw(ram_gb=32)
        recs = recommend_models(hw)
        assert len(recs) > 1
        for i in range(len(recs) - 1):
            assert recs[i].fit_score >= recs[i + 1].fit_score

    def test_apple_silicon_unified_memory(self) -> None:
        """Apple Silicon should use unified memory for VRAM calculation."""
        hw = _make_hw(ram_gb=32, vram_gb=32, apple_silicon=True, unified_memory=True)
        recs = recommend_models(hw)
        # 32GB unified memory = 24GB effective VRAM, should fit several models
        assert len(recs) > 3

    def test_discrete_gpu(self) -> None:
        """Discrete GPU VRAM should be used for filtering."""
        hw = _make_hw(ram_gb=16, vram_gb=24)  # 24GB GPU
        recs = recommend_models(hw)
        # 24GB VRAM should fit the 32B models
        large_models = [r for r in recs if r.model.param_count in ("32B", "34B")]
        assert len(large_models) > 0

    def test_tool_call_noted(self) -> None:
        """Recommendations should note tool call support."""
        hw = _make_hw(ram_gb=16)
        recs = recommend_models(hw)
        for rec in recs:
            if rec.model.tool_call_support:
                assert "tool call" in rec.notes.lower()

    def test_fit_scores_are_valid(self) -> None:
        """All fit scores should be between 0 and 1."""
        hw = _make_hw(ram_gb=32)
        recs = recommend_models(hw)
        for rec in recs:
            assert 0.0 <= rec.fit_score <= 1.0

    def test_no_gpu_cpu_only(self) -> None:
        """No GPU should still return results using RAM-based inference."""
        hw = _make_hw(ram_gb=16)
        recs = recommend_models(hw)
        assert len(recs) > 0

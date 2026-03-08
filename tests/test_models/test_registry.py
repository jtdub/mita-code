"""Tests for the model registry."""

from mita.models.registry import (
    find_model,
    get_coding_models,
    get_embedding_models,
    get_registry,
)


class TestRegistry:
    def test_registry_not_empty(self) -> None:
        assert len(get_registry()) > 0

    def test_find_model_exists(self) -> None:
        model = find_model("qwen2.5-coder:7b")
        assert model is not None
        assert model.family == "qwen"
        assert model.param_count == "7B"

    def test_find_model_not_found(self) -> None:
        assert find_model("nonexistent:99b") is None

    def test_coding_models_exclude_embeddings(self) -> None:
        coding = get_coding_models()
        for m in coding:
            assert "embedding" not in m.tags

    def test_embedding_models(self) -> None:
        embeds = get_embedding_models()
        assert len(embeds) > 0
        for m in embeds:
            assert "embedding" in m.tags

    def test_all_models_have_required_fields(self) -> None:
        for m in get_registry():
            assert m.name
            assert m.family
            assert m.param_count
            assert m.min_ram_gb > 0
            assert m.min_vram_gb > 0
            assert m.context_window > 0

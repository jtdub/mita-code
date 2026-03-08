"""Curated registry of coding-focused models for Ollama."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelCard(BaseModel):
    """Metadata for a coding-focused model in the curated registry."""

    name: str  # Ollama model tag, e.g. "qwen2.5-coder:7b"
    family: str  # "qwen", "codellama", "deepseek", etc.
    param_count: str  # "7B", "14B", "32B"
    min_ram_gb: float
    min_vram_gb: float
    context_window: int
    quantization: str = "Q4_K_M"
    tool_call_support: bool = True
    description: str = ""
    tags: list[str] = Field(default_factory=list)


# Curated list of coding models that work well with Ollama
CODING_MODELS: list[ModelCard] = [
    # ── Small (< 4GB) ────────────────────────────────────────────
    ModelCard(
        name="qwen2.5-coder:1.5b",
        family="qwen",
        param_count="1.5B",
        min_ram_gb=2,
        min_vram_gb=2,
        context_window=32768,
        description="Tiny but capable code model. Good for quick completions.",
        tags=["code", "fast", "small"],
        tool_call_support=False,
    ),
    ModelCard(
        name="qwen2.5-coder:3b",
        family="qwen",
        param_count="3B",
        min_ram_gb=3,
        min_vram_gb=3,
        context_window=32768,
        description="Small code model. Good balance of speed and quality.",
        tags=["code", "fast"],
        tool_call_support=False,
    ),
    ModelCard(
        name="deepseek-coder-v2:lite",
        family="deepseek",
        param_count="2.4B",
        min_ram_gb=3,
        min_vram_gb=2,
        context_window=16384,
        description="Lightweight DeepSeek coder variant.",
        tags=["code", "fast", "small"],
        tool_call_support=False,
    ),
    # ── Medium (4–8GB) ───────────────────────────────────────────
    ModelCard(
        name="qwen2.5-coder:7b",
        family="qwen",
        param_count="7B",
        min_ram_gb=6,
        min_vram_gb=5,
        context_window=32768,
        description="Strong code model. Best default for 8GB+ machines.",
        tags=["code", "instruct", "recommended"],
    ),
    ModelCard(
        name="codellama:7b",
        family="codellama",
        param_count="7B",
        min_ram_gb=6,
        min_vram_gb=5,
        context_window=16384,
        description="Meta's Code Llama. Solid general-purpose code model.",
        tags=["code", "instruct"],
    ),
    ModelCard(
        name="deepseek-coder-v2:16b",
        family="deepseek",
        param_count="16B",
        min_ram_gb=10,
        min_vram_gb=10,
        context_window=65536,
        description="DeepSeek Coder V2. Strong MoE architecture.",
        tags=["code", "instruct"],
    ),
    # ── Large (8–16GB) ───────────────────────────────────────────
    ModelCard(
        name="qwen2.5-coder:14b",
        family="qwen",
        param_count="14B",
        min_ram_gb=10,
        min_vram_gb=10,
        context_window=32768,
        description="Excellent code model. Strong tool call support.",
        tags=["code", "instruct", "recommended"],
    ),
    ModelCard(
        name="codellama:13b",
        family="codellama",
        param_count="13B",
        min_ram_gb=10,
        min_vram_gb=10,
        context_window=16384,
        description="Larger Code Llama with better reasoning.",
        tags=["code", "instruct"],
    ),
    # ── XL (16–32GB) ─────────────────────────────────────────────
    ModelCard(
        name="qwen2.5-coder:32b",
        family="qwen",
        param_count="32B",
        min_ram_gb=20,
        min_vram_gb=20,
        context_window=32768,
        description="Top-tier local code model. Excellent tool call reliability.",
        tags=["code", "instruct", "recommended"],
    ),
    ModelCard(
        name="codellama:34b",
        family="codellama",
        param_count="34B",
        min_ram_gb=22,
        min_vram_gb=22,
        context_window=16384,
        description="Largest Code Llama. Near-cloud quality.",
        tags=["code", "instruct"],
    ),
    # ── Embedding models ─────────────────────────────────────────
    ModelCard(
        name="nomic-embed-text",
        family="nomic",
        param_count="137M",
        min_ram_gb=1,
        min_vram_gb=1,
        context_window=8192,
        description="Default embedding model for codebase indexing.",
        tags=["embedding"],
        tool_call_support=False,
    ),
]


def get_registry() -> list[ModelCard]:
    """Return the full curated model registry."""
    return list(CODING_MODELS)


def find_model(name: str) -> ModelCard | None:
    """Find a model in the registry by name (exact match)."""
    for model in CODING_MODELS:
        if model.name == name:
            return model
    return None


def get_coding_models() -> list[ModelCard]:
    """Return only coding models (exclude embedding models)."""
    return [m for m in CODING_MODELS if "embedding" not in m.tags]


def get_embedding_models() -> list[ModelCard]:
    """Return only embedding models."""
    return [m for m in CODING_MODELS if "embedding" in m.tags]

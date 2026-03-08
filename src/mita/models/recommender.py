"""Filter and rank models by hardware capability."""

from __future__ import annotations

from pydantic import BaseModel

from mita.models.hardware import HardwareInfo
from mita.models.registry import ModelCard, get_coding_models


class ModelRecommendation(BaseModel):
    """A model recommendation with a fit score and notes."""

    model: ModelCard
    fit_score: float  # 0.0–1.0
    notes: str


def recommend_models(hw: HardwareInfo) -> list[ModelRecommendation]:
    """Recommend coding models that fit the given hardware.

    Returns a list sorted by fit_score descending (best fit first).
    Only includes models that can actually run on the hardware.
    """
    available_vram = hw.available_vram_gb
    recommendations: list[ModelRecommendation] = []

    for model in get_coding_models():
        if model.min_vram_gb <= 0:
            continue
        if model.min_vram_gb > available_vram:
            continue
        if model.min_ram_gb > hw.ram_gb:
            continue

        fit_score, notes = _score_model(model, hw, available_vram)
        recommendations.append(ModelRecommendation(model=model, fit_score=fit_score, notes=notes))

    recommendations.sort(key=lambda r: r.fit_score, reverse=True)
    return recommendations


def _score_model(model: ModelCard, hw: HardwareInfo, available_vram: float) -> tuple[float, str]:
    """Score a model's fit for the hardware. Returns (score, notes)."""
    notes_parts: list[str] = []

    # Base score: how much headroom do we have?
    # Ratio of available VRAM to required VRAM
    headroom_ratio = available_vram / model.min_vram_gb
    if headroom_ratio >= 2.0:
        vram_score = 1.0
        notes_parts.append(f"Plenty of headroom ({available_vram:.0f}GB available)")
    elif headroom_ratio >= 1.5:
        vram_score = 0.9
        notes_parts.append(f"Comfortable fit ({available_vram:.0f}GB available)")
    elif headroom_ratio >= 1.2:
        vram_score = 0.7
        notes_parts.append(f"Fits with some room ({available_vram:.0f}GB available)")
    else:
        vram_score = 0.5
        notes_parts.append(f"Tight fit — uses ~{model.min_vram_gb:.0f}GB of {available_vram:.0f}GB")

    # Bonus for tool call support (important for agentic use)
    tool_bonus = 0.1 if model.tool_call_support else 0.0

    # Bonus for being a recommended model
    recommended_bonus = 0.05 if "recommended" in model.tags else 0.0

    # Penalty for Apple Silicon running very large models (thermal throttling)
    thermal_penalty = 0.0
    if hw.apple_silicon and model.min_vram_gb > hw.ram_gb * 0.5:
        thermal_penalty = 0.1
        notes_parts.append("May cause thermal throttling on Apple Silicon")

    # Context window bonus (larger is better for agentic coding)
    ctx_bonus = min(0.05, model.context_window / 1_000_000)

    score = min(1.0, vram_score + tool_bonus + recommended_bonus + ctx_bonus - thermal_penalty)

    if model.tool_call_support:
        notes_parts.append("Supports tool calls")
    else:
        notes_parts.append("No tool call support — will use JSON fallback")

    if hw.unified_memory:
        notes_parts.append("Unified memory — shared with system")

    return round(score, 2), ". ".join(notes_parts)

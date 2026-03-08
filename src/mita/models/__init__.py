"""Model management: hardware detection, registry, recommendations, Ollama client."""

from mita.models.hardware import HardwareInfo, detect_hardware
from mita.models.recommender import ModelRecommendation, recommend_models
from mita.models.registry import ModelCard

__all__ = [
    "HardwareInfo",
    "ModelCard",
    "ModelRecommendation",
    "detect_hardware",
    "recommend_models",
]

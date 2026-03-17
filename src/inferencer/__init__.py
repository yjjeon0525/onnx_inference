# src/inferencer/__init__.py
from src.config import ModelConfig
from src.inferencer.base import BaseInferencer
from src.inferencer.yolov8 import YOLOv8Inferencer

INFERENCER_REGISTRY = {
    "yolov8": YOLOv8Inferencer,
}


def create_inferencer(model_path: str, model_config: ModelConfig) -> BaseInferencer:
    cls = INFERENCER_REGISTRY.get(model_config.type)
    if cls is None:
        raise ValueError(
            f"Unknown inferencer type: '{model_config.type}'. "
            f"Available: {list(INFERENCER_REGISTRY.keys())}"
        )
    return cls(model_path, model_config)

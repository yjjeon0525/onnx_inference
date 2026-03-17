# src/inferencer/base.py
from abc import ABC, abstractmethod
import time
import numpy as np
import onnxruntime as ort
from src.config import ModelConfig


class BaseInferencer(ABC):
    def __init__(self, model_path: str, model_config: ModelConfig):
        self.model_config = model_config
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        input_shape = self.session.get_inputs()[0].shape
        self._input_size = (int(input_shape[2]), int(input_shape[3]))

    @abstractmethod
    def postprocess_raw(self, raw_output: list[np.ndarray]) -> np.ndarray:
        pass

    def infer(self, tensor: np.ndarray) -> tuple[list[np.ndarray], float]:
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return outputs, elapsed_ms

    def get_input_size(self) -> tuple[int, int]:
        return self._input_size

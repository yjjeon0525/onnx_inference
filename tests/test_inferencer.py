# tests/test_inferencer.py
import pytest
import numpy as np
from src.inferencer.base import BaseInferencer
from src.config import ModelConfig


class TestBaseInferencerIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseInferencer("fake.onnx", ModelConfig())

    def test_subclass_must_implement_postprocess_raw(self):
        class IncompleteInferencer(BaseInferencer):
            pass
        with pytest.raises(TypeError):
            IncompleteInferencer("fake.onnx", ModelConfig())


from src.inferencer.yolov8 import YOLOv8Inferencer


class TestYOLOv8PostprocessRaw:
    def test_output_shape(self):
        num_boxes = 100
        num_classes = 80
        boxes = np.random.rand(1, 4, num_boxes).astype(np.float32)
        cls_scores = np.random.rand(1, num_classes, num_boxes).astype(np.float32)
        inferencer = YOLOv8Inferencer.__new__(YOLOv8Inferencer)
        result = inferencer.postprocess_raw([boxes, cls_scores])
        assert result.shape == (num_boxes, 6)

    def test_bbox_format_xyxy(self):
        boxes = np.array([[[50.0], [50.0], [20.0], [30.0]]])
        cls_scores = np.array([[[0.9]]])
        inferencer = YOLOv8Inferencer.__new__(YOLOv8Inferencer)
        result = inferencer.postprocess_raw([boxes, cls_scores])
        assert result[0, 0] == pytest.approx(40.0)
        assert result[0, 1] == pytest.approx(35.0)
        assert result[0, 2] == pytest.approx(60.0)
        assert result[0, 3] == pytest.approx(65.0)
        assert result[0, 4] == pytest.approx(0.9)
        assert result[0, 5] == 0

    def test_best_class_selected(self):
        boxes = np.array([[[10.0], [10.0], [5.0], [5.0]]])
        cls_scores = np.array([[[0.1], [0.8], [0.3]]])
        inferencer = YOLOv8Inferencer.__new__(YOLOv8Inferencer)
        result = inferencer.postprocess_raw([boxes, cls_scores])
        assert result[0, 4] == pytest.approx(0.8)
        assert result[0, 5] == 1


from src.inferencer import create_inferencer


class TestInferencerFactory:
    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown inferencer type"):
            create_inferencer("fake.onnx", ModelConfig(type="unknown"))

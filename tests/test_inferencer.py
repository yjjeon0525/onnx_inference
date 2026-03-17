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


class TestYOLOv8GridBuild:
    def test_grid_size_matches_strides(self):
        """Grid for 256x480 input should produce 2520 cells."""
        inf = YOLOv8Inferencer.__new__(YOLOv8Inferencer)
        inf._input_size = (256, 480)
        inf._grid_cache = {}
        grid, strides = inf._build_grid(256, 480)
        # 32*60 + 16*30 + 8*15 = 1920 + 480 + 120 = 2520
        assert grid.shape == (2520, 2)
        assert strides.shape == (2520,)

    def test_grid_stride_values(self):
        """First 1920 cells should be stride 8, next 480 stride 16, last 120 stride 32."""
        inf = YOLOv8Inferencer.__new__(YOLOv8Inferencer)
        inf._input_size = (256, 480)
        inf._grid_cache = {}
        grid, strides = inf._build_grid(256, 480)
        assert np.all(strides[:1920] == 8)
        assert np.all(strides[1920:2400] == 16)
        assert np.all(strides[2400:] == 32)

    def test_grid_is_cached(self):
        inf = YOLOv8Inferencer.__new__(YOLOv8Inferencer)
        inf._input_size = (256, 480)
        inf._grid_cache = {}
        g1, s1 = inf._build_grid(256, 480)
        g2, s2 = inf._build_grid(256, 480)
        assert g1 is g2
        assert s1 is s2


class TestYOLOv8DecodeBoxes:
    def test_single_box_decode(self):
        """A box at grid cell (0,0) stride 8 with offsets [1,1,1,1]
        should decode to: cx=4, cy=4, then x1=4-8=−4, y1=4-8=−4, x2=4+8=12, y2=4+8=12"""
        inf = YOLOv8Inferencer.__new__(YOLOv8Inferencer)
        grid = np.array([[0.0, 0.0]])
        strides = np.array([8.0])
        raw_boxes = np.array([[1.0, 1.0, 1.0, 1.0]])
        result = inf._decode_boxes(raw_boxes, grid, strides)
        # cx = (0 + 0.5) * 8 = 4,  cy = (0 + 0.5) * 8 = 4
        # x1 = 4 - 1*8 = -4,  y1 = 4 - 1*8 = -4
        # x2 = 4 + 1*8 = 12,  y2 = 4 + 1*8 = 12
        assert result[0, 0] == pytest.approx(-4.0)
        assert result[0, 1] == pytest.approx(-4.0)
        assert result[0, 2] == pytest.approx(12.0)
        assert result[0, 3] == pytest.approx(12.0)

    def test_nonzero_grid_cell(self):
        """Grid cell (3, 2) at stride 16."""
        inf = YOLOv8Inferencer.__new__(YOLOv8Inferencer)
        grid = np.array([[3.0, 2.0]])
        strides = np.array([16.0])
        raw_boxes = np.array([[0.5, 0.5, 0.5, 0.5]])
        result = inf._decode_boxes(raw_boxes, grid, strides)
        # cx = (3 + 0.5) * 16 = 56,  cy = (2 + 0.5) * 16 = 40
        # x1 = 56 - 0.5*16 = 48,  y1 = 40 - 0.5*16 = 32
        # x2 = 56 + 0.5*16 = 64,  y2 = 40 + 0.5*16 = 48
        assert result[0, 0] == pytest.approx(48.0)
        assert result[0, 1] == pytest.approx(32.0)
        assert result[0, 2] == pytest.approx(64.0)
        assert result[0, 3] == pytest.approx(48.0)


class TestYOLOv8PostprocessRaw:
    def test_output_shape(self):
        """Full pipeline: cls + boxes -> decoded detections."""
        inf = YOLOv8Inferencer.__new__(YOLOv8Inferencer)
        inf._input_size = (256, 480)
        inf._grid_cache = {}
        num_boxes = 2520
        num_classes = 6
        cls_scores = np.random.rand(1, num_classes, num_boxes).astype(np.float32)
        boxes = np.random.rand(1, 4, num_boxes).astype(np.float32)
        result = inf.postprocess_raw([cls_scores, boxes])
        assert result.shape == (num_boxes, 6)

    def test_best_class_selected(self):
        """Verify argmax class selection works."""
        inf = YOLOv8Inferencer.__new__(YOLOv8Inferencer)
        inf._input_size = (64, 64)  # stride 8: 8x8=64 cells, stride 16: 4x4=16, stride 32: 2x2=4 = 84 total
        inf._grid_cache = {}
        num_boxes = 84
        cls_scores = np.zeros((1, 3, num_boxes), dtype=np.float32)
        cls_scores[0, 1, 0] = 0.95  # box 0 -> class 1
        boxes = np.ones((1, 4, num_boxes), dtype=np.float32)
        result = inf.postprocess_raw([cls_scores, boxes])
        assert result[0, 4] == pytest.approx(0.95)
        assert result[0, 5] == 1


from src.inferencer import create_inferencer


class TestInferencerFactory:
    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown inferencer type"):
            create_inferencer("fake.onnx", ModelConfig(type="unknown"))

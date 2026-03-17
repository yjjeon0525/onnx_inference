# tests/test_postprocessor.py
import numpy as np
import pytest
from src.config import PostprocessConfig
from src.preprocessor import PreprocessMetadata
from src.postprocessor import Postprocessor
import cv2


def make_metadata(
    crop_origin=(0, 0), crop_size=(640, 480), input_size=(640, 640)
) -> PreprocessMetadata:
    return PreprocessMetadata(
        original_shape=(480, 640),
        crop_origin=crop_origin,
        crop_size=crop_size,
        resize_scale=(crop_size[0] / input_size[1], crop_size[1] / input_size[0]),
        input_size=input_size,
        resize_method=cv2.INTER_LINEAR,
    )


class TestThresholdFiltering:
    def test_global_threshold(self):
        config = PostprocessConfig(conf_threshold=0.5)
        pp = Postprocessor(config)
        dets = np.array([
            [0, 0, 10, 10, 0.8, 0],
            [0, 0, 10, 10, 0.3, 0],
            [0, 0, 10, 10, 0.6, 1],
        ])
        result = pp.filter_by_threshold(dets)
        assert len(result) == 2
        assert all(result[:, 4] >= 0.5)

    def test_per_class_threshold(self):
        config = PostprocessConfig(
            conf_threshold=0.5, class_thresholds={0: 0.9}
        )
        pp = Postprocessor(config)
        dets = np.array([
            [0, 0, 10, 10, 0.8, 0],   # class 0, below 0.9 -> filtered
            [0, 0, 10, 10, 0.95, 0],   # class 0, above 0.9 -> kept
            [0, 0, 10, 10, 0.6, 1],    # class 1, above 0.5 -> kept
        ])
        result = pp.filter_by_threshold(dets)
        assert len(result) == 2

    def test_empty_detections(self):
        config = PostprocessConfig(conf_threshold=0.5)
        pp = Postprocessor(config)
        dets = np.empty((0, 6))
        result = pp.filter_by_threshold(dets)
        assert len(result) == 0


class TestNMS:
    def test_nms_removes_overlapping(self):
        config = PostprocessConfig(iou_threshold=0.5)
        pp = Postprocessor(config)
        dets = np.array([
            [10, 10, 50, 50, 0.9, 0],
            [12, 12, 52, 52, 0.8, 0],  # overlaps heavily
            [200, 200, 250, 250, 0.7, 0],  # separate
        ])
        result = pp.apply_nms(dets)
        assert len(result) == 2

    def test_nms_empty(self):
        config = PostprocessConfig()
        pp = Postprocessor(config)
        dets = np.empty((0, 6))
        result = pp.apply_nms(dets)
        assert len(result) == 0


class TestCoordinateReversion:
    def test_no_crop_reversion(self):
        config = PostprocessConfig()
        pp = Postprocessor(config)
        meta = make_metadata(crop_origin=(0, 0), crop_size=(640, 640), input_size=(640, 640))
        dets = np.array([[100, 100, 200, 200, 0.9, 0]])
        result = pp.revert_coordinates(dets, meta)
        assert result[0, 0] == pytest.approx(100.0)
        assert result[0, 1] == pytest.approx(100.0)

    def test_crop_offset_applied(self):
        config = PostprocessConfig()
        pp = Postprocessor(config)
        meta = make_metadata(
            crop_origin=(100, 50), crop_size=(320, 240), input_size=(640, 640)
        )
        dets = np.array([[100, 100, 200, 200, 0.9, 0]])
        result = pp.revert_coordinates(dets, meta)
        assert result[0, 0] == pytest.approx(100 * 0.5 + 100)
        assert result[0, 1] == pytest.approx(100 * 0.375 + 50)
        assert result[0, 2] == pytest.approx(200 * 0.5 + 100)
        assert result[0, 3] == pytest.approx(200 * 0.375 + 50)


class TestPostprocessorPipeline:
    def test_full_pipeline(self):
        config = PostprocessConfig(conf_threshold=0.3, iou_threshold=0.5)
        pp = Postprocessor(config)
        meta = make_metadata()
        dets = np.array([
            [10, 10, 50, 50, 0.9, 0],
            [10, 10, 50, 50, 0.1, 0],  # below threshold
        ])
        result = pp.process(dets, meta, nms_applied=False)
        assert len(result) == 1
        assert result[0, 4] == pytest.approx(0.9)

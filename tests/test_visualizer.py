# tests/test_visualizer.py
import numpy as np
import pytest
from src.visualizer import Visualizer


class TestDrawDetections:
    def test_returns_image_same_shape(self):
        vis = Visualizer()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = np.array([[10, 10, 100, 100, 0.9, 0]])
        result = vis.draw_detections(image, dets)
        assert result.shape == image.shape

    def test_empty_detections(self):
        vis = Visualizer()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = np.empty((0, 6))
        result = vis.draw_detections(image, dets)
        assert np.array_equal(result, image)

    def test_with_class_names(self):
        vis = Visualizer(class_names=["person", "car"])
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = np.array([[10, 10, 100, 100, 0.9, 0]])
        result = vis.draw_detections(image, dets)
        assert result.shape == image.shape


class TestComparisonModes:
    def test_overlay_returns_same_shape(self):
        vis = Visualizer(initial_mode="overlay")
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = {"model_a": np.array([[10, 10, 100, 100, 0.9, 0]]), "model_b": np.array([[50, 50, 150, 150, 0.8, 1]])}
        output = vis.render_comparison(image, results)
        assert output.shape == image.shape

    def test_side_by_side_doubles_width(self):
        vis = Visualizer(initial_mode="side_by_side")
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = {"model_a": np.array([[10, 10, 100, 100, 0.9, 0]]), "model_b": np.array([[50, 50, 150, 150, 0.8, 1]])}
        output = vis.render_comparison(image, results)
        assert output.shape == (480, 1280, 3)

    def test_toggle_mode(self):
        vis = Visualizer(initial_mode="overlay")
        assert vis.comparison_mode == "overlay"
        vis.toggle_mode()
        assert vis.comparison_mode == "side_by_side"
        vis.toggle_mode()
        assert vis.comparison_mode == "overlay"

# tests/test_comparator.py
import numpy as np
import pytest
from src.config import ComparisonConfig
from src.comparator import Comparator


class TestCosineSimilarity:
    def test_identical_outputs(self):
        comp = Comparator(ComparisonConfig())
        raw = [np.random.rand(1, 4, 100).astype(np.float32), np.random.rand(1, 80, 100).astype(np.float32)]
        result = comp.cosine_similarity(raw, raw)
        assert result["box_similarity"] == pytest.approx(1.0, abs=1e-5)
        assert result["cls_similarity"] == pytest.approx(1.0, abs=1e-5)

    def test_different_outputs(self):
        comp = Comparator(ComparisonConfig())
        raw_a = [np.ones((1, 4, 100), dtype=np.float32), np.ones((1, 80, 100), dtype=np.float32)]
        raw_b = [np.full((1, 4, 100), -1, dtype=np.float32), np.full((1, 80, 100), -1, dtype=np.float32)]
        result = comp.cosine_similarity(raw_a, raw_b)
        assert result["box_similarity"] == pytest.approx(-1.0, abs=1e-5)

    def test_different_shapes_truncates(self):
        comp = Comparator(ComparisonConfig())
        raw_a = [np.ones((1, 4, 100), dtype=np.float32), np.ones((1, 80, 100), dtype=np.float32)]
        raw_b = [np.ones((1, 4, 50), dtype=np.float32), np.ones((1, 80, 50), dtype=np.float32)]
        result = comp.cosine_similarity(raw_a, raw_b)
        assert result["box_similarity"] == pytest.approx(1.0, abs=1e-5)


class TestComputeMetrics:
    def test_identical_detections(self):
        comp = Comparator(ComparisonConfig())
        dets = np.array([[10, 10, 50, 50, 0.9, 0]])
        result = comp.compute_metrics(dets, dets)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)

    def test_no_detections_b(self):
        comp = Comparator(ComparisonConfig())
        det_a = np.array([[10, 10, 50, 50, 0.9, 0]])
        det_b = np.empty((0, 6))
        result = comp.compute_metrics(det_a, det_b)
        assert result["recall"] == pytest.approx(0.0)

    def test_no_detections_a(self):
        comp = Comparator(ComparisonConfig())
        det_a = np.empty((0, 6))
        det_b = np.array([[10, 10, 50, 50, 0.9, 0]])
        result = comp.compute_metrics(det_a, det_b)
        assert result["precision"] == pytest.approx(0.0)


class TestTimingComparison:
    def test_timing_stats(self):
        comp = Comparator(ComparisonConfig())
        times = {"model_a": [10.0, 12.0, 11.0], "model_b": [20.0, 22.0, 21.0]}
        result = comp.timing_comparison(times)
        assert result["model_a"]["mean_ms"] == pytest.approx(11.0)
        assert result["model_b"]["mean_ms"] == pytest.approx(21.0)
        assert "std_ms" in result["model_a"]
        assert "min_ms" in result["model_a"]
        assert "max_ms" in result["model_a"]

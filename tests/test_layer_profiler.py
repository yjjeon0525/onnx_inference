# tests/test_layer_profiler.py
import pytest
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from src.layer_profiler import LayerProfiler


def _make_two_layer_model(w1: np.ndarray, w2: np.ndarray) -> onnx.ModelProto:
    """Create a model: input -> Conv(w1) -> Relu -> Conv(w2) -> output"""
    init1 = numpy_helper.from_array(w1, "conv1_weight")
    init2 = numpy_helper.from_array(w2, "conv2_weight")

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)

    n1 = helper.make_node("Conv", ["input", "conv1_weight"], ["conv1_out"], name="conv1",
                          kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    n2 = helper.make_node("Relu", ["conv1_out"], ["relu_out"], name="relu1")
    n3 = helper.make_node("Conv", ["relu_out", "conv2_weight"], ["output"], name="conv2",
                          kernel_shape=[3, 3], pads=[1, 1, 1, 1])

    graph = helper.make_graph([n1, n2, n3], "test", [inp], [out], initializer=[init1, init2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model = onnx.shape_inference.infer_shapes(model)
    return model


class TestExtractIntermediateNames:
    def test_finds_all_intermediate_tensors(self, tmp_path):
        w1 = np.random.rand(8, 3, 3, 3).astype(np.float32)
        w2 = np.random.rand(4, 8, 3, 3).astype(np.float32)
        model = _make_two_layer_model(w1, w2)
        path = str(tmp_path / "model.onnx")
        onnx.save(model, path)
        profiler = LayerProfiler(path, path)
        names = profiler._extract_intermediate_names(model)
        assert "conv1_out" in names
        assert "relu_out" in names


class TestProfile:
    def test_identical_models_perfect_similarity(self, tmp_path):
        w1 = np.random.rand(8, 3, 3, 3).astype(np.float32)
        w2 = np.random.rand(4, 8, 3, 3).astype(np.float32)
        model = _make_two_layer_model(w1, w2)
        path_a = str(tmp_path / "a.onnx")
        path_b = str(tmp_path / "b.onnx")
        onnx.save(model, path_a)
        onnx.save(model, path_b)
        profiler = LayerProfiler(path_a, path_b)
        inp = np.random.rand(1, 3, 32, 32).astype(np.float32)
        report = profiler.profile(inp)
        for layer in report["layers"]:
            assert layer["cosine_similarity"] == pytest.approx(1.0, abs=1e-4)
            assert layer["l2_distance"] == pytest.approx(0.0, abs=1e-4)

    def test_different_weights_show_divergence(self, tmp_path):
        w1 = np.random.rand(8, 3, 3, 3).astype(np.float32)
        w2 = np.random.rand(4, 8, 3, 3).astype(np.float32)
        model_a = _make_two_layer_model(w1, w2)

        w1_b = w1 + np.random.rand(*w1.shape).astype(np.float32) * 0.5
        model_b = _make_two_layer_model(w1_b, w2)

        path_a = str(tmp_path / "a.onnx")
        path_b = str(tmp_path / "b.onnx")
        onnx.save(model_a, path_a)
        onnx.save(model_b, path_b)
        profiler = LayerProfiler(path_a, path_b)
        inp = np.random.rand(1, 3, 32, 32).astype(np.float32)
        report = profiler.profile(inp)
        # conv1_out should diverge since w1 differs
        conv1_layer = next(l for l in report["layers"] if "conv1" in l["name"])
        assert conv1_layer["cosine_similarity"] < 1.0
        assert conv1_layer["l2_distance"] > 0.0

    def test_report_has_summary(self, tmp_path):
        w1 = np.random.rand(8, 3, 3, 3).astype(np.float32)
        w2 = np.random.rand(4, 8, 3, 3).astype(np.float32)
        model = _make_two_layer_model(w1, w2)
        path = str(tmp_path / "model.onnx")
        onnx.save(model, path)
        profiler = LayerProfiler(path, path)
        inp = np.random.rand(1, 3, 32, 32).astype(np.float32)
        report = profiler.profile(inp)
        assert "summary" in report
        assert "total_compared" in report["summary"]
        assert "most_divergent" in report["summary"]
        assert "least_divergent" in report["summary"]

    def test_report_layers_sorted_by_divergence(self, tmp_path):
        w1 = np.random.rand(8, 3, 3, 3).astype(np.float32)
        w2 = np.random.rand(4, 8, 3, 3).astype(np.float32)
        w1_b = w1 + np.random.rand(*w1.shape).astype(np.float32) * 0.5
        model_a = _make_two_layer_model(w1, w2)
        model_b = _make_two_layer_model(w1_b, w2)
        path_a = str(tmp_path / "a.onnx")
        path_b = str(tmp_path / "b.onnx")
        onnx.save(model_a, path_a)
        onnx.save(model_b, path_b)
        profiler = LayerProfiler(path_a, path_b)
        inp = np.random.rand(1, 3, 32, 32).astype(np.float32)
        report = profiler.profile(inp)
        cosines = [l["cosine_similarity"] for l in report["layers"]]
        assert cosines == sorted(cosines)  # ascending = most divergent first


class TestSaveReport:
    def test_save_json(self, tmp_path):
        w1 = np.random.rand(8, 3, 3, 3).astype(np.float32)
        w2 = np.random.rand(4, 8, 3, 3).astype(np.float32)
        model = _make_two_layer_model(w1, w2)
        path = str(tmp_path / "model.onnx")
        onnx.save(model, path)
        profiler = LayerProfiler(path, path)
        inp = np.random.rand(1, 3, 32, 32).astype(np.float32)
        report = profiler.profile(inp)
        out_path = str(tmp_path / "report.json")
        profiler.save_report(report, out_path)
        import json, os
        assert os.path.isfile(out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert "layers" in data
        assert "summary" in data

# tests/test_model_analyzer.py
import pytest
import numpy as np
import tempfile
import os

import onnx
from onnx import helper, TensorProto, numpy_helper

from src.model_analyzer import OnnxModelAnalyzer


def _make_simple_model(weights: dict[str, np.ndarray], op_type: str = "Conv") -> onnx.ModelProto:
    """Create a minimal ONNX model with given initializer weights."""
    initializers = []
    for name, arr in weights.items():
        initializers.append(numpy_helper.from_array(arr, name=name))

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 62, 62])

    node = helper.make_node(
        op_type,
        inputs=["input"] + list(weights.keys()),
        outputs=["output"],
        name="conv1",
    )

    graph = helper.make_graph([node], "test_graph", [input_tensor], [output_tensor], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model


def _save_model(model: onnx.ModelProto, path: str):
    onnx.save(model, path)


class TestQuickStructureCheck:
    def test_identical_models(self, tmp_path):
        w = {"weight": np.random.rand(16, 3, 3, 3).astype(np.float32)}
        model = _make_simple_model(w)
        path_a = str(tmp_path / "a.onnx")
        path_b = str(tmp_path / "b.onnx")
        _save_model(model, path_a)
        _save_model(model, path_b)
        analyzer = OnnxModelAnalyzer(path_a, path_b)
        result = analyzer.quick_structure_check()
        assert result["identical"] is True
        assert result["op_count_a"] == result["op_count_b"]

    def test_different_op_count(self, tmp_path):
        w = {"weight": np.random.rand(16, 3, 3, 3).astype(np.float32)}
        model_a = _make_simple_model(w, "Conv")

        # Model B with extra node
        init = [numpy_helper.from_array(w["weight"], "weight")]
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])
        mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, None)
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 62, 62])
        n1 = helper.make_node("Conv", ["input", "weight"], ["mid"], name="conv1")
        n2 = helper.make_node("Relu", ["mid"], ["output"], name="relu1")
        graph = helper.make_graph([n1, n2], "test", [inp], [out], initializer=init)
        model_b = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        path_a = str(tmp_path / "a.onnx")
        path_b = str(tmp_path / "b.onnx")
        _save_model(model_a, path_a)
        _save_model(model_b, path_b)
        analyzer = OnnxModelAnalyzer(path_a, path_b)
        result = analyzer.quick_structure_check()
        assert result["identical"] is False
        assert result["op_count_a"] == 1
        assert result["op_count_b"] == 2


class TestDetailedStructureDiff:
    def test_added_node(self, tmp_path):
        w = {"weight": np.random.rand(16, 3, 3, 3).astype(np.float32)}
        model_a = _make_simple_model(w, "Conv")

        init = [numpy_helper.from_array(w["weight"], "weight")]
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 62, 62])
        n1 = helper.make_node("Conv", ["input", "weight"], ["mid"], name="conv1")
        n2 = helper.make_node("Relu", ["mid"], ["output"], name="relu1")
        graph = helper.make_graph([n1, n2], "test", [inp], [out], initializer=init)
        model_b = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        path_a = str(tmp_path / "a.onnx")
        path_b = str(tmp_path / "b.onnx")
        _save_model(model_a, path_a)
        _save_model(model_b, path_b)
        analyzer = OnnxModelAnalyzer(path_a, path_b)
        result = analyzer.detailed_structure_diff()
        assert len(result["matched_nodes"]) == 1
        assert len(result["added_nodes"]) == 1
        assert result["added_nodes"][0]["name"] == "relu1"

    def test_changed_op_type(self, tmp_path):
        w = {"weight": np.random.rand(16, 3, 3, 3).astype(np.float32)}
        model_a = _make_simple_model(w, "Conv")
        model_b = _make_simple_model(w, "ConvInteger")

        path_a = str(tmp_path / "a.onnx")
        path_b = str(tmp_path / "b.onnx")
        _save_model(model_a, path_a)
        _save_model(model_b, path_b)
        analyzer = OnnxModelAnalyzer(path_a, path_b)
        result = analyzer.detailed_structure_diff()
        assert len(result["changed_nodes"]) == 1
        assert result["changed_nodes"][0]["op_type_a"] == "Conv"
        assert result["changed_nodes"][0]["op_type_b"] == "ConvInteger"


class TestCompareWeights:
    def test_identical_weights(self, tmp_path):
        w = {"weight": np.random.rand(16, 3, 3, 3).astype(np.float32)}
        model = _make_simple_model(w)
        path_a = str(tmp_path / "a.onnx")
        path_b = str(tmp_path / "b.onnx")
        _save_model(model, path_a)
        _save_model(model, path_b)
        analyzer = OnnxModelAnalyzer(path_a, path_b)
        result = analyzer.compare_weights()
        assert result["summary"]["matched_count"] == 1
        wr = result["weights"][0]
        assert wr["cosine_similarity"] == pytest.approx(1.0, abs=1e-5)
        assert wr["l2_distance"] == pytest.approx(0.0, abs=1e-5)

    def test_different_weights(self, tmp_path):
        w_a = {"weight": np.ones((16, 3, 3, 3), dtype=np.float32)}
        w_b = {"weight": np.ones((16, 3, 3, 3), dtype=np.float32) * 2.0}
        model_a = _make_simple_model(w_a)
        model_b = _make_simple_model(w_b)
        path_a = str(tmp_path / "a.onnx")
        path_b = str(tmp_path / "b.onnx")
        _save_model(model_a, path_a)
        _save_model(model_b, path_b)
        analyzer = OnnxModelAnalyzer(path_a, path_b)
        result = analyzer.compare_weights()
        wr = result["weights"][0]
        assert wr["cosine_similarity"] == pytest.approx(1.0, abs=1e-5)  # same direction
        assert wr["l2_distance"] > 0
        assert wr["stats"]["mean_diff"] == pytest.approx(1.0, abs=1e-5)

    def test_unmatched_weights(self, tmp_path):
        w_a = {"weight_a": np.random.rand(16, 3, 3, 3).astype(np.float32)}
        w_b = {"weight_b": np.random.rand(16, 3, 3, 3).astype(np.float32)}
        model_a = _make_simple_model(w_a)
        model_b = _make_simple_model(w_b)
        path_a = str(tmp_path / "a.onnx")
        path_b = str(tmp_path / "b.onnx")
        _save_model(model_a, path_a)
        _save_model(model_b, path_b)
        analyzer = OnnxModelAnalyzer(path_a, path_b)
        result = analyzer.compare_weights()
        # Fallback matching by shape should match them
        assert result["summary"]["matched_count"] == 1


class TestAnalyze:
    def test_full_report(self, tmp_path):
        w = {"weight": np.random.rand(16, 3, 3, 3).astype(np.float32)}
        model = _make_simple_model(w)
        path_a = str(tmp_path / "a.onnx")
        path_b = str(tmp_path / "b.onnx")
        _save_model(model, path_a)
        _save_model(model, path_b)
        analyzer = OnnxModelAnalyzer(path_a, path_b)
        report = analyzer.analyze()
        assert "structure" in report
        assert "weights" in report
        assert report["structure"]["quick"]["identical"] is True


class TestPrintAndSave:
    def test_save_json(self, tmp_path):
        w = {"weight": np.random.rand(16, 3, 3, 3).astype(np.float32)}
        model = _make_simple_model(w)
        path_a = str(tmp_path / "a.onnx")
        path_b = str(tmp_path / "b.onnx")
        _save_model(model, path_a)
        _save_model(model, path_b)
        analyzer = OnnxModelAnalyzer(path_a, path_b)
        report = analyzer.analyze()
        out_path = str(tmp_path / "report.json")
        analyzer.save_report(report, out_path)
        assert os.path.isfile(out_path)
        import json
        with open(out_path) as f:
            data = json.load(f)
        assert "structure" in data

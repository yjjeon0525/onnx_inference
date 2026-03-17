# tests/test_config.py
import pytest
import tempfile
import os
import yaml
from src.config import (
    ModelConfig, PreprocessConfig, PostprocessConfig,
    OutputConfig, ComparisonConfig, VideoConfig, Config,
    load_config, apply_cli_overrides,
)


def _write_yaml(data: dict, path: str):
    with open(path, "w") as f:
        yaml.dump(data, f)


def make_minimal_yaml(tmp_path: str) -> str:
    cfg = {
        "models": [
            {"name": "m1", "path": "onnx_files/org.onnx", "type": "yolov8"}
        ],
    }
    p = os.path.join(tmp_path, "cfg.yaml")
    _write_yaml(cfg, p)
    return p


class TestLoadConfig:
    def test_minimal_config_loads(self, tmp_path):
        path = make_minimal_yaml(str(tmp_path))
        config = load_config(path)
        assert len(config.models) == 1
        assert config.models[0].name == "m1"
        assert config.models[0].type == "yolov8"

    def test_defaults_are_applied(self, tmp_path):
        path = make_minimal_yaml(str(tmp_path))
        config = load_config(path)
        assert config.postprocess.conf_threshold == 0.25
        assert config.postprocess.iou_threshold == 0.45
        assert config.preprocess.crop.enabled is False
        assert config.preprocess.resize.method == "INTER_LINEAR"
        assert config.output.display is True

    def test_model_defaults(self, tmp_path):
        path = make_minimal_yaml(str(tmp_path))
        config = load_config(path)
        m = config.models[0]
        assert m.nms_applied is False
        assert m.conf_threshold is None
        assert m.iou_threshold is None
        assert m.class_thresholds is None


class TestCliOverrides:
    def test_dot_notation_override(self):
        data = {"postprocess": {"conf_threshold": 0.25}}
        result = apply_cli_overrides(data, ["--postprocess.conf_threshold", "0.5"])
        assert result["postprocess"]["conf_threshold"] == 0.5

    def test_nested_override(self):
        data = {"preprocess": {"crop": {"enabled": False}}}
        result = apply_cli_overrides(data, ["--preprocess.crop.enabled", "true"])
        assert result["preprocess"]["crop"]["enabled"] is True

    def test_list_override(self):
        data = {"preprocess": {"crop": {"region": [0.0, 0.0, 1.0, 1.0]}}}
        result = apply_cli_overrides(
            data, ["--preprocess.crop.region", "0.1,0.2,0.8,0.7"]
        )
        assert result["preprocess"]["crop"]["region"] == [0.1, 0.2, 0.8, 0.7]

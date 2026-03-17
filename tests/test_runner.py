# tests/test_runner.py
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.config import Config, ModelConfig, PreprocessConfig, PostprocessConfig, OutputConfig, ComparisonConfig, VideoConfig


class TestRunnerInit:
    def test_missing_model_file_raises(self, tmp_path):
        config = Config(
            models=[ModelConfig(name="m", path=str(tmp_path / "nonexistent.onnx"), type="yolov8")],
        )
        from src.runner import Runner
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            Runner(config)

    def test_per_model_threshold_merging(self, tmp_path):
        fake_onnx = str(tmp_path / "fake.onnx")
        open(fake_onnx, "w").close()
        config = Config(
            models=[ModelConfig(
                name="m", path=fake_onnx, type="yolov8",
                conf_threshold=0.8, class_thresholds={0: 0.95},
            )],
            postprocess=PostprocessConfig(conf_threshold=0.25, class_thresholds={1: 0.5}),
        )
        with patch("src.runner.create_inferencer") as mock_create:
            mock_inf = MagicMock()
            mock_inf.get_input_size.return_value = (640, 640)
            mock_create.return_value = mock_inf
            from src.runner import Runner
            runner = Runner(config)
            pp = runner.postprocessors["m"]
            assert pp.conf_threshold == 0.8
            assert pp.class_thresholds[0] == 0.95
            assert pp.class_thresholds[1] == 0.5


class TestRunnerInputDetection:
    def test_unsupported_extension_raises(self, tmp_path):
        fake_onnx = str(tmp_path / "fake.onnx")
        open(fake_onnx, "w").close()
        bad_file = str(tmp_path / "file.xyz")
        open(bad_file, "w").close()
        config = Config(
            models=[ModelConfig(name="m", path=fake_onnx, type="yolov8")],
            output=OutputConfig(display=False, save_video=False, save_json=False),
        )
        with patch("src.runner.create_inferencer") as mock_create:
            mock_inf = MagicMock()
            mock_inf.get_input_size.return_value = (640, 640)
            mock_create.return_value = mock_inf
            from src.runner import Runner
            runner = Runner(config)
            with pytest.raises(ValueError, match="Unsupported file type"):
                runner.run(bad_file)

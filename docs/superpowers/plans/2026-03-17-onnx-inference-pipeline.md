# ONNX Inference Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular ONNX inference pipeline supporting single/multi-model inference on images and videos with configurable preprocessing, YOLOv8 postprocessing, visual comparison, and metrics.

**Architecture:** Modular component design — Config, Preprocessor, BaseInferencer/YOLOv8Inferencer, Postprocessor, Visualizer, Comparator, Runner. Each component has a single responsibility and communicates via well-defined interfaces (PreprocessMetadata, unified detection format `(N, 6)`).

**Tech Stack:** Python 3.13, onnxruntime, opencv-python, numpy, pyyaml

**Spec:** `docs/superpowers/specs/2026-03-17-onnx-inference-pipeline-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `requirements.txt` | Python dependencies |
| `configs/default.yaml` | Default configuration template |
| `src/__init__.py` | Package init |
| `src/config.py` | YAML loading, CLI override merging, config dataclasses |
| `src/preprocessor.py` | PreprocessMetadata dataclass, Preprocessor (crop, resize, normalize, batch dim) |
| `src/inferencer/__init__.py` | Inferencer factory function |
| `src/inferencer/base.py` | BaseInferencer ABC |
| `src/inferencer/yolov8.py` | YOLOv8Inferencer (dual-output parsing) |
| `src/postprocessor.py` | Postprocessor (NMS, thresholds, coordinate reversion) |
| `src/visualizer.py` | Visualizer (draw bboxes, overlay, side-by-side, toggle) |
| `src/comparator.py` | Comparator (cosine similarity, metrics, timing) |
| `src/runner.py` | Runner orchestrator (image/video/directory) |
| `main.py` | CLI entry point (argparse) |
| `tests/test_config.py` | Config tests |
| `tests/test_preprocessor.py` | Preprocessor tests |
| `tests/test_inferencer.py` | Inferencer tests |
| `tests/test_postprocessor.py` | Postprocessor tests |
| `tests/test_visualizer.py` | Visualizer tests |
| `tests/test_comparator.py` | Comparator tests |

---

## Chunk 1: Project Setup + Config

### Task 1: Project scaffolding and dependencies

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/inferencer/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Install dependencies into conda env**

```bash
conda run -n onnx_inf pip install onnxruntime opencv-python numpy pyyaml pytest
```

- [ ] **Step 2: Create requirements.txt**

```
onnxruntime
opencv-python
numpy
pyyaml
pytest
```

- [ ] **Step 3: Create package init files**

`src/__init__.py` — empty file
`src/inferencer/__init__.py` — empty for now (factory added in Task 5)
`tests/__init__.py` — empty file

- [ ] **Step 4: Verify setup**

```bash
conda run -n onnx_inf python -c "import onnxruntime, cv2, numpy, yaml; print('All imports OK')"
```

Expected: `All imports OK`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/__init__.py src/inferencer/__init__.py tests/__init__.py
git commit -m "chore: scaffold project structure and install dependencies"
```

---

### Task 2: Config dataclasses and YAML loading

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for config**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n onnx_inf python -m pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 3: Implement config.py**

```python
# src/config.py
from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class CropConfig:
    enabled: bool = False
    region: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])


@dataclass
class ResizeConfig:
    method: str = "INTER_LINEAR"
    input_size: Optional[list[int]] = None


@dataclass
class PreprocessConfig:
    crop: CropConfig = field(default_factory=CropConfig)
    resize: ResizeConfig = field(default_factory=ResizeConfig)


@dataclass
class PostprocessConfig:
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    class_thresholds: dict[int, float] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = ""
    path: str = ""
    type: str = "yolov8"
    nms_applied: bool = False
    conf_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    class_thresholds: Optional[dict[int, float]] = None


@dataclass
class OutputConfig:
    save_video: bool = True
    save_json: bool = False
    display: bool = True
    output_dir: str = "output/"


@dataclass
class ComparisonConfig:
    mode: str = "overlay"
    metrics: list[str] = field(
        default_factory=lambda: [
            "cosine_similarity", "precision", "recall", "inference_time"
        ]
    )


@dataclass
class VideoConfig:
    codec: str = "mp4v"
    preserve_fps: bool = True


@dataclass
class Config:
    models: list[ModelConfig] = field(default_factory=list)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    class_names: Optional[list[str]] = None
    video: VideoConfig = field(default_factory=VideoConfig)


def _parse_value(value: str):
    """Parse a CLI string value into the appropriate Python type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null" or value.lower() == "none":
        return None
    if "," in value:
        return [_parse_value(v.strip()) for v in value.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_cli_overrides(data: dict, cli_args: list[str]) -> dict:
    """Apply --dot.notation overrides to a nested dict.
    Note: Cannot set dict values (e.g. class_thresholds) via CLI. Use YAML for those."""
    i = 0
    while i < len(cli_args):
        arg = cli_args[i]
        if arg.startswith("--") and "." in arg:
            key_path = arg[2:].split(".")
            value = cli_args[i + 1]
            parsed = _parse_value(value)
            d = data
            for k in key_path[:-1]:
                d = d.setdefault(k, {})
            d[key_path[-1]] = parsed
            i += 2
        else:
            i += 1
    return data


def _build_model_config(raw: dict) -> ModelConfig:
    return ModelConfig(
        name=raw.get("name", ""),
        path=raw.get("path", ""),
        type=raw.get("type", "yolov8"),
        nms_applied=raw.get("nms_applied", False),
        conf_threshold=raw.get("conf_threshold", None),
        iou_threshold=raw.get("iou_threshold", None),
        class_thresholds=raw.get("class_thresholds", None),
    )


def load_config(yaml_path: str, cli_args: list[str] | None = None) -> Config:
    """Load YAML config and merge CLI overrides."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    if cli_args:
        data = apply_cli_overrides(data, cli_args)

    models = [_build_model_config(m) for m in data.get("models", [])]

    pre_raw = data.get("preprocess", {})
    crop_raw = pre_raw.get("crop", {})
    resize_raw = pre_raw.get("resize", {})
    preprocess = PreprocessConfig(
        crop=CropConfig(
            enabled=crop_raw.get("enabled", False),
            region=crop_raw.get("region", [0.0, 0.0, 1.0, 1.0]),
        ),
        resize=ResizeConfig(
            method=resize_raw.get("method", "INTER_LINEAR"),
            input_size=resize_raw.get("input_size", None),
        ),
    )

    post_raw = data.get("postprocess", {})
    postprocess = PostprocessConfig(
        conf_threshold=post_raw.get("conf_threshold", 0.25),
        iou_threshold=post_raw.get("iou_threshold", 0.45),
        class_thresholds=post_raw.get("class_thresholds", {}),
    )

    out_raw = data.get("output", {})
    output = OutputConfig(
        save_video=out_raw.get("save_video", True),
        save_json=out_raw.get("save_json", False),
        display=out_raw.get("display", True),
        output_dir=out_raw.get("output_dir", "output/"),
    )

    comp_raw = data.get("comparison", {})
    comparison = ComparisonConfig(
        mode=comp_raw.get("mode", "overlay"),
        metrics=comp_raw.get(
            "metrics",
            ["cosine_similarity", "precision", "recall", "inference_time"],
        ),
    )

    vid_raw = data.get("video", {})
    video = VideoConfig(
        codec=vid_raw.get("codec", "mp4v"),
        preserve_fps=vid_raw.get("preserve_fps", True),
    )

    return Config(
        models=models,
        preprocess=preprocess,
        postprocess=postprocess,
        output=output,
        comparison=comparison,
        class_names=data.get("class_names", None),
        video=video,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n onnx_inf python -m pytest tests/test_config.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add config dataclasses with YAML loading and CLI overrides"
```

---

### Task 3: Default YAML config file

**Files:**
- Create: `configs/default.yaml`

- [ ] **Step 1: Create default.yaml**

```yaml
models:
  - name: "org"
    path: "onnx_files/org.onnx"
    type: "yolov8"
    nms_applied: false
    conf_threshold: null
    iou_threshold: null
    class_thresholds: null

preprocess:
  crop:
    enabled: false
    region: [0.0, 0.0, 1.0, 1.0]
  resize:
    method: "INTER_LINEAR"
    input_size: null

postprocess:
  conf_threshold: 0.25
  iou_threshold: 0.45
  class_thresholds: {}

output:
  save_video: true
  save_json: false
  display: true
  output_dir: "output/"

comparison:
  mode: "overlay"
  metrics: ["cosine_similarity", "precision", "recall", "inference_time"]

class_names: null

video:
  codec: "mp4v"
  preserve_fps: true
```

- [ ] **Step 2: Test it loads**

```bash
conda run -n onnx_inf python -c "from src.config import load_config; c = load_config('configs/default.yaml'); print(f'Loaded {len(c.models)} model(s): {c.models[0].name}')"
```

Expected: `Loaded 1 model(s): org`

- [ ] **Step 3: Commit**

```bash
git add configs/default.yaml
git commit -m "feat: add default YAML config"
```

---

## Chunk 2: Preprocessor

### Task 4: Preprocessor with crop, resize, normalize

**Files:**
- Create: `src/preprocessor.py`
- Create: `tests/test_preprocessor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_preprocessor.py
import numpy as np
import cv2
import pytest
from src.config import PreprocessConfig, CropConfig, ResizeConfig
from src.preprocessor import Preprocessor, PreprocessMetadata


class TestPreprocessMetadata:
    def test_metadata_fields(self):
        m = PreprocessMetadata(
            original_shape=(480, 640),
            crop_origin=(0, 0),
            crop_size=(640, 480),
            resize_scale=(1.0, 1.0),
            input_size=(640, 640),
            resize_method=cv2.INTER_LINEAR,
        )
        assert m.original_shape == (480, 640)
        assert m.crop_origin == (0, 0)


class TestPreprocessorNoCrop:
    def setup_method(self):
        config = PreprocessConfig(
            crop=CropConfig(enabled=False),
            resize=ResizeConfig(method="INTER_LINEAR"),
        )
        self.preprocessor = Preprocessor(config)

    def test_output_shape_matches_input_size(self):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, meta = self.preprocessor.process(image, input_size=(640, 640))
        assert tensor.shape == (1, 3, 640, 640)

    def test_output_dtype_float32(self):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, _ = self.preprocessor.process(image, input_size=(640, 640))
        assert tensor.dtype == np.float32

    def test_output_range_0_to_1(self):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, _ = self.preprocessor.process(image, input_size=(640, 640))
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_metadata_no_crop(self):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, meta = self.preprocessor.process(image, input_size=(640, 640))
        assert meta.original_shape == (480, 640)
        assert meta.crop_origin == (0, 0)
        assert meta.crop_size == (640, 480)

    def test_rgb_conversion(self):
        # Blue pixel in BGR -> should become (0, 0, 255) in RGB -> channel 0 = 0
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[:, :] = [255, 0, 0]  # BGR blue
        tensor, _ = self.preprocessor.process(image, input_size=(2, 2))
        assert tensor[0, 0, 0, 0] == pytest.approx(0.0, abs=0.01)  # R channel
        assert tensor[0, 2, 0, 0] == pytest.approx(1.0, abs=0.01)  # B channel


class TestPreprocessorWithCrop:
    def setup_method(self):
        config = PreprocessConfig(
            crop=CropConfig(enabled=True, region=[0.25, 0.25, 0.5, 0.5]),
            resize=ResizeConfig(method="INTER_LINEAR"),
        )
        self.preprocessor = Preprocessor(config)

    def test_crop_metadata(self):
        image = np.random.randint(0, 255, (400, 800, 3), dtype=np.uint8)
        _, meta = self.preprocessor.process(image, input_size=(640, 640))
        assert meta.crop_origin == (200, 100)  # x=0.25*800, y=0.25*400
        assert meta.crop_size == (400, 200)    # w=0.5*800, h=0.5*400

    def test_resize_scale(self):
        image = np.random.randint(0, 255, (400, 800, 3), dtype=np.uint8)
        _, meta = self.preprocessor.process(image, input_size=(640, 640))
        # scale_x = crop_w / input_w = 400 / 640
        # scale_y = crop_h / input_h = 200 / 640
        assert meta.resize_scale[0] == pytest.approx(400 / 640)
        assert meta.resize_scale[1] == pytest.approx(200 / 640)


class TestResizeMethods:
    def test_inter_cubic(self):
        config = PreprocessConfig(
            crop=CropConfig(enabled=False),
            resize=ResizeConfig(method="INTER_CUBIC"),
        )
        p = Preprocessor(config)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        tensor, meta = p.process(image, input_size=(64, 64))
        assert tensor.shape == (1, 3, 64, 64)
        assert meta.resize_method == cv2.INTER_CUBIC
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n onnx_inf python -m pytest tests/test_preprocessor.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.preprocessor'`

- [ ] **Step 3: Implement preprocessor.py**

```python
# src/preprocessor.py
from dataclasses import dataclass
import cv2
import numpy as np
from src.config import PreprocessConfig


RESIZE_METHODS = {
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
}


@dataclass
class PreprocessMetadata:
    original_shape: tuple[int, int]
    crop_origin: tuple[int, int]
    crop_size: tuple[int, int]
    resize_scale: tuple[float, float]
    input_size: tuple[int, int]
    resize_method: int


class Preprocessor:
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.cv2_method = RESIZE_METHODS.get(
            config.resize.method, cv2.INTER_LINEAR
        )

    def process(
        self, image: np.ndarray, input_size: tuple[int, int]
    ) -> tuple[np.ndarray, PreprocessMetadata]:
        h, w = image.shape[:2]
        original_shape = (h, w)

        if self.config.crop.enabled:
            rx, ry, rw, rh = self.config.crop.region
            crop_x = int(rx * w)
            crop_y = int(ry * h)
            crop_w = int(rw * w)
            crop_h = int(rh * h)
            cropped = image[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        else:
            crop_x, crop_y = 0, 0
            crop_w, crop_h = w, h
            cropped = image

        input_h, input_w = input_size
        resized = cv2.resize(cropped, (input_w, input_h), interpolation=self.cv2_method)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(chw, axis=0)

        scale_x = crop_w / input_w
        scale_y = crop_h / input_h

        metadata = PreprocessMetadata(
            original_shape=original_shape,
            crop_origin=(crop_x, crop_y),
            crop_size=(crop_w, crop_h),
            resize_scale=(scale_x, scale_y),
            input_size=input_size,
            resize_method=self.cv2_method,
        )

        return tensor, metadata
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n onnx_inf python -m pytest tests/test_preprocessor.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/preprocessor.py tests/test_preprocessor.py
git commit -m "feat: add preprocessor with crop, resize, normalize pipeline"
```

---

## Chunk 3: Inferencer (Base + YOLOv8)

### Task 5: BaseInferencer abstract class

**Files:**
- Create: `src/inferencer/base.py`
- Create: `tests/test_inferencer.py`

- [ ] **Step 1: Write failing test for base inferencer**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n onnx_inf python -m pytest tests/test_inferencer.py::TestBaseInferencerIsAbstract -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement base.py**

```python
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
        # input_shape typically [1, 3, H, W]
        self._input_size = (int(input_shape[2]), int(input_shape[3]))

    @abstractmethod
    def postprocess_raw(self, raw_output: list[np.ndarray]) -> np.ndarray:
        """Convert raw model output to unified (N, 6) [x1, y1, x2, y2, conf, class_id]."""
        pass

    def infer(self, tensor: np.ndarray) -> tuple[list[np.ndarray], float]:
        """Run inference. Returns (raw_outputs, inference_time_ms)."""
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return outputs, elapsed_ms

    def get_input_size(self) -> tuple[int, int]:
        """Return (H, W) expected by the model."""
        return self._input_size
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n onnx_inf python -m pytest tests/test_inferencer.py::TestBaseInferencerIsAbstract -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/inferencer/base.py tests/test_inferencer.py
git commit -m "feat: add BaseInferencer abstract class"
```

---

### Task 6: YOLOv8Inferencer

**Files:**
- Create: `src/inferencer/yolov8.py`
- Modify: `tests/test_inferencer.py`

- [ ] **Step 1: Write failing tests for YOLOv8 postprocess_raw**

Append to `tests/test_inferencer.py`:

```python
from src.inferencer.yolov8 import YOLOv8Inferencer


class TestYOLOv8PostprocessRaw:
    def test_output_shape(self):
        """Simulate dual output: boxes (1,4,N) + cls (1,C,N)."""
        num_boxes = 100
        num_classes = 80
        boxes = np.random.rand(1, 4, num_boxes).astype(np.float32)
        cls_scores = np.random.rand(1, num_classes, num_boxes).astype(np.float32)
        inferencer = YOLOv8Inferencer.__new__(YOLOv8Inferencer)
        result = inferencer.postprocess_raw([boxes, cls_scores])
        assert result.shape == (num_boxes, 6)

    def test_bbox_format_xyxy(self):
        """cx=50, cy=50, w=20, h=30 -> x1=40, y1=35, x2=60, y2=65."""
        boxes = np.array([[[50.0], [50.0], [20.0], [30.0]]])  # (1, 4, 1)
        cls_scores = np.array([[[0.9]]])  # (1, 1, 1)
        inferencer = YOLOv8Inferencer.__new__(YOLOv8Inferencer)
        result = inferencer.postprocess_raw([boxes, cls_scores])
        assert result[0, 0] == pytest.approx(40.0)  # x1
        assert result[0, 1] == pytest.approx(35.0)  # y1
        assert result[0, 2] == pytest.approx(60.0)  # x2
        assert result[0, 3] == pytest.approx(65.0)  # y2
        assert result[0, 4] == pytest.approx(0.9)   # conf
        assert result[0, 5] == 0                     # class_id

    def test_best_class_selected(self):
        boxes = np.array([[[10.0], [10.0], [5.0], [5.0]]])  # (1, 4, 1)
        cls_scores = np.array([[[0.1], [0.8], [0.3]]])       # (1, 3, 1)
        inferencer = YOLOv8Inferencer.__new__(YOLOv8Inferencer)
        result = inferencer.postprocess_raw([boxes, cls_scores])
        assert result[0, 4] == pytest.approx(0.8)  # best score
        assert result[0, 5] == 1                    # class index 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n onnx_inf python -m pytest tests/test_inferencer.py::TestYOLOv8PostprocessRaw -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.inferencer.yolov8'`

- [ ] **Step 3: Implement yolov8.py**

```python
# src/inferencer/yolov8.py
import numpy as np
from src.inferencer.base import BaseInferencer
from src.config import ModelConfig


class YOLOv8Inferencer(BaseInferencer):
    def postprocess_raw(self, raw_output: list[np.ndarray]) -> np.ndarray:
        """Parse YOLOv8 dual output.

        raw_output[0]: boxes (1, 4, num_boxes) — [cx, cy, w, h]
        raw_output[1]: cls   (1, num_classes, num_boxes)

        Returns: (num_boxes, 6) — [x1, y1, x2, y2, confidence, class_id]
        """
        boxes = raw_output[0][0].T    # (num_boxes, 4)
        cls = raw_output[1][0].T      # (num_boxes, num_classes)

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        class_ids = np.argmax(cls, axis=1).astype(np.float32)
        confidences = np.max(cls, axis=1)

        detections = np.stack([x1, y1, x2, y2, confidences, class_ids], axis=1)
        return detections
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n onnx_inf python -m pytest tests/test_inferencer.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/inferencer/yolov8.py tests/test_inferencer.py
git commit -m "feat: add YOLOv8Inferencer with dual-output parsing"
```

---

### Task 7: Inferencer factory

**Files:**
- Modify: `src/inferencer/__init__.py`

- [ ] **Step 1: Add factory test to tests/test_inferencer.py**

```python
from src.inferencer import create_inferencer


class TestInferencerFactory:
    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown inferencer type"):
            create_inferencer("fake.onnx", ModelConfig(type="unknown"))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n onnx_inf python -m pytest tests/test_inferencer.py::TestInferencerFactory -v
```

Expected: FAIL — `ImportError: cannot import name 'create_inferencer'`

- [ ] **Step 3: Implement factory in __init__.py**

```python
# src/inferencer/__init__.py
from src.config import ModelConfig
from src.inferencer.base import BaseInferencer
from src.inferencer.yolov8 import YOLOv8Inferencer

INFERENCER_REGISTRY = {
    "yolov8": YOLOv8Inferencer,
}


def create_inferencer(model_path: str, model_config: ModelConfig) -> BaseInferencer:
    cls = INFERENCER_REGISTRY.get(model_config.type)
    if cls is None:
        raise ValueError(
            f"Unknown inferencer type: '{model_config.type}'. "
            f"Available: {list(INFERENCER_REGISTRY.keys())}"
        )
    return cls(model_path, model_config)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n onnx_inf python -m pytest tests/test_inferencer.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/inferencer/__init__.py tests/test_inferencer.py
git commit -m "feat: add inferencer factory with registry"
```

---

## Chunk 4: Postprocessor

### Task 8: Postprocessor — thresholds, NMS, coordinate reversion

**Files:**
- Create: `src/postprocessor.py`
- Create: `tests/test_postprocessor.py`

- [ ] **Step 1: Write failing tests**

```python
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
        # Crop at (100, 50), crop size 320x240, model input 640x640
        meta = make_metadata(
            crop_origin=(100, 50), crop_size=(320, 240), input_size=(640, 640)
        )
        # scale_x = 320/640 = 0.5, scale_y = 240/640 = 0.375
        dets = np.array([[100, 100, 200, 200, 0.9, 0]])
        result = pp.revert_coordinates(dets, meta)
        assert result[0, 0] == pytest.approx(100 * 0.5 + 100)   # x1
        assert result[0, 1] == pytest.approx(100 * 0.375 + 50)   # y1
        assert result[0, 2] == pytest.approx(200 * 0.5 + 100)    # x2
        assert result[0, 3] == pytest.approx(200 * 0.375 + 50)   # y2


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n onnx_inf python -m pytest tests/test_postprocessor.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement postprocessor.py**

```python
# src/postprocessor.py
import cv2
import numpy as np
from src.config import PostprocessConfig
from src.preprocessor import PreprocessMetadata


class Postprocessor:
    def __init__(self, config: PostprocessConfig):
        self.conf_threshold = config.conf_threshold
        self.iou_threshold = config.iou_threshold
        self.class_thresholds = config.class_thresholds or {}

    def filter_by_threshold(self, detections: np.ndarray) -> np.ndarray:
        if len(detections) == 0:
            return detections

        mask = detections[:, 4] >= self.conf_threshold

        for cls_id, cls_thresh in self.class_thresholds.items():
            cls_mask = detections[:, 5] == cls_id
            mask[cls_mask] = detections[cls_mask, 4] >= cls_thresh

        return detections[mask]

    def apply_nms(self, detections: np.ndarray) -> np.ndarray:
        if len(detections) == 0:
            return detections

        boxes = detections[:, :4].tolist()
        scores = detections[:, 4].tolist()

        # cv2.dnn.NMSBoxes expects [x, y, w, h]
        boxes_xywh = []
        for x1, y1, x2, y2 in boxes:
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

        # score_threshold=0.0 since filter_by_threshold already ran
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh, scores, 0.0, self.iou_threshold
        )

        if len(indices) == 0:
            return np.empty((0, 6))

        indices = np.array(indices).flatten()
        return detections[indices]

    def revert_coordinates(
        self, detections: np.ndarray, metadata: PreprocessMetadata
    ) -> np.ndarray:
        if len(detections) == 0:
            return detections

        result = detections.copy()
        scale_x, scale_y = metadata.resize_scale
        offset_x, offset_y = metadata.crop_origin

        result[:, 0] = detections[:, 0] * scale_x + offset_x
        result[:, 1] = detections[:, 1] * scale_y + offset_y
        result[:, 2] = detections[:, 2] * scale_x + offset_x
        result[:, 3] = detections[:, 3] * scale_y + offset_y

        return result

    def process(
        self,
        detections: np.ndarray,
        metadata: PreprocessMetadata,
        nms_applied: bool,
    ) -> np.ndarray:
        detections = self.filter_by_threshold(detections)
        if not nms_applied:
            detections = self.apply_nms(detections)
        detections = self.revert_coordinates(detections, metadata)
        return detections
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n onnx_inf python -m pytest tests/test_postprocessor.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/postprocessor.py tests/test_postprocessor.py
git commit -m "feat: add postprocessor with NMS, thresholds, coordinate reversion"
```

---

## Chunk 5: Visualizer

### Task 9: Visualizer — draw, overlay, side-by-side, toggle

**Files:**
- Create: `src/visualizer.py`
- Create: `tests/test_visualizer.py`

- [ ] **Step 1: Write failing tests**

```python
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
        results = {
            "model_a": np.array([[10, 10, 100, 100, 0.9, 0]]),
            "model_b": np.array([[50, 50, 150, 150, 0.8, 1]]),
        }
        output = vis.render_comparison(image, results)
        assert output.shape == image.shape

    def test_side_by_side_doubles_width(self):
        vis = Visualizer(initial_mode="side_by_side")
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = {
            "model_a": np.array([[10, 10, 100, 100, 0.9, 0]]),
            "model_b": np.array([[50, 50, 150, 150, 0.8, 1]]),
        }
        output = vis.render_comparison(image, results)
        assert output.shape == (480, 1280, 3)

    def test_toggle_mode(self):
        vis = Visualizer(initial_mode="overlay")
        assert vis.comparison_mode == "overlay"
        vis.toggle_mode()
        assert vis.comparison_mode == "side_by_side"
        vis.toggle_mode()
        assert vis.comparison_mode == "overlay"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n onnx_inf python -m pytest tests/test_visualizer.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement visualizer.py**

```python
# src/visualizer.py
import cv2
import numpy as np


class Visualizer:
    # Distinct colors for up to 10 models (BGR)
    MODEL_COLORS = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
        (255, 128, 0), (128, 0, 255),
    ]

    def __init__(
        self,
        class_names: list[str] | None = None,
        initial_mode: str = "overlay",
    ):
        self.class_names = class_names
        self.comparison_mode = initial_mode

    def toggle_mode(self):
        if self.comparison_mode == "overlay":
            self.comparison_mode = "side_by_side"
        else:
            self.comparison_mode = "overlay"

    def _label(self, class_id: int, confidence: float, model_name: str | None = None) -> str:
        if self.class_names and int(class_id) < len(self.class_names):
            name = self.class_names[int(class_id)]
        else:
            name = f"cls_{int(class_id)}"
        label = f"{name}: {confidence:.2f}"
        if model_name:
            label = f"[{model_name}] {label}"
        return label

    def draw_detections(
        self,
        image: np.ndarray,
        detections: np.ndarray,
        model_name: str | None = None,
        color: tuple | None = None,
    ) -> np.ndarray:
        img = image.copy()
        if len(detections) == 0:
            return img

        draw_color = color or (0, 255, 0)

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.rectangle(img, pt1, pt2, draw_color, 2)
            label = self._label(cls_id, conf, model_name)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, pt1, (pt1[0] + tw, pt1[1] - th - 4), draw_color, -1)
            cv2.putText(
                img, label, (pt1[0], pt1[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            )

        return img

    def overlay_multi(
        self, image: np.ndarray, results: dict[str, np.ndarray]
    ) -> np.ndarray:
        img = image.copy()
        for i, (model_name, dets) in enumerate(results.items()):
            color = self.MODEL_COLORS[i % len(self.MODEL_COLORS)]
            img = self.draw_detections(img, dets, model_name=model_name, color=color)
        return img

    def side_by_side(
        self, image: np.ndarray, results: dict[str, np.ndarray]
    ) -> np.ndarray:
        panels = []
        for i, (model_name, dets) in enumerate(results.items()):
            color = self.MODEL_COLORS[i % len(self.MODEL_COLORS)]
            panel = self.draw_detections(image, dets, model_name=model_name, color=color)
            panels.append(panel)
        return np.concatenate(panels, axis=1)

    def render_comparison(
        self, image: np.ndarray, results: dict[str, np.ndarray]
    ) -> np.ndarray:
        if self.comparison_mode == "overlay":
            return self.overlay_multi(image, results)
        else:
            return self.side_by_side(image, results)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n onnx_inf python -m pytest tests/test_visualizer.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/visualizer.py tests/test_visualizer.py
git commit -m "feat: add visualizer with overlay, side-by-side, and runtime toggle"
```

---

## Chunk 6: Comparator

### Task 10: Comparator — cosine similarity, metrics, timing

**Files:**
- Create: `src/comparator.py`
- Create: `tests/test_comparator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_comparator.py
import numpy as np
import pytest
from src.config import ComparisonConfig
from src.comparator import Comparator


class TestCosineSimilarity:
    def test_identical_outputs(self):
        comp = Comparator(ComparisonConfig())
        raw = [np.random.rand(1, 4, 100).astype(np.float32),
               np.random.rand(1, 80, 100).astype(np.float32)]
        result = comp.cosine_similarity(raw, raw)
        assert result["box_similarity"] == pytest.approx(1.0, abs=1e-5)
        assert result["cls_similarity"] == pytest.approx(1.0, abs=1e-5)

    def test_different_outputs(self):
        comp = Comparator(ComparisonConfig())
        raw_a = [np.ones((1, 4, 100), dtype=np.float32),
                 np.ones((1, 80, 100), dtype=np.float32)]
        raw_b = [np.full((1, 4, 100), -1, dtype=np.float32),
                 np.full((1, 80, 100), -1, dtype=np.float32)]
        result = comp.cosine_similarity(raw_a, raw_b)
        assert result["box_similarity"] == pytest.approx(-1.0, abs=1e-5)

    def test_different_shapes_truncates(self):
        comp = Comparator(ComparisonConfig())
        raw_a = [np.ones((1, 4, 100), dtype=np.float32),
                 np.ones((1, 80, 100), dtype=np.float32)]
        raw_b = [np.ones((1, 4, 50), dtype=np.float32),
                 np.ones((1, 80, 50), dtype=np.float32)]
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n onnx_inf python -m pytest tests/test_comparator.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement comparator.py**

```python
# src/comparator.py
import numpy as np
from src.config import ComparisonConfig


class Comparator:
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self._frame_results: list[dict] = []
        self._timing: dict[str, list[float]] = {}

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.flatten()
        b_flat = b.flatten()
        min_len = min(len(a_flat), len(b_flat))
        a_flat = a_flat[:min_len]
        b_flat = b_flat[:min_len]
        dot = np.dot(a_flat, b_flat)
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def cosine_similarity(
        self, raw_a: list[np.ndarray], raw_b: list[np.ndarray]
    ) -> dict:
        return {
            "box_similarity": self._cosine_sim(raw_a[0], raw_b[0]),
            "cls_similarity": self._cosine_sim(raw_a[1], raw_b[1]),
        }

    def _iou(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def compute_metrics(
        self, det_a: np.ndarray, det_b: np.ndarray, iou_threshold: float = 0.5
    ) -> dict:
        """Model A is reference. Precision = matched_b / total_b, Recall = matched_a / total_a."""
        if len(det_a) == 0 and len(det_b) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched": 0}
        if len(det_a) == 0:
            return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "matched": 0}
        if len(det_b) == 0:
            return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "matched": 0}

        matched_a = set()
        matched_b = set()

        for i, da in enumerate(det_a):
            best_iou = 0.0
            best_j = -1
            for j, db in enumerate(det_b):
                if j in matched_b:
                    continue
                if da[5] != db[5]:  # same class only
                    continue
                iou = self._iou(da[:4], db[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_threshold and best_j >= 0:
                matched_a.add(i)
                matched_b.add(best_j)

        n_matched = len(matched_a)
        precision = n_matched / len(det_b) if len(det_b) > 0 else 0.0
        recall = n_matched / len(det_a) if len(det_a) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched": n_matched,
        }

    def timing_comparison(self, times: dict[str, list[float]]) -> dict:
        result = {}
        for model_name, t_list in times.items():
            arr = np.array(t_list)
            result[model_name] = {
                "mean_ms": float(np.mean(arr)),
                "std_ms": float(np.std(arr)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
            }
        return result

    def add_frame_result(
        self, raw_outputs: dict[str, list[np.ndarray]],
        detections: dict[str, np.ndarray],
        times: dict[str, float],
    ):
        """Accumulate per-frame results for final summary."""
        for name, t in times.items():
            self._timing.setdefault(name, []).append(t)

        frame = {"detections": detections}

        model_names = list(raw_outputs.keys())
        if len(model_names) >= 2:
            name_a, name_b = model_names[0], model_names[1]
            if "cosine_similarity" in self.config.metrics:
                frame["cosine_similarity"] = self.cosine_similarity(
                    raw_outputs[name_a], raw_outputs[name_b]
                )
            det_metrics = {}
            for metric in ["precision", "recall"]:
                if metric in self.config.metrics:
                    det_metrics = self.compute_metrics(
                        detections[name_a], detections[name_b]
                    )
                    break
            if det_metrics:
                frame["detection_metrics"] = det_metrics

        self._frame_results.append(frame)

    def summarize(self) -> dict:
        summary: dict = {}

        if self._timing and "inference_time" in self.config.metrics:
            summary["timing"] = self.timing_comparison(self._timing)

        if not self._frame_results:
            return summary

        cos_sims = [f["cosine_similarity"] for f in self._frame_results if "cosine_similarity" in f]
        if cos_sims:
            summary["cosine_similarity"] = {
                "box": float(np.mean([c["box_similarity"] for c in cos_sims])),
                "cls": float(np.mean([c["cls_similarity"] for c in cos_sims])),
            }

        det_metrics = [f["detection_metrics"] for f in self._frame_results if "detection_metrics" in f]
        if det_metrics:
            summary["detection_metrics"] = {
                "precision": float(np.mean([m["precision"] for m in det_metrics])),
                "recall": float(np.mean([m["recall"] for m in det_metrics])),
                "f1": float(np.mean([m["f1"] for m in det_metrics])),
            }

        return summary
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n onnx_inf python -m pytest tests/test_comparator.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/comparator.py tests/test_comparator.py
git commit -m "feat: add comparator with cosine similarity, detection metrics, timing"
```

---

## Chunk 7: Runner + main.py

### Task 11: Runner orchestrator

**Files:**
- Create: `src/runner.py`

- [ ] **Step 1: Implement runner.py**

```python
# src/runner.py
import os
import json
import cv2
import numpy as np
from src.config import Config
from src.preprocessor import Preprocessor
from src.inferencer import create_inferencer
from src.inferencer.base import BaseInferencer
from src.postprocessor import Postprocessor
from src.visualizer import Visualizer
from src.comparator import Comparator


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


class Runner:
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = Preprocessor(config.preprocess)

        self.inferencers: dict[str, BaseInferencer] = {}
        self.postprocessors: dict[str, Postprocessor] = {}

        for mc in config.models:
            if not os.path.isfile(mc.path):
                raise FileNotFoundError(f"Model file not found: {mc.path}")
            inferencer = create_inferencer(mc.path, mc)
            self.inferencers[mc.name] = inferencer

            post_config = config.postprocess
            # Per-model overrides
            from src.config import PostprocessConfig
            pp_config = PostprocessConfig(
                conf_threshold=mc.conf_threshold if mc.conf_threshold is not None else post_config.conf_threshold,
                iou_threshold=mc.iou_threshold if mc.iou_threshold is not None else post_config.iou_threshold,
                class_thresholds={**(post_config.class_thresholds or {}), **(mc.class_thresholds or {})},
            )
            self.postprocessors[mc.name] = Postprocessor(pp_config)

        self.visualizer = Visualizer(
            class_names=config.class_names,
            initial_mode=config.comparison.mode,
        )

        self.comparator = Comparator(config.comparison) if len(config.models) > 1 else None
        self.multi_model = len(config.models) > 1

    def _get_model_config(self, name: str):
        for mc in self.config.models:
            if mc.name == name:
                return mc
        return None

    def _process_frame(self, image: np.ndarray) -> tuple[dict, dict, np.ndarray]:
        """Process one frame through all models. Returns (detections, raw_outputs, vis_image)."""
        all_detections: dict[str, np.ndarray] = {}
        all_raw: dict[str, list[np.ndarray]] = {}
        all_times: dict[str, float] = {}

        for name, inferencer in self.inferencers.items():
            input_size = inferencer.get_input_size()
            tensor, metadata = self.preprocessor.process(image, input_size)
            raw_output, elapsed = inferencer.infer(tensor)
            detections = inferencer.postprocess_raw(raw_output)

            mc = self._get_model_config(name)
            nms_applied = mc.nms_applied if mc else False
            detections = self.postprocessors[name].process(detections, metadata, nms_applied)

            all_detections[name] = detections
            all_raw[name] = raw_output
            all_times[name] = elapsed

        if self.comparator:
            self.comparator.add_frame_result(all_raw, all_detections, all_times)

        # Visualize
        if self.multi_model:
            vis_image = self.visualizer.render_comparison(image, all_detections)
        else:
            name = list(all_detections.keys())[0]
            vis_image = self.visualizer.draw_detections(image, all_detections[name])

        return all_detections, all_times, vis_image

    def _save_json(self, data: dict, input_path: str):
        os.makedirs(self.config.output.output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        path = os.path.join(self.config.output.output_dir, f"{base}_results.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"JSON saved: {path}")

    def _detections_to_dict(self, dets: np.ndarray, model_name: str, time_ms: float) -> dict:
        det_list = []
        for d in dets:
            entry = {
                "bbox": [float(d[0]), float(d[1]), float(d[2]), float(d[3])],
                "confidence": float(d[4]),
                "class_id": int(d[5]),
            }
            if self.config.class_names and int(d[5]) < len(self.config.class_names):
                entry["class_name"] = self.config.class_names[int(d[5])]
            det_list.append(entry)
        return {"detections": det_list, "inference_time_ms": time_ms}

    def run_image(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        all_dets, all_times, vis_image = self._process_frame(image)

        if self.config.output.display:
            cv2.imshow("ONNX Inference", vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if self.config.output.save_video:
            os.makedirs(self.config.output.output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(self.config.output.output_dir, f"{base}_result.png")
            cv2.imwrite(out_path, vis_image)
            print(f"Image saved: {out_path}")

        if self.config.output.save_json:
            json_data = {"input": image_path, "frames": []}
            frame_models = {}
            for name, dets in all_dets.items():
                frame_models[name] = self._detections_to_dict(dets, name, all_times[name])
            json_data["frames"].append({"frame_id": 0, "models": frame_models})
            if self.comparator:
                json_data["comparison"] = self.comparator.summarize()
            self._save_json(json_data, image_path)

    def run_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) if self.config.video.preserve_fps else 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if self.config.output.save_video:
            os.makedirs(self.config.output.output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(video_path))[0]
            out_path = os.path.join(self.config.output.output_dir, f"{base}_result.mp4")
            fourcc = cv2.VideoWriter_fourcc(*self.config.video.codec)
            # Writer size will be set on first frame (may be wider for side-by-side)
            writer_size = None

        json_frames = [] if self.config.output.save_json else None
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            all_dets, all_times, vis_image = self._process_frame(frame)

            if self.config.output.save_video and writer is None:
                vh, vw = vis_image.shape[:2]
                writer = cv2.VideoWriter(out_path, fourcc, fps, (vw, vh))

            if writer is not None:
                writer.write(vis_image)

            if self.config.output.display:
                cv2.imshow("ONNX Inference (t=toggle, q=quit)", vis_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("t"):
                    # Only toggle display mode; if saving video, toggle would
                    # change frame dimensions and corrupt the output file, so
                    # the display toggles but the writer keeps original mode.
                    if writer is None:
                        self.visualizer.toggle_mode()
                    else:
                        print("Toggle disabled while recording video")

            if json_frames is not None:
                frame_models = {}
                for name, dets in all_dets.items():
                    frame_models[name] = self._detections_to_dict(dets, name, all_times[name])
                json_frames.append({"frame_id": frame_id, "models": frame_models})

            frame_id += 1

        cap.release()
        if writer is not None:
            writer.release()
            print(f"Video saved: {out_path}")
        if self.config.output.display:
            cv2.destroyAllWindows()

        if self.config.output.save_json:
            json_data = {"input": video_path, "frames": json_frames}
            if self.comparator:
                json_data["comparison"] = self.comparator.summarize()
            self._save_json(json_data, video_path)

        if self.comparator:
            summary = self.comparator.summarize()
            print("\n=== Comparison Summary ===")
            for key, val in summary.items():
                print(f"  {key}: {val}")

    def run(self, input_path: str):
        if os.path.isdir(input_path):
            files = sorted(os.listdir(input_path))
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTS:
                    print(f"Processing: {f}")
                    self.run_image(os.path.join(input_path, f))
        else:
            ext = os.path.splitext(input_path)[1].lower()
            if ext in IMAGE_EXTS:
                self.run_image(input_path)
            elif ext in VIDEO_EXTS:
                self.run_video(input_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
```

- [ ] **Step 2: Verify import chain works**

```bash
conda run -n onnx_inf python -c "from src.runner import Runner; print('Runner import OK')"
```

Expected: `Runner import OK`

- [ ] **Step 3: Commit**

```bash
git add src/runner.py
git commit -m "feat: add runner orchestrator for image/video/directory processing"
```

### Task 11b: Runner unit tests

**Files:**
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write Runner tests with mocked inferencers**

```python
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
        """Verify per-model thresholds override global defaults."""
        # Create a fake onnx file so path check passes
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
            assert pp.conf_threshold == 0.8       # per-model override
            assert pp.class_thresholds[0] == 0.95  # per-model class threshold
            assert pp.class_thresholds[1] == 0.5   # global class threshold preserved


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
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
conda run -n onnx_inf python -m pytest tests/test_runner.py -v
```

Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_runner.py
git commit -m "test: add runner unit tests for init validation and threshold merging"
```

---

### Task 12: CLI entry point (main.py)

**Files:**
- Create: `main.py`

- [ ] **Step 1: Implement main.py**

```python
# main.py
import argparse
import sys
from src.config import load_config
from src.runner import Runner


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="ONNX Inference Pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--input", required=True, help="Path to image, video, or directory")
    return parser.parse_known_args()


def main():
    args, extra_args = parse_args()
    config = load_config(args.config, cli_args=extra_args)
    runner = Runner(config)
    runner.run(args.input)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test CLI help works**

```bash
conda run -n onnx_inf python main.py --help
```

Expected: Shows usage with `--config` and `--input` options

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add CLI entry point with argparse"
```

---

## Chunk 8: Integration Test

### Task 13: End-to-end smoke test with real ONNX models

- [ ] **Step 1: Create a test image for integration testing**

```bash
conda run -n onnx_inf python -c "
import numpy as np, cv2, os
os.makedirs('test_data', exist_ok=True)
img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
cv2.imwrite('test_data/test_image.jpg', img)
print('Test image created')
"
```

- [ ] **Step 2: Create a two-model config for testing**

Create `configs/compare.yaml`:

```yaml
models:
  - name: "org"
    path: "onnx_files/org.onnx"
    type: "yolov8"
    nms_applied: false
  - name: "qat"
    path: "onnx_files/qat.onnx"
    type: "yolov8"
    nms_applied: false

preprocess:
  crop:
    enabled: false
    region: [0.0, 0.0, 1.0, 1.0]
  resize:
    method: "INTER_LINEAR"
    input_size: null

postprocess:
  conf_threshold: 0.25
  iou_threshold: 0.45
  class_thresholds: {}

output:
  save_video: true
  save_json: true
  display: false
  output_dir: "output/"

comparison:
  mode: "overlay"
  metrics: ["cosine_similarity", "precision", "recall", "inference_time"]

class_names: null

video:
  codec: "mp4v"
  preserve_fps: true
```

- [ ] **Step 3: Run single-model inference**

```bash
conda run -n onnx_inf python main.py --config configs/default.yaml --input test_data/test_image.jpg --output.display false --output.save_json true
```

Expected: Image saved to `output/test_image_result.png`, JSON to `output/test_image_results.json`

- [ ] **Step 4: Run multi-model comparison**

```bash
conda run -n onnx_inf python main.py --config configs/compare.yaml --input test_data/test_image.jpg
```

Expected: Comparison summary printed with cosine similarity, timing stats, and detection metrics

- [ ] **Step 5: Verify JSON output structure**

```bash
conda run -n onnx_inf python -c "
import json
with open('output/test_image_results.json') as f:
    data = json.load(f)
assert 'frames' in data
assert 'models' in data['frames'][0]
print('JSON structure valid')
"
```

- [ ] **Step 6: Commit test data and compare config**

```bash
git add configs/compare.yaml test_data/
git commit -m "test: add integration test data and comparison config"
```

---

### Task 14: Run all unit tests and verify

- [ ] **Step 1: Run full test suite**

```bash
conda run -n onnx_inf python -m pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 2: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address any test failures from integration"
```

---

## Execution Summary

| Chunk | Tasks | Description |
|-------|-------|-------------|
| 1 | 1-3 | Project setup, config system, default YAML |
| 2 | 4 | Preprocessor (crop, resize, normalize) |
| 3 | 5-7 | BaseInferencer, YOLOv8Inferencer, factory |
| 4 | 8 | Postprocessor (NMS, thresholds, coord revert) |
| 5 | 9 | Visualizer (draw, overlay, side-by-side, toggle) |
| 6 | 10 | Comparator (cosine sim, metrics, timing) |
| 7 | 11-11b-12 | Runner orchestrator, Runner tests, CLI entry point |
| 8 | 13-14 | Integration test, full test suite |

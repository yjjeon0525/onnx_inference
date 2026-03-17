# ONNX Inference Pipeline

A modular ONNX inference pipeline for object detection models. Supports single/multi-model inference on images and videos, with model comparison, visualization, and JSON export. Includes static model analysis and per-layer activation profiling for quantization debugging.

## Features

- **Single & Multi-Model Inference** — Run one or multiple ONNX models and compare results
- **Image / Video / Directory** — Accepts `.jpg`, `.png`, `.bmp`, `.mp4`, `.avi`, `.mov`, `.mkv`, or a directory of images
- **Configurable Preprocessing** — ROI crop (normalized coordinates) and resize with selectable OpenCV interpolation method
- **Coordinate Reversion** — Bounding boxes are mapped back to original image coordinates after crop/resize
- **NMS Support** — Apply NMS internally or skip if the model already includes it
- **Threshold Control** — Global, per-model, and per-class confidence thresholds
- **Comparison Mode** — Cosine similarity on raw outputs, precision/recall/F1 on detections, inference timing stats
- **Visualization** — Overlay (all models on one image) or side-by-side view, togglable at runtime with `t` key
- **JSON Export** — Optionally save per-frame detections and comparison summary
- **Static Model Analysis** — Compare two ONNX files by graph structure and weight values without running inference
- **Layer Profiling** — Compare per-layer intermediate activations between two models to debug quantization drift
- **Extensible** — Add new model types (YOLOX, EfficientDet, etc.) by subclassing `BaseInferencer`

## Setup

```bash
# Create conda environment
conda create -n onnx_inf python=3.13 -y
conda activate onnx_inf

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- `onnxruntime`
- `opencv-python`
- `numpy`
- `pyyaml`
- `onnx`
- `pytest`

## Quick Start

### Single Model Inference

```bash
python main.py --config configs/default.yaml --input path/to/image.jpg
```

### Multi-Model Comparison

```bash
python main.py --config configs/compare.yaml --input path/to/video.mp4
```

### Directory of Images

```bash
python main.py --config configs/default.yaml --input path/to/image_dir/
```

### Static Model Comparison

Compare two ONNX files by graph structure and weight values (no inference needed):

```bash
python main.py --compare-models onnx_files/org.onnx onnx_files/qat.onnx
```

Save the comparison report as JSON:

```bash
python main.py --compare-models onnx_files/org.onnx onnx_files/qat.onnx \
  --output-json output/model_comparison.json
```

### Per-Layer Activation Profiling

Compare intermediate layer outputs between two models given an input image. Useful for finding where quantization drift starts:

```bash
python main.py --compare-models onnx_files/org.onnx onnx_files/qat.onnx \
  --profile-layers --input test_data/test_image.jpg
```

With config and JSON output:

```bash
python main.py --compare-models onnx_files/org.onnx onnx_files/qat.onnx \
  --profile-layers --input test_data/test_image.jpg \
  --config configs/default.yaml \
  --output-json output/report.json
```

## Configuration

Configuration is done via YAML file + optional CLI overrides.

### YAML Config Structure

```yaml
models:
  - name: "org"
    path: "onnx_files/org.onnx"
    type: "yolov8"              # Model type (yolov8, etc.)
    nms_applied: false           # true if model already includes NMS
    conf_threshold: null         # Per-model override (null = use global)
    iou_threshold: null          # Per-model override (null = use global)
    class_thresholds: null       # Per-model per-class override

preprocess:
  crop:
    enabled: false
    region: [0.0, 0.0, 1.0, 1.0]  # Normalized [x, y, w, h] (0~1)
  resize:
    method: "INTER_LINEAR"         # INTER_LINEAR, INTER_CUBIC, INTER_NEAREST, INTER_AREA, INTER_LANCZOS4
    input_size: null               # Auto-detected from ONNX model if null

postprocess:
  conf_threshold: 0.25    # Global confidence threshold
  iou_threshold: 0.45     # Global NMS IoU threshold
  class_thresholds: {}    # Per-class thresholds, e.g. {0: 0.7, 2: 0.3}

output:
  save_video: true        # Save annotated image/video to output_dir
  save_json: false        # Save detection results as JSON
  display: true           # Show real-time display window
  output_dir: "output/"

comparison:
  mode: "overlay"         # Initial mode: "overlay" or "side_by_side"
  metrics: ["cosine_similarity", "precision", "recall", "inference_time"]

class_names: null         # Optional list of class names, e.g. ["person", "car", "bus"]

video:
  codec: "mp4v"
  preserve_fps: true
```

### CLI Overrides

Override any config value using dot-notation:

```bash
# Change confidence threshold
python main.py --config configs/default.yaml --input img.jpg --postprocess.conf_threshold 0.5

# Enable crop with region
python main.py --config configs/default.yaml --input img.jpg \
  --preprocess.crop.enabled true \
  --preprocess.crop.region 0.1,0.2,0.8,0.7

# Disable display, enable JSON output
python main.py --config configs/default.yaml --input video.mp4 \
  --output.display false \
  --output.save_json true

# Change resize method
python main.py --config configs/default.yaml --input img.jpg \
  --preprocess.resize.method INTER_CUBIC
```

## CLI Reference

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to YAML config file (required for inference mode) |
| `--input PATH` | Path to image, video, or directory |
| `--compare-models A B` | Compare two ONNX model files (structure + weights) |
| `--profile-layers` | Compare per-layer activations (requires `--compare-models` and `--input`) |
| `--output-json PATH` | Save comparison report as JSON |

## Keyboard Controls (Video Mode)

| Key | Action |
|-----|--------|
| `t` | Toggle between overlay and side-by-side view |
| `q` | Quit |

## Extending with New Model Types

To add a new model type (e.g., YOLOX), create a new inferencer:

```python
# src/inferencer/yolox.py
from src.inferencer.base import BaseInferencer

class YOLOXInferencer(BaseInferencer):
    def postprocess_raw(self, raw_output: list) -> np.ndarray:
        # Parse model-specific output format
        # Return (N, 6) array: [x1, y1, x2, y2, confidence, class_id]
        ...
```

Register it in `src/inferencer/__init__.py`:

```python
from src.inferencer.yolox import YOLOXInferencer

INFERENCER_REGISTRY = {
    "yolov8": YOLOv8Inferencer,
    "yolox": YOLOXInferencer,
}
```

Then use `type: "yolox"` in your YAML config.

## Project Structure

```
onnx_inference/
├── main.py                  # CLI entry point
├── requirements.txt
├── configs/
│   ├── default.yaml         # Single model config
│   └── compare.yaml         # Multi-model comparison config
├── onnx_files/              # Place ONNX models here
├── src/
│   ├── config.py            # Config dataclasses + YAML/CLI loading
│   ├── preprocessor.py      # Crop, resize, normalize
│   ├── postprocessor.py     # NMS, threshold filter, coordinate revert
│   ├── visualizer.py        # Draw bboxes, overlay, side-by-side, toggle
│   ├── comparator.py        # Cosine similarity, metrics, timing (runtime)
│   ├── model_analyzer.py    # Static ONNX model comparison (structure + weights)
│   ├── layer_profiler.py    # Per-layer activation comparison
│   ├── runner.py            # Pipeline orchestrator
│   └── inferencer/
│       ├── __init__.py      # Factory (create_inferencer)
│       ├── base.py          # BaseInferencer (ABC)
│       └── yolov8.py        # YOLOv8 dual-output parser with grid/stride decode
├── tests/
│   ├── test_config.py
│   ├── test_preprocessor.py
│   ├── test_inferencer.py
│   ├── test_postprocessor.py
│   ├── test_visualizer.py
│   ├── test_comparator.py
│   ├── test_runner.py
│   ├── test_model_analyzer.py
│   └── test_layer_profiler.py
└── output/                  # Generated results
```

## Running Tests

```bash
conda activate onnx_inf
python -m pytest tests/ -v
```

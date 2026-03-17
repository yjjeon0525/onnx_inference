# ONNX Inference Pipeline — Design Spec

## Overview

A modular ONNX inference pipeline for running YOLOv8 (and future object detection models) on images and videos. Supports single-model inference, multi-model comparison with visual overlay and metrics, configurable preprocessing (ROI crop + resize), and coordinate reversion in postprocessing.

## Project Structure

```
onnx_inference/
├── configs/
│   └── default.yaml
├── onnx_files/
│   ├── org.onnx
│   └── qat.onnx
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessor.py
│   ├── inferencer/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── yolov8.py
│   ├── postprocessor.py
│   ├── visualizer.py
│   ├── comparator.py
│   └── runner.py
├── main.py
└── requirements.txt
```

## Dependencies

- `onnxruntime` — ONNX model inference
- `opencv-python` — image/video I/O, resize, display, NMS
- `numpy` — tensor operations
- `pyyaml` — config loading

## Configuration

### YAML Config Structure

```yaml
models:
  - name: "model_a"
    path: "onnx_files/org.onnx"
    type: "yolov8"
    nms_applied: false
    conf_threshold: null      # per-model override, null = use global
    iou_threshold: null
    class_thresholds: null    # e.g., {0: 0.7, 1: 0.3}

preprocess:
  crop:
    enabled: false
    region: [0.0, 0.0, 1.0, 1.0]   # normalized [x, y, w, h]
  resize:
    method: "INTER_LINEAR"
    input_size: null                 # inferred from model if null

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
  mode: "overlay"                    # "overlay" | "side_by_side"
  metrics: ["cosine_similarity", "precision", "recall", "inference_time"]

class_names: null                    # list of class names, e.g., ["person", "car", ...]
                                     # if null, use class indices as labels

video:
  codec: "mp4v"                      # fourcc codec for VideoWriter
  preserve_fps: true                 # use source FPS, else default 30
```

### CLI Override

Dot-notation overrides on top of YAML config:

```bash
python main.py --config configs/default.yaml --input path/to/video.mp4 \
  --preprocess.crop.region 0.1,0.1,0.8,0.8 \
  --postprocess.conf_threshold 0.3
```

`--input` is a required CLI argument (not in YAML) specifying the image, video, or directory path.

### Threshold Priority (highest to lowest)

1. Per-class threshold (`class_thresholds`)
2. Per-model threshold (`models[].conf_threshold`)
3. Global threshold (`postprocess.conf_threshold`)

## Components

### 1. Config (`src/config.py`)

Loads YAML config, merges CLI overrides via dot-notation, and produces typed config dataclasses.

- `load_config(yaml_path, cli_overrides) -> Config`
- Dot-notation parser: `"preprocess.crop.region"` -> nested dict access
- Validates required fields (model path existence, valid resize method, etc.)

### 2. Preprocessor (`src/preprocessor.py`)

Transforms input image into model-ready tensor and emits metadata for coordinate reversion.

```python
@dataclass
class PreprocessMetadata:
    original_shape: tuple[int, int]         # (H, W)
    crop_origin: tuple[int, int]            # (x, y) absolute pixels — top-left of crop
    crop_size: tuple[int, int]              # (w, h) absolute pixels — size of cropped region
    resize_scale: tuple[float, float]       # (scale_x, scale_y) = (crop_w / input_w, crop_h / input_h)
                                            # This is the reversion scale: multiply model coords to get crop-space coords
    input_size: tuple[int, int]             # (H, W) model input
    resize_method: int                      # cv2 interpolation enum

class Preprocessor:
    def __init__(self, config: PreprocessConfig)
    def process(self, image: np.ndarray) -> tuple[np.ndarray, PreprocessMetadata]
```

**Pipeline:** Crop (normalized ROI -> absolute pixel slice) -> Resize (configurable cv2 interpolation) -> Normalize (HWC BGR uint8 -> CHW RGB float32 [0,1]) -> Add batch dim (expand to `(1, C, H, W)`)

**Crop conversion:** Normalized `[x, y, w, h]` where each value is in [0, 1] is converted to absolute pixels:
- `crop_x = int(x * image_width)`, `crop_y = int(y * image_height)`
- `crop_w = int(w * image_width)`, `crop_h = int(h * image_height)`
- Slice: `image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]`

**Per-model preprocessing:** Each model may have a different `input_size` (read from ONNX metadata). The `Preprocessor` accepts `input_size` per call so the `Runner` can invoke it once per model if input sizes differ. If all models share the same input size, preprocessing is done once and shared.

**Supported resize methods:** `INTER_LINEAR`, `INTER_CUBIC`, `INTER_NEAREST`, `INTER_AREA`, `INTER_LANCZOS4`

### 3. BaseInferencer (`src/inferencer/base.py`)

Abstract base class for all model types.

```python
class BaseInferencer(ABC):
    def __init__(self, model_path: str, model_config: ModelConfig):
        # Load ONNX via onnxruntime.InferenceSession
        # Extract input_name, output_names, input_size from model metadata

    @abstractmethod
    def postprocess_raw(self, raw_output: list[np.ndarray]) -> np.ndarray:
        """Convert model-specific raw output to unified (N, 6) format.
        Returns: [x1, y1, x2, y2, confidence, class_id]"""

    def infer(self, tensor: np.ndarray) -> tuple[list[np.ndarray], float]:
        """Run onnxruntime session.run(). Returns (raw_outputs, inference_time_ms)."""

    def get_input_size(self) -> tuple[int, int]:
        """Return (H, W) from ONNX model metadata."""
```

**Extension pattern:** New networks subclass `BaseInferencer` and implement `postprocess_raw()` only.

### 4. YOLOv8Inferencer (`src/inferencer/yolov8.py`)

Handles YOLOv8 dual-output format.

```python
class YOLOv8Inferencer(BaseInferencer):
    def postprocess_raw(self, raw_output: list[np.ndarray]) -> np.ndarray:
        # raw_output[0]: boxes (1, 4, num_boxes) -> transpose -> (num_boxes, 4) [cx, cy, w, h]
        # raw_output[1]: cls   (1, num_classes, num_boxes) -> transpose -> (num_boxes, num_classes)
        #
        # 1. Squeeze batch dim, transpose
        # 2. argmax + max on cls scores -> class_id, confidence
        # 3. Convert (cx, cy, w, h) -> (x1, y1, x2, y2)
        # 4. Return (N, 6) [x1, y1, x2, y2, conf, class_id]
```

### 5. Postprocessor (`src/postprocessor.py`)

Filters and transforms detections back to original image coordinates.

```python
class Postprocessor:
    def __init__(self, config: PostprocessConfig)

    def apply_nms(self, detections: np.ndarray) -> np.ndarray:
        """cv2.dnn.NMSBoxes. Skipped if nms_applied=True."""

    def filter_by_threshold(self, detections: np.ndarray) -> np.ndarray:
        """Apply global -> per-model -> per-class thresholds."""

    def revert_coordinates(self, detections: np.ndarray, metadata: PreprocessMetadata) -> np.ndarray:
        """Map bbox coords from model input space back to original image space.
        resize_scale = (crop_w / input_w, crop_h / input_h)
        x_orig = x_model * resize_scale_x + crop_origin_x
        y_orig = y_model * resize_scale_y + crop_origin_y
        Where crop_origin is the absolute pixel offset of the crop top-left corner."""

    def process(self, detections: np.ndarray, metadata: PreprocessMetadata,
                nms_applied: bool) -> np.ndarray:
        """Full pipeline: threshold filter -> NMS (conditional) -> coordinate revert.
        Confidence filtering first reduces candidates before NMS for performance.
        Returns (M, 6) in original image coords."""
```

### 6. Visualizer (`src/visualizer.py`)

Draws detections and supports runtime mode toggle.

```python
class Visualizer:
    def __init__(self, class_names: list[str] = None, initial_mode: str = "overlay"):
        # class_names loaded from config's class_names field
        # If None, display class index (e.g., "cls_0: 0.85")
        self.comparison_mode = initial_mode  # from config

    def toggle_mode(self):
        """Switch overlay <-> side_by_side."""

    def draw_detections(self, image: np.ndarray, detections: np.ndarray,
                        model_name: str = None, color: tuple = None) -> np.ndarray

    def overlay_multi(self, image: np.ndarray, results: dict[str, np.ndarray]) -> np.ndarray:
        """All models' bboxes on single image, distinct colors per model."""

    def side_by_side(self, image: np.ndarray, results: dict[str, np.ndarray]) -> np.ndarray:
        """Horizontal concat, one panel per model."""

    def render_comparison(self, image: np.ndarray, results: dict[str, np.ndarray]) -> np.ndarray:
        """Dispatches to overlay_multi or side_by_side based on current mode."""
```

**Runtime toggle:** Press `t` during video display to switch modes. Press `q` to quit.

### 7. Comparator (`src/comparator.py`)

Computes similarity and performance metrics between models.

```python
class Comparator:
    def __init__(self, config: ComparisonConfig)

    def cosine_similarity(self, raw_a: list[np.ndarray], raw_b: list[np.ndarray]) -> dict:
        """Cosine similarity on raw box and cls outputs separately.
        Both models must produce same-shaped outputs (same input size, same architecture).
        If shapes differ, flatten and truncate to min length before computing.
        Returns: {box_similarity: float, cls_similarity: float}"""

    def compute_metrics(self, det_a: np.ndarray, det_b: np.ndarray,
                        iou_threshold: float = 0.5) -> dict:
        """Compare two detection sets using the first model as reference.
        Matches detections by IoU, computes precision/recall/F1 of model_b vs model_a.
        No ground truth required — this is model-vs-model comparison."""

    def timing_comparison(self, times: dict[str, list[float]]) -> dict:
        """Mean, std, min, max inference time per model."""

    def summarize(self) -> dict:
        """Full comparison report, dict (printable + JSON serializable)."""
```

### 8. Runner (`src/runner.py`)

Orchestrates the full pipeline.

```python
class Runner:
    def __init__(self, config: Config):
        # Initialize: Preprocessor, Inferencers (by type), Postprocessors (per-model),
        # Visualizer, Comparator (if multi-model)

    def run_image(self, image_path: str)
    def run_video(self, video_path: str)
    def run(self, input_path: str):
        """Auto-detect image/video/directory by extension."""
```

**Single model flow:**
```
Input -> Preprocess -> Infer -> PostprocessRaw -> Postprocess -> Visualize -> Output
```

**Multi model flow:**
```
Input -> Preprocess -> [Infer per model] -> [PostprocessRaw per model]
      -> [Postprocess per model] -> Compare(raw + final) -> Visualize(toggle) -> Output
```

**Supported input formats:**
- Images: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Videos: `.mp4`, `.avi`, `.mov`, `.mkv`
- Directory of images

**Output:**
- Annotated video/image with bboxes (if `save_video` or `display`)
- JSON detection results per frame (if `save_json`)
- Comparison summary JSON (if multi-model)

## Data Flow

```
[Image/Frame]
    |
    v
[Preprocessor] -- emits PreprocessMetadata
    |                     |
    v                     |
[Inferencer.infer()]      |
    |                     |
    v                     |
[Inferencer.postprocess_raw()]  -- raw outputs also sent to Comparator
    |                     |
    v                     |
[Postprocessor.process()] <-- uses PreprocessMetadata for coord revert
    |
    v
[Visualizer] -- draw on original image
    |
    v
[Display / Save]
```

## Key Design Decisions

1. **One abstract method per network:** `postprocess_raw()` is the only thing new model types implement. Everything else is shared.
2. **Metadata-driven coordinate reversion:** Preprocessing emits metadata, postprocessing consumes it. No global state.
3. **Layered thresholds:** Global -> per-model -> per-class, cleanly merged at config load time.
4. **Runtime toggle:** Visualization mode switchable via keypress during playback, initial mode from config.
5. **Config-first with CLI overrides:** YAML for reproducibility, CLI for quick experiments.
6. **Threshold filter before NMS:** Filter low-confidence detections first to reduce NMS candidate set for performance.

## JSON Output Schema

When `save_json: true`, output is saved per-input as `{output_dir}/{input_name}_results.json`:

```json
{
  "input": "path/to/input.mp4",
  "frames": [
    {
      "frame_id": 0,
      "models": {
        "model_a": {
          "detections": [
            {"bbox": [x1, y1, x2, y2], "confidence": 0.92, "class_id": 0, "class_name": "person"}
          ],
          "inference_time_ms": 12.3
        }
      }
    }
  ],
  "comparison": {
    "cosine_similarity": {"box": 0.98, "cls": 0.95},
    "timing": {"model_a": {"mean_ms": 12.1}, "model_b": {"mean_ms": 13.4}}
  }
}
```

## Directory Input

When input is a directory:
- Process files sorted alphabetically
- Non-recursive (top-level only)
- Skip non-image files silently
- Each image processed independently, results saved individually

## Video Output

- Codec: configurable via `video.codec` (default `mp4v`)
- FPS: preserved from source if `video.preserve_fps` is true, else default 30
- Resolution: original image resolution (not model input size); for side-by-side mode, width is `original_width * num_models`

## Error Handling

- **Model file not found:** Raise clear error at startup with path, do not silently skip
- **Invalid ONNX model:** Raise error at `InferenceSession` load time
- **Zero detections on frame:** Draw empty frame (no boxes), continue processing
- **Incompatible model outputs for comparison:** Log warning, skip cosine similarity for that pair
- **Invalid video codec / cannot open video:** Raise error at startup before processing loop

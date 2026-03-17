# Layer Profiler — Per-Layer Activation Comparison

## Purpose

Debug quantization drift by comparing intermediate activations (layer outputs) between two ONNX models given the same input. Identifies which layer first diverges significantly between original and quantized models.

## Architecture

### New Module: `src/layer_profiler.py`

A standalone `LayerProfiler` class that modifies ONNX models to expose intermediate tensors, runs both with the same input, and compares activations layer by layer.

### How It Works

1. Load both ONNX files via `onnx.load()`
2. Extract all intermediate tensor names (node output names) from both graphs
3. Find common tensor names (set intersection) — skip nodes unique to one model (e.g., Q/DQ nodes in QAT)
4. Clone both models and register common intermediate tensors as graph outputs
5. Create `onnxruntime.InferenceSession` from the modified models
6. Run both with the same input tensor, collect all intermediate activations
7. Compare each matched layer and produce a report

### Class Interface

```python
class LayerProfiler:
    def __init__(self, model_path_a: str, model_path_b: str):
        # Load, find common layers, modify models, create sessions

    def _extract_intermediate_names(self, model: ModelProto) -> set[str]:
        """Get all node output tensor names from the graph."""

    def _add_intermediate_outputs(self, model: ModelProto, tensor_names: list[str]) -> ModelProto:
        """Clone model, add selected intermediate tensors as graph outputs.
        Uses onnx.shape_inference to resolve tensor types for new outputs."""

    def profile(self, input_tensor: np.ndarray) -> dict:
        """Run both models with same input, compare all matched layer outputs.
        Returns:
            layers: list of per-layer dicts sorted by divergence (most divergent first)
            summary: {total_compared, most_divergent, least_divergent}
        """

    def print_report(self, report: dict):
        """Pretty-print table to console."""

    def save_report(self, report: dict, output_path: str):
        """Save as JSON."""
```

### Per-Layer Metrics

For each matched intermediate tensor:
- `name`: tensor name
- `shape`: activation shape
- `cosine_similarity`: cosine sim between flattened activations
- `l2_distance`: L2 norm of the difference
- `mean_abs_diff`: mean of absolute element-wise differences
- `std_diff`: standard deviation of the difference
- `max_abs_diff`: maximum absolute element-wise difference
- `snr`: signal-to-noise ratio (norm of signal / norm of difference)

### Layer Matching

Match by intermediate tensor name — only compare layers that have the same output tensor name in both models. Layers unique to one model (e.g., QuantizeLinear/DequantizeLinear in QAT) are skipped.

## CLI Integration

Extends `--compare-models` with a `--profile-layers` flag:

```bash
python main.py --compare-models org.onnx qat.onnx --profile-layers --input test_image.jpg
```

- `--compare-models` alone: static weight/structure comparison (existing)
- `--compare-models` + `--profile-layers` + `--input`: static comparison + per-layer activation comparison using the given input image
- The input image is preprocessed through the existing `Preprocessor` (using default config or `--config` if provided)

### Console Output Format

```
=== Layer Activation Comparison ===
  Compared: 142 layers | Input: test_image.jpg

  Name                              Shape              Cosine    L2 Dist   Mean Diff  Max Diff      SNR
  -------------------------------- ------------------- -------- --------- ---------- --------- --------
  /model.22/cv3/2/2/Conv_output    [1,6,32,60]         0.9812    1.4560    0.0234    0.0891     42.3
  /model.22/cv3/1/2/Conv_output    [1,6,16,30]         0.9856    1.2340    0.0198    0.0756     48.1
  ...
  /model.0/conv/Conv_output        [1,16,128,240]      1.0000    0.0001    0.0000    0.0001   9999.9

  Most divergent: /model.22/cv3/2/2/Conv_output (cosine=0.9812)
  Least divergent: /model.0/conv/Conv_output (cosine=1.0000)
```

## Dependencies

- `onnx` (already installed)
- `onnxruntime` (already installed)
- `numpy` (already installed)

## Files

- Create: `src/layer_profiler.py`
- Create: `tests/test_layer_profiler.py`
- Modify: `main.py` (add `--profile-layers` flag)

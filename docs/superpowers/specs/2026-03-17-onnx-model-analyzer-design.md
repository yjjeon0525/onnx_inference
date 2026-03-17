# ONNX Model Analyzer — Static Model Comparison

## Purpose

Compare two ONNX files directly at the weight and graph structure level, without running inference. Useful for comparing original vs quantized (QAT) models, verifying export correctness, and diagnosing weight drift.

## Architecture

### New Module: `src/model_analyzer.py`

A standalone `OnnxModelAnalyzer` class that loads two ONNX files via the `onnx` library (not onnxruntime) and performs static comparison.

### Structure Comparison

**Quick check** (`quick_structure_check`):
- Returns whether the graph topology is identical or different
- Compares: op count, input shapes, output shapes
- Result: `{identical: bool, op_count_a, op_count_b, input_shapes_match, output_shapes_match}`

**Detailed diff** (`detailed_structure_diff`):
- Layer-by-layer diff showing matched, added, removed, and changed nodes
- Each node entry includes: name, op_type, input shapes, output shapes
- Result: `{matched_nodes: [...], added_nodes: [...], removed_nodes: [...], changed_nodes: [...]}`

### Weight Comparison

**Per-initializer metrics** (`compare_weights`):
- Cosine similarity between matched weight tensors
- L2 distance between matched weight tensors
- Statistical summary: mean_diff, std_diff, min_diff, max_diff, abs_max_diff
- Summary: total params per model, matched count, unmatched per model

### Layer Matching Strategy

1. **Primary: Name matching** — exact initializer/node name match
2. **Fallback: Shape + position** — for unmatched layers, match by shape and topological position in the graph

### Full Report

`analyze()` runs all comparisons and returns a single dict containing structure and weight results.

## CLI Integration

New `--compare-models` flag in `main.py` as a separate mode from inference:

```
python main.py --compare-models model_a.onnx model_b.onnx
python main.py --compare-models model_a.onnx model_b.onnx --output-json output/comparison.json
```

When `--compare-models` is provided, the inference pipeline is skipped entirely.

### Console Output Format

```
=== Structure Comparison ===
  Identical: No
  Model A: 142 ops | Model B: 148 ops
  Matched: 140 | Added: 8 | Removed: 2 | Changed: 3

  Changed nodes:
    conv2d_15: Conv -> ConvInteger (quantized)
    ...

=== Weight Comparison ===
  Total params A: 3,012,416 | B: 3,012,416
  Matched weights: 87/89 | Unmatched A: 2 | Unmatched B: 4

  Per-weight summary (top divergent):
    Name                    Shape       Cosine   L2 Dist  Max Diff
    backbone.conv1.weight   [64,3,3,3]  0.9998   0.0123   0.0045
    head.cls.weight         [6,128,1,1] 0.9812   0.1456   0.0891
```

### JSON Output

Same data as console, saved as structured JSON when `--output-json` is provided.

## Dependencies

- `onnx` package (new dependency, added to requirements.txt)
- `numpy` (existing)

## Files

- Create: `src/model_analyzer.py`
- Create: `tests/test_model_analyzer.py`
- Modify: `main.py` (add `--compare-models` and `--output-json` args)
- Modify: `requirements.txt` (add `onnx`)

## Relationship to Existing Code

This is **separate from the existing `Comparator`** in `src/comparator.py`. The `Comparator` compares inference outputs at runtime (cosine similarity on predictions, detection metrics, timing). The `OnnxModelAnalyzer` compares model files statically (graph structure, weight values). They serve different purposes and do not interact.

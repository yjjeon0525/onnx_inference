# src/layer_profiler.py
import json
import os
import copy
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort


class LayerProfiler:
    def __init__(self, model_path_a: str, model_path_b: str):
        self.path_a = model_path_a
        self.path_b = model_path_b

        model_a = onnx.load(model_path_a)
        model_b = onnx.load(model_path_b)

        names_a = self._extract_intermediate_names(model_a)
        names_b = self._extract_intermediate_names(model_b)
        self.common_names = sorted(names_a & names_b)

        modified_a = self._add_intermediate_outputs(model_a, self.common_names)
        modified_b = self._add_intermediate_outputs(model_b, self.common_names)

        self.session_a = ort.InferenceSession(modified_a.SerializeToString())
        self.session_b = ort.InferenceSession(modified_b.SerializeToString())

        self.input_name_a = self.session_a.get_inputs()[0].name
        self.input_name_b = self.session_b.get_inputs()[0].name

    def _extract_intermediate_names(self, model: onnx.ModelProto) -> set:
        """Get all node output tensor names, excluding graph inputs and initializers."""
        graph_input_names = {inp.name for inp in model.graph.input}
        init_names = {init.name for init in model.graph.initializer}
        graph_output_names = {out.name for out in model.graph.output}
        exclude = graph_input_names | init_names

        names = set()
        for node in model.graph.node:
            for output in node.output:
                if output and output not in exclude:
                    names.add(output)
        return names

    def _add_intermediate_outputs(
        self, model: onnx.ModelProto, tensor_names: list
    ) -> onnx.ModelProto:
        """Clone model and add intermediate tensors as graph outputs."""
        model = copy.deepcopy(model)

        # Run shape inference to get tensor types
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception:
            pass  # Some models may fail shape inference; proceed anyway

        # Build a map of known tensor value_info
        known_vi = {}
        for vi in model.graph.value_info:
            known_vi[vi.name] = vi
        for vi in model.graph.output:
            known_vi[vi.name] = vi

        existing_output_names = {out.name for out in model.graph.output}

        for name in tensor_names:
            if name in existing_output_names:
                continue
            if name in known_vi:
                model.graph.output.append(known_vi[name])
            else:
                # Fallback: create output without shape info
                out_vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
                model.graph.output.append(out_vi)

        return model

    def _compute_metrics(self, act_a: np.ndarray, act_b: np.ndarray) -> dict:
        a = act_a.astype(np.float64).flatten()
        b = act_b.astype(np.float64).flatten()
        diff = a - b

        # Cosine similarity
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            cosine = 0.0
        else:
            cosine = float(dot / (norm_a * norm_b))

        l2 = float(np.linalg.norm(diff))
        mean_abs = float(np.mean(np.abs(diff)))
        std = float(np.std(diff))
        max_abs = float(np.max(np.abs(diff)))

        # SNR: signal norm / noise norm
        signal_norm = float(np.linalg.norm(a))
        noise_norm = float(np.linalg.norm(diff))
        snr = signal_norm / noise_norm if noise_norm > 0 else float("inf")

        return {
            "cosine_similarity": cosine,
            "l2_distance": l2,
            "mean_abs_diff": mean_abs,
            "std_diff": std,
            "max_abs_diff": max_abs,
            "snr": snr,
        }

    def profile(self, input_tensor: np.ndarray) -> dict:
        output_names_a = [o.name for o in self.session_a.get_outputs()]
        output_names_b = [o.name for o in self.session_b.get_outputs()]

        results_a = self.session_a.run(output_names_a, {self.input_name_a: input_tensor})
        results_b = self.session_b.run(output_names_b, {self.input_name_b: input_tensor})

        map_a = dict(zip(output_names_a, results_a))
        map_b = dict(zip(output_names_b, results_b))

        layers = []
        for name in self.common_names:
            if name in map_a and name in map_b:
                act_a = map_a[name]
                act_b = map_b[name]
                if act_a.shape != act_b.shape:
                    continue
                metrics = self._compute_metrics(act_a, act_b)
                metrics["name"] = name
                metrics["shape"] = list(act_a.shape)
                layers.append(metrics)

        # Sort by cosine similarity ascending (most divergent first)
        layers.sort(key=lambda x: x["cosine_similarity"])

        most_div = layers[0]["name"] if layers else ""
        least_div = layers[-1]["name"] if layers else ""

        return {
            "layers": layers,
            "summary": {
                "total_compared": len(layers),
                "most_divergent": most_div,
                "least_divergent": least_div,
            },
        }

    def print_report(self, report: dict):
        layers = report["layers"]
        summary = report["summary"]

        print(f"\n=== Layer Activation Comparison ===")
        print(f"  Compared: {summary['total_compared']} layers\n")

        header = f"  {'Name':<45} {'Shape':<22} {'Cosine':>8} {'L2 Dist':>10} {'Mean Diff':>10} {'Max Diff':>10} {'SNR':>8}"
        print(header)
        print(f"  {'-'*45} {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

        for layer in layers:
            shape_str = str(layer["shape"])
            snr_str = f"{layer['snr']:.1f}" if layer["snr"] != float("inf") else "inf"
            print(
                f"  {layer['name']:<45} {shape_str:<22} {layer['cosine_similarity']:>8.4f} "
                f"{layer['l2_distance']:>10.4f} {layer['mean_abs_diff']:>10.6f} "
                f"{layer['max_abs_diff']:>10.6f} {snr_str:>8}"
            )

        print(f"\n  Most divergent:  {summary['most_divergent']}")
        print(f"  Least divergent: {summary['least_divergent']}")

    def save_report(self, report: dict, output_path: str):
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        # Convert inf to string for JSON serialization
        def convert(obj):
            if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
                return str(obj)
            return obj

        serializable = json.loads(json.dumps(report, default=str))
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Report saved: {output_path}")

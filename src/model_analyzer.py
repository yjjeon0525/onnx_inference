# src/model_analyzer.py
import json
import numpy as np
import onnx
from onnx import numpy_helper


class OnnxModelAnalyzer:
    def __init__(self, model_path_a: str, model_path_b: str):
        self.model_a = onnx.load(model_path_a)
        self.model_b = onnx.load(model_path_b)
        self.path_a = model_path_a
        self.path_b = model_path_b

    # --- Helpers ---

    def _get_initializers(self, model: onnx.ModelProto) -> dict[str, np.ndarray]:
        return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

    def _get_nodes(self, model: onnx.ModelProto) -> list[dict]:
        nodes = []
        for node in model.graph.node:
            nodes.append({
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
            })
        return nodes

    def _get_io_shapes(self, model: onnx.ModelProto, io_list) -> list[dict]:
        result = []
        for io in io_list:
            shape = []
            if io.type.tensor_type.HasField("shape"):
                for dim in io.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        shape.append(dim.dim_param)
                    else:
                        shape.append(dim.dim_value)
            result.append({"name": io.name, "shape": shape})
        return result

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)
        dot = np.dot(a_flat, b_flat)
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    # --- Structure ---

    def quick_structure_check(self) -> dict:
        nodes_a = self._get_nodes(self.model_a)
        nodes_b = self._get_nodes(self.model_b)
        inputs_a = self._get_io_shapes(self.model_a, self.model_a.graph.input)
        inputs_b = self._get_io_shapes(self.model_b, self.model_b.graph.input)
        outputs_a = self._get_io_shapes(self.model_a, self.model_a.graph.output)
        outputs_b = self._get_io_shapes(self.model_b, self.model_b.graph.output)

        identical = (
            len(nodes_a) == len(nodes_b)
            and all(na["op_type"] == nb["op_type"] and na["name"] == nb["name"]
                    for na, nb in zip(nodes_a, nodes_b))
            and inputs_a == inputs_b
            and outputs_a == outputs_b
        )

        return {
            "identical": identical,
            "op_count_a": len(nodes_a),
            "op_count_b": len(nodes_b),
            "input_shapes_match": inputs_a == inputs_b,
            "output_shapes_match": outputs_a == outputs_b,
        }

    def detailed_structure_diff(self) -> dict:
        nodes_a = self._get_nodes(self.model_a)
        nodes_b = self._get_nodes(self.model_b)

        names_a = {n["name"]: n for n in nodes_a}
        names_b = {n["name"]: n for n in nodes_b}

        matched = []
        changed = []
        removed = []
        added = []

        for name, node_a in names_a.items():
            if name in names_b:
                node_b = names_b[name]
                if node_a["op_type"] == node_b["op_type"]:
                    matched.append({"name": name, "op_type": node_a["op_type"]})
                else:
                    changed.append({
                        "name": name,
                        "op_type_a": node_a["op_type"],
                        "op_type_b": node_b["op_type"],
                    })
            else:
                removed.append(node_a)

        for name, node_b in names_b.items():
            if name not in names_a:
                added.append(node_b)

        return {
            "matched_nodes": matched,
            "added_nodes": added,
            "removed_nodes": removed,
            "changed_nodes": changed,
        }

    # --- Weights ---

    def compare_weights(self) -> dict:
        inits_a = self._get_initializers(self.model_a)
        inits_b = self._get_initializers(self.model_b)

        # Step 1: Match by name
        matched = {}
        unmatched_a = {}
        unmatched_b = {}

        for name in inits_a:
            if name in inits_b:
                matched[name] = (inits_a[name], inits_b[name])
            else:
                unmatched_a[name] = inits_a[name]

        for name in inits_b:
            if name not in inits_a:
                unmatched_b[name] = inits_b[name]

        # Step 2: Fallback — match unmatched by shape + position
        if unmatched_a and unmatched_b:
            ua_list = list(unmatched_a.items())
            ub_list = list(unmatched_b.items())
            used_b = set()
            newly_matched_a = []
            for name_a, arr_a in ua_list:
                best_j = -1
                for j, (name_b, arr_b) in enumerate(ub_list):
                    if j in used_b:
                        continue
                    if arr_a.shape == arr_b.shape:
                        best_j = j
                        break
                if best_j >= 0:
                    name_b = ub_list[best_j][0]
                    matched[f"{name_a} <-> {name_b}"] = (arr_a, ub_list[best_j][1])
                    used_b.add(best_j)
                    newly_matched_a.append(name_a)
            for name_a in newly_matched_a:
                del unmatched_a[name_a]
            unmatched_b = {name: arr for j, (name, arr) in enumerate(ub_list) if j not in used_b}

        # Step 3: Compute metrics for matched weights
        weight_results = []
        total_params_a = sum(arr.size for arr in inits_a.values())
        total_params_b = sum(arr.size for arr in inits_b.values())

        for name, (arr_a, arr_b) in matched.items():
            diff = arr_a.astype(np.float64) - arr_b.astype(np.float64)
            weight_results.append({
                "name": name,
                "shape": list(arr_a.shape),
                "dtype_a": str(arr_a.dtype),
                "dtype_b": str(arr_b.dtype),
                "cosine_similarity": self._cosine_sim(arr_a, arr_b),
                "l2_distance": float(np.linalg.norm(diff)),
                "stats": {
                    "mean_diff": float(np.mean(np.abs(diff))),
                    "std_diff": float(np.std(diff)),
                    "min_diff": float(np.min(diff)),
                    "max_diff": float(np.max(diff)),
                    "abs_max_diff": float(np.max(np.abs(diff))),
                },
            })

        # Sort by L2 distance descending (most divergent first)
        weight_results.sort(key=lambda x: x["l2_distance"], reverse=True)

        return {
            "weights": weight_results,
            "summary": {
                "total_params_a": total_params_a,
                "total_params_b": total_params_b,
                "matched_count": len(matched),
                "unmatched_a": list(unmatched_a.keys()),
                "unmatched_b": list(unmatched_b.keys()),
            },
        }

    # --- Full Report ---

    def analyze(self) -> dict:
        return {
            "model_a": self.path_a,
            "model_b": self.path_b,
            "structure": {
                "quick": self.quick_structure_check(),
                "detailed": self.detailed_structure_diff(),
            },
            "weights": self.compare_weights(),
        }

    def print_report(self, report: dict):
        sq = report["structure"]["quick"]
        sd = report["structure"]["detailed"]
        wc = report["weights"]

        print("=== Structure Comparison ===")
        print(f"  Identical: {'Yes' if sq['identical'] else 'No'}")
        print(f"  Model A: {sq['op_count_a']} ops | Model B: {sq['op_count_b']} ops")
        print(f"  Matched: {len(sd['matched_nodes'])} | Added: {len(sd['added_nodes'])} | "
              f"Removed: {len(sd['removed_nodes'])} | Changed: {len(sd['changed_nodes'])}")

        if sd["changed_nodes"]:
            print("\n  Changed nodes:")
            for cn in sd["changed_nodes"]:
                print(f"    {cn['name']}: {cn['op_type_a']} -> {cn['op_type_b']}")

        if sd["added_nodes"]:
            print("\n  Added nodes:")
            for an in sd["added_nodes"]:
                print(f"    {an['name']}: {an['op_type']}")

        if sd["removed_nodes"]:
            print("\n  Removed nodes:")
            for rn in sd["removed_nodes"]:
                print(f"    {rn['name']}: {rn['op_type']}")

        print("\n=== Weight Comparison ===")
        ws = wc["summary"]
        print(f"  Total params A: {ws['total_params_a']:,} | B: {ws['total_params_b']:,}")
        print(f"  Matched weights: {ws['matched_count']} | "
              f"Unmatched A: {len(ws['unmatched_a'])} | Unmatched B: {len(ws['unmatched_b'])}")

        if wc["weights"]:
            print(f"\n  {'Name':<40} {'Shape':<20} {'Cosine':>8} {'L2 Dist':>10} {'Max Diff':>10}")
            print(f"  {'-'*40} {'-'*20} {'-'*8} {'-'*10} {'-'*10}")
            for wr in wc["weights"]:
                shape_str = str(wr["shape"])
                print(f"  {wr['name']:<40} {shape_str:<20} {wr['cosine_similarity']:>8.4f} "
                      f"{wr['l2_distance']:>10.4f} {wr['stats']['abs_max_diff']:>10.6f}")

    def save_report(self, report: dict, output_path: str):
        import os
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved: {output_path}")

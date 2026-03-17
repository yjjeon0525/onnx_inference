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

    def cosine_similarity(self, raw_a: list[np.ndarray], raw_b: list[np.ndarray]) -> dict:
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

    def compute_metrics(self, det_a: np.ndarray, det_b: np.ndarray, iou_threshold: float = 0.5) -> dict:
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
                if da[5] != db[5]:
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
        return {"precision": precision, "recall": recall, "f1": f1, "matched": n_matched}

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

    def add_frame_result(self, raw_outputs: dict[str, list[np.ndarray]], detections: dict[str, np.ndarray], times: dict[str, float]):
        for name, t in times.items():
            self._timing.setdefault(name, []).append(t)
        frame = {"detections": detections}
        model_names = list(raw_outputs.keys())
        if len(model_names) >= 2:
            name_a, name_b = model_names[0], model_names[1]
            if "cosine_similarity" in self.config.metrics:
                frame["cosine_similarity"] = self.cosine_similarity(raw_outputs[name_a], raw_outputs[name_b])
            det_metrics = {}
            for metric in ["precision", "recall"]:
                if metric in self.config.metrics:
                    det_metrics = self.compute_metrics(detections[name_a], detections[name_b])
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

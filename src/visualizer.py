# src/visualizer.py
import cv2
import numpy as np


def _generate_class_colors(num_classes: int = 80) -> list[tuple[int, int, int]]:
    colors = []
    for i in range(num_classes):
        hue = int(180 * i / num_classes)
        hsv = np.array([[[hue, 255, 220]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


CLASS_COLORS = _generate_class_colors(80)


class Visualizer:
    MODEL_COLORS = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
        (255, 128, 0), (128, 0, 255),
    ]

    def __init__(self, class_names: list[str] | None = None, initial_mode: str = "overlay"):
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

    def draw_detections(self, image: np.ndarray, detections: np.ndarray, model_name: str | None = None, color: tuple | None = None) -> np.ndarray:
        img = image.copy()
        if len(detections) == 0:
            return img
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            draw_color = color or CLASS_COLORS[int(cls_id) % len(CLASS_COLORS)]
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.rectangle(img, pt1, pt2, draw_color, 2)
            label = self._label(cls_id, conf, model_name)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, pt1, (pt1[0] + tw, pt1[1] - th - 4), draw_color, -1)
            cv2.putText(img, label, (pt1[0], pt1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return img

    def overlay_multi(self, image: np.ndarray, results: dict[str, np.ndarray]) -> np.ndarray:
        img = image.copy()
        for i, (model_name, dets) in enumerate(results.items()):
            color = self.MODEL_COLORS[i % len(self.MODEL_COLORS)]
            img = self.draw_detections(img, dets, model_name=model_name, color=color)
        return img

    def side_by_side(self, image: np.ndarray, results: dict[str, np.ndarray]) -> np.ndarray:
        panels = []
        for i, (model_name, dets) in enumerate(results.items()):
            color = self.MODEL_COLORS[i % len(self.MODEL_COLORS)]
            panel = self.draw_detections(image, dets, model_name=model_name, color=color)
            panels.append(panel)
        return np.concatenate(panels, axis=1)

    def render_comparison(self, image: np.ndarray, results: dict[str, np.ndarray]) -> np.ndarray:
        if self.comparison_mode == "overlay":
            return self.overlay_multi(image, results)
        else:
            return self.side_by_side(image, results)

# src/inferencer/yolov8.py
import numpy as np
from src.inferencer.base import BaseInferencer
from src.config import ModelConfig


class YOLOv8Inferencer(BaseInferencer):
    def postprocess_raw(self, raw_output: list[np.ndarray]) -> np.ndarray:
        # Model outputs: [cls_scores (1, num_classes, N), boxes (1, 4, N)]
        cls = raw_output[0][0].T      # (num_boxes, num_classes)
        boxes = raw_output[1][0].T    # (num_boxes, 4)

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        class_ids = np.argmax(cls, axis=1).astype(np.float32)
        confidences = np.max(cls, axis=1)

        detections = np.stack([x1, y1, x2, y2, confidences, class_ids], axis=1)
        return detections

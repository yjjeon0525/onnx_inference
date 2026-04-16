# src/inferencer/yolov8_dfl.py
import numpy as np
from src.inferencer.yolov8 import YOLOv8Inferencer
from src.config import ModelConfig


class YOLOv8DFLInferencer(YOLOv8Inferencer):
    """YOLOv8 inferencer with DFL (Distribution Focal Loss) box decoding.

    Handles models that output raw DFL distributions instead of
    pre-integrated box offsets. The box output has shape (1, 4*reg_max, N)
    where reg_max bins encode a discrete probability distribution over
    each of the 4 box distances (left, top, right, bottom).
    """

    def __init__(self, model_path: str, model_config: ModelConfig):
        super().__init__(model_path, model_config)
        self.reg_max = model_config.reg_max

    def _dfl_decode(self, raw_boxes: np.ndarray) -> np.ndarray:
        """Apply DFL integral to convert distributions to distance values.

        Args:
            raw_boxes: (N, 4 * reg_max) raw DFL logits

        Returns:
            (N, 4) decoded distance values [left, top, right, bottom]
        """
        n = raw_boxes.shape[0]
        # Reshape to (N, 4, reg_max)
        raw_boxes = raw_boxes.reshape(n, 4, self.reg_max)

        # Softmax over reg_max dimension
        exp = np.exp(raw_boxes - raw_boxes.max(axis=2, keepdims=True))
        softmax = exp / exp.sum(axis=2, keepdims=True)

        # Weighted sum with bin indices [0, 1, ..., reg_max-1]
        bins = np.arange(self.reg_max, dtype=np.float32)
        return (softmax * bins).sum(axis=2)  # (N, 4)

    def postprocess_raw(self, raw_output: list[np.ndarray]) -> np.ndarray:
        """Parse YOLOv8 DFL output and decode to pixel coordinates.

        Model outputs:
            raw_output[0]: cls scores (1, num_classes, N)
            raw_output[1]: DFL box logits (1, 4*reg_max, N)

        Returns:
            (N, 6) array of [x1, y1, x2, y2, confidence, class_id]
        """
        cls = raw_output[0][0].T          # (N, num_classes)
        raw_dfl = raw_output[1][0].T      # (N, 4*reg_max)

        # DFL integral: distributions -> distances
        raw_boxes = self._dfl_decode(raw_dfl)  # (N, 4)

        input_h, input_w = self._input_size
        grid, strides = self._build_grid(input_h, input_w)

        boxes = self._decode_boxes(raw_boxes, grid, strides)

        class_ids = np.argmax(cls, axis=1).astype(np.float32)
        confidences = np.max(cls, axis=1)

        detections = np.stack(
            [boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], confidences, class_ids],
            axis=1,
        )
        return detections

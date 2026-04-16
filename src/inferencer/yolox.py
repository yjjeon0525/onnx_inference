# src/inferencer/yolox.py
import numpy as np
from src.inferencer.base import BaseInferencer
from src.config import ModelConfig


class YOLOXInferencer(BaseInferencer):
    """YOLOX inferencer with grid/stride anchor decoding.

    Handles YOLOX output format where:
        reg_output: (1, 4, N) — raw box offsets (center-x, center-y, w, h)
        conf_output: (1, num_classes, N) — pre-multiplied obj * cls scores
    """

    def __init__(self, model_path: str, model_config: ModelConfig):
        super().__init__(model_path, model_config)
        self.strides = model_config.strides
        self._grid_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _build_grid(self, input_h: int, input_w: int) -> tuple[np.ndarray, np.ndarray]:
        """Build grid cell positions and stride arrays for all feature map levels.

        Returns:
            grid: (N, 2) array of (gx, gy) grid cell positions
            strides: (N,) array of stride values per cell
        """
        key = (input_h, input_w)
        if key in self._grid_cache:
            return self._grid_cache[key]

        grids = []
        stride_arr = []

        for stride in self.strides:
            feat_h = input_h // stride
            feat_w = input_w // stride
            yv, xv = np.meshgrid(
                np.arange(feat_h, dtype=np.float32),
                np.arange(feat_w, dtype=np.float32),
                indexing="ij",
            )
            grid = np.stack([xv.ravel(), yv.ravel()], axis=1)  # (feat_h*feat_w, 2)
            grids.append(grid)
            stride_arr.append(np.full(feat_h * feat_w, stride, dtype=np.float32))

        grid = np.concatenate(grids, axis=0)        # (N, 2)
        strides = np.concatenate(stride_arr, axis=0)  # (N,)

        self._grid_cache[key] = (grid, strides)
        return grid, strides

    def _decode_boxes(
        self, raw_boxes: np.ndarray, grid: np.ndarray, strides: np.ndarray
    ) -> np.ndarray:
        """Decode YOLOX raw box predictions to pixel coordinates.

        YOLOX box format: (cx_offset, cy_offset, w, h) in grid-relative space.
        Decoded as:
            cx = (grid_x + cx_offset) * stride
            cy = (grid_y + cy_offset) * stride
            w  = exp(w) * stride
            h  = exp(h) * stride

        Returns:
            (N, 4) decoded boxes in [x1, y1, x2, y2] pixel coordinates
        """
        cx = (grid[:, 0] + raw_boxes[:, 0]) * strides
        cy = (grid[:, 1] + raw_boxes[:, 1]) * strides
        w = np.exp(raw_boxes[:, 2]) * strides
        h = np.exp(raw_boxes[:, 3]) * strides
        # w = raw_boxes[:, 2] * strides
        # h = raw_boxes[:, 3] * strides

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        return np.stack([x1, y1, x2, y2], axis=1)

    def postprocess_raw(self, raw_output: list[np.ndarray]) -> np.ndarray:
        """Parse YOLOX outputs and decode to pixel coordinates.

        Model outputs:
            raw_output[0]: reg_output (1, 4, N) — raw box offsets
            raw_output[1]: conf_output (1, num_classes, N) — pre-multiplied obj * cls

        Returns:
            (N, 6) array of [x1, y1, x2, y2, confidence, class_id]
        """
        raw_boxes = raw_output[1][0].T   # (N, 4)
        cls = raw_output[0][0].T         # (N, num_classes)

        # print(raw_boxes.shape)
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

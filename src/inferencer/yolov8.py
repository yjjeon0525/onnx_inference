# src/inferencer/yolov8.py
import numpy as np
from src.inferencer.base import BaseInferencer
from src.config import ModelConfig


class YOLOv8Inferencer(BaseInferencer):
    """YOLOv8 inferencer with grid/stride anchor decoding.

    Handles the raw model output where box predictions are grid-relative
    offsets (left, top, right, bottom) that need to be decoded using
    grid cell positions and stride values.
    """

    STRIDES = [8, 16, 32]

    def __init__(self, model_path: str, model_config: ModelConfig):
        super().__init__(model_path, model_config)
        self._grid_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _build_grid(self, input_h: int, input_w: int) -> tuple[np.ndarray, np.ndarray]:
        """Build grid cell centers and stride arrays for all feature map levels.

        Returns:
            grid: (N, 2) array of (cx, cy) grid cell center coordinates
            strides: (N,) array of stride values per cell
        """
        key = (input_h, input_w)
        if key in self._grid_cache:
            return self._grid_cache[key]

        grids = []
        stride_arr = []

        for stride in self.STRIDES:
            feat_h = input_h // stride
            feat_w = input_w // stride
            # Grid cell centers: offset by 0.5 and scaled by stride
            yv, xv = np.meshgrid(
                np.arange(feat_h, dtype=np.float32),
                np.arange(feat_w, dtype=np.float32),
                indexing="ij",
            )
            grid = np.stack([xv.ravel(), yv.ravel()], axis=1)  # (feat_h*feat_w, 2)
            grids.append(grid)
            stride_arr.append(np.full(feat_h * feat_w, stride, dtype=np.float32))

        grid = np.concatenate(grids, axis=0)       # (N, 2)
        strides = np.concatenate(stride_arr, axis=0)  # (N,)

        self._grid_cache[key] = (grid, strides)
        return grid, strides

    def _decode_boxes(
        self, raw_boxes: np.ndarray, grid: np.ndarray, strides: np.ndarray
    ) -> np.ndarray:
        """Decode raw box predictions to pixel coordinates.

        Args:
            raw_boxes: (N, 4) raw predictions [left, top, right, bottom] distances
            grid: (N, 2) grid cell positions [gx, gy]
            strides: (N,) stride per cell

        Returns:
            (N, 4) decoded boxes in [x1, y1, x2, y2] pixel coordinates
        """
        strides_2d = strides[:, None]  # (N, 1)

        # Grid center in pixel space
        cx = (grid[:, 0] + 0.5) * strides
        cy = (grid[:, 1] + 0.5) * strides

        # Decode: center +/- distance * stride
        x1 = cx - raw_boxes[:, 0] * strides
        y1 = cy - raw_boxes[:, 1] * strides
        x2 = cx + raw_boxes[:, 2] * strides
        y2 = cy + raw_boxes[:, 3] * strides

        return np.stack([x1, y1, x2, y2], axis=1)

    def postprocess_raw(self, raw_output: list[np.ndarray]) -> np.ndarray:
        """Parse YOLOv8 dual output and decode to pixel coordinates.

        Model outputs:
            raw_output[0]: cls scores (1, num_classes, N)
            raw_output[1]: box offsets (1, 4, N) — raw grid-relative distances

        Returns:
            (N, 6) array of [x1, y1, x2, y2, confidence, class_id]
        """
        cls = raw_output[0][0].T       # (N, num_classes)
        raw_boxes = raw_output[1][0].T  # (N, 4)

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

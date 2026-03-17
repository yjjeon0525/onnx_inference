# src/postprocessor.py
import cv2
import numpy as np
from src.config import PostprocessConfig
from src.preprocessor import PreprocessMetadata


class Postprocessor:
    def __init__(self, config: PostprocessConfig):
        self.conf_threshold = config.conf_threshold
        self.iou_threshold = config.iou_threshold
        self.class_thresholds = config.class_thresholds or {}

    def filter_by_threshold(self, detections: np.ndarray) -> np.ndarray:
        if len(detections) == 0:
            return detections
        mask = detections[:, 4] >= self.conf_threshold
        for cls_id, cls_thresh in self.class_thresholds.items():
            cls_mask = detections[:, 5] == cls_id
            mask[cls_mask] = detections[cls_mask, 4] >= cls_thresh
        return detections[mask]

    def apply_nms(self, detections: np.ndarray) -> np.ndarray:
        if len(detections) == 0:
            return detections
        boxes = detections[:, :4].tolist()
        scores = detections[:, 4].tolist()
        boxes_xywh = []
        for x1, y1, x2, y2 in boxes:
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
        # score_threshold=0.0 since filter_by_threshold already ran
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh, scores, 0.0, self.iou_threshold
        )
        if len(indices) == 0:
            return np.empty((0, 6))
        indices = np.array(indices).flatten()
        return detections[indices]

    def revert_coordinates(
        self, detections: np.ndarray, metadata: PreprocessMetadata
    ) -> np.ndarray:
        if len(detections) == 0:
            return detections
        result = detections.copy()
        scale_x, scale_y = metadata.resize_scale
        offset_x, offset_y = metadata.crop_origin
        result[:, 0] = detections[:, 0] * scale_x + offset_x
        result[:, 1] = detections[:, 1] * scale_y + offset_y
        result[:, 2] = detections[:, 2] * scale_x + offset_x
        result[:, 3] = detections[:, 3] * scale_y + offset_y
        return result

    def process(
        self,
        detections: np.ndarray,
        metadata: PreprocessMetadata,
        nms_applied: bool,
    ) -> np.ndarray:
        detections = self.filter_by_threshold(detections)
        if not nms_applied:
            detections = self.apply_nms(detections)
        detections = self.revert_coordinates(detections, metadata)
        return detections

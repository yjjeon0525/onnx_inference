# src/preprocessor.py
from dataclasses import dataclass
import cv2
import numpy as np
from src.config import PreprocessConfig


RESIZE_METHODS = {
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
}


@dataclass
class PreprocessMetadata:
    original_shape: tuple[int, int]
    crop_origin: tuple[int, int]
    crop_size: tuple[int, int]
    resize_scale: tuple[float, float]
    input_size: tuple[int, int]
    resize_method: int


class Preprocessor:
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.cv2_method = RESIZE_METHODS.get(
            config.resize.method, cv2.INTER_LINEAR
        )

    def process(
        self, image: np.ndarray, input_size: tuple[int, int]
    ) -> tuple[np.ndarray, PreprocessMetadata]:
        h, w = image.shape[:2]
        original_shape = (h, w)

        if self.config.crop.enabled:
            rx, ry, rw, rh = self.config.crop.region
            crop_x = int(rx * w)
            crop_y = int(ry * h)
            crop_w = int(rw * w)
            crop_h = int(rh * h)
            cropped = image[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        else:
            crop_x, crop_y = 0, 0
            crop_w, crop_h = w, h
            cropped = image

        input_h, input_w = input_size
        resized = cv2.resize(cropped, (input_w, input_h), interpolation=self.cv2_method)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(chw, axis=0)

        scale_x = crop_w / input_w
        scale_y = crop_h / input_h

        metadata = PreprocessMetadata(
            original_shape=original_shape,
            crop_origin=(crop_x, crop_y),
            crop_size=(crop_w, crop_h),
            resize_scale=(scale_x, scale_y),
            input_size=input_size,
            resize_method=self.cv2_method,
        )

        return tensor, metadata

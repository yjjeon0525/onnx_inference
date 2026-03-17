# tests/test_preprocessor.py
import numpy as np
import cv2
import pytest
from src.config import PreprocessConfig, CropConfig, ResizeConfig
from src.preprocessor import Preprocessor, PreprocessMetadata


class TestPreprocessMetadata:
    def test_metadata_fields(self):
        m = PreprocessMetadata(
            original_shape=(480, 640),
            crop_origin=(0, 0),
            crop_size=(640, 480),
            resize_scale=(1.0, 1.0),
            input_size=(640, 640),
            resize_method=cv2.INTER_LINEAR,
        )
        assert m.original_shape == (480, 640)
        assert m.crop_origin == (0, 0)


class TestPreprocessorNoCrop:
    def setup_method(self):
        config = PreprocessConfig(
            crop=CropConfig(enabled=False),
            resize=ResizeConfig(method="INTER_LINEAR"),
        )
        self.preprocessor = Preprocessor(config)

    def test_output_shape_matches_input_size(self):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, meta = self.preprocessor.process(image, input_size=(640, 640))
        assert tensor.shape == (1, 3, 640, 640)

    def test_output_dtype_float32(self):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, _ = self.preprocessor.process(image, input_size=(640, 640))
        assert tensor.dtype == np.float32

    def test_output_range_0_to_1(self):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, _ = self.preprocessor.process(image, input_size=(640, 640))
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_metadata_no_crop(self):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, meta = self.preprocessor.process(image, input_size=(640, 640))
        assert meta.original_shape == (480, 640)
        assert meta.crop_origin == (0, 0)
        assert meta.crop_size == (640, 480)

    def test_rgb_conversion(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[:, :] = [255, 0, 0]  # BGR blue
        tensor, _ = self.preprocessor.process(image, input_size=(2, 2))
        assert tensor[0, 0, 0, 0] == pytest.approx(0.0, abs=0.01)  # R channel
        assert tensor[0, 2, 0, 0] == pytest.approx(1.0, abs=0.01)  # B channel


class TestPreprocessorWithCrop:
    def setup_method(self):
        config = PreprocessConfig(
            crop=CropConfig(enabled=True, region=[0.25, 0.25, 0.5, 0.5]),
            resize=ResizeConfig(method="INTER_LINEAR"),
        )
        self.preprocessor = Preprocessor(config)

    def test_crop_metadata(self):
        image = np.random.randint(0, 255, (400, 800, 3), dtype=np.uint8)
        _, meta = self.preprocessor.process(image, input_size=(640, 640))
        assert meta.crop_origin == (200, 100)  # x=0.25*800, y=0.25*400
        assert meta.crop_size == (400, 200)    # w=0.5*800, h=0.5*400

    def test_resize_scale(self):
        image = np.random.randint(0, 255, (400, 800, 3), dtype=np.uint8)
        _, meta = self.preprocessor.process(image, input_size=(640, 640))
        assert meta.resize_scale[0] == pytest.approx(400 / 640)
        assert meta.resize_scale[1] == pytest.approx(200 / 640)


class TestResizeMethods:
    def test_inter_cubic(self):
        config = PreprocessConfig(
            crop=CropConfig(enabled=False),
            resize=ResizeConfig(method="INTER_CUBIC"),
        )
        p = Preprocessor(config)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        tensor, meta = p.process(image, input_size=(64, 64))
        assert tensor.shape == (1, 3, 64, 64)
        assert meta.resize_method == cv2.INTER_CUBIC

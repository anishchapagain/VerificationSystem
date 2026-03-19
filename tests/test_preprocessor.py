"""
Unit Tests — SignaturePreprocessor
=====================================
Tests each pipeline step independently using synthetic numpy arrays.
"""
import numpy as np
import pytest
from pathlib import Path
import cv2
import tempfile

from backend.services.preprocessor import SignaturePreprocessor, PreprocessingResult
from backend.core.exceptions import ImagePreprocessingError, InvalidImageFormatError


@pytest.fixture
def preprocessor():
    return SignaturePreprocessor(target_width=256, target_height=128)


@pytest.fixture
def synthetic_bgr():
    """128×256 white BGR image with a black rectangle simulating a signature."""
    img = np.ones((128, 256, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (30, 20), (200, 100), (0, 0, 0), thickness=2)
    return img


class TestPreprocessorPipeline:

    def test_grayscale_converts_bgr_to_single_channel(self, preprocessor, synthetic_bgr):
        gray = preprocessor._to_grayscale(synthetic_bgr)
        assert gray.ndim == 2
        assert gray.shape == (128, 256)

    def test_grayscale_passthrough_for_already_gray(self, preprocessor):
        gray_in = np.ones((100, 200), dtype=np.uint8) * 200
        gray_out = preprocessor._to_grayscale(gray_in)
        assert gray_out.ndim == 2
        np.testing.assert_array_equal(gray_in, gray_out)

    def test_denoise_preserves_shape(self, preprocessor, synthetic_bgr):
        gray = preprocessor._to_grayscale(synthetic_bgr)
        blurred = preprocessor._denoise(gray)
        assert blurred.shape == gray.shape

    def test_binarize_produces_binary_image(self, preprocessor, synthetic_bgr):
        gray = preprocessor._to_grayscale(synthetic_bgr)
        blurred = preprocessor._denoise(gray)
        binary = preprocessor._binarize(blurred)
        unique_vals = np.unique(binary)
        assert set(unique_vals).issubset({0, 255}), f"Expected binary, got {unique_vals}"

    def test_resize_produces_correct_dimensions(self, preprocessor):
        dummy = np.zeros((50, 80), dtype=np.uint8)
        resized = preprocessor._resize(dummy)
        assert resized.shape == (128, 256)

    def test_normalize_output_range(self, preprocessor):
        dummy = (np.random.rand(128, 256) * 255).astype(np.uint8)
        normalized = preprocessor._normalize(dummy)
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_run_from_array_full_pipeline(self, preprocessor, synthetic_bgr):
        result = preprocessor.run_from_array(synthetic_bgr)
        assert isinstance(result, PreprocessingResult)
        assert result.image.shape == (128, 256)
        assert result.image.dtype == np.float32
        assert result.image.min() >= 0.0
        assert result.image.max() <= 1.0

    def test_invalid_extension_raises(self, preprocessor):
        with pytest.raises(InvalidImageFormatError):
            preprocessor.run("signature.pdf")

    def test_run_saves_debug_steps_when_enabled(self, synthetic_bgr):
        p = SignaturePreprocessor(debug=True)
        result = p.run_from_array(synthetic_bgr)
        assert len(result.steps) > 0

    def test_run_from_file(self, preprocessor, synthetic_bgr):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, synthetic_bgr)
            result = preprocessor.run(tmp.name)
        assert result.image.shape == (128, 256)


class TestMorphologicalCleanup:

    def test_cleanup_removes_single_noise_pixel(self, preprocessor):
        binary = np.zeros((128, 256), dtype=np.uint8)
        binary[64, 128] = 255  # single noise pixel
        cleaned = preprocessor._morphological_cleanup(binary)
        # After opening, isolated single pixel should be removed
        assert cleaned.sum() == 0 or cleaned[64, 128] == 0

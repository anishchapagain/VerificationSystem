"""
Image Preprocessing Pipeline
==============================
Module  : services/preprocessor.py
Purpose : Transform raw signature images into clean, normalized binary arrays
          ready for embedding extraction. Each step is a discrete method to
          allow unit testing and debugging in isolation.

Author  : Signature Verifier Team
Version : 1.0.0
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from backend.config import get_settings
from backend.core.exceptions import (
    ImageLoadError,
    ImagePreprocessingError,
    InvalidImageFormatError,
)
from backend.core.logger import get_logger

log = get_logger("preprocessor")
settings = get_settings()


@dataclass
class PreprocessingResult:
    """
    Value object returned by the PreprocessingPipeline.

    Attributes:
        image      : Final normalized grayscale numpy array (H×W, float32, [0,1]).
        original   : Original loaded image before any processing.
        steps      : Ordered list of (step_name, array) snapshots for debugging.
        width      : Final image width in pixels.
        height     : Final image height in pixels.
    """
    image: np.ndarray
    original: np.ndarray
    steps: list = field(default_factory=list)

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    def __repr__(self) -> str:
        return f"<PreprocessingResult shape={self.image.shape} dtype={self.image.dtype}>"


class SignaturePreprocessor:
    """
    Multi-step OpenCV pipeline that converts a raw signature image into
    a clean, normalized binary array suitable for the Siamese Network.

    Pipeline Steps
    --------------
    1. Load         — Read the image file from disk.
    2. Grayscale    — Convert BGR → single channel.
    3. Denoise      — Gaussian blur to smooth scan/photo artifacts.
    4. Binarize     — Otsu's adaptive thresholding → pure black/white.
    5. Morphology   — Close small holes; open pepper noise.
    6. Crop         — Tight bounding box around the ink region.
    7. Resize       — Scale to the target (W × H) expected by the model.
    8. Normalize    — Pixel values from uint8 [0,255] → float32 [0.0,1.0].

    Usage
    -----
        preprocessor = SignaturePreprocessor()
        result = preprocessor.run(image_path="/path/to/sig.png")
        tensor_input = result.image  # shape (128, 256), float32
    """

    ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def __init__(
        self,
        target_width: int = None,
        target_height: int = None,
        blur_kernel: Tuple[int, int] = (5, 5),
        morph_kernel_size: int = 3,
        debug: bool = False,
    ) -> None:
        """
        Initialize the preprocessor with configurable parameters.

        Args:
            target_width      : Output image width (default from settings).
            target_height     : Output image height (default from settings).
            blur_kernel       : Gaussian blur kernel size (must be odd × odd).
            morph_kernel_size : Morphological operation kernel radius.
            debug             : If True, saves intermediate steps in result.steps.
        """
        self.target_width = target_width or settings.IMAGE_TARGET_WIDTH
        self.target_height = target_height or settings.IMAGE_TARGET_HEIGHT
        self.blur_kernel = blur_kernel
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self.use_binarization = settings.USE_BINARIZATION
        self.use_cropping = settings.USE_CROPPING
        self.use_clahe = settings.USE_CLAHE
        self.use_aspect_ratio_resize = settings.USE_ASPECT_RATIO_RESIZE
        self.clahe_clip_limit = settings.CLAHE_CLIP_LIMIT
        self.debug = debug
        log.debug(
            f"SignaturePreprocessor initialized | "
            f"target=({self.target_width}×{self.target_height}) | "
            f"blur={blur_kernel} | debug={debug}"
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    def run(self, image_path: str) -> PreprocessingResult:
        """
        Execute the full preprocessing pipeline on a file path.

        Args:
            image_path (str): Absolute or relative path to the signature image.

        Returns:
            PreprocessingResult: Contains the final normalized array and debug info.

        Raises:
            InvalidImageFormatError : Unsupported file extension.
            ImageLoadError          : File cannot be read by OpenCV.
            ImagePreprocessingError : Any intermediate step fails.
        """
        path = Path(image_path)
        self._validate_extension(path)

        log.info(f"Starting preprocessing | file={path.name}")
        steps = []

        try:
            original = self._load(path)
            if self.debug:
                steps.append(("original", original.copy()))

            gray = self._to_grayscale(original)
            if self.debug:
                steps.append(("grayscale", gray.copy()))

            # Optimized Step: Contrast Normalization
            if self.use_clahe:
                processed = self._apply_clahe(gray)
                if self.debug:
                    steps.append(("clahe", processed.copy()))
            else:
                processed = gray

            blurred = self._denoise(processed)
            if self.debug:
                steps.append(("denoised", blurred.copy()))

            if self.use_binarization:
                binary = self._binarize(blurred)
                if self.debug:
                    steps.append(("binarized", binary.copy()))

                cleaned = self._morphological_cleanup(binary)
                if self.debug:
                    steps.append(("morphology", cleaned.copy()))
                
                input_for_crop = cleaned
            else:
                input_for_crop = blurred

            if self.use_cropping:
                cropped = self._crop_to_signature(input_for_crop)
                if self.debug:
                    steps.append(("cropped", cropped.copy()))
                input_for_resize = cropped
            else:
                input_for_resize = input_for_crop

            # Optimized Step: Aspect-Ratio Preserving Resize
            if self.use_aspect_ratio_resize:
                resized = self._resize_with_aspect_ratio(input_for_resize)
            else:
                resized = self._resize(input_for_resize)
                
            if self.debug:
                steps.append(("resized", resized.copy()))

            normalized = self._normalize(resized)

            log.info(
                f"Preprocessing complete | file={path.name} | "
                f"output_shape={normalized.shape}"
            )
            return PreprocessingResult(
                image=normalized, original=original, steps=steps
            )

        except (InvalidImageFormatError, ImageLoadError, ImagePreprocessingError):
            raise
        except Exception as exc:
            log.error(f"Unexpected preprocessing error | file={path.name} | error={exc}")
            raise ImagePreprocessingError("unknown", detail=str(exc)) from exc

    def run_from_array(self, array: np.ndarray) -> PreprocessingResult:
        """
        Execute the pipeline directly on a numpy array (e.g., a video frame).

        Args:
            array (np.ndarray): BGR or grayscale image as a numpy array.

        Returns:
            PreprocessingResult: Contains the final normalized array.
        """
        try:
            original = array.copy()
            gray = self._to_grayscale(array) if array.ndim == 3 else array
            blurred = self._denoise(gray)
            binary = self._binarize(blurred)
            cleaned = self._morphological_cleanup(binary)
            cropped = self._crop_to_signature(cleaned)
            resized = self._resize(cropped)
            normalized = self._normalize(resized)
            return PreprocessingResult(image=normalized, original=original)
        except Exception as exc:
            log.error(f"run_from_array failed | error={exc}")
            raise ImagePreprocessingError("array_input", detail=str(exc)) from exc

    # ─── Pipeline Steps ───────────────────────────────────────────────────────

    def _validate_extension(self, path: Path) -> None:
        """Raise InvalidImageFormatError if the file extension is not supported."""
        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise InvalidImageFormatError(path.name)

    def _load(self, path: Path) -> np.ndarray:
        """
        Load an image from disk using OpenCV.

        Raises:
            ImageLoadError: If the file does not exist or cannot be decoded.
        """
        if not path.exists():
            raise ImageLoadError(str(path), detail="File does not exist")

        image = cv2.imread(str(path))
        if image is None:
            raise ImageLoadError(str(path), detail="OpenCV returned None — file may be corrupted")

        log.debug(f"Image loaded | shape={image.shape} | file={path.name}")
        return image

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert a BGR image to single-channel grayscale."""
        try:
            if image.ndim == 2:
                return image  # Already grayscale
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as exc:
            raise ImagePreprocessingError("grayscale", detail=str(exc)) from exc

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to reduce noise and smooth pen strokes."""
        try:
            return cv2.GaussianBlur(gray, self.blur_kernel, sigmaX=0)
        except Exception as exc:
            raise ImagePreprocessingError("denoise", detail=str(exc)) from exc

    def _binarize(self, blurred: np.ndarray) -> np.ndarray:
        """
        Apply Otsu's global thresholding to produce a binary image.

        Otsu's algorithm automatically finds the optimal threshold that
        separates the ink (foreground) from the paper (background).
        The image is inverted so ink = white (255), paper = black (0).
        """
        try:
            _, binary = cv2.threshold(
                blurred, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            return binary
        except Exception as exc:
            raise ImagePreprocessingError("binarize", detail=str(exc)) from exc

    def _morphological_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """
        Apply morphological closing then opening to clean the binary image.

        - Closing (dilate then erode): fills small holes within strokes.
        - Opening (erode then dilate): removes isolated noise pixels.
        """
        try:
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, self.morph_kernel)
            return opened
        except Exception as exc:
            raise ImagePreprocessingError("morphology", detail=str(exc)) from exc

    def _crop_to_signature(self, binary: np.ndarray) -> np.ndarray:
        """
        Compute the tight bounding box around the signature ink and crop to it.

        Adds a small 5-pixel padding so edge strokes are not cut off.
        Falls back to the full image if no contours are detected.
        """
        try:
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                log.warning("No contours found — using full image without crop")
                return binary

            # Merge all contours into one bounding rectangle
            x, y, w, h = cv2.boundingRect(np.vstack(contours))

            # Add padding (clamped to image boundaries)
            # Tighter padding (2px) for aggressive logic
            pad = 2
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, binary.shape[1])
            y2 = min(y + h + pad, binary.shape[0])

            cropped = binary[y1:y2, x1:x2]
            log.debug(f"Cropped to signature | bbox=({x1},{y1},{x2},{y2})")
            return cropped
        except Exception as exc:
            raise ImagePreprocessingError("crop", detail=str(exc)) from exc

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize the cropped image to the model's expected input dimensions."""
        try:
            return cv2.resize(
                image,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_AREA,
            )
        except Exception as exc:
            raise ImagePreprocessingError("resize", detail=str(exc)) from exc

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Convert pixel values from uint8 [0, 255] to float32 [0.0, 1.0]."""
        try:
            return image.astype(np.float32) / 255.0
        except Exception as exc:
            raise ImagePreprocessingError("normalize", detail=str(exc)) from exc

    def _apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        try:
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8))
            return clahe.apply(gray)
        except Exception as exc:
            raise ImagePreprocessingError("clahe", detail=str(exc)) from exc

    def _resize_with_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        """Resize while preserving aspect ratio by adding zero-padding."""
        try:
            h, w = image.shape[:2]
            target_w, target_h = self.target_width, self.target_height
            
            # Calculate scaling factor
            aspect = w / h
            target_aspect = target_w / target_h
            
            if aspect > target_aspect:
                # Width is the limiting factor
                new_w = target_w
                new_h = int(new_w / aspect)
            else:
                # Height is the limiting factor
                new_h = target_h
                new_w = int(new_h * aspect)
                
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Pad to target size (centering)
            top = (target_h - new_h) // 2
            bottom = target_h - new_h - top
            left = (target_w - new_w) // 2
            right = target_w - new_w - left
            
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right, 
                cv2.BORDER_CONSTANT, value=0
            )
            return padded
        except Exception as exc:
            raise ImagePreprocessingError("aspect_resize", detail=str(exc)) from exc

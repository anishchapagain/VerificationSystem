"""
Video Frame Extraction Service
================================
Module  : services/video_handler.py
Purpose : Extract the sharpest signature frame(s) from a video file,
          then pipe them into the preprocessing pipeline.

Author  : Signature Verifier Team
Version : 1.0.0
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from backend.config import get_settings
from backend.core.exceptions import NoUsableFrameError, VideoLoadError
from backend.core.logger import get_logger
from backend.services.preprocessor import PreprocessingResult, SignaturePreprocessor

log = get_logger("video_handler")
settings = get_settings()


@dataclass
class FrameCandidate:
    """A single extracted video frame with its quality score."""
    frame_index: int
    array: np.ndarray
    sharpness: float

    def __repr__(self) -> str:
        return (
            f"<FrameCandidate idx={self.frame_index} "
            f"shape={self.array.shape} sharpness={self.sharpness:.2f}>"
        )


class VideoSignatureExtractor:
    """
    Reads a video file, scores every Nth frame by sharpness (Laplacian variance),
    selects the top K sharpest frames, and returns their preprocessed results.

    Sharpness Metric
    ----------------
    Laplacian variance: the variance of pixel values after applying a Laplacian
    kernel. Higher variance = more high-frequency detail = sharper image.
    This naturally penalizes motion-blurred frames.

    Usage
    -----
        extractor = VideoSignatureExtractor()
        results = extractor.extract(video_path="sig_video.mp4")
        # results is a list of PreprocessingResult (one per top frame)
    """

    MIN_SHARPNESS_THRESHOLD = 50.0  # Frames below this are too blurry to use

    def __init__(
        self,
        stride: int = None,
        top_frames: int = None,
        preprocessor: SignaturePreprocessor = None,
    ) -> None:
        """
        Args:
            stride       : Process every Nth frame (default from settings).
            top_frames   : Return top K sharpest frames (default from settings).
            preprocessor : Injected SignaturePreprocessor (creates one if None).
        """
        self.stride = stride or settings.VIDEO_FRAME_STRIDE
        self.top_frames = top_frames or settings.VIDEO_TOP_FRAMES
        self.preprocessor = preprocessor or SignaturePreprocessor()
        log.info(
            f"VideoSignatureExtractor initialized | "
            f"stride={self.stride} | top_frames={self.top_frames}"
        )

    def extract(self, video_path: str) -> List[PreprocessingResult]:
        """
        Full pipeline: open video → sample frames → rank by sharpness →
        preprocess top K → return results.

        Args:
            video_path (str): Path to the video file.

        Returns:
            List[PreprocessingResult]: One result per selected top frame.

        Raises:
            VideoLoadError      : If the file cannot be opened.
            NoUsableFrameError  : If no frame meets the minimum sharpness.
        """
        path = Path(video_path)
        log.info(f"Extracting frames | file={path.name}")

        cap = self._open_video(path)
        try:
            candidates = self._sample_frames(cap)
        finally:
            cap.release()

        if not candidates:
            log.warning(f"No usable frames found | file={path.name}")
            raise NoUsableFrameError()

        top = self._select_top_frames(candidates)
        results = self._preprocess_frames(top)

        log.info(
            f"Extraction complete | file={path.name} | "
            f"sampled={len(candidates)} | selected={len(top)}"
        )
        return results

    # ─── Internal Methods ──────────────────────────────────────────────────────

    def _open_video(self, path: Path) -> cv2.VideoCapture:
        """Open a VideoCapture or raise VideoLoadError."""
        if not path.exists():
            raise VideoLoadError(str(path), detail="File not found")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoLoadError(str(path), detail="OpenCV could not open the video")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        log.debug(
            f"Video opened | file={path.name} | frames={total_frames} | fps={fps:.1f}"
        )
        return cap

    def _sample_frames(self, cap: cv2.VideoCapture) -> List[FrameCandidate]:
        """
        Read every Nth frame and score it by Laplacian variance.

        Frames below MIN_SHARPNESS_THRESHOLD are discarded early.
        """
        candidates: List[FrameCandidate] = []
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % self.stride == 0:
                sharpness = self._laplacian_variance(frame)
                if sharpness >= self.MIN_SHARPNESS_THRESHOLD:
                    candidates.append(
                        FrameCandidate(
                            frame_index=frame_index,
                            array=frame,
                            sharpness=sharpness,
                        )
                    )
                    log.debug(
                        f"Frame accepted | idx={frame_index} | sharpness={sharpness:.2f}"
                    )
                else:
                    log.debug(
                        f"Frame rejected (too blurry) | idx={frame_index} | "
                        f"sharpness={sharpness:.2f}"
                    )

            frame_index += 1

        log.info(f"Sampled {frame_index} frames | accepted {len(candidates)}")
        return candidates

    def _laplacian_variance(self, frame: np.ndarray) -> float:
        """
        Compute the Laplacian variance of a frame as a sharpness proxy.

        Higher = sharper. Motion-blurred frames produce very low variance.

        Args:
            frame: BGR image array.

        Returns:
            float: Laplacian variance score.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _select_top_frames(
        self, candidates: List[FrameCandidate]
    ) -> List[FrameCandidate]:
        """Sort by sharpness descending and return top K frames."""
        sorted_candidates = sorted(candidates, key=lambda c: c.sharpness, reverse=True)
        top = sorted_candidates[: self.top_frames]
        log.info(
            f"Top {len(top)} frames selected | "
            f"sharpness range: {top[-1].sharpness:.1f}–{top[0].sharpness:.1f}"
        )
        return top

    def _preprocess_frames(
        self, frames: List[FrameCandidate]
    ) -> List[PreprocessingResult]:
        """Run each selected frame through the preprocessing pipeline."""
        results = []
        for candidate in frames:
            try:
                result = self.preprocessor.run_from_array(candidate.array)
                results.append(result)
                log.debug(
                    f"Frame preprocessed | idx={candidate.frame_index} | "
                    f"output_shape={result.image.shape}"
                )
            except Exception as exc:
                log.warning(
                    f"Frame preprocessing failed | idx={candidate.frame_index} | "
                    f"error={exc} — skipping"
                )

        if not results:
            raise NoUsableFrameError()

        return results

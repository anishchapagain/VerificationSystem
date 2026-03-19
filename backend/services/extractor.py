"""
Embedding extractor service.

Loads the trained SiameseNetwork at startup and provides a thread-safe
interface for extracting L2-normalized embedding vectors from preprocessed
signature images. Implements a singleton pattern so the model is loaded
only once per process.

Author: Signature Verifier Team
"""

import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from backend.config import get_settings
from backend.core.exceptions import ModelNotLoadedError, EmbeddingExtractionError
from backend.core.logger import logger
from backend.models.siamese_net import SiameseNetwork, load_model
from backend.services.preprocessor import PreprocessedImage

settings = get_settings()


class EmbeddingExtractor:
    """
    Thread-safe service that wraps the SiameseNetwork for inference.

    The model is loaded once via initialize() and then reused across all
    requests. Thread safety is ensured by a threading.Lock during model
    loading (inference itself is read-only and thread-safe in eval mode).

    Attributes:
        _model      : Loaded SiameseNetwork instance.
        _device     : Torch device ('cuda' or 'cpu').
        _is_loaded  : Whether the model has been successfully loaded.

    Usage:
        extractor = EmbeddingExtractor()
        extractor.initialize()                   # Called once on startup
        embedding = extractor.extract(preprocessed_image)
    """

    def __init__(self) -> None:
        self._model: SiameseNetwork | None = None
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_loaded: bool = False
        self._lock = threading.Lock()

    # ─── Lifecycle ───────────────────────────────────────────────────────────

    def initialize(self, weights_path: str | Path | None = None) -> None:
        """
        Load the SiameseNetwork from the checkpoint file.

        If no weights_path is provided, uses the path from application config.
        This method is idempotent — calling it twice does nothing.

        Args:
            weights_path: Optional override for the model weights file path.

        Raises:
            ModelNotLoadedError: If the weights file is missing.
            EmbeddingExtractionError: If the model fails to load.
        """
        with self._lock:
            if self._is_loaded:
                logger.debug("Model already loaded, skipping initialization")
                return

            path = str(weights_path or settings.MODEL_WEIGHTS_PATH)
            logger.info(f"Loading SiameseNetwork from '{path}' on device='{self._device}'")

            try:
                if not Path(path).exists():
                    logger.warning(
                        f"Weights file not found at '{path}'. "
                        "Starting with a fresh untrained model. "
                        "Run ml/train.py to generate trained weights."
                    )
                    self._model = SiameseNetwork(embedding_dim=settings.EMBEDDING_DIM)
                    self._model.to(self._device)
                    self._model.eval()
                else:
                    self._model = load_model(
                        weights_path=path,
                        embedding_dim=settings.EMBEDDING_DIM,
                        device=self._device,
                    )

                self._is_loaded = True
                param_count = sum(p.numel() for p in self._model.parameters())
                logger.info(
                    f"Model loaded successfully — "
                    f"params={param_count:,}, device={self._device}"
                )

            except FileNotFoundError as exc:
                raise ModelNotLoadedError() from exc
            except Exception as exc:
                logger.error(f"Failed to load model: {exc}")
                raise EmbeddingExtractionError(f"Model loading failed: {exc}") from exc

    def shutdown(self) -> None:
        """Release model from memory. Call during application shutdown."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
                self._is_loaded = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Model unloaded from memory")

    # ─── Inference ───────────────────────────────────────────────────────────

    def extract(self, preprocessed: PreprocessedImage) -> np.ndarray:
        """
        Extract a 512-D L2-normalized embedding from a preprocessed image.

        Args:
            preprocessed: Result from SignaturePreprocessor.run().

        Returns:
            float32 numpy array of shape (embedding_dim,).

        Raises:
            ModelNotLoadedError      : If initialize() has not been called.
            EmbeddingExtractionError : On inference failure.
        """
        if not self._is_loaded or self._model is None:
            raise ModelNotLoadedError()

        try:
            tensor = self._array_to_tensor(preprocessed.image_array)

            with torch.no_grad():
                embedding = self._model.embed(tensor)  # (1, D)

            result = embedding.cpu().numpy().squeeze(0)  # (D,)
            logger.debug(f"Embedding extracted: shape={result.shape}, norm={np.linalg.norm(result):.4f}")
            return result.astype(np.float32)

        except ModelNotLoadedError:
            raise
        except Exception as exc:
            logger.error(f"Embedding extraction failed: {exc}")
            raise EmbeddingExtractionError(f"Inference error: {exc}") from exc

    def extract_from_array(self, image_array: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a raw float32 numpy array (H, W).

        Convenience method when you already have a normalized array
        without a full PreprocessedImage wrapper.

        Args:
            image_array: float32 array of shape (H, W) in [0, 1].

        Returns:
            float32 numpy array of shape (embedding_dim,).
        """
        if not self._is_loaded or self._model is None:
            raise ModelNotLoadedError()

        try:
            tensor = self._array_to_tensor(image_array)
            with torch.no_grad():
                embedding = self._model.embed(tensor)
            return embedding.cpu().numpy().squeeze(0).astype(np.float32)
        except Exception as exc:
            raise EmbeddingExtractionError(f"Inference error: {exc}") from exc

    def compute_similarity(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two embedding vectors.

        Because embeddings are L2-normalized, cosine similarity equals
        the dot product: sim = a · b ∈ [-1, 1], where 1 = identical.

        Args:
            embedding_a: float32 array of shape (D,).
            embedding_b: float32 array of shape (D,).

        Returns:
            Cosine similarity score in [-1.0, 1.0].
        """
        try:
            a = torch.tensor(embedding_a, dtype=torch.float32).unsqueeze(0)
            b = torch.tensor(embedding_b, dtype=torch.float32).unsqueeze(0)
            similarity = F.cosine_similarity(a, b, dim=1).item()
            return float(similarity)
        except Exception as exc:
            raise EmbeddingExtractionError(f"Similarity computation failed: {exc}") from exc

    # ─── Properties ──────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """True if the model has been successfully loaded."""
        return self._is_loaded

    @property
    def device(self) -> str:
        """The torch device the model is running on."""
        return self._device

    # ─── Private Helpers ─────────────────────────────────────────────────────

    def _array_to_tensor(self, image_array: np.ndarray) -> torch.Tensor:
        """
        Convert a (H, W) float32 numpy array to a (1, 1, H, W) torch tensor.

        Args:
            image_array: Normalized float32 numpy array.

        Returns:
            Torch tensor on the model's device.
        """
        # Add batch and channel dimensions: (H, W) → (1, 1, H, W)
        tensor = torch.from_numpy(image_array).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return tensor.to(self._device)


# ─── Application Singleton ────────────────────────────────────────────────────

embedding_extractor = EmbeddingExtractor()

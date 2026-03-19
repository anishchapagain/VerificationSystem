"""
Siamese Neural Network — Architecture + Inference Manager
===========================================================
Module  : models/siamese_net.py
Purpose : Define the twin CNN that maps signature images to 512-D embedding
          vectors, and the ModelManager that loads checkpoints and serves
          inference requests from FastAPI routes.

CHECKPOINT FORMAT (produced by ml/train.py and scripts/generate_weights.py):
    {
        "epoch":       int,
        "model_state": OrderedDict,   # model.state_dict()
        "best_eer":    float | None,
        "config": {
            "embedding_dim": int,
            "architecture":  str,
            "input_shape":   list,
        }
    }

Author  : Signature Verifier Team
Version : 2.0.0
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.core.exceptions import EmbeddingExtractionError, ModelNotLoadedError
from backend.core.logger import get_logger

log = get_logger("siamese_net")


# ─── Building Blocks ──────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Conv2d → BatchNorm2d → ReLU → (optional MaxPool).

    Keeping this as a named block makes the architecture easy to visualise,
    swap out (e.g. replace MaxPool with stride-2 Conv), and unit-test.

    Args:
        in_ch  : Input feature map channels.
        out_ch : Output feature map channels.
        pool   : If True, append MaxPool2d(2×2) after ReLU.
    """

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True) -> None:
        super().__init__()
        layers: list = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─── Encoder (shared CNN backbone) ───────────────────────────────────────────

class SignatureEncoder(nn.Module):
    """
    Shared CNN backbone: signature image → L2-normalised embedding vector.

    Input shape : (B, 1, 128, 256) — batch × grayscale × H × W
    Output shape: (B, embedding_dim) — L2-normalised (unit sphere)

    Spatial reduction after 4× MaxPool(2):
        128 → 64 → 32 → 16 → 8     (height)
        256 → 128 → 64 → 32 → 16   (width)
    Feature maps after 4 ConvBlocks: 256 channels × 8 × 16 = 32,768

    Args:
        embedding_dim : Dimension of the output vector. Default 512.
        dropout       : FC dropout probability. Default 0.4.
    """

    def __init__(self, embedding_dim: int = 512, dropout: float = 0.4) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            ConvBlock(1,   32,  pool=True),   # → (B, 32,  64, 128)
            ConvBlock(32,  64,  pool=True),   # → (B, 64,  32,  64)
            ConvBlock(64,  128, pool=True),   # → (B, 128, 16,  32)
            ConvBlock(128, 256, pool=True),   # → (B, 256,  8,  16)
        )

        self._flat_dim = 256 * 8 * 16   # = 32,768

        self.fc = nn.Sequential(
            nn.Linear(self._flat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 128, 256) float32 in [0, 1]
        Returns:
            (B, embedding_dim) L2-normalised
        """
        features  = self.cnn(x)
        flat      = features.view(features.size(0), -1)
        embedding = self.fc(flat)
        return F.normalize(embedding, p=2, dim=1)


# ─── Siamese Network ─────────────────────────────────────────────────────────

class SiameseNetwork(nn.Module):
    """
    Twin network sharing a single SignatureEncoder backbone.

    Weight sharing ensures both branches learn the same embedding space.

    During training  : call forward(img1, img2) → (emb1, emb2)
    During inference : call forward_one(img)    → emb

    Args:
        embedding_dim : Passed through to SignatureEncoder.
        dropout       : Passed through to SignatureEncoder.
    """

    def __init__(self, embedding_dim: int = 512, dropout: float = 0.4) -> None:
        super().__init__()
        self.encoder = SignatureEncoder(embedding_dim=embedding_dim, dropout=dropout)

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Single branch — used for inference."""
        return self.encoder(x)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Both branches — used for training with contrastive loss."""
        return self.encoder(x1), self.encoder(x2)


# ─── Model Manager (Inference Service) ───────────────────────────────────────

class ModelManager:
    """
    Loads a SiameseNetwork checkpoint and serves embedding extraction
    for FastAPI routes via extract_embedding().

    Handles two checkpoint formats:
      1. Full checkpoint dict (produced by train.py / generate_weights.py):
             {"model_state": ..., "epoch": ..., "config": ...}
      2. Bare state_dict (legacy, produced by older training scripts):
             OrderedDict of layer weights

    GPU acceleration: if CUDA is available the model runs on GPU automatically.
    Inference uses torch.no_grad() for speed and AMP autocast for efficiency.

    Usage:
        manager = ModelManager("weights/siamese_best.pt")
        manager.load()
        embedding = manager.extract_embedding(preprocessed_image_array)
    """

    def __init__(
        self,
        weights_path: str,
        embedding_dim: int = 512,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            weights_path  : Path to the .pt checkpoint file.
            embedding_dim : Must match the value used during training.
            device        : "cuda" | "cpu" | None (auto-detect).
        """
        self.weights_path  = Path(weights_path)
        self.embedding_dim = embedding_dim
        self.device        = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model: Optional[SiameseNetwork] = None
        self._amp_enabled  = (self.device.type == "cuda")

        log.info(
            f"ModelManager configured | device={self.device} | "
            f"amp={self._amp_enabled} | weights={weights_path}"
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device_info(self) -> dict:
        """Return a summary dict for the /health endpoint."""
        info = {"device": str(self.device), "amp": self._amp_enabled}
        if self.device.type == "cuda":
            info["gpu_name"]     = torch.cuda.get_device_name(0)
            info["vram_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
            info["vram_used_gb"] = round(
                torch.cuda.memory_allocated(0) / 1e9, 3
            )
        return info

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load the model from the checkpoint file.

        Handles both full checkpoint dicts and bare state_dicts.
        Sets the model to eval mode and warms it up with a dummy forward pass.

        Raises:
            ModelNotLoadedError      : Weights file does not exist.
            EmbeddingExtractionError : Checkpoint incompatible with architecture.
        """
        if not self.weights_path.exists():
            log.error(f"Weights file not found: {self.weights_path}")
            raise ModelNotLoadedError()

        try:
            log.info(f"Loading checkpoint | path={self.weights_path}")

            raw = torch.load(
                self.weights_path,
                map_location=self.device,
                weights_only=False,   # Needed to load our dict checkpoints
            )

            # ── Detect checkpoint format ──────────────────────────────────────
            if isinstance(raw, dict) and "model_state" in raw:
                state_dict = raw["model_state"]
                cfg        = raw.get("config", {})
                epoch      = raw.get("epoch", 0)
                eer        = raw.get("best_eer")

                # Validate embedding_dim matches
                saved_dim = cfg.get("embedding_dim")
                if saved_dim and saved_dim != self.embedding_dim:
                    log.warning(
                        f"embedding_dim mismatch: checkpoint={saved_dim}, "
                        f"config={self.embedding_dim}. Using checkpoint value."
                    )
                    self.embedding_dim = saved_dim

                log.info(
                    f"Checkpoint metadata | epoch={epoch} | "
                    f"eer={f'{eer:.4f}' if eer is not None else 'N/A (random init)'} | "
                    f"embedding_dim={self.embedding_dim}"
                )
            else:
                # Bare state_dict (legacy format)
                state_dict = raw
                log.info("Bare state_dict detected (legacy format)")

            model = SiameseNetwork(embedding_dim=self.embedding_dim, dropout=0.0)
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()

            # Warm-up pass: JIT-compiles cuDNN kernels for our input size
            self._warmup(model)

            self._model = model
            total_params = sum(p.numel() for p in model.parameters())
            log.info(
                f"Model loaded | params={total_params:,} | "
                f"device={self.device} | amp={self._amp_enabled}"
            )
            if self.device.type == "cuda":
                used_gb = torch.cuda.memory_allocated(0) / 1e9
                log.info(f"VRAM after load: {used_gb:.2f} GB")

        except ModelNotLoadedError:
            raise
        except Exception as exc:
            log.error(f"Checkpoint load failed: {exc}")
            raise EmbeddingExtractionError(
                detail=f"Failed to load checkpoint: {exc}"
            ) from exc

    def _warmup(self, model: SiameseNetwork) -> None:
        """
        Run one silent dummy forward pass to trigger cuDNN kernel compilation.
        This makes the first real inference request fast instead of slow.
        """
        try:
            dummy = torch.zeros(1, 1, 128, 256, device=self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self._amp_enabled):
                    _ = model.forward_one(dummy)
            log.debug("Model warm-up completed")
        except Exception as exc:
            log.warning(f"Warm-up failed (non-fatal): {exc}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def extract_embedding(self, image_array: np.ndarray) -> np.ndarray:
        """
        Run inference on a preprocessed image and return the embedding vector.

        The image_array must already be preprocessed by SignaturePreprocessor:
            - Shape: (128, 256) or (H, W) float32
            - Values: [0.0, 1.0]
            - Convention: ink = light (normalised from binary)

        Args:
            image_array : float32 numpy array of shape (H, W).

        Returns:
            np.ndarray  : 1-D float32 array of shape (embedding_dim,),
                          L2-normalised (unit sphere).

        Raises:
            ModelNotLoadedError      : If load() has not been called.
            EmbeddingExtractionError : If inference fails for any reason.
        """
        if not self.is_loaded:
            raise ModelNotLoadedError()

        try:
            # (H, W) → (1, 1, H, W)
            tensor = (
                torch.from_numpy(image_array)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device, non_blocking=True)
            )

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self._amp_enabled):
                    embedding = self._model.forward_one(tensor)

            result: np.ndarray = embedding.squeeze(0).cpu().float().numpy()
            norm = float(np.linalg.norm(result))

            log.debug(
                f"Embedding extracted | shape={result.shape} | "
                f"norm={norm:.6f} | device={self.device}"
            )
            return result

        except ModelNotLoadedError:
            raise
        except Exception as exc:
            log.error(f"extract_embedding failed: {exc}")
            raise EmbeddingExtractionError(detail=str(exc)) from exc

    def extract_batch(self, image_arrays: list) -> np.ndarray:
        """
        Extract embeddings for a batch of images in one forward pass.

        More efficient than calling extract_embedding() in a loop when
        processing multiple video frames.

        Args:
            image_arrays : List of float32 numpy arrays, each (H, W).

        Returns:
            np.ndarray : Shape (N, embedding_dim), each row L2-normalised.
        """
        if not self.is_loaded:
            raise ModelNotLoadedError()

        try:
            batch = torch.stack([
                torch.from_numpy(arr).unsqueeze(0)
                for arr in image_arrays
            ]).to(self.device, non_blocking=True)  # (N, 1, H, W)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self._amp_enabled):
                    embeddings = self._model.forward_one(batch)

            return embeddings.cpu().float().numpy()

        except ModelNotLoadedError:
            raise
        except Exception as exc:
            log.error(f"extract_batch failed: {exc}")
            raise EmbeddingExtractionError(detail=str(exc)) from exc
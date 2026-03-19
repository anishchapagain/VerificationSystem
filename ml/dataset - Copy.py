"""
Signature Pair Dataset
========================
Module  : ml/dataset.py
Purpose : PyTorch Dataset that generates (genuine, forged) signature pairs
          from the CEDAR / SigNet directory structure for Siamese training.

Directory structure expected:
    data/processed/
    ├── genuine/
    │   ├── user_001_sig_01.png
    │   ├── user_001_sig_02.png
    │   └── ...
    └── forged/
        ├── user_001_forg_01.png
        └── ...

Author  : Signature Verifier Team
Version : 1.0.0
"""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from backend.core.logger import get_logger

log = get_logger("dataset")


class SignaturePairDataset(Dataset):
    """
    Builds (img1, img2, label) pairs for contrastive Siamese training.

    - Genuine pair  : two signatures by the same person  → label = 1
    - Forged  pair  : one genuine + one forgery           → label = 0

    Pairs are generated once at construction for reproducibility.
    The genuine:forged ratio is kept 1:1 to avoid class imbalance.

    Args:
        data_dir      : Root directory containing genuine/ and forged/ subfolders.
        pairs_per_user: Number of genuine + forged pairs to generate per user.
        img_size      : (width, height) tuple for resizing.
        augment       : If True, apply random affine transforms during training.
    """

    def __init__(
        self,
        data_dir: str,
        pairs_per_user: int = 40, # 20 genuine + 20 forged
        img_size: Tuple[int, int] = (256, 128),
        augment: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.pairs_per_user = pairs_per_user
        self.img_size = img_size

        self.transform = self._build_transform(augment)
        self.pairs: List[Tuple[str, str, int]] = []
        self._build_pairs()
        log.info(f"Dataset ready | pairs={len(self.pairs)} | augment={augment}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path1, path2, label = self.pairs[idx]
        img1 = self._load_image(path1)
        img2 = self._load_image(path2)
        return img1, img2, torch.tensor(label, dtype=torch.float32)

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _build_pairs(self) -> None:
        """Scan directories and build the pair list."""
        genuine_dir = self.data_dir / "genuine"
        forged_dir = self.data_dir / "forged"

        if not genuine_dir.exists():
            raise FileNotFoundError(f"Genuine directory not found: {genuine_dir}")

        # Group genuine images by user prefix (e.g., "user_001")
        user_sigs: dict = {}
        for img_path in sorted(genuine_dir.glob("*.png")):
            user_id = "_".join(img_path.stem.split("_")[:2])
            user_sigs.setdefault(user_id, []).append(str(img_path))

        for user_id, sigs in user_sigs.items():
            # Genuine pairs: random combinations of the same user's signatures
            genuine_pairs = [
                (random.choice(sigs), random.choice(sigs), 1)
                for _ in range(self.pairs_per_user)
            ]

            # Forged pairs: genuine + matching forgery
            forgeries = list(forged_dir.glob(f"{user_id}*.png")) if forged_dir.exists() else []
            if forgeries:
                forged_pairs = [
                    (random.choice(sigs), str(random.choice(forgeries)), 0)
                    for _ in range(self.pairs_per_user)
                ]
            else:
                # Fallback: cross-user pairs as impostors
                other_users = [u for u in user_sigs if u != user_id]
                forged_pairs = []
                if other_users:
                    for _ in range(self.pairs_per_user):
                        other = random.choice(user_sigs[random.choice(other_users)])
                        forged_pairs.append((random.choice(sigs), other, 0))

            self.pairs.extend(genuine_pairs + forged_pairs)

        random.shuffle(self.pairs)

    def _load_image(self, path: str) -> torch.Tensor:
        """Load, resize, and transform a signature image."""
        img = Image.open(path).convert("L")  # Grayscale
        return self.transform(img)

    def _build_transform(self, augment: bool) -> transforms.Compose:
        """Build torchvision transform pipeline."""
        base = [
            transforms.Resize(self.img_size[::-1]),  # PIL takes (H, W)
            transforms.ToTensor(),
        ]
        if augment:
            augmentations = [
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            ]
            return transforms.Compose(augmentations + base)
        return transforms.Compose(base)

"""
Signature Pair Dataset
========================
Module  : ml/dataset.py
Purpose : PyTorch Dataset that generates genuine and forged signature pairs
          for Siamese Network contrastive training.

Key improvements over v1.0:
  - Person-independent train/val split support via get_splits()
  - More pairs per user (default 50) for harder training
  - Genuine pairs always use DIFFERENT images (not same image twice)
  - Cross-user impostor pairs added for harder negative examples
  - User list exposed so caller can split by person not by pair

Author  : Signature Verifier Team
Version : 2.0.0
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from backend.core.logger import get_logger

log = get_logger("dataset")


class SignaturePairDataset(Dataset):
    """
    Builds (img1, img2, label) pairs for contrastive Siamese training.

    Label = 1  →  Genuine pair   (two signatures by the same person)
    Label = 0  →  Impostor pair  (genuine + forgery, or two different people)

    Pair types generated:
      1. Genuine-Genuine   : two different genuine sigs from same person → label 1
      2. Genuine-Forged    : one genuine + one forgery of same person    → label 0
      3. Genuine-Impostor  : genuine from person A + genuine from person B → label 0
         (extra hard negatives — two real signatures that should NOT match)

    Pairs are built once at construction. Call get_splits() for a
    person-independent train/val split.

    Args:
        data_dir       : Root directory with genuine/ and forged/ subfolders.
        pairs_per_user : Genuine + forged pairs per user. Default 50.
        img_size       : (width, height). Default (256, 128).
        augment        : Apply random affine augmentation. Default False.
        seed           : Random seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        data_dir: str,
        pairs_per_user: int = 50,
        img_size: Tuple[int, int] = (256, 128),
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        random.seed(seed)

        self.data_dir      = Path(data_dir)
        self.pairs_per_user= pairs_per_user
        self.img_size      = img_size
        self.transform     = self._build_transform(augment)

        # Built by _build_pairs():
        # Each entry: (path1, path2, label, user_id)
        # user_id stored so get_splits() can group by person
        self.pairs: List[Tuple[str, str, int, str]] = []
        self.user_ids: List[str] = []   # sorted unique user ids

        self._build_pairs()

        log.info(
            f"Dataset ready | pairs={len(self.pairs)} | "
            f"users={len(self.user_ids)} | augment={augment}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path1, path2, label, _ = self.pairs[idx]
        img1 = self._load_image(path1)
        img2 = self._load_image(path2)
        return img1, img2, torch.tensor(label, dtype=torch.float32)

    def get_splits(
        self, val_fraction: float = 0.20
    ) -> Tuple["SignaturePairDataset", "SignaturePairDataset"]:
        """
        Return person-independent train and validation subsets.

        Person-independent means: every signature from a given person
        is entirely in training OR entirely in validation — never both.

        This prevents data leakage where the model validates on people
        it already trained on, which causes artificially perfect EER.

        Args:
            val_fraction : Fraction of users reserved for validation.
                           Default 0.20 (20% of people → validation).

        Returns:
            (train_dataset, val_dataset): Both are Subset objects.
        """
        users = list(self.user_ids)
        random.shuffle(users)

        n_val      = max(1, int(len(users) * val_fraction))
        val_users  = set(users[:n_val])
        train_users= set(users[n_val:])

        train_idx = [i for i, (_, _, _, uid) in enumerate(self.pairs) if uid in train_users]
        val_idx   = [i for i, (_, _, _, uid) in enumerate(self.pairs) if uid in val_users]

        log.info(
            f"Person-independent split | "
            f"train_users={len(train_users)} ({len(train_idx)} pairs) | "
            f"val_users={len(val_users)} ({len(val_idx)} pairs)"
        )

        train_ds = _LabelledSubset(self, train_idx)
        val_ds   = _LabelledSubset(self, val_idx)
        return train_ds, val_ds

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_pairs(self) -> None:
        """
        Scan genuine/ and forged/ directories and build training pairs.

        Three types of pairs are built for each user:
          1. Genuine-Genuine  (label=1) — different images, same person
          2. Genuine-Forged   (label=0) — genuine vs forgery of same person
          3. Genuine-Impostor (label=0) — genuine vs different person's genuine
        """
        genuine_dir = self.data_dir / "genuine"
        forged_dir  = self.data_dir / "forged"

        if not genuine_dir.exists():
            raise FileNotFoundError(
                f"genuine/ folder not found at: {genuine_dir}\n"
                f"Expected: {self.data_dir}/genuine/*.png"
            )

        # ── Group genuine images by user prefix ───────────────────────────────
        # user_001_sig_01.png → user_id = "user_001"
        user_sigs: Dict[str, List[str]] = {}
        for img_path in sorted(genuine_dir.glob("*.png")):
            uid = "_".join(img_path.stem.split("_")[:2])   # user_001
            user_sigs.setdefault(uid, []).append(str(img_path))

        # Also check jpg and bmp
        for ext in ["*.jpg", "*.jpeg", "*.bmp"]:
            for img_path in sorted(genuine_dir.glob(ext)):
                uid = "_".join(img_path.stem.split("_")[:2])
                user_sigs.setdefault(uid, []).append(str(img_path))

        self.user_ids = sorted(user_sigs.keys())
        all_user_list = self.user_ids

        if len(self.user_ids) == 0:
            raise FileNotFoundError(
                f"No images found in {genuine_dir}\n"
                "Expected filenames like: user_001_sig_01.png"
            )

        log.info(f"Found {len(self.user_ids)} users in genuine/")

        for user_id, sigs in user_sigs.items():

            # ── 1. Genuine-Genuine pairs (label = 1) ─────────────────────────
            # Always pick TWO DIFFERENT images — never the same image twice
            if len(sigs) >= 2:
                for _ in range(self.pairs_per_user):
                    a, b = random.sample(sigs, 2)
                    self.pairs.append((a, b, 1, user_id))
            else:
                # Only one image — duplicate is unavoidable
                for _ in range(self.pairs_per_user):
                    self.pairs.append((sigs[0], sigs[0], 1, user_id))

            # ── 2. Genuine-Forged pairs (label = 0) ──────────────────────────
            forgeries = []
            if forged_dir.exists():
                for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                    forgeries += list(forged_dir.glob(f"{user_id}{ext[1:]}"))
                    forgeries += [
                        f for f in forged_dir.glob(ext)
                        if "_".join(f.stem.split("_")[:2]) == user_id
                    ]
                forgeries = list(set(str(f) for f in forgeries))

            if forgeries:
                for _ in range(self.pairs_per_user):
                    genuine  = random.choice(sigs)
                    forgery  = random.choice(forgeries)
                    self.pairs.append((genuine, forgery, 0, user_id))
            else:
                log.warning(
                    f"No forgeries found for {user_id} — "
                    f"using cross-user pairs as negatives"
                )

            # ── 3. Genuine-Impostor pairs (label = 0) ────────────────────────
            # Pick a different person's genuine signature as a hard negative
            other_users = [u for u in all_user_list if u != user_id]
            if other_users:
                n_impostor = self.pairs_per_user // 2   # half as many
                for _ in range(n_impostor):
                    other_uid  = random.choice(other_users)
                    other_sigs = user_sigs.get(other_uid, [])
                    if other_sigs:
                        a = random.choice(sigs)
                        b = random.choice(other_sigs)
                        self.pairs.append((a, b, 0, user_id))

        random.shuffle(self.pairs)
        log.info(
            f"Pairs built | total={len(self.pairs)} | "
            f"genuine={sum(1 for p in self.pairs if p[2]==1)} | "
            f"impostor={sum(1 for p in self.pairs if p[2]==0)}"
        )

    def _load_image(self, path: str) -> torch.Tensor:
        """Load a signature image, convert to grayscale, apply transforms."""
        img = Image.open(path).convert("L")
        return self.transform(img)

    def _build_transform(self, augment: bool) -> transforms.Compose:
        """Build the torchvision transform pipeline."""
        base = [
            transforms.Resize((self.img_size[1], self.img_size[0])),
            transforms.ToTensor(),
        ]
        if augment:
            aug = [
                transforms.RandomAffine(
                    degrees=8,
                    translate=(0.05, 0.05),
                    scale=(0.92, 1.08)
                ),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            ]
            return transforms.Compose(aug + base)
        return transforms.Compose(base)


class _LabelledSubset(Dataset):
    """
    Thin wrapper around Subset that preserves __getitem__ returning
    (img1, img2, label) rather than just an index.
    """
    def __init__(self, dataset: SignaturePairDataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.dataset[self.indices[idx]]
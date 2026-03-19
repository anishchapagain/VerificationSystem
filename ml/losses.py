"""
Custom Loss Functions
=======================
Module  : ml/losses.py
Purpose : Contrastive loss for Siamese Network training.

Author  : Signature Verifier Team
Version : 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Network training.

    For a pair (emb1, emb2) with label:
        label = 1  →  genuine pair  (same person)
        label = 0  →  forged pair   (different person)

    Loss = label × d²  +  (1 - label) × max(margin - d, 0)²

    Where d = Euclidean distance between the two L2-normalised embeddings.

    The margin parameter defines how far apart impostor pairs should be pushed.
    A margin of 1.0 works well when embeddings are L2-normalised (unit sphere).

    Args:
        margin (float): Minimum desired distance between impostor pairs.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        if margin <= 0:
            raise ValueError(f"margin must be > 0, got {margin}")
        self.margin = margin

    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss for a batch of pairs.

        Args:
            embedding1 : (B, D) — first embeddings, L2-normalised.
            embedding2 : (B, D) — second embeddings, L2-normalised.
            label      : (B,)   — 1.0 for genuine, 0.0 for forged.

        Returns:
            Scalar mean loss for the batch.
        """
        distance = F.pairwise_distance(embedding1, embedding2, p=2)
        genuine_loss = label * distance.pow(2)
        impostor_loss = (1.0 - label) * F.relu(self.margin - distance).pow(2)
        loss = (genuine_loss + impostor_loss).mean()
        return loss

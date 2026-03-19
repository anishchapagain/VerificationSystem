"""
Generate Initialised Weights for Immediate API Testing
========================================================
Module  : scripts/generate_weights.py
Purpose : Create a valid siamese_best.pt checkpoint using random initialisation
          so you can run and test the full FastAPI inference pipeline IMMEDIATELY,
          before training on real signature data.

WHAT THIS GIVES YOU:
  - A structurally valid .pt checkpoint the API can load right now
  - Full API functionality: /register, /verify, /health all work
  - Preprocessing, embedding extraction, FAISS all run end-to-end
  - Scores will be RANDOM (meaningless) until replaced with trained weights

WHAT YOU DO NEXT:
  1. Run this script        → API works immediately
  2. Start training         → python -m ml.train --batch_size 64
  3. Training auto-replaces → weights/siamese_best.pt with real weights
  4. Restart the API        → Now scores are accurate

Usage:
    python scripts/generate_weights.py
    python scripts/generate_weights.py --embedding_dim 512 --output weights/siamese_best.pt

Author  : Signature Verifier Team
Version : 1.0.0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from backend.core.logger import get_logger, setup_logger
from backend.models.siamese_net import SiameseNetwork

setup_logger(log_level="INFO")
log = get_logger("generate_weights")


def generate_weights(output_path: str = "weights/siamese_best.pt", embedding_dim: int = 512) -> None:
    """
    Create a randomly-initialised SiameseNetwork checkpoint.

    Uses PyTorch's default weight init:
      - Conv2d layers: Kaiming uniform (good for ReLU activations)
      - Linear layers: Xavier uniform
      - BatchNorm: weight=1, bias=0

    The checkpoint format is IDENTICAL to what train.py produces,
    so ModelManager.load() accepts it without modification.

    Args:
        output_path   : Where to save the .pt file.
        embedding_dim : Must match EMBEDDING_DIM in your .env.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device          : {device}")
    if device.type == "cuda":
        log.info(f"GPU             : {torch.cuda.get_device_name(0)}")
    log.info(f"Embedding dim   : {embedding_dim}")
    log.info(f"Output path     : {output_path}")
    log.info("")

    # ── Build model ───────────────────────────────────────────────────────────
    model = SiameseNetwork(embedding_dim=embedding_dim, dropout=0.4)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {total_params:,}")

    # ── Smoke test ────────────────────────────────────────────────────────────
    # Verify the model produces (1, embedding_dim) L2-normalised output
    dummy = torch.randn(1, 1, 128, 256, device=device)
    with torch.no_grad():
        emb = model.forward_one(dummy)

    assert emb.shape == (1, embedding_dim), f"Shape mismatch: {emb.shape}"
    norm = emb.norm().item()
    assert abs(norm - 1.0) < 1e-4, f"Not L2-normalised: norm={norm:.6f}"
    log.info(f"Smoke test      : PASSED | shape={tuple(emb.shape)} | norm={norm:.6f}")

    # ── Test pair forward pass ────────────────────────────────────────────────
    img1 = torch.randn(4, 1, 128, 256, device=device)
    img2 = torch.randn(4, 1, 128, 256, device=device)
    with torch.no_grad():
        e1, e2 = model(img1, img2)
        import torch.nn.functional as F
        sim = F.cosine_similarity(e1, e2)
    log.info(f"Pair test       : PASSED | cosine_sim range=[{sim.min():.3f}, {sim.max():.3f}]")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    checkpoint = {
        "epoch":       0,
        "model_state": model.state_dict(),
        "best_eer":    None,
        "config": {
            "embedding_dim": embedding_dim,
            "architecture":  "SiameseNetwork",
            "input_shape":   [1, 128, 256],
            "status":        "random_init — replace with trained weights",
        },
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(out))

    size_mb = out.stat().st_size / 1e6
    log.info(f"File size       : {size_mb:.1f} MB")
    log.info("")
    log.info("=" * 62)
    log.info("  ✅  weights/siamese_best.pt created successfully!")
    log.info("  ⚡  FastAPI server can now load and run inference.")
    log.info("  ⚠️  Verification SCORES are random until training completes.")
    log.info("")
    log.info("  NEXT STEPS:")
    log.info("  1. Start the API:  uvicorn backend.main:app --reload")
    log.info("  2. Check health:   curl http://localhost:8000/health")
    log.info("  3. Start training: python -m ml.train --batch_size 64")
    log.info("     (runs in background, replaces this file when done)")
    log.info("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate initialised model weights")
    parser.add_argument("--output",        default="weights/siamese_best.pt")
    parser.add_argument("--embedding_dim", type=int, default=512)
    args = parser.parse_args()
    generate_weights(args.output, args.embedding_dim)

"""
Siamese Network Training Script — GPU + AMP Optimised
========================================================
Module  : ml/train.py
Purpose : Full training loop with CUDA AMP (Automatic Mixed Precision),
          early stopping, checkpoint saving, and per-epoch evaluation.

Designed for: 8GB VRAM GPU, 24GB RAM (e.g. RTX 3070 / RTX 4060 Ti)

Key GPU optimisations applied:
  - torch.cuda.amp.GradScaler + autocast  → FP16 forward pass (2× speed, ½ VRAM)
  - torch.backends.cudnn.benchmark = True → fastest conv algorithm per input size
  - num_workers=6 + pin_memory=True       → overlap CPU data prep with GPU compute
  - gradient clipping                     → prevents exploding gradients with FP16
  - torch.compile() (PyTorch 2.x)         → JIT-compile model for extra ~20% speedup

Usage:
    python -m ml.train --data_dir data/processed --epochs 100 --batch_size 64

Author  : Signature Verifier Team
Version : 1.0.0
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.logger import get_logger, setup_logger
from backend.models.siamese_net import SiameseNetwork
from ml.dataset import SignaturePairDataset
from ml.evaluate import compute_eer, accuracy_at_threshold
from ml.losses import ContrastiveLoss

setup_logger(log_level="INFO")
log = get_logger("trainer")


# ─── GPU Tuning ───────────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True   # Let cuDNN find fastest conv algorithm
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Siamese Signature Network (GPU + AMP)")
    p.add_argument("--data_dir",      default="data/processed")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=64,    help="64 is ideal for 8GB VRAM")
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--embedding_dim", type=int,   default=512)
    p.add_argument("--margin",        type=float, default=1.0)
    p.add_argument("--val_split",     type=float, default=0.15)
    p.add_argument("--output",        default="weights/siamese_best.pt")
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--workers",       type=int,   default=6,     help="DataLoader workers")
    p.add_argument("--amp",           action="store_true", default=True,  help="Use AMP FP16")
    p.add_argument("--compile",       action="store_true", default=False, help="torch.compile (PyTorch 2.x)")
    p.add_argument("--resume",        default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


# ─── Training Epoch ───────────────────────────────────────────────────────────

def train_epoch(
    model: SiameseNetwork,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: ContrastiveLoss,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """
    Run one full training epoch with AMP.

    Returns:
        (mean_loss, epoch_time_seconds)
    """
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for img1, img2, labels in loader:
        img1   = img1.to(device, non_blocking=True)
        img2   = img2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # Faster than zero_grad()

        # ── AMP forward pass ──────────────────────────────────────────────────
        with autocast(enabled=use_amp):
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, labels)

        # ── Scaled backward pass ──────────────────────────────────────────────
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader), time.time() - t0


# ─── Validation Epoch ─────────────────────────────────────────────────────────

@torch.no_grad()
def validate_epoch(
    model: SiameseNetwork,
    loader: DataLoader,
    criterion: ContrastiveLoss,
    device: torch.device,
    use_amp: bool,
    threshold: float = 0.85,
) -> dict:
    """
    Run validation. Returns dict with loss, EER, and accuracy metrics.
    """
    import torch.nn.functional as F

    model.eval()
    total_loss  = 0.0
    all_scores  = []
    all_labels  = []

    for img1, img2, labels in loader:
        img1   = img1.to(device, non_blocking=True)
        img2   = img2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, labels)

        total_loss += loss.item()
        scores = F.cosine_similarity(emb1, emb2)
        all_scores.extend(scores.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    eer     = compute_eer(all_labels, all_scores)
    metrics = accuracy_at_threshold(all_labels, all_scores, threshold)

    return {
        "val_loss":  total_loss / len(loader),
        "eer":       eer,
        "accuracy":  metrics["accuracy"],
        "far":       metrics["far"],
        "frr":       metrics["frr"],
        "f1":        metrics["f1"],
    }


# ─── VRAM Reporter ────────────────────────────────────────────────────────────

def log_gpu_stats(device: torch.device) -> None:
    """Log current GPU VRAM usage (allocated and reserved)."""
    if device.type != "cuda":
        return
    alloc   = torch.cuda.memory_allocated(device)  / 1e9
    reserved = torch.cuda.memory_reserved(device)  / 1e9
    total   = torch.cuda.get_device_properties(device).total_memory / 1e9
    log.info(
        f"GPU VRAM | allocated={alloc:.2f}GB | "
        f"reserved={reserved:.2f}GB | total={total:.1f}GB"
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("=" * 60)
    log.info("Siamese Network Training — GPU + AMP")
    log.info(f"Device      : {device}")
    if device.type == "cuda":
        log.info(f"GPU         : {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log.info(f"AMP FP16    : {args.amp}")
    log.info(f"Batch size  : {args.batch_size}")
    log.info(f"Epochs      : {args.epochs}")
    log.info(f"LR          : {args.lr}")
    log.info(f"Embedding   : {args.embedding_dim}-D")
    log.info("=" * 60)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset  = SignaturePairDataset(args.data_dir, pairs_per_user=40, augment=True) # pairs_per_user=20 -> 20 genuine + 20 forged pairs per user
    val_size = int(len(dataset) * args.val_split)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(args.workers // 2, 2),
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )
    log.info(f"Dataset | train={len(train_ds)} pairs | val={len(val_ds)} pairs")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SiameseNetwork(embedding_dim=args.embedding_dim, dropout=0.4).to(device)

    if args.compile:
        log.info("Compiling model with torch.compile() — this takes ~1 min first run...")
        model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {total_params:,}")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    best_eer    = float("inf")

    if args.resume and Path(args.resume).exists():
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            start_epoch = checkpoint.get("epoch", 1) + 1
            best_eer    = checkpoint.get("best_eer", float("inf"))
            log.info(f"Resumed from {args.resume} | epoch={start_epoch} | best_eer={best_eer:.4f}")
        else:
            model.load_state_dict(checkpoint)
            log.info(f"Loaded weights from {args.resume}")

    # ── Optimiser, Loss, Scheduler ────────────────────────────────────────────
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler    = GradScaler(enabled=(args.amp and device.type == "cuda"))

    log_gpu_stats(device)

    # ── Training Loop ─────────────────────────────────────────────────────────
    output_path     = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patience_count  = 0

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, epoch_time = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, args.amp
        )
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, args.amp
        )
        scheduler.step()

        log.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"EER={val_metrics['eer']:.4f} | "
            f"acc={val_metrics['accuracy']:.4f} | "
            f"FAR={val_metrics['far']:.4f} | "
            f"FRR={val_metrics['frr']:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        # Save best checkpoint
        if val_metrics["eer"] < best_eer:
            best_eer       = val_metrics["eer"]
            patience_count = 0

            checkpoint = {
                "epoch":       epoch,
                "model_state": model.state_dict()
                               if not args.compile
                               else model._orig_mod.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "best_eer":    best_eer,
                "config": {
                    "embedding_dim": args.embedding_dim,
                    "architecture":  "SiameseNetwork",
                    "input_size":    [1, 128, 256],
                },
            }
            torch.save(checkpoint, str(output_path))
            log.info(
                f"  ✅ Best checkpoint saved | EER={best_eer:.4f} | "
                f"acc={val_metrics['accuracy']:.4f} | path={output_path}"
            )
        else:
            patience_count += 1
            if patience_count >= args.patience:
                log.info(
                    f"Early stopping at epoch {epoch} | "
                    f"best EER={best_eer:.4f} | patience={args.patience}"
                )
                break

        if epoch % 10 == 0:
            log_gpu_stats(device)

    log.info("=" * 60)
    log.info(f"Training complete | best EER={best_eer:.4f} | saved to {output_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

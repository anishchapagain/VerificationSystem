"""
Siamese Network Training Script — GPU + AMP Optimised
========================================================
Module  : ml/train.py
Purpose : Full training loop with person-independent validation split,
          CUDA AMP, early stopping, and checkpoint saving.

Key fix in v3.0:
  Uses dataset.get_splits() for person-independent train/val separation.
  This prevents data leakage that caused EER=0.0000 from epoch 1.

Usage:
    python -m ml.train --data_dir data/processed --epochs 100 --batch_size 64

Author  : Signature Verifier Team
Version : 3.0.0
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.logger import get_logger, setup_logger
from backend.models.siamese_net import SiameseNetwork
from ml.dataset import SignaturePairDataset
from ml.evaluate import compute_eer, accuracy_at_threshold
from ml.losses import ContrastiveLoss

setup_logger(log_level="INFO")
log = get_logger("trainer")

torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Siamese Signature Network")
    p.add_argument("--data_dir",      default="data/processed")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--embedding_dim", type=int,   default=512)
    p.add_argument("--margin",        type=float, default=1.0)
    p.add_argument("--val_split",     type=float, default=0.20,
                   help="Fraction of PEOPLE reserved for validation (person-independent)")
    p.add_argument("--pairs_per_user",type=int,   default=50,
                   help="Genuine + forged pairs generated per user")
    p.add_argument("--output",        default="weights/siamese_best.pt")
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--workers",       type=int,   default=6)
    p.add_argument("--amp",           action="store_true", default=True)
    p.add_argument("--compile",       action="store_true", default=False)
    p.add_argument("--resume",        default=None)
    return p.parse_args()


def train_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for img1, img2, labels in loader:
        img1   = img1.to(device, non_blocking=True)
        img2   = img2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader), time.time() - t0


@torch.no_grad()
def validate_epoch(model, loader, criterion, device, use_amp, threshold=0.85):
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
        "val_loss": total_loss / len(loader),
        "eer":      eer,
        "accuracy": metrics["accuracy"],
        "far":      metrics["far"],
        "frr":      metrics["frr"],
        "f1":       metrics["f1"],
    }


def log_gpu_stats(device):
    if device.type != "cuda":
        return
    alloc   = torch.cuda.memory_allocated(device) / 1e9
    total   = torch.cuda.get_device_properties(device).total_memory / 1e9
    log.info(f"GPU VRAM | allocated={alloc:.2f}GB | total={total:.1f}GB")


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("=" * 60)
    log.info("Siamese Network Training  v3.0")
    log.info(f"Device        : {device}")
    if device.type == "cuda":
        log.info(f"GPU           : {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    log.info(f"AMP FP16      : {args.amp and device.type == 'cuda'}")
    log.info(f"Batch size    : {args.batch_size}")
    log.info(f"Epochs        : {args.epochs}")
    log.info(f"LR            : {args.lr}")
    log.info(f"Pairs/user    : {args.pairs_per_user}")
    log.info(f"Val split     : {args.val_split} (person-independent)")
    log.info("=" * 60)

    # ── Dataset — person-independent split ───────────────────────────────────
    dataset  = SignaturePairDataset(
        args.data_dir,
        pairs_per_user=args.pairs_per_user,
        augment=True,
        seed=42,
    )

    # KEY FIX: split by person, not by pair
    train_ds, val_ds = dataset.get_splits(val_fraction=args.val_split)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(args.workers // 2, 0),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    log.info(
        f"Split | train={len(train_ds)} pairs | "
        f"val={len(val_ds)} pairs"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SiameseNetwork(embedding_dim=args.embedding_dim, dropout=0.4).to(device)

    if args.compile:
        log.info("Compiling model with torch.compile()...")
        model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {total_params:,}")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_eer    = float("inf")

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            start_epoch = ckpt.get("epoch", 1) + 1
            best_eer    = ckpt.get("best_eer") or float("inf")
            log.info(
                f"Resumed | path={args.resume} | "
                f"epoch={start_epoch} | best_eer={best_eer}"
            )
        else:
            model.load_state_dict(ckpt)
            log.info(f"Loaded weights from {args.resume}")

    # ── Optimiser, Loss, Scheduler ────────────────────────────────────────────
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler    = GradScaler(enabled=(args.amp and device.type == "cuda"))

    log_gpu_stats(device)

    # ── Training loop ─────────────────────────────────────────────────────────
    output_path    = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patience_count = 0

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, epoch_time = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, args.amp and device.type == "cuda"
        )
        val_m = validate_epoch(
            model, val_loader, criterion, device,
            args.amp and device.type == "cuda"
        )
        scheduler.step()

        log.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_m['val_loss']:.4f} | "
            f"EER={val_m['eer']:.4f} | "
            f"acc={val_m['accuracy']:.4f} | "
            f"FAR={val_m['far']:.4f} | "
            f"FRR={val_m['frr']:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        if val_m["eer"] < best_eer:
            best_eer       = val_m["eer"]
            patience_count = 0

            state = model.state_dict() if not args.compile \
                    else model._orig_mod.state_dict()

            checkpoint = {
                "epoch":       epoch,
                "model_state": state,
                "optimizer":   optimizer.state_dict(),
                "best_eer":    float(best_eer),   # always a real float
                "config": {
                    "embedding_dim": args.embedding_dim,
                    "architecture":  "SiameseNetwork",
                    "input_shape":   [1, 128, 256],
                },
            }
            torch.save(checkpoint, str(output_path))
            log.info(
                f"  ✅ Best checkpoint saved | "
                f"EER={best_eer:.4f} | "
                f"acc={val_m['accuracy']:.4f} | "
                f"path={output_path}"
            )
        else:
            patience_count += 1
            if patience_count >= args.patience:
                log.info(
                    f"Early stopping at epoch {epoch} | "
                    f"best EER={best_eer:.4f}"
                )
                break

        if epoch % 10 == 0:
            log_gpu_stats(device)

    log.info("=" * 60)
    log.info(f"Training complete | best EER={best_eer:.4f} | saved to {output_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
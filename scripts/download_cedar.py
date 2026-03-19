"""
CEDAR Dataset Organiser
========================
Module  : scripts/download_cedar.py
Purpose : Scan the extracted CEDAR folder and copy images into the
          genuine/ and forged/ structure expected by SignaturePairDataset.

CEDAR Dataset Details:
  - 55 signers
  - 24 genuine signatures per signer
  - 24 skilled forgeries per signer
  - Total: 2,640 images
  - Source: http://www.cedar.buffalo.edu/NIJ/data/signatures.rar

Expected folder structure after extraction:
    signatures_cedar/
        full_org/
            original_1_1.png     ← user 1, signature 1
            original_1_2.png     ← user 1, signature 2
            original_1_3.png
            ...
            original_55_24.png   ← user 55, signature 24
        full_forg/
            forgeries_1_1.png    ← user 1, forgery 1
            forgeries_1_2.png
            ...
            forgeries_55_24.png  ← user 55, forgery 24

Output structure:
    data/processed/
        genuine/
            user_001_sig_01.png
            user_001_sig_02.png
            ...
        forged/
            user_001_forg_01.png
            ...

Usage:
    # signatures_cedar is in the project root (default)
    python scripts/download_cedar.py

    # Custom path
    python scripts/download_cedar.py --cedar_dir C:\\Downloads\\signatures_cedar

Author  : Signature Verifier Team
Version : 4.0.0
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.logger import get_logger, setup_logger

setup_logger(log_level="INFO")
log = get_logger("download_cedar")


def organize_cedar(cedar_dir: str, output_dir: str = "data/processed") -> None:
    """
    Read flat image files from full_org/ and full_forg/ and copy them
    into genuine/ and forged/ with a clean consistent naming convention.

    File naming pattern inside CEDAR:
        original_{user}_{sig}.png     e.g. original_1_1.png
        forgeries_{user}_{sig}.png    e.g. forgeries_1_1.png

    Output naming:
        user_001_sig_01.png
        user_001_forg_01.png

    Args:
        cedar_dir  : Path to the extracted CEDAR root folder.
        output_dir : Destination for genuine/ and forged/ folders.
    """
    source = Path(cedar_dir)
    out    = Path(output_dir)

    # ── Validate ──────────────────────────────────────────────────────────────
    if not source.exists():
        log.error(f"Folder not found: {source.resolve()}")
        log.error(
            "Make sure the path is correct.\n"
            "Example: python scripts/download_cedar.py --cedar_dir signatures_cedar"
        )
        sys.exit(1)

    full_org  = source / "full_org"
    full_forg = source / "full_forg"

    if not full_org.exists():
        log.error(
            f"Could not find full_org/ inside {source.resolve()}\n"
            f"Contents of {source.name}/:\n" +
            "\n".join(f"  {p.name}" for p in sorted(source.iterdir()))
        )
        sys.exit(1)

    if not full_forg.exists():
        log.error(
            f"Could not find full_forg/ inside {source.resolve()}\n"
            f"Contents of {source.name}/:\n" +
            "\n".join(f"  {p.name}" for p in sorted(source.iterdir()))
        )
        sys.exit(1)

    # ── Create output directories ─────────────────────────────────────────────
    genuine_dir = out / "genuine"
    forged_dir  = out / "forged"
    genuine_dir.mkdir(parents=True, exist_ok=True)
    forged_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Source   : {source.resolve()}")
    log.info(f"Genuine  : {full_org.resolve()}")
    log.info(f"Forged   : {full_forg.resolve()}")
    log.info(f"Output   : {out.resolve()}")

    # ── Process genuine — full_org/ ───────────────────────────────────────────
    log.info("Processing genuine signatures from full_org/ ...")
    genuine_count, genuine_errors = _copy_images(
        source_dir  = full_org,
        dest_dir    = genuine_dir,
        prefix      = "original",
        dest_infix  = "sig",
    )

    # ── Process forged — full_forg/ ───────────────────────────────────────────
    log.info("Processing forged signatures from full_forg/ ...")
    forged_count, forged_errors = _copy_images(
        source_dir  = full_forg,
        dest_dir    = forged_dir,
        prefix      = "forgeries",
        dest_infix  = "forg",
    )

    # ── Nothing found ─────────────────────────────────────────────────────────
    if genuine_count == 0 and forged_count == 0:
        log.error(
            "\nNo images were copied.\n"
            "Expected filenames like:\n"
            "  full_org/original_1_1.png\n"
            "  full_forg/forgeries_1_1.png\n\n"
            "Sample of files found in full_org/:\n" +
            "\n".join(
                f"  {p.name}"
                for p in sorted(full_org.iterdir())[:10]
                if p.is_file()
            )
        )
        sys.exit(1)

    # ── Warnings ──────────────────────────────────────────────────────────────
    if genuine_count != 1320:
        log.warning(
            f"Expected 1320 genuine images (55 × 24) but copied {genuine_count}."
        )
    if forged_count != 1320:
        log.warning(
            f"Expected 1320 forged images (55 × 24) but copied {forged_count}."
        )
    if genuine_errors or forged_errors:
        log.warning(
            f"{genuine_errors + forged_errors} files had unexpected names and were skipped."
        )

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  CEDAR Dataset Organised Successfully")
    print("=" * 62)
    print(f"  Genuine images  : {genuine_count}")
    print(f"  Forged images   : {forged_count}")
    print(f"  Total           : {genuine_count + forged_count}")
    print(f"  Output          : {out.resolve()}")
    print("=" * 62)
    print("\n  Run training now:\n")
    print("  GPU:")
    print("    python -m ml.train --data_dir data/processed \\")
    print("        --batch_size 64 --epochs 100\n")
    print("  CPU:")
    print("    python -m ml.train --data_dir data/processed \\")
    print("        --batch_size 16 --epochs 50 --workers 2\n")


def _copy_images(
    source_dir: Path,
    dest_dir: Path,
    prefix: str,
    dest_infix: str,
) -> tuple[int, int]:
    """
    Copy all images from source_dir into dest_dir using a clean naming convention.

    Source filename pattern : {prefix}_{user}_{sig}.png
                              e.g. original_1_1.png
                              e.g. forgeries_55_24.png

    Destination filename    : user_{user:03d}_{dest_infix}_{sig:02d}.png
                              e.g. user_001_sig_01.png
                              e.g. user_055_forg_24.png

    Args:
        source_dir  : Folder containing flat image files.
        dest_dir    : Output folder to copy into.
        prefix      : Expected filename prefix (original or forgeries).
        dest_infix  : Infix for output filename (sig or forg).

    Returns:
        (copied_count, error_count)
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    copied = 0
    errors = 0

    all_images = sorted(
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    )

    if not all_images:
        log.warning(f"No image files found in {source_dir}")
        return 0, 0

    log.info(f"  Found {len(all_images)} images in {source_dir.name}/")

    for img in all_images:
        stem   = img.stem          # e.g. "original_1_1" or "forgeries_55_24"
        parts  = stem.split("_")

        # ── Parse filename ────────────────────────────────────────────────────
        # Expected: {prefix}_{user}_{sig}
        # Parts after split:
        #   original_1_1    → ['original', '1', '1']
        #   forgeries_55_24 → ['forgeries', '55', '24']
        #   original_1_10   → ['original', '1', '10']

        # Handle both 2-part and 1-part prefixes
        # "original"  → 1 part → user=parts[1], sig=parts[2]
        # "forgeries" → 1 part → user=parts[1], sig=parts[2]

        try:
            if len(parts) < 3:
                raise ValueError(f"Too few parts: {parts}")

            # The last two parts are always user_num and sig_num
            # Everything before is the prefix (handles multi-word prefixes)
            user_num = int(parts[-2])
            sig_num  = int(parts[-1])

        except (ValueError, IndexError):
            log.warning(f"  Skipped (unexpected name): {img.name}")
            errors += 1
            continue

        dest_name = f"user_{user_num:03d}_{dest_infix}_{sig_num:02d}{img.suffix.lower()}"
        dest_path = dest_dir / dest_name

        shutil.copy2(str(img), str(dest_path))
        copied += 1

    log.info(f"  Copied {copied} images → {dest_dir}")
    return copied, errors


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Organise the extracted CEDAR signature folder for model training.\n"
            "Reads from full_org/ and full_forg/ and outputs into genuine/ and forged/."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_cedar.py\n"
            "  python scripts/download_cedar.py --cedar_dir signatures_cedar\n"
            "  python scripts/download_cedar.py --cedar_dir C:\\Downloads\\cedar\n"
            "  python scripts/download_cedar.py --cedar_dir signatures_cedar --output_dir data/processed\n"
        )
    )
    parser.add_argument(
        "--cedar_dir",
        default="signatures_cedar",
        help=(
            "Path to the extracted CEDAR root folder.\n"
            "Must contain full_org/ and full_forg/ subfolders.\n"
            "Default: signatures_cedar  (project root)"
        )
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed",
        help=(
            "Destination for genuine/ and forged/ output folders.\n"
            "Default: data/processed"
        )
    )
    args = parser.parse_args()
    organize_cedar(args.cedar_dir, args.output_dir)
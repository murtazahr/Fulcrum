"""Fetch all datasets used by the NSAV experiments.

Datasets:
- CIFAR-10 (Setting C) — torchvision built-in, ~200 MB.
- Fed-Heart-Disease (Setting B) — FLamby auto-downloads on first use, ~1 MB.
- Fed-ISIC2019 (Setting A) — requires manual download from the ISIC archive
  (account-gated). This script prints the manual steps and verifies the
  expected layout.

Usage:  python scripts/download_data.py [--data-root path] [--skip-isic]

All datasets are placed under <data-root> (default: ./data).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------

def download_cifar10(data_root: Path) -> None:
    """Download CIFAR-10 train+test via torchvision."""
    import torchvision

    target = data_root / "cifar10"
    target.mkdir(parents=True, exist_ok=True)
    print(f"==> CIFAR-10 → {target}")
    torchvision.datasets.CIFAR10(root=str(target), train=True, download=True)
    torchvision.datasets.CIFAR10(root=str(target), train=False, download=True)
    print("    OK")


# ---------------------------------------------------------------------------
# Fed-Heart-Disease (FLamby)
# ---------------------------------------------------------------------------

def download_fed_heart_disease(data_root: Path) -> None:
    """Trigger FLamby's auto-download for Fed-Heart-Disease."""
    target = data_root / "fed_heart_disease"
    target.mkdir(parents=True, exist_ok=True)
    print(f"==> Fed-Heart-Disease → {target}")

    # FLamby downloads to its own cache by default. Point it at our data root.
    os.environ.setdefault("FLAMBY_DATASET_PATH", str(target))

    try:
        from flamby.datasets.fed_heart_disease import FedHeartDisease
    except ImportError as exc:
        print(f"    SKIP: FLamby not installed ({exc}). Run scripts/setup_env.sh first.")
        return

    # Iterating over the dataset triggers the download lazily on first access
    ds = FedHeartDisease(center=0, train=True)
    _ = len(ds)
    print(f"    OK ({len(ds)} samples in center 0, train split)")


# ---------------------------------------------------------------------------
# Fed-ISIC2019 (FLamby) — manual download required
# ---------------------------------------------------------------------------

ISIC_INSTRUCTIONS = """\
==> Fed-ISIC2019 requires manual download (the ISIC challenge data is
    account-gated by the International Skin Imaging Collaboration).

    Steps:

    1. Create a free account at https://challenge.isic-archive.com/
    2. Download the ISIC 2019 challenge files:
       - ISIC_2019_Training_Input.zip  (~9 GB)
       - ISIC_2019_Training_GroundTruth.csv
       - ISIC_2019_Training_Metadata.csv
    3. Place all three files under  {target}/
    4. From the FLamby checkout, run the preprocessing:

       cd .flamby_src/flamby/datasets/fed_isic2019/dataset_creation_scripts
       python download_isic.py --output-folder {target}
       python resize_images.py --input-folder {target} --output-folder {target}/resized

    5. Re-run this script to verify the layout.

    Verification target:  {target}/resized/  (containing per-center splits)

    See https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_isic2019/README.md
"""


def download_fed_isic2019(data_root: Path) -> None:
    """Verify Fed-ISIC2019 is in place; print manual instructions otherwise."""
    target = data_root / "fed_isic2019"
    target.mkdir(parents=True, exist_ok=True)

    expected = target / "resized"
    if expected.exists() and any(expected.iterdir()):
        print(f"==> Fed-ISIC2019 → {target}  (already present)")
        return

    print(ISIC_INSTRUCTIONS.format(target=target))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Root directory for downloaded datasets (default: ./data)",
    )
    parser.add_argument(
        "--skip-isic",
        action="store_true",
        help="Skip Fed-ISIC2019 (it needs manual download anyway).",
    )
    args = parser.parse_args()

    args.data_root.mkdir(parents=True, exist_ok=True)
    print(f"Data root: {args.data_root}\n")

    download_cifar10(args.data_root)
    print()
    download_fed_heart_disease(args.data_root)
    print()
    if not args.skip_isic:
        download_fed_isic2019(args.data_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())

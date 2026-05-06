"""Fetch all datasets used by the Fulcrum experiments.

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
    """Run FLamby's Fed-Heart-Disease download script.

    FLamby does NOT auto-download on dataset instantiation — each dataset has
    its own ``dataset_creation_scripts/download.py`` that fetches data + writes
    a ``dataset_location.yaml`` config so subsequent imports can find it.
    """
    target = data_root / "fed_heart_disease"
    target.mkdir(parents=True, exist_ok=True)
    print(f"==> Fed-Heart-Disease → {target}")

    # Locate FLamby's bundled download script (cloned by scripts/setup_env.sh)
    repo_root = Path(__file__).resolve().parent.parent
    script = (
        repo_root / ".flamby_src" / "flamby" / "datasets" / "fed_heart_disease"
        / "dataset_creation_scripts" / "download.py"
    )
    if not script.exists():
        print(f"    SKIP: FLamby download script not found at {script}")
        print( "          Re-run scripts/setup_env.sh to clone FLamby.")
        return

    import subprocess
    cmd = [sys.executable, str(script), "--output-folder", str(target)]
    print(f"    Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=script.parent)
    if result.returncode != 0:
        print(f"    FAILED with exit code {result.returncode}")
        return

    # Sanity-check by instantiating the dataset
    try:
        from flamby.datasets.fed_heart_disease import FedHeartDisease
        ds = FedHeartDisease(center=0, train=True)
        print(f"    OK ({len(ds)} samples in center 0, train split)")
    except Exception as exc:
        print(f"    Download succeeded but FedHeartDisease instantiation failed: {exc}")
        print( "    Check that FLamby's dataset_location.yaml was written correctly.")


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

"""
scripts/download_dataset.py

Downloads the ToothFairy2 dataset from the official source.
Dataset homepage: https://ditto.ing.unimore.it/toothfairy2

Usage:
    python scripts/download_dataset.py --output data/raw/
"""

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Dataset Meta Data 
TOOTHFAIRY2_INFO = {
    "name": "ToothFairy2",
    "homepage": "https://ditto.ing.unimore.it/toothfairy2",
    "challenge": "https://toothfairy.grand-challenge.org/",
    "description": (
        "500 CBCT volumes with expert annotations for 42 tooth labels. "
        "Provided by the University of Modena and Reggio Emilia (UNIMORE)."
    ),
    "license": "CC BY 4.0",
    "citation": (
        "Cipriano M. et al., 'ToothFairy2: A Large-Scale Dataset "
        "and Benchmark for Tooth Segmentation in CBCT Volumes', "
        "IEEE TMI, 2024."
    ),
    # Official dataset requires registration at Grand Challenge.
    # After registration you can download via:
    #   https://grand-challenge.org/datasets/
    # Below URL is a placeholder — replace with your download token URL.
    "download_url": "https://zenodo.org/record/PLACEHOLDER/files/ToothFairy2.zip",
    "md5": "PLACEHOLDER_MD5",
    "size_gb": 12.4,
}


def download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    """Stream-download url → dest with a progress bar."""
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as fh, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest.name,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                bar.update(len(chunk))



def verify_md5(path: Path, expected: str) -> bool:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def extract_zip(zip_path: Path, dest: Path) -> None:
    print(f"Extracting {zip_path.name} → {dest} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="Extracting"):
            zf.extract(member, dest)
    print("Extraction complete.")


def parse_args():
    p = argparse.ArgumentParser(description="Download ToothFairy2 dataset")
    p.add_argument("--output", default="data/raw/", help="Output directory")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip download, only extract (zip must already exist)")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip MD5 verification")
    p.add_argument("--keep-zip", action="store_true",
                   help="Keep the .zip file after extraction")
    return p.parse_args()



def print_manual_instructions(output_dir: Path) -> None:
    """Print instructions for manual download when automated download fails."""
    print("\n" + "=" * 65)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 65)
    print()
    print("ToothFairy2 requires registration at Grand Challenge.")
    print("Follow these steps:")
    print()
    print("  1. Create a free account at: https://grand-challenge.org/")
    print("  2. Visit the challenge page:")
    print("       https://toothfairy.grand-challenge.org/")
    print("  3. Join the challenge and accept the data license (CC BY 4.0)")
    print("  4. Download the dataset ZIP (~12.4 GB)")
    print(f"  5. Place the ZIP in: {output_dir}")
    print("  6. Re-run this script with --skip-download")
    print()
    print("Alternative: Use the Zenodo mirror if available:")
    print("  https://zenodo.org/record/toothfairy2")
    print()
    print("Dataset structure expected after extraction:")
    print(f"  {output_dir}/ToothFairy2/")
    print("    ├── images/     # CBCT volumes (.mha or .nii.gz)")
    print("    ├── labels/     # Segmentation masks (.mha or .nii.gz)")
    print("    └── metadata.json")
    print("=" * 65 + "\n")


def validate_dataset_structure(dataset_dir: Path) -> bool:
    """Check that the extracted dataset has the expected layout."""
    required = [
        dataset_dir / "images",
        dataset_dir / "labels",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print(f"[WARNING] Expected directories not found: {missing}")
        print("  The dataset may use a different directory structure.")
        print("  Please check ToothFairy2 documentation and adjust paths.")
        return False
    
    n_images = len(list((dataset_dir / "images").glob("*")))
    n_labels = len(list((dataset_dir / "labels").glob("*")))
    print(f"[OK] Found {n_images} images and {n_labels} labels.")
    return True


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "ToothFairy2.zip"
    dataset_dir = output_dir / "ToothFairy2"

    # ── Print dataset info ──────────────────────────────────────
    print("\n📦 Dataset: ToothFairy2")
    print(f"   Homepage : {TOOTHFAIRY2_INFO['homepage']}")
    print(f"   License  : {TOOTHFAIRY2_INFO['license']}")
    print(f"   Size     : ~{TOOTHFAIRY2_INFO['size_gb']} GB")
    print(f"   Citation : {TOOTHFAIRY2_INFO['citation']}")
    print()

    # ── Download ────────────────────────────────────────────────
    if not args.skip_download:
        if zip_path.exists():
            print(f"[SKIP] ZIP already exists at {zip_path}")
        else:
            url = TOOTHFAIRY2_INFO["download_url"]
            if "PLACEHOLDER" in url:
                print("[INFO] Automated download URL not yet configured.")
                print_manual_instructions(output_dir)
                print("After placing the ZIP file, re-run with --skip-download")
                sys.exit(0)
            print(f"Downloading from {url} ...")
            try:
                download_file(url, zip_path)
            except Exception as e:
                print(f"[ERROR] Download failed: {e}")
                print_manual_instructions(output_dir)
                sys.exit(1)

    # ── Verify ──────────────────────────────────────────────────
    if zip_path.exists() and not args.skip_verify:
        expected_md5 = TOOTHFAIRY2_INFO["md5"]
        if "PLACEHOLDER" not in expected_md5:
            print("Verifying MD5 checksum ...")
            if verify_md5(zip_path, expected_md5):
                print("[OK] MD5 checksum verified.")
            else:
                print("[ERROR] MD5 mismatch — download may be corrupted.")
                sys.exit(1)
        else:
            print("[SKIP] MD5 verification skipped (no checksum on file).")

    # ── Extract ─────────────────────────────────────────────────
    if dataset_dir.exists():
        print(f"[SKIP] Dataset directory already exists at {dataset_dir}")
    elif zip_path.exists():
        extract_zip(zip_path, output_dir)
    else:
        print("[ERROR] No ZIP file found. Please download manually.")
        print_manual_instructions(output_dir)
        sys.exit(1)

    # ── Validate ────────────────────────────────────────────────
    validate_dataset_structure(dataset_dir)

    # ── Cleanup ─────────────────────────────────────────────────
    if zip_path.exists() and not args.keep_zip:
        print(f"Removing {zip_path} ...")
        zip_path.unlink()

    print("\n✅ Dataset ready.")
    print(f"   Location: {dataset_dir.resolve()}")
    print("   Next step: python scripts/generate_splits.py")
    print()


if __name__ == "__main__":
    main()
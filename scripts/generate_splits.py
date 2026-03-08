#!/usr/bin/env python3
"""
scripts/generate_splits.py

Creates reproducible train / val / test splits for ToothFairy2.

Split strategy
--------------
- 400 train / 50 val / 50 test  (80 / 10 / 10)
- Stratified by jaw completeness (full arch vs partial)
- Random seed = 42 for reproducibility
- Writes plaintext lists to data/splits/

Usage:
    python scripts/generate_splits.py \
        --dataset_dir data/raw/ToothFairy2 \
        --splits_dir  data/splits/
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

SEED = 42
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}


def discover_cases(dataset_dir: Path) -> list[str]:
    """
    Return sorted list of case IDs found in the dataset directory.
    Handles common ToothFairy2 layouts:
      - dataset_dir/images/*.nii.gz
      - dataset_dir/*/scan.nii.gz  (per-case subdirs)
    """
    image_dir = dataset_dir / "images"
    if image_dir.exists():
        files = sorted(image_dir.glob("*.nii.gz")) + sorted(image_dir.glob("*.mha"))
        case_ids = [f.stem.replace(".nii", "") for f in files]
    else:
        # Per-case subdirectory layout
        subdirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
        case_ids = [d.name for d in subdirs]

    if not case_ids:
        raise FileNotFoundError(
            f"No cases found in {dataset_dir}. "
            "Please download the dataset first with scripts/download_dataset.py"
        )
    return case_ids


def stratify_by_jaw(case_ids: list[str], dataset_dir: Path) -> dict[str, list[str]]:
    """
    Attempt simple stratification: cases with both jaws vs single jaw.
    Falls back to single group if label files are unavailable.
    """
    label_dir = dataset_dir / "labels"
    strata = defaultdict(list)

    for cid in case_ids:
        # Try to infer jaw completeness from label stats
        # If unavailable, fall back to single stratum
        label_candidates = [
            label_dir / f"{cid}.nii.gz",
            label_dir / f"{cid}.mha",
            dataset_dir / cid / "label.nii.gz",
        ]
        label_path = next((p for p in label_candidates if p.exists()), None)

        if label_path is not None:
            try:
                import nibabel as nib
                import SimpleITK as sitk

                if str(label_path).endswith(".mha"):
                    img = sitk.ReadImage(str(label_path))
                    arr = sitk.GetArrayFromImage(img)
                else:
                    arr = nib.load(str(label_path)).get_fdata()

                unique_labels = set(np.unique(arr).astype(int)) - {0}
                # Upper jaw: FDI 11-28 → labels 1-16
                # Lower jaw: FDI 31-48 → labels 17-32
                has_upper = any(1 <= l <= 16 for l in unique_labels)
                has_lower = any(17 <= l <= 32 for l in unique_labels)

                if has_upper and has_lower:
                    strata["both_jaws"].append(cid)
                elif has_upper:
                    strata["upper_only"].append(cid)
                elif has_lower:
                    strata["lower_only"].append(cid)
                else:
                    strata["unknown"].append(cid)
            except Exception:
                strata["unknown"].append(cid)
        else:
            strata["unknown"].append(cid)

    return dict(strata)


def stratified_split(
    strata: dict[str, list[str]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[str]]:
    """Split each stratum proportionally, then combine."""
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for group, ids in strata.items():
        ids_shuffled = ids.copy()
        rng.shuffle(ids_shuffled)
        n = len(ids_shuffled)
        n_val = max(1, round(n * ratios["val"]))
        n_test = max(1, round(n * ratios["test"]))
        n_train = n - n_val - n_test

        splits["train"].extend(ids_shuffled[:n_train])
        splits["val"].extend(ids_shuffled[n_train: n_train + n_val])
        splits["test"].extend(ids_shuffled[n_train + n_val:])

    # Final shuffle within each split
    for k in splits:
        rng.shuffle(splits[k])
        splits[k].sort()  # deterministic order for git diffs

    return splits


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate train/val/test splits")
    p.add_argument("--dataset_dir", default="data/raw/ToothFairy2")
    p.add_argument("--splits_dir", default="data/splits/")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--train_ratio", type=float, default=0.80)
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--test_ratio", type=float, default=0.10)
    return p.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    assert abs(sum(ratios.values()) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    print(f"Discovering cases in {dataset_dir} ...")
    case_ids = discover_cases(dataset_dir)
    print(f"Found {len(case_ids)} cases.")

    print("Stratifying by jaw completeness ...")
    strata = stratify_by_jaw(case_ids, dataset_dir)
    for k, v in strata.items():
        print(f"  {k}: {len(v)} cases")

    print(f"Generating splits (seed={args.seed}) ...")
    splits = stratified_split(strata, ratios, seed=args.seed)

    # Validate no overlap
    sets = {k: set(v) for k, v in splits.items()}
    assert sets["train"].isdisjoint(sets["val"]), "OVERLAP: train ∩ val"
    assert sets["train"].isdisjoint(sets["test"]), "OVERLAP: train ∩ test"
    assert sets["val"].isdisjoint(sets["test"]), "OVERLAP: val ∩ test"
    covered = sets["train"] | sets["val"] | sets["test"]
    assert covered == set(case_ids), "Some cases missing from splits"

    # Write split files
    for split_name, ids in splits.items():
        out_path = splits_dir / f"{split_name}.txt"
        out_path.write_text("\n".join(ids) + "\n")
        print(f"  Wrote {len(ids):3d} cases → {out_path}")

    # Write metadata JSON
    meta = {
        "seed": args.seed,
        "ratios": ratios,
        "n_total": len(case_ids),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "strata": {k: len(v) for k, v in strata.items()},
        "dataset": "ToothFairy2",
        "notes": (
            "Stratified by jaw completeness (both/upper/lower/unknown). "
            "No external data used."
        ),
    }
    meta_path = splits_dir / "splits_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Wrote metadata → {meta_path}")

    print("\n✅ Splits ready.")
    print(f"   Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
    print("   Next step: python src/preprocessing/preprocess.py")


if __name__ == "__main__":
    main()
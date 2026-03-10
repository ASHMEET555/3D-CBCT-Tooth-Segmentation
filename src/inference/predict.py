"""
src/inference/predict.py

End-to-end inference pipeline:
  1. Load any input format (NIfTI / MHA / DICOM)
  2. Preprocess (resample, clip, normalize)
  3. Sliding-window prediction
  4. Postprocess (CCA, FDI assignment, jaw separation)
  5. Save mask + labels JSON + HTML viewer

Usage:
    python src/inference/predict.py \
        --input  path/to/scan.nii.gz \
        --weights weights/best_model.pth \
        --output results/

    # With config override:
    python src/inference/predict.py \
        --input  path/to/scan/ \   # DICOM directory
        --config configs/inference_config.yaml \
        --output results/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.io_utils import load_volume, save_volume
from src.preprocessing.preprocess import CBCTPreprocessor
from src.models.nnunet_resencl import ResEncLUNet
from src.inference.sliding_window import SlidingWindowPredictor
from src.inference.postprocess import ToothSegPostprocessor, save_labels_json
from src.visualization.html_viewer import generate_html_viewer


# ──────────────────────────────────────────────────────────────
# Default config
# ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "model": {
        "architecture": "nnunet_resencl",
        "num_classes": 43,
        "weights": "weights/best_model.pth",
    },
    "inference": {
        "patch_size": [128, 128, 128],
        "patch_overlap": 0.5,
        "batch_size": 1,
        "tta": True,
        "tta_axes": [0, 1, 2],
        "mixed_precision": True,
    },
    "postprocessing": {
        "min_voxel_count": 100,
        "apply_jaw_separation": True,
        "assign_fdi": True,
        "classify_restorations": True,
    },
    "output": {
        "save_mask": True,
        "save_labels_json": True,
        "save_viewer_html": True,
    },
}


# ──────────────────────────────────────────────────────────────
# Inference pipeline
# ──────────────────────────────────────────────────────────────

class InferencePipeline:
    def __init__(self, config: dict, weights_path: str):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference device: {self.device}")

        # Load model
        self.model = self._load_model(weights_path)
        self.model.eval()

        # Preprocessor
        self.preprocessor = CBCTPreprocessor(
            target_spacing=(0.4, 0.4, 0.4),
            hu_range=(-1000.0, 3000.0),
        )

        # Sliding window predictor
        inf_cfg = config["inference"]
        self.predictor = SlidingWindowPredictor(
            patch_size=tuple(inf_cfg["patch_size"]),
            overlap=inf_cfg.get("patch_overlap", 0.5),
            num_classes=config["model"]["num_classes"],
            batch_size=inf_cfg.get("batch_size", 1),
            use_gaussian=True,
            tta=inf_cfg.get("tta", True),
            tta_axes=tuple(inf_cfg.get("tta_axes", [0, 1, 2])),
        )

        # Postprocessor
        pp_cfg = config["postprocessing"]
        self.postprocessor = ToothSegPostprocessor(
            min_voxels=pp_cfg.get("min_voxel_count", 100),
        )

    def predict(self, input_path: str, output_dir: str) -> dict:
        """Run full inference on a single volume."""
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        case_id = input_path.stem.replace(".nii", "")

        t_start = time.time()

        # ── 1. Load ─────────────────────────────────────────
        logger.info(f"Loading: {input_path}")
        vol = load_volume(input_path)
        raw_array = vol["array"].copy()
        original_spacing = vol["spacing"]
        origin = vol["origin"]
        direction = vol["direction"]

        # ── 2. Preprocess ────────────────────────────────────
        logger.info("Preprocessing ...")
        tmp_img = output_dir / "_tmp_img.nii.gz"
        meta = self.preprocessor.process(input_path, out_image_path=tmp_img)
        pp_vol = load_volume(tmp_img)
        pp_array = pp_vol["array"]
        pp_spacing = pp_vol["spacing"]
        tmp_img.unlink(missing_ok=True)

        # ── 3. Sliding-window inference ───────────────────────
        logger.info(f"Running sliding-window inference on volume {pp_array.shape} ...")
        pred_mask = self.predictor.predict(pp_array, self.model, self.device)
        logger.info(f"Unique predicted labels: {np.unique(pred_mask)}")

        # ── 4. Postprocess ───────────────────────────────────
        logger.info("Postprocessing ...")
        # Re-load raw HU for restoration detection (at preprocessed spacing)
        hu_clipped = np.clip(pp_array, -1000, 3000)
        # Denormalize roughly (approximate, for HU threshold)
        # In practice, keep raw_array at original spacing

        clean_mask, tooth_info = self.postprocessor.run(
            raw_mask=pred_mask,
            raw_image=hu_clipped,
            spacing=pp_spacing,
        )

        t_elapsed = time.time() - t_start
        logger.info(f"Detected {len(tooth_info)} teeth in {t_elapsed:.1f}s")

        # ── 5. Save outputs ──────────────────────────────────
        out_cfg = self.cfg["output"]

        # Mask (NIfTI)
        mask_path = None
        if out_cfg.get("save_mask", True):
            mask_path = output_dir / f"{case_id}_mask.nii.gz"
            save_volume(clean_mask.astype(np.float32), pp_spacing,
                        origin, direction, mask_path, is_label=True)
            logger.info(f"Saved mask → {mask_path}")

        # Labels JSON
        json_path = None
        if out_cfg.get("save_labels_json", True):
            json_path = output_dir / f"{case_id}_labels.json"
            save_labels_json(tooth_info, json_path)
            logger.info(f"Saved labels → {json_path}")

        # Save a copy of the scan for the viewer
        scan_path = output_dir / f"{case_id}_scan.nii.gz"
        save_volume(pp_array, pp_spacing, origin, direction, scan_path, is_label=False)

        # HTML viewer
        viewer_path = None
        if out_cfg.get("save_viewer_html", True):
            viewer_path = output_dir / f"{case_id}_viewer.html"
            generate_html_viewer(
                scan_path=scan_path,
                mask_path=mask_path,
                tooth_info=tooth_info,
                out_path=viewer_path,
            )
            logger.info(f"Saved viewer → {viewer_path}")

        return {
            "case_id": case_id,
            "n_teeth": len(tooth_info),
            "teeth": tooth_info,
            "mask_path": str(mask_path),
            "json_path": str(json_path),
            "viewer_path": str(viewer_path),
            "elapsed_s": round(t_elapsed, 2),
        }

    def _load_model(self, weights_path: str) -> ResEncLUNet:
        """Load model weights from checkpoint."""
        n_classes = self.cfg["model"]["num_classes"]
        model = ResEncLUNet(
            in_channels=1,
            num_classes=n_classes,
            deep_supervision=False,  # inference mode
        ).to(self.device)

        weights_path = Path(weights_path)
        if not weights_path.exists():
            logger.warning(
                f"Weights not found at {weights_path}. "
                "Running with random weights (for demo/testing only)."
            )
            return model

        logger.info(f"Loading weights from {weights_path} ...")
        checkpoint = torch.load(weights_path, map_location=self.device)

        # Handle various checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)
        logger.info("✅ Weights loaded successfully.")
        return model


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run CBCT tooth segmentation inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/inference/predict.py \\
      --input data/sample/scan.nii.gz \\
      --weights weights/best_model.pth \\
      --output results/

  python src/inference/predict.py \\
      --input data/sample/DICOM_DIR/ \\
      --output results/ --no-tta
        """
    )
    p.add_argument("--input", required=True,
                   help="Input file (.nii.gz, .mha) or DICOM directory")
    p.add_argument("--output", default="results/",
                   help="Output directory (default: results/)")
    p.add_argument("--weights", default="weights/best_model.pth",
                   help="Path to model weights (.pth)")
    p.add_argument("--config", default=None,
                   help="Optional YAML config (overrides defaults)")
    p.add_argument("--no-tta", action="store_true",
                   help="Disable test-time augmentation")
    p.add_argument("--no-viewer", action="store_true",
                   help="Skip HTML viewer generation")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    config = DEFAULT_CONFIG.copy()
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            user_cfg = yaml.safe_load(f)
        # Deep merge
        for k, v in user_cfg.items():
            if isinstance(v, dict) and k in config:
                config[k].update(v)
            else:
                config[k] = v

    # CLI overrides
    if args.no_tta:
        config["inference"]["tta"] = False
    if args.no_viewer:
        config["output"]["save_viewer_html"] = False

    # Run
    pipeline = InferencePipeline(config, args.weights)
    result = pipeline.predict(args.input, args.output)

    # Print summary
    print("\n" + "=" * 50)
    print("INFERENCE COMPLETE")
    print("=" * 50)
    print(f"Case ID       : {result['case_id']}")
    print(f"Teeth detected: {result['n_teeth']}")
    print(f"Time elapsed  : {result['elapsed_s']}s")
    print(f"Mask          : {result['mask_path']}")
    print(f"Labels JSON   : {result['json_path']}")
    print(f"HTML viewer   : {result['viewer_path']}")
    print("=" * 50 + "\n")

    # Print detected teeth table
    print(f"{'FDI':>5} {'Jaw':>6} {'Voxels':>8} {'Vol(mm³)':>10} {'Restoration':>12}")
    print("-" * 50)
    for t in result["teeth"]:
        fdi = str(t["fdi"]) if t["fdi"] else "??"
        jaw = t["jaw"]
        print(
            f"{fdi:>5} {jaw:>6} {t['voxel_count']:>8} "
            f"{t['volume_mm3']:>10.1f} {'Yes' if t['is_restoration'] else 'No':>12}"
        )


if __name__ == "__main__":
    main()
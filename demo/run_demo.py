"""
demo/run_demo.py

Runs inference on the bundled sample volume and opens the HTML viewer.

Usage:
    python demo/run_demo.py
    python demo/run_demo.py --volume demo/sample_scan.nii.gz
"""

import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Run CBCT segmentation demo")
    p.add_argument("--volume", default=None,
                   help="Input volume (default: demo/sample_scan.nii.gz)")
    p.add_argument("--weights", default="weights/best_model.pth",
                   help="Model weights path")
    p.add_argument("--output", default="demo/output/",
                   help="Output directory")
    p.add_argument("--no-browser", action="store_true",
                   help="Don't open viewer in browser")
    return p.parse_args()


def create_synthetic_demo():
    """
    Creates a synthetic CBCT-like volume for demo purposes
    when no real volume is available.
    """
    print("[INFO] Creating synthetic demo volume ...")
    try:
        import numpy as np
        import nibabel as nib

        vol_path = Path("demo/sample_scan.nii.gz")
        vol_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a synthetic dental arch
        D, H, W = 128, 128, 128
        vol = np.full((D, H, W), -800.0, dtype=np.float32)  # air background

        # Simulated jaw bone (ellipsoid)
        z, y, x = np.mgrid[0:D, 0:H, 0:W]
        jaw = ((x - W//2)**2/(W//3)**2 + (y - H//2)**2/(H//3)**2 + (z - D//2)**2/(D//4)**2) < 1
        vol[jaw] = 300.0  # bone HU

        # Simulated teeth (small spheres in an arch)
        import math
        n_teeth = 14
        for i in range(n_teeth):
            angle = (i / n_teeth) * math.pi
            tx = int(W//2 + math.cos(angle) * W//4)
            ty = int(H//2 + math.sin(angle) * H//6 - H//10)
            tz = D // 2 + np.random.randint(-5, 5)
            tr = 8  # tooth radius in voxels
            tooth = ((x - tx)**2 + (y - ty)**2 + (z - tz)**2) < tr**2
            vol[tooth] = 800.0  # tooth HU

        # Save NIfTI
        affine = np.diag([0.4, 0.4, 0.4, 1.0])
        img = nib.Nifti1Image(np.transpose(vol, (2, 1, 0)), affine)
        nib.save(img, str(vol_path))
        print(f"[OK] Synthetic volume saved → {vol_path}")
        return vol_path

    except ImportError as e:
        print(f"[ERROR] Cannot create synthetic volume: {e}")
        print("Please install nibabel: pip install nibabel")
        return None


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine input volume
    if args.volume:
        volume_path = Path(args.volume)
        if not volume_path.exists():
            print(f"[ERROR] Volume not found: {volume_path}")
            sys.exit(1)
    else:
        volume_path = Path("demo/sample_scan.nii.gz")
        if not volume_path.exists():
            volume_path = create_synthetic_demo()
            if volume_path is None:
                sys.exit(1)

    print(f"\n{'='*50}")
    print("CBCT Tooth Segmentation — Demo")
    print(f"{'='*50}")
    print(f"Input  : {volume_path}")
    print(f"Weights: {args.weights}")
    print(f"Output : {output_dir}")
    print(f"{'='*50}\n")

    # Run inference
    cmd = [
        sys.executable,
        str(ROOT / "src" / "inference" / "predict.py"),
        "--input", str(volume_path),
        "--weights", args.weights,
        "--output", str(output_dir),
    ]

    print("Running inference ...")
    result = subprocess.run(cmd, cwd=str(ROOT))

    if result.returncode != 0:
        print("[ERROR] Inference failed. Check logs above.")
        sys.exit(1)

    # Find viewer HTML
    viewer_files = list(output_dir.glob("*_viewer.html"))
    if viewer_files:
        viewer_path = viewer_files[0]
        print(f"\n✅ Viewer ready: {viewer_path.resolve()}")
        if not args.no_browser:
            print("Opening in browser ...")
            webbrowser.open(f"file://{viewer_path.resolve()}")
    else:
        print("[WARNING] No viewer HTML generated.")


if __name__ == "__main__":
    main()
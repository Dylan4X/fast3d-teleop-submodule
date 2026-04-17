#!/usr/bin/env python3
"""Download all required model assets from ModelScope.

Usage::

    # Download to Fast-SAM-3D-Body project root
    python scripts/download_assets.py --project-root /path/to/Fast-SAM-3D-Body

    # Or via CLI entry point (after pip install)
    fast3d-download-assets --project-root /path/to/Fast-SAM-3D-Body

    # Also download SMPL model (requires prior agreement to SMPL license)
    fast3d-download-assets --project-root /path/to/Fast-SAM-3D-Body --smpl-dir ./smpl
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

MODELSCOPE_REPO = "XDRRRR/fast3d-teleop-assets"

# (src_relative_in_repo, dst_relative_in_project_root)
ASSET_MAP: list[tuple[str, str]] = [
    # SAM-3D-Body main model
    ("checkpoints/sam-3d-body-dinov3/model.ckpt",
     "checkpoints/sam-3d-body-dinov3/model.ckpt"),
    ("checkpoints/sam-3d-body-dinov3/model_config.yaml",
     "checkpoints/sam-3d-body-dinov3/model_config.yaml"),
    ("checkpoints/sam-3d-body-dinov3/mhr_model.pt",
     "checkpoints/sam-3d-body-dinov3/mhr_model.pt"),
    # YOLO-Pose
    ("checkpoints/yolo/yolo11m-pose.pt",
     "checkpoints/yolo/yolo11m-pose.pt"),
    # MoGe2 ViT-S
    ("checkpoints/moge-2-vits-normal/model.pt",
     "checkpoints/moge-2-vits-normal/model.pt"),
    # MHR→SMPL mapping & models
    ("mhr2smpl/data/mhr2smpl_mapping.npz",
     "mhr2smpl/data/mhr2smpl_mapping.npz"),
    ("mhr2smpl/data/mhr_face_mask.ply",
     "mhr2smpl/data/mhr_face_mask.ply"),
    ("mhr2smpl/experiments/multiview_n30000_e500/best_model.pth",
     "mhr2smpl/experiments/multiview_n30000_e500/best_model.pth"),
    ("mhr2smpl/experiments/multiview_n30000_e500/config.json",
     "mhr2smpl/experiments/multiview_n30000_e500/config.json"),
    ("mhr2smpl/experiments/smoother_w5/smoother_best.pth",
     "mhr2smpl/experiments/smoother_w5/smoother_best.pth"),
    ("mhr2smpl/experiments/smoother_w5/smoother_config.json",
     "mhr2smpl/experiments/smoother_w5/smoother_config.json"),
]


def download_from_modelscope(project_root: Path, repo_id: str = MODELSCOPE_REPO) -> None:
    """Download model assets from ModelScope and place into project_root."""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("ERROR: modelscope package not installed. Run:")
        print("  pip install modelscope")
        sys.exit(1)

    print(f"Downloading from ModelScope: {repo_id} ...")
    cache_dir = snapshot_download(repo_id)
    cache_dir = Path(cache_dir)
    print(f"Cached at: {cache_dir}")

    for src_rel, dst_rel in ASSET_MAP:
        src = cache_dir / src_rel
        dst = project_root / dst_rel
        if dst.exists():
            print(f"  [skip] {dst_rel} (already exists)")
            continue
        if not src.exists():
            print(f"  [WARN] {src_rel} not found in ModelScope repo")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Use symlink to save disk space (cache already has the file)
        try:
            dst.symlink_to(src.resolve())
            print(f"  [link] {dst_rel}")
        except OSError:
            shutil.copy2(src, dst)
            print(f"  [copy] {dst_rel}")

    # Create assets/mhr_model.pt symlink (upstream code expects this path)
    assets_dir = project_root / "checkpoints" / "sam-3d-body-dinov3" / "assets"
    assets_mhr = assets_dir / "mhr_model.pt"
    mhr_src = project_root / "checkpoints" / "sam-3d-body-dinov3" / "mhr_model.pt"
    if mhr_src.exists() and not assets_mhr.exists():
        assets_dir.mkdir(parents=True, exist_ok=True)
        try:
            assets_mhr.symlink_to(mhr_src.resolve())
            print("  [link] checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
        except OSError:
            shutil.copy2(mhr_src, assets_mhr)
            print("  [copy] checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")


def check_smpl(smpl_dir: Path | None) -> None:
    """Check if SMPL_NEUTRAL.pkl exists."""
    if smpl_dir is None:
        print("\n[INFO] SMPL model not specified. You need SMPL_NEUTRAL.pkl for inference.")
        print("       Download from: https://smpl.is.tue.mpg.de/")
        print("       Then pass --smpl-model-path /path/to/SMPL_NEUTRAL.pkl to fast3d-mocap-server")
        return
    pkl = smpl_dir / "SMPL_NEUTRAL.pkl"
    if pkl.exists():
        print(f"\n[OK] SMPL model found: {pkl}")
    else:
        print(f"\n[WARN] SMPL_NEUTRAL.pkl not found in {smpl_dir}")
        print("       Download from: https://smpl.is.tue.mpg.de/")


def verify_assets(project_root: Path) -> bool:
    """Verify all required files are present."""
    print("\n--- Asset verification ---")
    ok = True
    for _, dst_rel in ASSET_MAP:
        dst = project_root / dst_rel
        status = "OK" if dst.exists() else "MISSING"
        if status == "MISSING":
            ok = False
        print(f"  [{status}] {dst_rel}")
    return ok


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Fast3D Teleop assets from ModelScope")
    p.add_argument("--project-root", type=str, required=True,
                   help="Fast-SAM-3D-Body project root directory")
    p.add_argument("--smpl-dir", type=str, default=None,
                   help="Directory containing SMPL_NEUTRAL.pkl (optional)")
    p.add_argument("--repo", type=str, default=MODELSCOPE_REPO,
                   help=f"ModelScope repo ID (default: {MODELSCOPE_REPO})")
    p.add_argument("--verify-only", action="store_true",
                   help="Only check if assets exist, don't download")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()

    if not project_root.exists():
        print(f"ERROR: project root does not exist: {project_root}")
        sys.exit(1)

    if args.verify_only:
        ok = verify_assets(project_root)
        check_smpl(Path(args.smpl_dir) if args.smpl_dir else None)
        sys.exit(0 if ok else 1)

    download_from_modelscope(project_root, args.repo)
    ok = verify_assets(project_root)
    check_smpl(Path(args.smpl_dir) if args.smpl_dir else None)

    if ok:
        print("\n All assets ready!")
        print("\nNext step: build TRT engines for maximum performance:")
        print(f"  fast3d-build-trt --project-root {project_root}")
    else:
        print("\n Some assets are missing!")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build TensorRT engines for maximum inference performance.

Usage::

    # Build all TRT engines (backbone, MoGe, YOLO)
    fast3d-build-trt --project-root /path/to/Fast-SAM-3D-Body

    # Build specific engine only
    fast3d-build-trt --project-root /path/to/Fast-SAM-3D-Body --only backbone
    fast3d-build-trt --project-root /path/to/Fast-SAM-3D-Body --only moge
    fast3d-build-trt --project-root /path/to/Fast-SAM-3D-Body --only yolo

Engines are GPU-architecture-specific (.engine files) and must be rebuilt
when switching between different GPU types (e.g. RTX 4090 → RTX 3090).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ENGINES = {
    "backbone": {
        "script": "convert_backbone_tensorrt.py",
        "args": ["--all"],
        "check": "checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine",
        "desc": "DINOv3 Backbone (fp16)",
    },
    "moge": {
        "script": "convert_moge_encoder_trt.py",
        "args": ["--all"],
        "check": "checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine",
        "desc": "MoGe2 DINOv2 Encoder (fp16)",
    },
    "yolo": {
        "script": "convert_yolo_pose_trt.py",
        "args": ["--model", "yolo11m-pose.pt", "--imgsz", "640", "--half"],
        "check": "checkpoints/yolo/yolo11m-pose.engine",
        "desc": "YOLO11m-Pose",
    },
}


def build_engine(project_root: Path, name: str, gpu_id: int = 0) -> bool:
    """Build a single TRT engine."""
    cfg = ENGINES[name]
    engine_path = project_root / cfg["check"]
    script_path = project_root / cfg["script"]

    if engine_path.exists():
        print(f"  [{name}] SKIP — engine already exists: {engine_path.name}")
        return True

    if not script_path.exists():
        print(f"  [{name}] ERROR — conversion script not found: {script_path}")
        return False

    print(f"  [{name}] Building {cfg['desc']}...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [sys.executable, str(script_path)] + cfg["args"]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            env=env,
            capture_output=False,
            timeout=600,  # 10 min per engine
        )
        if result.returncode != 0:
            print(f"  [{name}] FAILED (exit code {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [{name}] TIMEOUT (>10min)")
        return False

    if engine_path.exists():
        size_mb = engine_path.stat().st_size / (1024 * 1024)
        print(f"  [{name}] OK — {engine_path.name} ({size_mb:.0f}MB)")
        return True
    else:
        print(f"  [{name}] WARN — script finished but engine file not found")
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TensorRT engines for Fast3D Teleop")
    p.add_argument("--project-root", type=str, required=True,
                   help="Fast-SAM-3D-Body project root directory")
    p.add_argument("--only", type=str, default=None,
                   choices=list(ENGINES.keys()),
                   help="Build only the specified engine")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU device ID (default: 0)")
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if engine already exists")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()

    if not (project_root / "sam_3d_body").is_dir():
        print(f"ERROR: {project_root} doesn't look like a Fast-SAM-3D-Body repo")
        sys.exit(1)

    targets = [args.only] if args.only else list(ENGINES.keys())

    if args.force:
        for name in targets:
            engine_path = project_root / ENGINES[name]["check"]
            if engine_path.exists():
                engine_path.unlink()
                print(f"  [{name}] Removed existing engine")

    print(f"Building TRT engines on GPU {args.gpu}...")
    print(f"Project root: {project_root}\n")

    results = {}
    for name in targets:
        results[name] = build_engine(project_root, name, args.gpu)

    print("\n--- Summary ---")
    all_ok = True
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        if not ok:
            all_ok = False
        print(f"  [{status}] {ENGINES[name]['desc']}")

    if all_ok:
        print("\n All TRT engines ready!")
    else:
        print("\n Some engines failed to build.")
        print("You can still run without TRT (slower, uses PyTorch fallback).")
        sys.exit(1)


if __name__ == "__main__":
    main()

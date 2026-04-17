#!/usr/bin/env python3
"""Upload staged assets to ModelScope.

Usage::

    # Login first (one-time)
    python scripts/upload_to_modelscope.py --login

    # Upload assets
    python scripts/upload_to_modelscope.py \
      --stage-dir /amax/xuedingrong/tmp/modelscope_upload/fast3d-teleop-assets \
      --repo Dylan4X/fast3d-teleop-assets
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def do_login():
    """Interactive ModelScope login."""
    from modelscope.hub.api import HubApi
    api = HubApi()
    token = input("Enter your ModelScope API token: ").strip()
    api.login(token)
    print("Login successful!")


def do_upload(stage_dir: Path, repo_id: str):
    """Upload staged assets to ModelScope model repo."""
    from modelscope.hub.api import HubApi

    api = HubApi()

    # Create repo if it doesn't exist
    namespace, name = repo_id.split("/", 1)
    try:
        api.create_model(
            model_id=repo_id,
            visibility=1,  # public
            chinese_name="Fast3D Teleop Assets",
        )
        print(f"Created model repo: {repo_id}")
    except Exception as e:
        if "already exist" in str(e).lower() or "409" in str(e):
            print(f"Model repo already exists: {repo_id}")
        else:
            print(f"Warning creating repo: {e}")

    # Upload all files
    files = sorted(stage_dir.rglob("*"))
    files = [f for f in files if f.is_file()]
    print(f"\nUploading {len(files)} files from {stage_dir} ...")

    for f in files:
        rel = f.relative_to(stage_dir)
        print(f"  Uploading: {rel} ({f.stat().st_size / (1024*1024):.1f}MB)")
        try:
            api.push_model(
                model_id=repo_id,
                model_dir=str(stage_dir),
                commit_message=f"Upload {rel}",
            )
            print("  Done!")
            break  # push_model uploads the entire directory
        except Exception as e:
            print(f"  Error: {e}")
            return

    print(f"\n Upload complete! View at: https://modelscope.cn/models/{repo_id}")


def main():
    p = argparse.ArgumentParser(description="Upload assets to ModelScope")
    p.add_argument("--login", action="store_true", help="Login to ModelScope")
    p.add_argument("--stage-dir", type=str,
                   default="/amax/xuedingrong/tmp/modelscope_upload/fast3d-teleop-assets",
                   help="Directory with staged assets")
    p.add_argument("--repo", type=str, default="Dylan4X/fast3d-teleop-assets",
                   help="ModelScope repo ID")
    args = p.parse_args()

    if args.login:
        do_login()
        return

    stage_dir = Path(args.stage_dir)
    if not stage_dir.exists():
        print(f"ERROR: stage dir not found: {stage_dir}")
        sys.exit(1)

    do_upload(stage_dir, args.repo)


if __name__ == "__main__":
    main()

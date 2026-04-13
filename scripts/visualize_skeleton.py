#!/usr/bin/env python3
"""ZMQ subscriber that visualizes the 3D skeleton output in real time.

Usage:
    # 1. Start the mocap server in one terminal:
    fast3d-mocap-server --source video.mp4 --loop-video --smpl-model-path ...

    # 2. In another terminal, run this visualizer:
    python scripts/visualize_skeleton.py
    python scripts/visualize_skeleton.py --host 192.168.1.100 --port 5555
    python scripts/visualize_skeleton.py --save skeleton.mp4
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import matplotlib
matplotlib.use("Agg")  # headless by default; switched to TkAgg if --live
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import msgpack
import numpy as np

# 24 SMPL body joint names (same as Teleopit/conversion.py)
JOINT_NAMES = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
]

# SMPL kinematic-tree parent indices
JOINT_PARENTS = [
    -1, 0, 0, 0, 1, 2,
    3, 4, 5, 6, 7, 8,
    9, 12, 12, 12, 13, 14,
    16, 17, 18, 19, 20, 21,
]

# Bone colors by body region
BONE_COLORS = {}
_LEFT = ["Left_Hip", "Left_Knee", "Left_Ankle", "Left_Foot",
         "Left_Collar", "Left_Shoulder", "Left_Elbow", "Left_Wrist", "Left_Hand"]
_RIGHT = ["Right_Hip", "Right_Knee", "Right_Ankle", "Right_Foot",
          "Right_Collar", "Right_Shoulder", "Right_Elbow", "Right_Wrist", "Right_Hand"]
for j in _LEFT:
    BONE_COLORS[j] = "tab:blue"
for j in _RIGHT:
    BONE_COLORS[j] = "tab:red"


def parse_args():
    p = argparse.ArgumentParser(description="Visualize 3D skeleton from ZMQ mocap server")
    p.add_argument("--host", default="127.0.0.1", help="ZMQ PUB host")
    p.add_argument("--port", type=int, default=5555, help="ZMQ PUB port")
    p.add_argument("--topic", default="mocap", help="ZMQ topic")
    p.add_argument("--save", default=None, help="Save to video file (e.g. skeleton.mp4)")
    p.add_argument("--max-frames", type=int, default=300, help="Max frames to record (default: 300 = ~30s)")
    p.add_argument("--live", action="store_true", help="Show live matplotlib window (requires display)")
    p.add_argument("--figsize", type=float, nargs=2, default=[8, 8], help="Figure size in inches")
    return p.parse_args()


def draw_skeleton(ax, positions: dict[str, np.ndarray], seq: int, fps: float):
    """Draw 3D skeleton on a matplotlib Axes3D."""
    ax.cla()

    # Collect all points for axis limits
    all_pts = np.array([positions[j] for j in JOINT_NAMES if j in positions])
    if len(all_pts) == 0:
        return

    # Draw bones
    for i, jname in enumerate(JOINT_NAMES):
        parent_idx = JOINT_PARENTS[i]
        if parent_idx < 0 or jname not in positions:
            continue
        parent_name = JOINT_NAMES[parent_idx]
        if parent_name not in positions:
            continue
        p1 = positions[jname]
        p2 = positions[parent_name]
        color = BONE_COLORS.get(jname, "tab:green")
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=color, linewidth=2.5, solid_capstyle="round")

    # Draw joints
    ax.scatter(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2],
               c="black", s=20, zorder=5)

    # Label key joints
    for key_joint in ["Head", "Pelvis", "Left_Hand", "Right_Hand", "Left_Foot", "Right_Foot"]:
        if key_joint in positions:
            p = positions[key_joint]
            ax.text(p[0], p[1], p[2] + 0.02, key_joint.replace("_", "\n"),
                    fontsize=6, ha="center", va="bottom", color="gray")

    # Axis setup
    center = all_pts.mean(axis=0)
    max_range = max(all_pts.max(axis=0) - all_pts.min(axis=0)) * 0.6 + 0.1
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"seq={seq}  fps={fps:.1f}", fontsize=10)
    ax.view_init(elev=15, azim=-60)


def main():
    args = parse_args()

    if args.live:
        matplotlib.use("TkAgg")
        plt.ion()

    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.setsockopt(zmq.CONFLATE, 1)  # only keep latest
    sock.connect(f"tcp://{args.host}:{args.port}")
    sock.subscribe(args.topic.encode())
    print(f"Subscribed to tcp://{args.host}:{args.port} topic={args.topic!r}")
    print("Waiting for frames...")

    fig = plt.figure(figsize=args.figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Video writer (if saving)
    writer = None
    if args.save:
        import cv2
        dpi = fig.dpi
        w = int(args.figsize[0] * dpi)
        h = int(args.figsize[1] * dpi)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 10.0, (w, h))
        print(f"Saving to {args.save} ({w}x{h})")

    fps_window: deque[float] = deque(maxlen=30)
    frame_count = 0

    try:
        while frame_count < args.max_frames:
            try:
                raw = sock.recv()
            except zmq.Again:
                print("No data received (timeout), retrying...")
                continue

            t0 = time.monotonic()

            # Parse: topic + space + msgpack
            sep = raw.index(b" ")
            payload = msgpack.unpackb(raw[sep + 1:], raw=False)

            ts = payload.pop("_ts", None)
            seq = payload.pop("_seq", 0)
            payload.pop("_control_events", None)
            payload.pop("control_events", None)

            # Extract positions (ignore quaternions for visualization)
            positions = {}
            for name, (pos, _quat) in payload.items():
                positions[name] = np.asarray(pos, dtype=np.float64)

            draw_skeleton(ax, positions, seq, np.mean(fps_window) if fps_window else 0.0)

            fig.canvas.draw()

            if writer is not None:
                import cv2
                buf = fig.canvas.buffer_rgba()
                img = np.asarray(buf)[:, :, :3]  # RGBA → RGB
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                writer.write(img_bgr)

            if args.live:
                plt.pause(0.001)

            dt = time.monotonic() - t0
            fps_window.append(dt)
            frame_count += 1

            if frame_count % 30 == 0:
                avg_fps = 1.0 / np.mean(fps_window) if fps_window else 0
                print(f"[frame {frame_count}] seq={seq} render_fps={avg_fps:.1f}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if writer is not None:
            writer.release()
            print(f"Saved {frame_count} frames to {args.save}")
        sock.close()
        ctx.term()
        plt.close(fig)


if __name__ == "__main__":
    main()

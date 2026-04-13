#!/usr/bin/env python3
"""Side-by-side visualization: original video (left) + 3D skeleton (right).

Requires the mocap server to be running. Reads the original video locally
and subscribes to ZMQ for skeleton data.

Usage:
    python scripts/visualize_sidebyside.py \
        --video /path/to/video.mp4 \
        --save /tmp/sidebyside.mp4 \
        --max-frames 200
"""

from __future__ import annotations

import argparse
import io
import time
from collections import deque

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import msgpack
import numpy as np

JOINT_NAMES = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
]

JOINT_PARENTS = [
    -1, 0, 0, 0, 1, 2,
    3, 4, 5, 6, 7, 8,
    9, 12, 12, 12, 13, 14,
    16, 17, 18, 19, 20, 21,
]

_LEFT = {"Left_Hip", "Left_Knee", "Left_Ankle", "Left_Foot",
         "Left_Collar", "Left_Shoulder", "Left_Elbow", "Left_Wrist", "Left_Hand"}
_RIGHT = {"Right_Hip", "Right_Knee", "Right_Ankle", "Right_Foot",
          "Right_Collar", "Right_Shoulder", "Right_Elbow", "Right_Wrist", "Right_Hand"}


def parse_args():
    p = argparse.ArgumentParser(description="Side-by-side original video + skeleton")
    p.add_argument("--video", required=True, help="Original video file")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--topic", default="mocap")
    p.add_argument("--save", default="/tmp/sidebyside.mp4", help="Output video path")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--height", type=int, default=600, help="Output frame height")
    return p.parse_args()


def render_skeleton_to_image(fig, ax, positions: dict, seq: int, fps: float, target_h: int) -> np.ndarray:
    """Render 3D skeleton to a numpy image (BGR)."""
    ax.cla()

    all_pts = np.array([positions[j] for j in JOINT_NAMES if j in positions])
    if len(all_pts) == 0:
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        return cv2.cvtColor(cv2.resize(buf, (target_h, target_h)), cv2.COLOR_RGB2BGR)

    # Draw bones
    for i, jname in enumerate(JOINT_NAMES):
        pid = JOINT_PARENTS[i]
        if pid < 0 or jname not in positions:
            continue
        pname = JOINT_NAMES[pid]
        if pname not in positions:
            continue
        p1, p2 = positions[jname], positions[pname]
        color = "tab:blue" if jname in _LEFT else ("tab:red" if jname in _RIGHT else "tab:green")
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=color, linewidth=2.5, solid_capstyle="round")

    ax.scatter(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2], c="black", s=20, zorder=5)

    # Label key joints
    for kj in ["Head", "Pelvis", "Left_Hand", "Right_Hand"]:
        if kj in positions:
            p = positions[kj]
            ax.text(p[0], p[1], p[2] + 0.03, kj.replace("_", "\n"),
                    fontsize=7, ha="center", va="bottom", color="gray")

    center = all_pts.mean(axis=0)
    rng = max(all_pts.max(axis=0) - all_pts.min(axis=0)) * 0.6 + 0.1
    ax.set_xlim(center[0] - rng, center[0] + rng)
    ax.set_ylim(center[1] - rng, center[1] + rng)
    ax.set_zlim(center[2] - rng, center[2] + rng)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"seq={seq}  fps={fps:.1f}", fontsize=10)
    ax.view_init(elev=15, azim=-60)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    img = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    # Resize to target height (square)
    img = cv2.resize(img, (target_h, target_h))
    return img


def main():
    args = parse_args()
    import zmq

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open video: {args.video}")
        return
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0

    # Scale video to target height
    scale = args.height / vid_h
    frame_w = int(vid_w * scale)
    frame_h = args.height

    # ZMQ subscriber
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(f"tcp://{args.host}:{args.port}")
    sock.subscribe(args.topic.encode())
    print(f"Subscribed to tcp://{args.host}:{args.port} topic={args.topic!r}")

    # Matplotlib figure
    dpi = 100
    fig_size = args.height / dpi
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Output video: video frame (left) + skeleton (right)
    out_w = frame_w + args.height  # video width + skeleton square
    out_h = args.height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.save, fourcc, min(vid_fps, 10.0), (out_w, out_h))
    print(f"Output: {args.save} ({out_w}x{out_h})")

    fps_window: deque[float] = deque(maxlen=30)
    frame_count = 0

    try:
        while frame_count < args.max_frames:
            # Receive ZMQ first to get the video position
            try:
                raw = sock.recv()
            except zmq.Again:
                print("No ZMQ data, retrying...")
                continue

            t0 = time.monotonic()

            # Parse
            sep = raw.index(b" ")
            payload = msgpack.unpackb(raw[sep + 1:], raw=False)
            payload.pop("_ts", None)
            seq = payload.pop("_seq", 0)
            video_pos = payload.pop("_video_pos", None)
            payload.pop("_control_events", None)
            payload.pop("control_events", None)

            # Read the matching video frame using _video_pos
            if video_pos is not None:
                # _video_pos is the frame AFTER the one processed (CAP_PROP_POS_FRAMES advances after read)
                target_frame = max(0, int(video_pos) - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ok, vid_frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, vid_frame = cap.read()
                if not ok:
                    break

            positions = {}
            for name, (pos, _quat) in payload.items():
                positions[name] = np.asarray(pos, dtype=np.float64)

            # Render skeleton
            fps_val = 1.0 / np.mean(fps_window) if fps_window else 0.0
            skel_img = render_skeleton_to_image(fig, ax, positions, seq, fps_val, args.height)

            # Resize video frame
            vid_resized = cv2.resize(vid_frame, (frame_w, frame_h))

            # Concatenate side by side
            combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            combined[:, :frame_w] = vid_resized
            combined[:, frame_w:] = skel_img

            # Add labels
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
            cv2.putText(combined, "3D Skeleton", (frame_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

            writer.write(combined)

            dt = time.monotonic() - t0
            fps_window.append(dt)
            frame_count += 1

            if frame_count % 30 == 0:
                avg = 1.0 / np.mean(fps_window) if fps_window else 0
                print(f"[frame {frame_count}] seq={seq} render_fps={avg:.1f}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        writer.release()
        print(f"Saved {frame_count} frames to {args.save}")
        sock.close()
        ctx.term()
        cap.release()
        plt.close(fig)


if __name__ == "__main__":
    main()

"""ZMQ mocap server — broadcasts 3D body joints for Teleopit.

Usage::

    fast3d-mocap-server --smpl-model-path /path/to/SMPL_NEUTRAL.pkl
    fast3d-mocap-server --source 0 --port 5555 --smpl-model-path ...
    fast3d-mocap-server --source video.mp4 --loop-video --smpl-model-path ...
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

import cv2
import msgpack
import numpy as np
import zmq

from .conversion import packet_to_human_frame
from .env_setup import enable_tf32, setup_env

logger = logging.getLogger("fast3d_mocap")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fast-SAM-3D-Body vision mocap ZMQ server for Teleopit",
    )
    # Input
    p.add_argument("--source", type=str, default="0",
                    help="Camera index (e.g. 0) or video file path")
    # Model
    p.add_argument("--smpl-model-path", type=str, required=True,
                    help="Path to SMPL_NEUTRAL.pkl")
    p.add_argument("--project-root", type=str, default=None,
                    help="Fast-SAM-3D-Body project root (auto-detected)")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--mode", type=str, default="body_only",
                    choices=["body_only", "full_body_hands"])
    # ZMQ
    p.add_argument("--host", type=str, default="0.0.0.0",
                    help="ZMQ PUB bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=5555,
                    help="ZMQ PUB port (default: 5555)")
    p.add_argument("--topic", type=str, default="mocap",
                    help="ZMQ topic string (default: mocap)")
    # Options
    p.add_argument("--show-preview", action="store_true",
                    help="Show an OpenCV preview window")
    p.add_argument("--fps-limit", type=float, default=0,
                    help="Cap output FPS (0 = unlimited)")
    p.add_argument("--loop-video", action="store_true",
                    help="Loop video when reaching the end")
    p.add_argument("--warmup-frames", type=int, default=30,
                    help="Frames to run before publishing (default: 30)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- Environment + TF32 ----
    setup_env(image_size=args.image_size)
    enable_tf32()

    # ---- Load model ----
    logger.info("Loading model...")
    t0 = time.monotonic()

    from .core import Fast3DTeleopSubmodule, Fast3DTeleopSubmoduleConfig

    config = Fast3DTeleopSubmoduleConfig(
        mode=args.mode,
        image_size=args.image_size,
        smpl_model_path=args.smpl_model_path,
        project_root=args.project_root,
    )
    submodule = Fast3DTeleopSubmodule(config)
    logger.info("Model loaded in %.1fs", time.monotonic() - t0)

    # ---- ZMQ PUB ----
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.bind(f"tcp://{args.host}:{args.port}")
    topic_bytes = args.topic.encode("utf-8")
    logger.info("ZMQ PUB  tcp://%s:%d  topic=%r", args.host, args.port, args.topic)

    # ---- Video capture ----
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open video source: %s", args.source)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info("Source: %s  (%d frames, %.1f fps)", args.source,
                total_frames, source_fps)

    # ---- Warmup ----
    if args.warmup_frames > 0:
        logger.info("Warmup (%d frames)...", args.warmup_frames)
        t_wu = time.monotonic()

        # Force full-pipeline compile warmup
        try:
            submodule.warmup(width=640, height=480)
        except Exception as e:
            logger.warning("Estimator warmup failed: %s", e)

        # Process real frames to warm caches
        wu_ok = 0
        for _ in range(args.warmup_frames):
            ok, frame_bgr = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame_bgr = cap.read()
            if not ok:
                break
            pkt = submodule.process_frame(frame_bgr, timestamp=time.monotonic())
            if pkt is not None:
                wu_ok += 1
        import torch
        torch.cuda.synchronize()
        logger.info("Warmup done: %d/%d valid in %.1fs",
                     wu_ok, args.warmup_frames, time.monotonic() - t_wu)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ---- Main loop ----
    seq = 0
    running = True
    min_interval = 1.0 / args.fps_limit if args.fps_limit > 0 else 0
    recent_times: list[float] = []

    def _on_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    logger.info("Streaming... Ctrl+C to stop")
    t_start = time.monotonic()

    try:
        while running:
            t_frame = time.monotonic()

            ok, frame_bgr = cap.read()
            if not ok:
                if args.loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame_bgr = cap.read()
                if not ok:
                    logger.info("End of video")
                    break

            packet = submodule.process_frame(
                frame_bgr, timestamp=time.monotonic()
            )
            if packet is None:
                continue

            # Convert → publish
            human_frame = packet_to_human_frame(packet)
            human_frame["_ts"] = time.monotonic()
            human_frame["_seq"] = seq
            payload = msgpack.packb(human_frame, use_bin_type=True)
            sock.send(topic_bytes + b" " + payload)
            seq += 1

            # Preview
            if args.show_preview:
                info = f"seq={seq}  infer={packet.infer_sec*1000:.0f}ms"
                cv2.putText(frame_bgr, info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Fast3D Mocap", frame_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            # FPS limit
            elapsed = time.monotonic() - t_frame
            if min_interval > 0 and elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            # Rolling 30-frame FPS stats
            if seq % 30 == 0:
                recent_times.append(time.monotonic())
                if len(recent_times) > 2:
                    window = recent_times[-2:]
                    fps = 30.0 / max(window[1] - window[0], 1e-6)
                else:
                    fps = seq / max(time.monotonic() - t_start, 1e-6)
                logger.info(
                    "seq=%d  infer=%.0fms  convert=%.0fms  fps=%.1f",
                    seq,
                    packet.infer_sec * 1000,
                    packet.convert_sec * 1000,
                    fps,
                )
    finally:
        cap.release()
        sock.close()
        ctx.term()
        if args.show_preview:
            cv2.destroyAllWindows()
        logger.info("Stopped — published %d frames", seq)


if __name__ == "__main__":
    main()

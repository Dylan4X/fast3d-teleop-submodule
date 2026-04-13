#!/usr/bin/env python3
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

_THIS_FILE = Path(__file__).resolve()
PACKAGE_ROOT = _THIS_FILE.parent

from .env_setup import setup_env as _setup_env
_setup_env()


MHR_INDEX = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_hip": 9,
    "right_hip": 10,
    "left_knee": 11,
    "right_knee": 12,
    "left_ankle": 13,
    "right_ankle": 14,
    "right_wrist": 41,
    "left_wrist": 62,
    "neck": 69,
}


@dataclass
class Fast3DTeleopSubmoduleConfig:
    mode: str = "body_only"
    project_root: Optional[str] = None
    image_size: int = 512
    yolo_model_path: str = "./checkpoints/yolo/yolo11m-pose.engine"
    fov_model_size: str = "s"
    fov_resolution_level: int = 0
    fov_fixed_size: int = 512
    fov_fast_mode: bool = True
    smpl_model_path: str = ""
    multiview_model_dir: str = "mhr2smpl/experiments/multiview_n30000_e500"
    mhr2smpl_mapping_path: str = "mhr2smpl/data/mhr2smpl_mapping.npz"
    mhr_mesh_path: Optional[str] = "mhr2smpl/data/mhr_face_mask.ply"
    smoother_dir: Optional[str] = "mhr2smpl/experiments/smoother_w5"
    gravity_direction: tuple[float, float, float] = (0.0, 1.0, 0.0)
    min_person_confidence: float = 0.75
    require_single_person: bool = True
    hand_box_source: str = "body_decoder"
    device: str = "cuda"
    inference_type: str = "body"


@dataclass
class TeleopPosePacket:
    timestamp: Optional[float]
    body_quat_wxyz: np.ndarray
    smpl_joints_local: np.ndarray
    smpl_pose: np.ndarray
    body_quat_xyzw_cam: np.ndarray
    canonical_joints: np.ndarray
    bbox_xyxy: Optional[np.ndarray] = None
    focal_length: Optional[float] = None
    pred_cam_t: Optional[np.ndarray] = None
    body_keypoints_2d: Optional[np.ndarray] = None
    debug_named_points_cam: dict[str, list[float]] = field(default_factory=dict)
    infer_sec: float = 0.0
    convert_sec: float = 0.0

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "body_quat_wxyz": np.asarray(self.body_quat_wxyz, dtype=np.float64).tolist(),
            "smpl_joints_local": np.asarray(self.smpl_joints_local, dtype=np.float64).tolist(),
            "smpl_pose": np.asarray(self.smpl_pose, dtype=np.float64).tolist(),
            "body_quat_xyzw_cam": np.asarray(self.body_quat_xyzw_cam, dtype=np.float64).tolist(),
            "canonical_joints": np.asarray(self.canonical_joints, dtype=np.float64).tolist(),
            "bbox_xyxy": None if self.bbox_xyxy is None else np.asarray(self.bbox_xyxy, dtype=np.float64).tolist(),
            "focal_length": self.focal_length,
            "pred_cam_t": None if self.pred_cam_t is None else np.asarray(self.pred_cam_t, dtype=np.float64).tolist(),
            "body_keypoints_2d": None if self.body_keypoints_2d is None else np.asarray(self.body_keypoints_2d, dtype=np.float64).tolist(),
            "debug_named_points_cam": self.debug_named_points_cam,
            "infer_sec": self.infer_sec,
            "convert_sec": self.convert_sec,
        }


class Fast3DTeleopSubmodule:
    """Teleoperation-oriented Fast-SAM-3D-Body submodule.

    This follows the original repository's teleop path:
      estimator -> single-view MHR->SMPL conversion -> prepare_publish_pose

    The result is a control-friendly packet instead of a visualization-first output.
    """

    def __init__(self, config: Fast3DTeleopSubmoduleConfig):
        import sys

        self.config = self._normalize_config(config)
        self.project_root = self._discover_project_root(self.config.project_root)
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        # Upstream backbone code uses relative paths (./checkpoints/) — must chdir
        os.chdir(self.project_root)

        from .vendor.gravity_alignment import build_camera_to_world_rotation
        from .vendor.multiview_mhr2smpl import MultiViewFusionRunner
        from .vendor.pose_protocol import prepare_publish_pose

        self._prepare_publish_pose_fn = prepare_publish_pose
        self._fusion_runner = MultiViewFusionRunner(
            smpl_model_path=self._resolve_path(self.config.smpl_model_path),
            model_dir=self._resolve_path(self.config.multiview_model_dir),
            mapping_path=self._resolve_path(self.config.mhr2smpl_mapping_path),
            mhr_mesh_path=self._resolve_optional_path(self.config.mhr_mesh_path),
            smoother_dir=self._resolve_optional_path(self.config.smoother_dir),
        )
        self._estimator = self._build_estimator()

        gravity = np.asarray(self.config.gravity_direction, dtype=np.float64)
        gravity = gravity / max(np.linalg.norm(gravity), 1e-8)
        self.gravity_direction = gravity
        self.R_world_cam = build_camera_to_world_rotation(gravity)

    # ------------------------------------------------------------------
    # Gravity auto-calibration
    # ------------------------------------------------------------------
    def estimate_gravity_from_packets(
        self, packets: list[TeleopPosePacket]
    ) -> np.ndarray:
        """Estimate camera-frame gravity direction from body orientations.

        Assumes people are *approximately* upright.  The SMPL body frame has
        Y-up, so the body quaternion (camera frame) maps [0, 1, 0] to the
        "up" direction in camera space; gravity is opposite.
        """
        up_vectors = []
        for pkt in packets:
            R_body = Rotation.from_quat(pkt.body_quat_xyzw_cam)
            up_cam = R_body.apply(np.array([0.0, 1.0, 0.0]))
            up_vectors.append(up_cam)
        avg_up = np.mean(up_vectors, axis=0)
        norm = np.linalg.norm(avg_up)
        if norm < 1e-8:
            return self.gravity_direction
        return (-avg_up / norm).astype(np.float64)

    def update_gravity(self, gravity_cam: np.ndarray) -> None:
        """Overwrite the gravity direction and rebuild R_world_cam."""
        gravity_cam = np.asarray(gravity_cam, dtype=np.float64)
        gravity_cam = gravity_cam / max(np.linalg.norm(gravity_cam), 1e-8)
        self.gravity_direction = gravity_cam
        self.R_world_cam = self._build_cwrot(gravity_cam)

    @staticmethod
    def _build_cwrot(gravity_cam: np.ndarray):
        from .vendor.gravity_alignment import build_camera_to_world_rotation
        return build_camera_to_world_rotation(gravity_cam)

    @staticmethod
    def _normalize_config(config: Fast3DTeleopSubmoduleConfig) -> Fast3DTeleopSubmoduleConfig:
        mode = str(config.mode).lower().strip()
        if mode in {"body", "body_only", "teleop_fast"}:
            config.mode = "body_only"
            config.inference_type = "body"
            config.hand_box_source = "body_decoder"
        elif mode in {"full", "full_body_hands", "avatar_full"}:
            config.mode = "full_body_hands"
            config.inference_type = "full"
            if not config.hand_box_source:
                config.hand_box_source = "yolo_pose"
        else:
            raise ValueError(f"Unsupported mode: {config.mode}")
        return config

    @staticmethod
    def _discover_project_root(project_root: Optional[str]) -> Path:
        candidates: list[Path] = []
        if project_root:
            candidates.append(Path(project_root))

        env_root = os.environ.get("FAST3D_REPO_ROOT")
        if env_root:
            candidates.append(Path(env_root))

        candidates.extend(
            [
                Path.cwd(),
                Path.cwd() / "Fast-SAM-3D-Body",
                PACKAGE_ROOT,
                PACKAGE_ROOT.parent,
                PACKAGE_ROOT.parent.parent,
                PACKAGE_ROOT.parent.parent.parent,
            ]
        )

        for candidate in candidates:
            candidate = candidate.resolve()
            if (candidate / "sam_3d_body").is_dir() and (candidate / "tools").is_dir():
                return candidate
        raise FileNotFoundError(
            "Cannot locate the Fast-SAM-3D-Body repository root. "
            "Pass `project_root=...` or set FAST3D_REPO_ROOT."
        )

    @classmethod
    def body_only_config(cls, **kwargs) -> Fast3DTeleopSubmoduleConfig:
        return Fast3DTeleopSubmoduleConfig(mode="body_only", **kwargs)

    @classmethod
    def full_body_hands_config(cls, **kwargs) -> Fast3DTeleopSubmoduleConfig:
        return Fast3DTeleopSubmoduleConfig(mode="full_body_hands", hand_box_source="yolo_pose", inference_type="full", **kwargs)

    def _resolve_path(self, path_str: str) -> str:
        path = Path(path_str)
        if not path.is_absolute():
            path = self.project_root / path
        return str(path)

    def _resolve_optional_path(self, path_str: Optional[str]) -> Optional[str]:
        if not path_str:
            return None
        return self._resolve_path(path_str)

    def _build_estimator(self):
        """Build the SAM-3D-Body estimator (inlined from mocap/core/setup_estimator.py)."""
        import torch

        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
        from tools.build_detector import HumanDetector
        from tools.build_fov_estimator import FOVEstimator

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model, model_cfg = load_sam_3d_body(
            checkpoint_path=self._resolve_path("checkpoints/sam-3d-body-dinov3/model.ckpt"),
            device=device,
            mhr_path=self._resolve_path("checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"),
        )

        human_detector = HumanDetector(
            name="yolo_pose",
            device=device,
            model=self._resolve_path(self.config.yolo_model_path),
        )

        fov_estimator = FOVEstimator(name="moge2", device=device)

        return SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=human_detector,
            human_segmentor=None,
            fov_estimator=fov_estimator,
        )

    def warmup(self, width: int = 640, height: int = 480) -> None:
        dummy_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        warmup_bbox = np.array([[0.0, 0.0, float(width - 1), float(height - 1)]], dtype=np.float32)
        _ = self._estimator.process_one_image(
            dummy_img,
            bboxes=warmup_bbox,
            hand_box_source="body_decoder",
            inference_type=self.config.inference_type,
        )

    @staticmethod
    def _compute_body_quat(global_rot) -> np.ndarray:
        global_rot = np.asarray(global_rot, dtype=np.float64).reshape(3)
        rot = Rotation.from_euler("ZYX", global_rot)
        x180 = Rotation.from_euler("x", 180.0, degrees=True)
        return (x180 * rot).as_quat().astype(np.float64)

    @staticmethod
    def _extract_named_points_cam(person: dict[str, Any]) -> dict[str, list[float]]:
        named: dict[str, list[float]] = {}
        kps3d = person.get("pred_keypoints_3d")
        cam_t = person.get("pred_cam_t")
        if kps3d is None or cam_t is None:
            return named
        pts = np.asarray(kps3d, dtype=np.float64)
        cam_t = np.asarray(cam_t, dtype=np.float64).reshape(3)
        if pts.ndim != 2 or pts.shape[1] < 3:
            return named
        pts = pts.copy()
        pts[:, :3] += cam_t[None, :]
        for name, idx in MHR_INDEX.items():
            if idx < len(pts):
                named[name] = np.round(pts[idx, :3], 4).tolist()
        if "left_hip" in named and "right_hip" in named:
            l = np.asarray(named["left_hip"], dtype=np.float64)
            r = np.asarray(named["right_hip"], dtype=np.float64)
            named["pelvis"] = np.round(0.5 * (l + r), 4).tolist()
        return named

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        timestamp: Optional[float] = None,
        cam_intrinsics: Optional[np.ndarray] = None,
    ) -> Optional[TeleopPosePacket]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        outputs = self._estimator.process_one_image(
            frame_rgb,
            cam_int=cam_intrinsics,
            hand_box_source=self.config.hand_box_source,
            inference_type=self.config.inference_type,
        )
        infer_sec = time.perf_counter() - t0

        if self.config.require_single_person:
            if len(outputs) != 1:
                return None
            out = outputs[0]
        else:
            if not outputs:
                return None
            out = max(outputs, key=lambda x: float((np.asarray(x.get("bbox", [0, 0, 0, 0]))[2] - np.asarray(x.get("bbox", [0, 0, 0, 0]))[0]) * (np.asarray(x.get("bbox", [0, 0, 0, 0]))[3] - np.asarray(x.get("bbox", [0, 0, 0, 0]))[1])))

        required = ("pred_vertices", "pred_cam_t", "global_rot")
        if any(k not in out for k in required):
            return None

        t1 = time.perf_counter()
        pred_vertices = np.asarray(out["pred_vertices"], dtype=np.float32)
        pred_cam_t = np.asarray(out["pred_cam_t"], dtype=np.float32)
        body_quat_xyzw = self._compute_body_quat(out["global_rot"])
        smpl_pose, canonical_joints, _betas, _weights = self._fusion_runner.infer(
            [(pred_vertices, pred_cam_t)]
        )
        body_quat_wxyz, smpl_joints_local, smpl_pose = self._prepare_publish_pose_fn(
            body_quat_xyzw,
            canonical_joints,
            smpl_pose,
            self.R_world_cam,
        )
        convert_sec = time.perf_counter() - t1

        bbox = out.get("bbox")
        bbox_xyxy = None
        if bbox is not None:
            bbox_xyxy = np.asarray(bbox, dtype=np.float64).reshape(-1)[:4]

        body_keypoints_2d = None
        if out.get("pred_keypoints_2d") is not None:
            body_keypoints_2d = np.asarray(out["pred_keypoints_2d"], dtype=np.float64)

        return TeleopPosePacket(
            timestamp=timestamp,
            body_quat_wxyz=np.asarray(body_quat_wxyz, dtype=np.float64),
            smpl_joints_local=np.asarray(smpl_joints_local, dtype=np.float64),
            smpl_pose=np.asarray(smpl_pose, dtype=np.float64),
            body_quat_xyzw_cam=np.asarray(body_quat_xyzw, dtype=np.float64),
            canonical_joints=np.asarray(canonical_joints, dtype=np.float64),
            bbox_xyxy=bbox_xyxy,
            focal_length=None if out.get("focal_length") is None else float(out["focal_length"]),
            pred_cam_t=np.asarray(pred_cam_t, dtype=np.float64),
            body_keypoints_2d=body_keypoints_2d,
            debug_named_points_cam=self._extract_named_points_cam(out),
            infer_sec=infer_sec,
            convert_sec=convert_sec,
        )


class Fast3DTeleopPublisherAdapter:
    """Thin adapter that publishes protocol-compatible SMPL packets over ZMQ."""

    def __init__(self, addr: str = "tcp://*:5556"):
        import sys

        repo_root = Fast3DTeleopSubmodule._discover_project_root(None)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from mocap.realtime.publisher import ZMQPublisher

        self.publisher = ZMQPublisher(addr)

    def publish(self, packet: TeleopPosePacket) -> None:
        self.publisher.publish(
            packet.body_quat_wxyz,
            packet.smpl_joints_local,
            packet.smpl_pose,
        )

    def close(self) -> None:
        self.publisher.close()


def default_control_json(packet: Optional[TeleopPosePacket]) -> dict[str, Any]:
    if packet is None:
        return {
            "ready": False,
            "reason": "not_exactly_one_person_or_missing_required_keys",
        }
    return {
        "ready": True,
        "timestamp": packet.timestamp,
        "body_quat_wxyz": packet.body_quat_wxyz.tolist(),
        "smpl_joints_local": packet.smpl_joints_local.tolist(),
        "smpl_pose": packet.smpl_pose.tolist(),
        "debug": {
            "bbox_xyxy": None if packet.bbox_xyxy is None else packet.bbox_xyxy.tolist(),
            "focal_length": packet.focal_length,
            "pred_cam_t": None if packet.pred_cam_t is None else packet.pred_cam_t.tolist(),
            "named_points_cam": packet.debug_named_points_cam,
            "infer_sec": packet.infer_sec,
            "convert_sec": packet.convert_sec,
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fast-SAM-3D-Body teleop submodule smoke test on a video.")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--smpl-model-path", required=True)
    parser.add_argument("--project-root", default=os.environ.get("FAST3D_REPO_ROOT"))
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--subsample", type=int, default=1)
    args = parser.parse_args()

    cfg = Fast3DTeleopSubmoduleConfig(
        project_root=args.project_root,
        smpl_model_path=args.smpl_model_path,
    )
    submodule = Fast3DTeleopSubmodule(cfg)
    submodule.warmup()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with output_path.open("w", encoding="utf-8") as f:
        frame_idx = 0
        while kept < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % max(args.subsample, 1) != 0:
                frame_idx += 1
                continue
            packet = submodule.process_frame(frame, timestamp=float(frame_idx))
            f.write(json.dumps(default_control_json(packet), ensure_ascii=False) + "\n")
            kept += 1
            frame_idx += 1
    cap.release()

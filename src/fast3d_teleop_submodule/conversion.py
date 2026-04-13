"""Convert TeleopPosePacket to Teleopit-compatible HumanFrame."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from .core import TeleopPosePacket

# SMPL 24 body joint names — matches Teleopit's BODY_JOINT_NAMES
BODY_JOINT_NAMES: list[str] = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
]

# SMPL kinematic-tree parent indices
JOINT_PARENTS = np.array([
    -1, 0, 0, 0, 1, 2,
    3, 4, 5, 6, 7, 8,
    9, 12, 12, 12, 13, 14,
    16, 17, 18, 19, 20, 21,
], dtype=np.int32)


def packet_to_human_frame(packet: TeleopPosePacket) -> dict:
    """Convert TeleopPosePacket to Teleopit HumanFrame dict.

    Returns ``{joint_name: [pos_xyz_list, quat_wxyz_list], ...}``
    compatible with Teleopit's ``ZMQInputProvider._deserialize_frame()``.

    Positions and orientations are placed in the same coordinate frame as
    ``body_quat_wxyz`` (the published root orientation, including PICO-protocol
    adjustment rotations).
    """
    body_q_wxyz = np.asarray(packet.body_quat_wxyz, dtype=np.float64)
    joints_local = np.asarray(packet.smpl_joints_local, dtype=np.float64)
    smpl_pose = np.asarray(packet.smpl_pose, dtype=np.float64)

    # wxyz → xyzw for scipy
    body_q_xyzw = np.array([body_q_wxyz[1], body_q_wxyz[2],
                             body_q_wxyz[3], body_q_wxyz[0]])
    R_body = Rotation.from_quat(body_q_xyzw)

    # World positions: rotate body-local joints by body orientation
    world_positions = R_body.apply(joints_local)

    # Per-joint world quaternions via FK through kinematic chain
    n_joints = 24
    world_rots: list[Rotation] = [Rotation.identity()] * n_joints
    world_rots[0] = R_body

    n_pose = min(len(smpl_pose), 23)
    for j in range(1, n_joints):
        idx = j - 1
        local_rot = (
            Rotation.from_rotvec(smpl_pose[idx]) if idx < n_pose
            else Rotation.identity()
        )
        world_rots[j] = world_rots[JOINT_PARENTS[j]] * local_rot

    # Build HumanFrame dict
    frame: dict = {}
    for i, name in enumerate(BODY_JOINT_NAMES):
        pos = world_positions[i].tolist()
        q_xyzw = world_rots[i].as_quat()
        q_wxyz = [float(q_xyzw[3]), float(q_xyzw[0]),
                   float(q_xyzw[1]), float(q_xyzw[2])]
        frame[name] = [pos, q_wxyz]
    return frame

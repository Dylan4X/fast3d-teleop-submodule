"""Convert TeleopPosePacket to Teleopit-compatible HumanFrame.

Coordinate system notes
-----------------------
Our pipeline (gravity_alignment.py → pose_protocol.py) already produces output
in a **Z-up** world frame with PICO-protocol orientation adjustments applied.

Teleopit's Pico4InputProvider converts Pico4 SDK data (Y-up) to Z-up via
``_INPUT_TO_TELEOPIT_MATRIX = [[1,0,0],[0,0,-1],[0,1,0]]``.  Since our
pipeline independently produces Z-up output, we do NOT apply that matrix
again (it would double-transform).

If the retargetted pose looks mirrored or rotated 90°, set
``EXTRA_COORD_TRANSFORM`` to an appropriate 3×3 rotation to fix the axis
convention mismatch.
"""

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

# Post-FK coordinate transform: convert from Y-up (our pipeline output after
# GLOBAL_ORIENT_EXTRA_ROT in pose_protocol) to Z-up (Teleopit convention).
# Same matrix used by Teleopit's Pico4InputProvider._coordinate_transform_input().
# Maps [x, y, z] → [x, -z, y]:  Y-up → Z-up
EXTRA_COORD_TRANSFORM: np.ndarray = np.array(
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64
)


def packet_to_human_frame(packet: TeleopPosePacket) -> dict:
    """Convert TeleopPosePacket to Teleopit HumanFrame dict.

    Returns ``{joint_name: [pos_xyz_list, quat_wxyz_list], ...}``
    compatible with Teleopit's ``ZMQInputProvider._deserialize_frame()``.

    The output coordinate frame is Z-up, produced by gravity_alignment +
    pose_protocol transforms.  An optional ``EXTRA_COORD_TRANSFORM`` 3×3
    matrix is applied last if set (default: identity / no-op).
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

    # Build HumanFrame dict — apply optional extra coordinate transform
    R_extra = None
    if EXTRA_COORD_TRANSFORM is not None:
        R_extra = Rotation.from_matrix(EXTRA_COORD_TRANSFORM)

    frame: dict = {}
    for i, name in enumerate(BODY_JOINT_NAMES):
        pos = world_positions[i]
        q_xyzw = world_rots[i].as_quat()
        if R_extra is not None:
            pos = EXTRA_COORD_TRANSFORM @ pos
            q_xyzw = (R_extra * Rotation.from_quat(q_xyzw)).as_quat()
        q_wxyz = [float(q_xyzw[3]), float(q_xyzw[0]),
                   float(q_xyzw[1]), float(q_xyzw[2])]
        frame[name] = [pos.tolist(), q_wxyz]
    return frame

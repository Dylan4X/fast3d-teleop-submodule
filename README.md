# Fast3D Teleop Submodule

`Fast3D Teleop Submodule` is a teleoperation-facing extraction of [Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body).

It keeps the original teleop logic:

- `build_default_estimator`
- `MultiViewFusionRunner`
- `prepare_publish_pose`
- original SMPL-style publish fields

and exposes them as a cleaner perception module for humanoid teleoperation.

## What It Is

This repository is a perception front-end for a teleop stack:

- RGB frame in
- single-person 3D body understanding
- control-friendly SMPL packet out

It is not a demo viewer and it is not a mesh-rendering product.

It is designed to sit next to an existing `Fast-SAM-3D-Body` checkout and reuse the original project's weights, teleop conversion code, and SMPL pipeline.

## Output

For a valid single-person frame, the submodule returns:

- `body_quat_wxyz`
- `smpl_joints_local`
- `smpl_pose`

It also keeps a compact debug block:

- `bbox_xyxy`
- `focal_length`
- `pred_cam_t`
- `named_points_cam`
- `infer_sec`
- `convert_sec`

## Modes

### `body_only`

Recommended for teleoperation.

- single person
- no hand decoder
- uses the original model's native `inference_type="body"`
- preserves the original SMPL conversion and pose protocol path

### `full_body_hands`

Recommended when hand detail matters more than throughput.

- single person
- full body + hands
- richer avatar-oriented output
- lower throughput

## Performance On RTX 4090

Measured on an RTX 4090-class server using the optimized Fast-SAM-3D-Body environment.

### `body_only`

- about **7.1 fps** end-to-end submodule throughput
- about **7.6 fps** pure inference throughput

### `full_body_hands`

- about **4.3 fps** end-to-end submodule throughput
- about **4.5 fps** pure inference throughput

### Important Note

Removing visualization does **not** materially improve the `body_only` path. The bottleneck is the model body inference itself, not the JSON/control packaging.

## Repository Layout

```text
fast3d-teleop-submodule/
  README.md
  pyproject.toml
  src/
    fast3d_teleop_submodule/
      __init__.py
      core.py
```

## Minimal Example

```python
from fast3d_teleop_submodule import Fast3DTeleopSubmodule, Fast3DTeleopSubmoduleConfig

cfg = Fast3DTeleopSubmoduleConfig(
    mode="body_only",
    project_root="/path/to/Fast-SAM-3D-Body",
    smpl_model_path="/path/to/SMPL_NEUTRAL.pkl",
)

submodule = Fast3DTeleopSubmodule(cfg)
submodule.warmup()

packet = submodule.process_frame(frame_bgr, timestamp=timestamp)
if packet is not None:
    control = packet.to_jsonable()
```

## Positioning

This module should be treated as:

- a teleop perception submodule
- not a complete robot controller
- not a UI product
- not a mesh-rendering demo

The intended system boundary is:

1. RGB frame enters this submodule
2. the submodule emits SMPL-compatible control signals
3. downstream retargeting or robot control consumes those signals

## Integration Assumption

This package does not vendor the original `Fast-SAM-3D-Body` codebase or checkpoints.

You should point it at a prepared upstream checkout by either:

- setting `project_root=...` in `Fast3DTeleopSubmoduleConfig`, or
- setting the `FAST3D_REPO_ROOT` environment variable

## Current Recommendation

If the target is humanoid teleoperation on a 4090:

- use `body_only` as the main control path
- keep hand detail off the critical loop
- bring hands back only at lower frequency if needed

That gives the best balance we found between:

- stability
- faithfulness to the original method
- practical control-loop speed

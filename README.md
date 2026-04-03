# Fast3D Teleop Submodule

`Fast3D Teleop Submodule` 是一个面向 humanoid teleoperation 的感知子模块，基于 [Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body) 抽取而来。

它的作用很简单：

- 输入单帧 RGB
- 调用原始 `Fast-SAM-3D-Body` 主链路
- 输出可供下游控制或 retargeting 使用的 `SMPL-compatible` 姿态包

这个仓库不重新实现上游模型，也不替代原始仓库。它更像一个清晰、可集成、可交付的 teleop perception module，适合直接作为组内遥操前端子模块使用。

## 定位

这个模块适合作为遥操作系统中的感知前端。

它负责：

- 单人 3D 身体理解
- `MHR -> SMPL` 转换
- 输出结构化控制字段

它不负责：

- 机器人控制器
- retargeting
- 大量 mesh 可视化
- 完整 UI

## 保留的原生链路

本仓库尽量沿用原始 `Fast-SAM-3D-Body` 的 teleop 路线，包括：

- `build_default_estimator`
- `MultiViewFusionRunner`
- `prepare_publish_pose`
- 原始 `SMPL` 风格输出字段

因此，这个仓库更适合被理解为：

> 面向遥操作集成的 Fast-SAM-3D-Body 工程封装

## 输出

单人检测成功时，核心输出包括：

- `body_quat_wxyz`
- `smpl_joints_local`
- `smpl_pose`

附带少量调试字段：

- `bbox_xyxy`
- `focal_length`
- `pred_cam_t`
- `body_keypoints_2d`
- `named_points_cam`
- `infer_sec`
- `convert_sec`

## 模式

### `body_only`

推荐作为遥操作主链路使用，也是当前默认建议配置。

特点：

- 单人
- 不跑 hand decoder
- 使用原模型原生 `inference_type="body"`
- 保留完整 `MHR -> SMPL -> prepare_publish_pose` 路径

### `full_body_hands`

适合需要更完整 body + hand 表达的场景，但不建议直接作为当前阶段的主控制回路。

特点：

- 单人
- 保留 body + hands 更完整输出
- 更适合 richer avatar / 手部研究

## RTX 4090 实测性能

测试环境基于已经优化完成的 `Fast-SAM-3D-Body` 运行环境，包含：

- `flash_attn`
- `TensorRT`
- `YOLO pose engine`
- `MoGe TRT`
- `backbone TRT`
- `torch.compile`

### `body_only`

- 端到端约 **7.1 fps**
- 纯 inference 约 **7.6 fps**

### `full_body_hands`

- 端到端约 **4.3 fps**
- 纯 inference 约 **4.5 fps**

## 性能结论

- 当前最适合作为遥操主链路的是 `body_only`
- 去掉可视化后，速度几乎不会继续明显提升
- 当前主要瓶颈仍然是 `Fast-SAM-3D-Body` 的 body inference 本体

## 推荐用法

如果目标是 humanoid teleoperation，建议：

- 主链路使用 `body_only`
- 手部细节放到低频旁路或后续扩展
- 可视化不要放在控制主回路里

## 与上游仓库的关系

本仓库默认依赖一份已经准备好的上游 `Fast-SAM-3D-Body` 仓库，包括：

- 代码本体
- checkpoint
- `mhr2smpl` 资源
- `SMPL_NEUTRAL.pkl`

可通过两种方式接入：

- 在 `Fast3DTeleopSubmoduleConfig` 中传入 `project_root`
- 或设置环境变量 `FAST3D_REPO_ROOT`

## 仓库结构

```text
fast3d-teleop-submodule/
  README.md
  LICENSE
  pyproject.toml
  src/
    fast3d_teleop_submodule/
      __init__.py
      core.py
```

## 最小示例

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

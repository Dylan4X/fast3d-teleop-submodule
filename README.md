# Fast3D Teleop Submodule

`Fast3D Teleop Submodule` 是一个面向遥操作的感知子模块封装，基于原始项目 [Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body) 进行抽取与整理。

这个仓库的目标不是替代原始仓库，也不是重新实现一套 `Fast-SAM-3D-Body`，而是将其中最适合 humanoid teleoperation 的主链路抽成一个更清晰、更容易集成的模块，方便后续接入机器人控制、retargeting 或上层遥操系统。

## 1. 项目定位

这个仓库适合放在一个完整遥操作系统中的“感知前端”位置。

它负责的事情是：

- 输入单帧 RGB 图像
- 调用原始 `Fast-SAM-3D-Body` 感知与 `MHR -> SMPL` 转换链路
- 输出适合控制系统消费的 `SMPL-compatible` 结构化姿态包

它不负责的事情是：

- 机器人控制器本身
- 动作 retargeting
- 下游策略学习
- 大量可视化渲染
- 完整 UI 产品化界面

一句话理解：

> 这是一个把 `Fast-SAM-3D-Body` 收束成“遥操感知子模块”的工程化封装。

## 2. 设计原则

本仓库遵循两个核心原则：

### 2.1 严格贴近原始论文和原仓库

本仓库没有自创一套新的姿态协议，而是尽可能沿用原始 `Fast-SAM-3D-Body` 的 teleop 路径，包括：

- `build_default_estimator`
- `MultiViewFusionRunner`
- `prepare_publish_pose`
- 原始 `SMPL` 风格发布字段

也就是说，这个仓库的目标不是“看起来像”，而是尽可能保持与原始作者 teleop 设计的一致性。

### 2.2 优先服务遥操，而不是 demo 可视化

在遥操场景中，核心问题是：

- 姿态是否稳定
- 输出是否结构化
- 是否能进入控制链路
- 是否能在合理帧率下运行

因此，这个仓库优先保留了控制真正需要的内容，弱化了对 mesh 展示和重可视化的依赖。

## 3. 与原始 Fast-SAM-3D-Body 的关系

本仓库不是独立模型仓库。

它默认假设你已经有一份准备好的 `Fast-SAM-3D-Body` 上游仓库，包括：

- 代码本体
- checkpoint
- YOLO engine / TRT engine
- `mhr2smpl` 相关资源
- `SMPL_NEUTRAL.pkl`

本仓库通过两种方式接入上游：

- 在 `Fast3DTeleopSubmoduleConfig` 中显式传入 `project_root`
- 或设置环境变量 `FAST3D_REPO_ROOT`

因此，这个项目更准确地说是：

> 一个“依赖上游 Fast-SAM-3D-Body”的轻量集成仓库。

## 4. 输出内容

在单人检测成功且关键字段完整的情况下，模块会输出一个控制友好的姿态包。

核心输出字段包括：

- `body_quat_wxyz`
- `smpl_joints_local`
- `smpl_pose`

同时保留少量调试字段，方便系统联调：

- `bbox_xyxy`
- `focal_length`
- `pred_cam_t`
- `body_keypoints_2d`
- `named_points_cam`
- `infer_sec`
- `convert_sec`

这些字段的作用分别是：

- `body_quat_wxyz`
  - 根部朝向，适合下游身体整体姿态控制
- `smpl_joints_local`
  - 局部关节位置，适合做结构化控制或中间态观测
- `smpl_pose`
  - 完整的 `SMPL` 姿态参数，适合继续做 retargeting
- `pred_cam_t / focal_length`
  - 保留相机相关估计信息，便于调试与外部对齐
- `named_points_cam`
  - 提供少量便于直观理解的关键点调试输出

## 5. 两种运行模式

## `body_only`

这是当前推荐的遥操主链路模式。

特点：

- 单人
- 不跑 hand decoder
- 使用原模型原生的 `inference_type="body"`
- 保留原始 `MHR -> SMPL -> prepare_publish_pose` 主链路

这个模式的优点是：

- 明显更快
- 更适合作为控制主回路
- 对于全身遥操，已经能保留核心身体信息

这个模式的缺点是：

- 不保留完整手部细节
- 不适合追求最完整 avatar 表达的展示场景

## `full_body_hands`

这是更完整的 body + hands 模式。

特点：

- 单人
- 保留身体和手部更完整的输出能力
- 更接近完整 avatar 路线

这个模式的优点是：

- 表达更完整
- 更适合后续研究 richer avatar 或细粒度 body-hand 表达

这个模式的缺点是：

- 帧率显著低于 `body_only`
- 不适合作为当前阶段最优的控制主链路

## 6. RTX 4090 上的实际表现

以下性能结果来自实际服务器测试，硬件为 `RTX 4090`，并基于已经优化过的 `Fast-SAM-3D-Body` 环境：

- 已启用原仓库公开链路中的关键加速能力
  - `flash_attn`
  - `TensorRT`
  - `YOLO pose engine`
  - `MoGe TRT`
  - `backbone TRT`
  - `torch.compile`

### `body_only`

- 端到端吞吐约 **7.1 fps**
- 纯 inference 吞吐约 **7.6 fps**

### `full_body_hands`

- 端到端吞吐约 **4.3 fps**
- 纯 inference 吞吐约 **4.5 fps**

## 7. 性能结论

有几点结论是比较明确的：

### 7.1 `body_only` 是当前最适合遥操主链路的版本

从当前测试结果看，`body_only` 在 4090 上已经明显优于完整 body+hands 路线，并且保持了原始 teleop 输出结构。

### 7.2 可视化不是当前主瓶颈

我们额外测试过“完全去掉可视化，只传机器人控制所需 JSON”的版本。

结果显示：

- 速度几乎没有显著提升

这说明当前瓶颈主要在：

- `Fast-SAM-3D-Body` 的 body inference 本体

而不是：

- JSON 打包
- 控制字段组织
- 简单调试输出

### 7.3 继续压 `IMG_SIZE` 在当前公开模型配置下不可直接使用

我们尝试进一步减小输入尺寸，但在当前公开模型 / TRT / backbone 配置下会触发 shape mismatch。

因此，进一步靠缩小 `IMG_SIZE` 白拿速度，在当前公开配置下并不可行。

## 8. 为什么要抽这个子模块

原始 `Fast-SAM-3D-Body` 仓库同时承担了：

- 论文复现
- demo 展示
- publisher / debug
- 实时部署实验

对于真正的遥操作系统接入来说，直接拿原仓库做上层系统开发会有几个问题：

- 功能边界不够清晰
- demo 路线与控制主链路混在一起
- 不够像一个“可依赖的模块”

因此，把它抽成一个单独子模块的价值在于：

- 明确系统边界
- 保留原始 teleop 能力
- 更容易被 retargeting / controller 调用
- 更适合作为组内交付件和后续二次开发基础

## 9. 推荐使用方式

如果目标是 humanoid teleoperation，我当前建议：

- 主控制链路使用 `body_only`
- 可视化作为旁路，而不是主链路的一部分
- 若后续确实需要手部细节，再考虑低频更新手部，而不是每帧完整跑 `full_body_hands`

也就是说，比较合理的系统结构是：

1. RGB 帧进入本模块
2. 本模块输出 `SMPL-compatible` 控制字段
3. 下游 retargeting / controller 消费这些字段
4. 可视化与调试链路旁路运行

## 10. 仓库结构

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

## 11. 最小示例

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

## 12. 运行假设

本仓库默认你已经准备好了：

- `Fast-SAM-3D-Body` 上游仓库
- 对应 checkpoint
- `SMPL_NEUTRAL.pkl`
- 相关 `mhr2smpl` 资源
- 已经配置完成的运行环境

本仓库本身不重复打包这些大型资产。

## 13. 当前边界

这个仓库当前已经适合作为：

- 遥操感知前端子模块
- 组内交付件
- 后续 retargeting / controller 接口的上游输入

但它目前仍然不是：

- 完整机器人遥操系统
- 最终机器人控制器
- 全流程产品化前端

## 14. 当前建议结论

如果目标是基于 `Fast-SAM-3D-Body` 做 humanoid teleoperation，我建议：

- 当前阶段优先采用 `body_only`
- 将本仓库作为控制主链路中的感知子模块
- 将 `full_body_hands` 保留给后续 richer avatar / 手部精细控制研究

从现阶段结果来看，这样的划分在：

- 速度
- 稳定性
- 工程可集成性
- 与原始论文方向的一致性

之间达到了比较合理的平衡。

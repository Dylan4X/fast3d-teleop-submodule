# Fast3D Teleop Submodule

面向 humanoid teleoperation 的视觉动捕后端，基于 [Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body)。

**单帧 RGB → 3D SMPL 姿态 → ZMQ 广播 → [Teleopit](https://github.com/BotRunner64/Teleopit) 遥操作框架**

## 前提条件

1. **Fast-SAM-3D-Body 推理环境已部署**：
   - `sam_3d_body/`、`tools/` 代码可用
   - `checkpoints/` 含 model.ckpt、YOLO-Pose engine、MoGe2 TRT engine、backbone TRT engine
   - `mhr2smpl/` 数据文件（mapping、smoother）
2. **SMPL 模型**：`SMPL_NEUTRAL.pkl`（来自 [GVHMR](https://github.com/zhengyiluo/GVHMR) 或 [SMPL 官网](https://smpl.is.tue.mpg.de/)）
3. **Python ≥ 3.10**，PyTorch ≥ 2.0 + CUDA
4. **GPU**：NVIDIA Ampere+ (RTX 3090/4090/5090) 推荐，TF32 自动启用

## 安装

```bash
cd fast3d-teleop-submodule
pip install -e ".[server]"
```

## 使用

### CLI 启动 Mocap Server

```bash
# 摄像头实时推理
fast3d-mocap-server \
  --source 0 \
  --smpl-model-path /path/to/SMPL_NEUTRAL.pkl \
  --project-root /path/to/Fast-SAM-3D-Body

# 视频文件调试
fast3d-mocap-server \
  --source video.mp4 --loop-video \
  --smpl-model-path /path/to/SMPL_NEUTRAL.pkl \
  --project-root /path/to/Fast-SAM-3D-Body

# 自定义端口 + OpenCV 预览窗口
fast3d-mocap-server \
  --source 0 --port 5556 --show-preview \
  --smpl-model-path /path/to/SMPL_NEUTRAL.pkl \
  --project-root /path/to/Fast-SAM-3D-Body
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source` | `0` | 摄像头索引或视频文件路径 |
| `--smpl-model-path` | **必填** | SMPL_NEUTRAL.pkl 路径 |
| `--project-root` | 自动探测 | Fast-SAM-3D-Body 仓库根目录 |
| `--host` | `0.0.0.0` | ZMQ PUB 绑定地址 |
| `--port` | `5555` | ZMQ PUB 端口 |
| `--topic` | `mocap` | ZMQ 消息 topic |
| `--mode` | `body_only` | `body_only` 或 `full_body_hands` |
| `--image-size` | `512` | 推理分辨率 |
| `--warmup-frames` | `30` | 预热帧数（torch.compile 编译） |
| `--loop-video` | `false` | 视频播完后循环 |
| `--show-preview` | `false` | 显示 OpenCV 预览 |
| `--fps-limit` | `0` | 最大输出 FPS（0=不限） |

### Python API

```python
import time
from fast3d_teleop_submodule import (
    Fast3DTeleopSubmodule,
    Fast3DTeleopSubmoduleConfig,
    packet_to_human_frame,
)

cfg = Fast3DTeleopSubmoduleConfig(
    mode="body_only",
    project_root="/path/to/Fast-SAM-3D-Body",
    smpl_model_path="/path/to/SMPL_NEUTRAL.pkl",
)
submodule = Fast3DTeleopSubmodule(cfg)
submodule.warmup()

packet = submodule.process_frame(frame_bgr, timestamp=time.monotonic())
if packet is not None:
    human_frame = packet_to_human_frame(packet)
    # → {"Pelvis": [[x,y,z], [qw,qx,qy,qz]], ...}
```

## 接入 Teleopit

### 架构

```
[GPU 机器]                                  [控制机器]
fast3d-mocap-server                         Teleopit
  Camera → YOLO → MoGe → ViT → Decoder       ZMQInputProvider
  → MHR→SMPL → packet_to_human_frame()        → GMR Retargeter
  → ZMQ PUB (msgpack)  ──── 网络 ────→          → RL Policy
                                                → Robot
```

### 步骤

**1. GPU 机器启动 mocap server：**

```bash
fast3d-mocap-server \
  --source 0 \
  --smpl-model-path /path/to/SMPL_NEUTRAL.pkl \
  --project-root /path/to/Fast-SAM-3D-Body \
  --host 0.0.0.0 --port 5555
```

**2. 控制机器创建 Teleopit input 配置：**

```yaml
# teleopit/configs/input/zmq_fast3d.yaml
input:
  _target_: teleopit.inputs.zmq_provider.ZMQInputProvider
  host: "<GPU机器IP>"     # 如 192.168.1.100
  port: 5555
  topic: "mocap"
  human_format: "xrobot"  # 24 SMPL joints, pos+quat
```

**3. 启动 Teleopit：**

```bash
python -m teleopit.run \
  input=zmq_fast3d \
  retarget=gmr_g1 \
  policy=rl_walk
```

### ZMQ 消息格式

每帧发布一个 msgpack 消息（topic + payload），与 Teleopit `ZMQInputProvider._deserialize_frame()` 兼容：

```python
# topic: b"mocap"
# payload (msgpack):
{
    "Pelvis":         [[x, y, z], [qw, qx, qy, qz]],
    "Left_Hip":       [[x, y, z], [qw, qx, qy, qz]],
    "Right_Hip":      [[x, y, z], [qw, qx, qy, qz]],
    "Spine1":         [[x, y, z], [qw, qx, qy, qz]],
    # ... 共 24 个 SMPL 关节（见 conversion.py BODY_JOINT_NAMES）
    "_ts": 1234567.89,   # monotonic 时间戳 (float)
    "_seq": 42            # 帧序号 (int)
}
```

24 个关节名称按 SMPL 标准排列：
`Pelvis, Left_Hip, Right_Hip, Spine1, Left_Knee, Right_Knee, Spine2, Left_Ankle, Right_Ankle, Spine3, Left_Foot, Right_Foot, Neck, Left_Collar, Right_Collar, Head, Left_Shoulder, Right_Shoulder, Left_Elbow, Right_Elbow, Left_Wrist, Right_Wrist, Left_Hand, Right_Hand`

## 性能

RTX 4090 独占，`body_only` 模式，TF32 + torch.compile(reduce-overhead) + TRT：

| 阶段 | 耗时 |
|------|------|
| YOLO-Pose 人体检测 | ~5ms |
| MoGe2 FOV 估计 (TRT) | ~11ms |
| DINOv3 Backbone (TRT fp16) | ~32ms |
| Transformer Decoder (compiled) | ~50ms |
| MHR → SMPL 转换 | ~8ms |
| **端到端 p50** | **~93ms** |
| **FPS** | **~10.8** |

> 注意：多 GPU 共享供电时，单卡性能可能下降 30-50%。建议独占 GPU 运行。

## 仓库结构

```text
fast3d-teleop-submodule/
├── pyproject.toml          # 包配置，CLI entry point
├── README.md
├── LICENSE
└── src/
    └── fast3d_teleop_submodule/
        ├── __init__.py
        ├── core.py           # Fast3DTeleopSubmodule 主类
        ├── conversion.py     # TeleopPosePacket → HumanFrame (FK 链)
        ├── env_setup.py      # 环境变量管理（替代 run_demo.sh）
        ├── server.py         # ZMQ mocap 服务器 + CLI entry point
        └── vendor/           # 从 upstream mocap/ 提取的胶水代码
            ├── gravity_alignment.py      # 重力对齐 / 坐标变换
            ├── multiview_mhr2smpl.py     # MHR → SMPL 神经网络转换
            └── pose_protocol.py          # PICO 协议姿态输出
```

## 与上游 Fast-SAM-3D-Body 的关系

**已内化（不再依赖上游 `mocap/` 目录）：**
- MHR→SMPL 转换（`vendor/multiview_mhr2smpl.py`）
- 重力对齐 / 坐标变换（`vendor/gravity_alignment.py`）
- PICO 协议姿态输出（`vendor/pose_protocol.py`）
- 模型构建流程（`core.py` 内联 `build_default_estimator`）
- 30+ 环境变量（`env_setup.py` 替代 `run_demo.sh`）

**仍需上游：**
- `sam_3d_body/` — 模型定义（ViT backbone、Transformer decoder、MHR head）
- `tools/` — HumanDetector (YOLO-Pose)、FOVEstimator (MoGe2)
- `checkpoints/` — 模型权重 + TRT engines
- `mhr2smpl/data/` — SMPL 映射文件

通过 `--project-root` 或环境变量 `FAST3D_REPO_ROOT` 指定上游路径。

## License

MIT

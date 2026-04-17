# Fast3D Teleop Submodule

面向 humanoid teleoperation 的视觉动捕后端，基于 [Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body)。

**单帧 RGB → 3D SMPL 姿态 → ZMQ 广播 → [Teleopit](https://github.com/BotRunner64/Teleopit) 遥操作框架**

---

## 快速开始（3 步启动）

```bash
# 1. 安装
pip install -e ".[server,setup]"

# 2. 下载模型权重（从 ModelScope，约 2.5GB）
fast3d-download-assets --project-root /path/to/Fast-SAM-3D-Body

# 3. 启动
fast3d-mocap-server --source 0 \
  --smpl-model-path /path/to/SMPL_NEUTRAL.pkl \
  --project-root /path/to/Fast-SAM-3D-Body
```

---

## 详细安装指南

### 0. 硬件要求

| 项目 | 要求 |
|------|------|
| **GPU** | NVIDIA Ampere+ (RTX 3090 / 4090 / 5090)，显存 ≥ 16GB |
| **CUDA** | ≥ 12.0（推荐 12.4） |
| **TensorRT** | ≥ 8.6（可选，用于加速推理，需在目标 GPU 上本地构建 engine） |
| **内存** | ≥ 32GB |
| **磁盘** | ≥ 10GB（模型权重 + TRT engines） |

### 1. 克隆仓库

```bash
# 上游推理代码（模型定义、检测器、FOV 估计器）
git clone https://github.com/yangtiming/Fast-SAM-3D-Body.git

# 本仓库（遥操作封装、ZMQ 服务、坐标转换）
git clone https://github.com/Dylan4X/fast3d-teleop-submodule.git
```

### 2. 创建 Conda 环境

```bash
conda create -n fast3d python=3.11 -y
conda activate fast3d

# 安装 CUDA toolkit（detectron2 编译需要）
conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit -y
```

### 3. 安装 PyTorch

```bash
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124
```

### 4. 安装上游依赖

```bash
# 基础 Python 包
pip install pytorch-lightning opencv-python yacs scikit-image einops timm \
    dill pandas rich hydra-core pyrootutils webdataset networkx==3.2.1 roma \
    joblib huggingface_hub smplx chumpy numpy scipy tqdm

# Detectron2（需要 CUDA_HOME）
export CUDA_HOME=$CONDA_PREFIX
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
    --no-build-isolation --no-deps

# YOLO 人体检测
pip install ultralytics

# MoGe FOV 估计
pip install git+https://github.com/microsoft/MoGe.git

# TensorRT + ONNX（可选，用于构建 TRT 加速引擎）
pip install tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs \
    onnx onnxruntime-gpu
```

### 5. 安装本包

```bash
cd fast3d-teleop-submodule
pip install -e ".[server,setup]"
```

`pip install -e ".[server]"` 安装 ZMQ + msgpack。加 `setup` 会额外安装 modelscope。

### 6. 下载模型权重

#### 6a. 从 ModelScope 下载（推荐）

```bash
fast3d-download-assets --project-root /path/to/Fast-SAM-3D-Body
```

该命令自动从 [ModelScope: XDRRRR/fast3d-teleop-assets](https://modelscope.cn/models/XDRRRR/fast3d-teleop-assets) 下载所有模型权重并放置到正确目录。

下载内容（约 2.5GB）：

| 文件 | 大小 | 说明 |
|------|------|------|
| `checkpoints/sam-3d-body-dinov3/model.ckpt` | 1.6G | SAM-3D-Body 主模型（ViT-H backbone + Transformer decoder + MHR head） |
| `checkpoints/sam-3d-body-dinov3/model_config.yaml` | 4K | 模型配置 |
| `checkpoints/sam-3d-body-dinov3/mhr_model.pt` | 664M | MHR TorchScript 模型 |
| `checkpoints/yolo/yolo11m-pose.pt` | 41M | YOLO11m-Pose 人体检测（17 关键点） |
| `checkpoints/moge-2-vits-normal/model.pt` | 135M | MoGe2 ViT-S FOV 估计器 |
| `mhr2smpl/data/mhr2smpl_mapping.npz` | 168K | MHR→SMPL 重心坐标映射 |
| `mhr2smpl/data/mhr_face_mask.ply` | 760K | MHR 面部网格掩码 |
| `mhr2smpl/experiments/.../best_model.pth` | 9.4M | MHR→SMPL 融合网络权重 |
| `mhr2smpl/experiments/.../smoother_best.pth` | 1.3M | 时序平滑器权重 |

#### 6b. 验证下载

```bash
fast3d-download-assets --project-root /path/to/Fast-SAM-3D-Body --verify-only
```

### 7. 获取 SMPL 模型

SMPL 模型因许可证限制**不包含在自动下载中**，需手动获取：

1. 前往 [SMPL 官网](https://smpl.is.tue.mpg.de/) 注册并下载
2. 或从已有的 GVHMR / HumanML3D 等项目中找到 `SMPL_NEUTRAL.pkl`（236MB）
3. 记住文件路径，启动时通过 `--smpl-model-path` 传入

### 8. 构建 TensorRT 引擎（可选，提升 ~40% 性能）

TensorRT 引擎是 **GPU 架构特定的**（RTX 4090 构建的 engine 不能在 RTX 3090 上使用），需在目标 GPU 上本地构建：

```bash
# 构建所有 TRT 引擎（backbone + MoGe + YOLO）
fast3d-build-trt --project-root /path/to/Fast-SAM-3D-Body --gpu 0

# 或单独构建
fast3d-build-trt --project-root /path/to/Fast-SAM-3D-Body --only backbone
fast3d-build-trt --project-root /path/to/Fast-SAM-3D-Body --only moge
fast3d-build-trt --project-root /path/to/Fast-SAM-3D-Body --only yolo
```

构建产物：

| 引擎 | 大小 | 加速效果 |
|------|------|----------|
| `backbone_dinov3_fp16.engine` | ~3.6G | Backbone: ~50ms → ~32ms |
| `moge_dinov2_encoder_fp16.engine` | ~50M | MoGe: ~15ms → ~11ms |
| `yolo11m-pose.engine` | ~30M | YOLO: ~8ms → ~5ms |

> **没有 TRT 也能运行**：程序会自动回退到 PyTorch 推理，只是速度较慢（约 15 FPS → 10 FPS on RTX 4090）。

### 9. 启动！

```bash
fast3d-mocap-server \
  --source 0 \
  --smpl-model-path /path/to/SMPL_NEUTRAL.pkl \
  --project-root /path/to/Fast-SAM-3D-Body
```

---

## CLI 参数

### `fast3d-mocap-server`

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
| `--gravity` | `None` | 相机坐标系重力方向（逗号分隔，如 `0.05,0.99,-0.02`） |
| `--intrinsics-json` | `None` | 标定 JSON 文件路径（含 `gravity` 和可选 `camera_matrix`） |
| `--no-gravity-calibration` | `false` | 跳过 warmup 自动重力标定 |

### `fast3d-download-assets`

| 参数 | 说明 |
|------|------|
| `--project-root` | **必填**，Fast-SAM-3D-Body 目录 |
| `--smpl-dir` | 包含 SMPL_NEUTRAL.pkl 的目录（仅检查） |
| `--verify-only` | 只检查文件是否存在 |

### `fast3d-build-trt`

| 参数 | 说明 |
|------|------|
| `--project-root` | **必填**，Fast-SAM-3D-Body 目录 |
| `--gpu` | GPU 设备 ID（默认 0） |
| `--only` | 只构建指定引擎：`backbone` / `moge` / `yolo` |
| `--force` | 强制重建（删除已有 engine 后重新构建） |

---

## 使用示例

```bash
# 摄像头实时推理
fast3d-mocap-server \
  --source 0 \
  --smpl-model-path /path/to/SMPL_NEUTRAL.pkl \
  --project-root /path/to/Fast-SAM-3D-Body

# 视频文件循环调试
fast3d-mocap-server \
  --source video.mp4 --loop-video \
  --smpl-model-path /path/to/SMPL_NEUTRAL.pkl \
  --project-root /path/to/Fast-SAM-3D-Body

# 使用 RealSense 标定文件（含 IMU 重力 + 相机内参）
fast3d-mocap-server \
  --source 0 \
  --intrinsics-json /path/to/intrinsics.json \
  --smpl-model-path /path/to/SMPL_NEUTRAL.pkl \
  --project-root /path/to/Fast-SAM-3D-Body

# 手动指定重力方向（相机坐标系）
fast3d-mocap-server \
  --source 0 \
  --gravity 0.05,0.99,-0.02 \
  --smpl-model-path /path/to/SMPL_NEUTRAL.pkl \
  --project-root /path/to/Fast-SAM-3D-Body
```

### 重力方向设置

重力方向决定姿态输出的竖直对齐精度。支持三种来源（优先级从高到低）：

1. **`--gravity`** — 直接传入相机坐标系下的重力向量
2. **`--intrinsics-json`** — 从 `record_realsense.py` 输出的标定 JSON 文件读取
3. **Warmup 自动标定** — 无相机 IMU 时，利用 warmup 人体朝向估算重力

标定 JSON 格式：

```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "gravity": [gx, gy, gz]
}
```

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

---

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
provider: zmq_pico4
zmq_host: "<GPU机器IP>"     # 如 192.168.1.100
zmq_port: 5555
zmq_topic: "mocap"           # ⚠ 默认是 "pico4"，这里要改成 "mocap"
human_format: "xrobot"       # 24 SMPL joints, pos+quat → xrobot_to_g1.json IK
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

### 坐标系

输出为 **Z-up** 世界坐标系，与 Teleopit 一致。

变换链：
1. `gravity_alignment.py` → 相机帧 → 重力对齐 Z-up 世界帧
2. `pose_protocol.py` → `GLOBAL_ORIENT_EXTRA_ROT = Ry(90°) × Rx(-90°)` → Y-up
3. `conversion.py` → `EXTRA_COORD_TRANSFORM = [[1,0,0],[0,0,-1],[0,1,0]]` → Z-up

---

## 性能

RTX 4090 独占，`body_only` 模式：

| 配置 | 端到端 p50 | FPS |
|------|-----------|-----|
| PyTorch (无 TRT) + torch.compile + TF32 | ~140ms | ~7 |
| **TRT backbone + MoGe + YOLO + torch.compile + TF32** | **~93ms** | **~10.8** |

> 多 GPU 共享供电时，单卡性能可能下降 30-50%。建议**独占 GPU** 运行。

---

## 可视化

```bash
# 需先启动 mocap server

# 3D 骨架可视化（保存为视频）
python scripts/visualize_skeleton.py --save /tmp/skeleton.mp4

# 原视频 + 3D 骨架并排对比
python scripts/visualize_sidebyside.py \
  --video /path/to/test.mp4 \
  --save /tmp/sidebyside.mp4 \
  --max-frames 200
```

---

## 仓库结构

```text
fast3d-teleop-submodule/
├── pyproject.toml
├── README.md
├── scripts/
│   ├── download_assets.py          # 从 ModelScope 下载模型权重
│   ├── build_trt.py                # 构建 TensorRT 引擎
│   ├── visualize_skeleton.py       # 3D 骨架可视化
│   └── visualize_sidebyside.py     # 原视频 + 骨架对比
└── src/
    └── fast3d_teleop_submodule/
        ├── __init__.py
        ├── core.py                 # Fast3DTeleopSubmodule 主类
        ├── conversion.py           # TeleopPosePacket → HumanFrame (FK)
        ├── env_setup.py            # 环境变量管理
        ├── server.py               # ZMQ mocap 服务器 + CLI
        ├── _cli_download.py        # fast3d-download-assets 入口
        ├── _cli_build_trt.py       # fast3d-build-trt 入口
        └── vendor/
            ├── gravity_alignment.py
            ├── multiview_mhr2smpl.py
            └── pose_protocol.py
```

### 依赖关系

```
fast3d-teleop-submodule（本仓库）
  ↓ --project-root
Fast-SAM-3D-Body（上游，运行时 import）
  ├── sam_3d_body/     模型定义
  ├── tools/           HumanDetector, FOVEstimator
  └── checkpoints/     模型权重 + TRT engines
```

**已内化（不需要上游 `mocap/` 目录）：**
- MHR→SMPL 转换、重力对齐、PICO 协议姿态输出
- 模型构建流程（`core.py` 内联 `build_default_estimator`）
- 30+ 环境变量（`env_setup.py` 替代 `run_demo.sh`）

---

## 常见问题

### Q: TRT engine 能跨 GPU 使用吗？

不能。`.engine` 文件绑定到特定 GPU 架构（如 sm_89 = RTX 4090）和 TensorRT 版本。换 GPU 需要重新运行 `fast3d-build-trt`。

### Q: 不装 TensorRT 能用吗？

可以。设置环境变量关闭 TRT：
```bash
USE_TRT_BACKBONE=0 FOV_TRT=0 fast3d-mocap-server ...
```
或者不构建 TRT engine，程序会自动回退到 PyTorch。

### Q: SMPL_NEUTRAL.pkl 哪里下载？

- [SMPL 官网](https://smpl.is.tue.mpg.de/) — 注册后下载（推荐）
- 如果你有 GVHMR/HumanML3D/Human4DiT 等项目，它们通常自带此文件

### Q: 怎么指定使用哪张 GPU？

```bash
CUDA_VISIBLE_DEVICES=2 fast3d-mocap-server --source 0 ...
```

---

## License

MIT
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

# 使用 RealSense 标定文件（含重力方向 + 相机内参）
fast3d-mocap-server \
  --source 0 \
  --intrinsics-json /path/to/intrinsics.json \
  --smpl-model-path /path/to/SMPL_NEUTRAL.pkl \
  --project-root /path/to/Fast-SAM-3D-Body

# 手动指定重力方向（相机坐标系）
fast3d-mocap-server \
  --source 0 \
  --gravity 0.05,0.99,-0.02 \
  --smpl-model-path /path/to/SMPL_NEUTRAL.pkl \
  --project-root /path/to/Fast-SAM-3D-Body
```

#### 重力方向设置

重力方向决定了姿态输出的竖直对齐精度。支持三种来源（优先级从高到低）：

1. **`--gravity`** — 直接传入相机坐标系下的重力向量
2. **`--intrinsics-json`** — 从 `record_realsense.py` 输出的标定 JSON 文件读取（含 IMU 重力标定）
3. **Warmup 自动标定** — 无相机 IMU 时，利用 warmup 期间的人体朝向估算重力（假设人近似直立）

标定 JSON 格式（与上游 `record_realsense.py` 一致）：

```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "gravity": [gx, gy, gz]
}
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
| `--gravity` | `None` | 相机坐标系重力方向（逗号分隔，如 `0.05,0.99,-0.02`） |
| `--intrinsics-json` | `None` | 标定 JSON 文件路径（含 `gravity` 和可选 `camera_matrix`） |
| `--no-gravity-calibration` | `false` | 跳过 warmup 自动重力标定 |

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
provider: zmq_pico4
zmq_host: "<GPU机器IP>"     # 如 192.168.1.100
zmq_port: 5555
zmq_topic: "mocap"           # ⚠ 默认是 "pico4"，这里要改成 "mocap"
human_format: "xrobot"       # 24 SMPL joints, pos+quat → xrobot_to_g1.json IK
```

> **注意**：Teleopit 的 input provider 使用工厂模式（`factory.py` → `_build_input_provider()`），
> 而非 Hydra `_target_` 实例化。`provider: zmq_pico4` 会自动构建 `ZMQInputProvider`。

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
    "_seq": 42            # 帧序号 (int), Teleopit 自动 pop
}
```

24 个关节名称按 SMPL 标准排列：
`Pelvis, Left_Hip, Right_Hip, Spine1, Left_Knee, Right_Knee, Spine2, Left_Ankle, Right_Ankle, Spine3, Left_Foot, Right_Foot, Neck, Left_Collar, Right_Collar, Head, Left_Shoulder, Right_Shoulder, Left_Elbow, Right_Elbow, Left_Wrist, Right_Wrist, Left_Hand, Right_Hand`

### 坐标系

数据经过 `gravity_alignment.py` → `pose_protocol.py` → `EXTRA_COORD_TRANSFORM` 的变换链后，
输出为 **Z-up** 世界坐标系，与 Teleopit 期望一致。

变换链：
1. `gravity_alignment.py`：相机帧 → 重力对齐 Z-up 世界帧
2. `pose_protocol.py`：`GLOBAL_ORIENT_EXTRA_ROT = Ry(90°) × Rx(-90°)`，输出变为 Y-up
3. `conversion.py`：`EXTRA_COORD_TRANSFORM = [[1,0,0],[0,0,-1],[0,1,0]]`，Y-up → Z-up

最终 `EXTRA_COORD_TRANSFORM` 矩阵与 Teleopit 的 `_INPUT_TO_TELEOPIT_MATRIX` 完全一致。

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
├── scripts/
│   ├── visualize_skeleton.py       # 3D 骨架实时可视化
│   └── visualize_sidebyside.py     # 原视频 + 骨架对比可视化
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

## 可视化工具

需要先启动 mocap server，然后连接 ZMQ 订阅数据：

```bash
# 3D 骨架可视化（保存为视频）
python scripts/visualize_skeleton.py --save /tmp/skeleton.mp4

# 原视频 + 3D 骨架并排对比
python scripts/visualize_sidebyside.py \
  --video /path/to/test.mp4 \
  --save /tmp/sidebyside.mp4 \
  --max-frames 200
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

"""Centralized environment variable setup for Fast-SAM-3D-Body inference.

Calling ``setup_env()`` sets all required env vars with sensible defaults
so that downstream code (sam_3d_body, mocap, tools) works correctly
**without** needing to ``source run_demo.sh`` first.

All values are set via ``os.environ.setdefault`` so user overrides are
respected.
"""

from __future__ import annotations

import os


_DEFAULTS: dict[str, str] = {
    # ---- TF32 ----
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
    # ---- torch.compile ----
    "TORCHINDUCTOR_CUDAGRAPH_TREES": "0",
    "USE_COMPILE": "1",
    "USE_COMPILE_BACKBONE": "1",
    "DECODER_COMPILE": "1",
    "COMPILE_MODE": "reduce-overhead",
    "COMPILE_WARMUP_BATCH_SIZES": "1",
    "MHR_USE_CUDA_GRAPH": "0",
    # ---- Model / pipeline ----
    "IMG_SIZE": "512",
    "LAYER_DTYPE": "fp32",
    "GPU_HAND_PREP": "1",
    "SKIP_KEYPOINT_PROMPT": "1",
    "KEYPOINT_PROMPT_INTERM_INTERVAL": "999",
    "BODY_INTERM_PRED_LAYERS": "0,1,2",
    "HAND_INTERM_PRED_LAYERS": "0,1",
    "MHR_NO_CORRECTIVES": "1",
    # ---- TensorRT ----
    "USE_TRT_BACKBONE": "1",
    "FOV_TRT": "1",
    "FOV_FAST": "1",
    "FOV_MODEL": "s",
    "FOV_LEVEL": "0",
    "FOV_SIZE": "512",
    # ---- Debug (off) ----
    "DEBUG_NAN": "0",
    "DEBUG_HAND_PREP": "0",
    "DEBUG_BACKBONE_INPUT": "0",
    "INTERM_TIMING": "0",
}


def setup_env(*, image_size: int | None = None) -> None:
    """Apply default env vars. Call once before any upstream import."""
    for k, v in _DEFAULTS.items():
        os.environ.setdefault(k, v)
    if image_size is not None:
        os.environ["IMG_SIZE"] = str(image_size)


def enable_tf32() -> None:
    """Enable TF32 for matmul and cuDNN (requires Ampere+ GPU)."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except ImportError:
        pass

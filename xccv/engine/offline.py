"""基准权重离线加载。

平台随包分发全部基准模型权重（torchvision ImageNet 骨干 + ultralytics
YOLO 官方预训练），构建模型时优先使用本地文件，完全离线、不联网下载。

目录结构（开发态在 xccv/engine/weights/，打包后复制到 exe 同级 weights/）：
weights/
  torch/hub/checkpoints/*.pth     ← torchvision 预训练（TORCH_HOME 重定向）
  ultralytics/*.pt                ← YOLO 官方预训练（yolo.py 直接取用）
download_weights.py 为一次性下载脚本，可随时重新补齐。
"""

import os
import sys
from pathlib import Path

from ..app_paths import app_root, is_installed

# 安装态/冻结态：运行根下 weights/（打包脚本负责复制）；
# 开发态：引擎包内 xccv/engine/weights/
_here = Path(__file__).resolve()
if is_installed() or getattr(sys, "frozen", False):
    WEIGHTS_DIR = app_root() / "weights"
else:
    WEIGHTS_DIR = _here.parent / "weights"

_setup_done = False


def setup_offline_weights() -> None:
    """在构造任何模型之前调用（engine/__init__ 已自动调用）。

    - TORCH_HOME 重定向到随包权重目录，torchvision 全部走本地缓存；
    - YOLO_CONFIG_DIR 隔离到 platform_data，不写用户 AppData。
    """
    global _setup_done
    if _setup_done:
        return
    _setup_done = True

    torch_dir = WEIGHTS_DIR / "torch"
    if torch_dir.is_dir():
        os.environ["TORCH_HOME"] = str(torch_dir)

    try:
        base = app_root() / "platform_data"
        (base / "ultralytics").mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(base / "ultralytics"))
    except Exception:
        pass  # 目录不可写时退回 ultralytics 默认配置位置

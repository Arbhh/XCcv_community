"""E2（V0.7）：训练 run_manifest.json——权重可追溯元数据。

工业现场必问的三件事：这个权重由哪份数据、哪个环境、哪套依赖训出来的。
训练收尾写入输出目录 run_manifest.json：
- environment：python/torch/torchvision/ultralytics/tensorrt/CUDA 版本 +
  GPU 名称与算力（替代建议书的 git commit 字段——主区无 git，铁律 4）；
- lock：requirements.lock 的 sha1（存在时；路径可用 XCCV_LOCK_PATH 覆盖，
  E1 生成物默认 E:\\zcode\\XCcv_dist\\build_tools\\requirements.lock）；
- dataset：数据集指纹（复用 dataset_fingerprint）+ 路径；
- config：完整 TrainConfig 快照（asdict）；
- products：best.pt/metrics.json 相对名 + 最终指标摘要；
- application：__version__ + 时间。
失败不阻塞训练收尾（best-effort，与 metrics.json 同级兜底）。
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

_LOCK_DEFAULT = Path(__file__).resolve().parents[3] / "requirements.lock"


def _lock_hash() -> str | None:
    p = Path(os.environ.get("XCCV_LOCK_PATH") or _LOCK_DEFAULT)
    try:
        if p.is_file():
            return hashlib.sha1(p.read_bytes()).hexdigest()
    except Exception:
        pass
    return None


def _env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cudnn"] = getattr(torch.backends.cudnn, "version", lambda: None)()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            info["compute_capability"] = list(cap)
    except Exception:
        info["torch"] = None
    for mod, key in (("torchvision", "torchvision"), ("ultralytics", "ultralytics"),
                     ("tensorrt", "tensorrt"), ("cv2", "opencv")):
        try:
            m = __import__(mod)
            info[key] = getattr(m, "__version__", None)
        except Exception:
            info[key] = None
    return info


def build_run_manifest(cfg, metrics: dict) -> dict:
    """TrainConfig + 最终指标 → manifest dict（纯函数，便于测试）。

    TrainConfig 含 Path 字段（dataset_dir/output_dir）——asdict 后统一
    转 str，否则 json.dumps 直接 TypeError（f7e 门禁实锤：写入静默失败）。
    """
    from dataclasses import asdict

    from ..dataset.prepare import dataset_fingerprint

    try:
        import xccv as _app

        version = getattr(_app, "__version__", "")
    except Exception:
        version = ""
    final = metrics.get("final") or {}

    def _plain(obj):
        if isinstance(obj, dict):
            return {k: _plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_plain(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    manifest = {
        "schema": "xccv-run-manifest",
        "schema_version": 1,
        "application": {"version": version,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        "environment": _env_info(),
        "lock_sha1": _lock_hash(),
        "dataset": {"dir": str(cfg.dataset_dir),
                    "fingerprint": dataset_fingerprint(cfg.dataset_dir)},
        "config": _plain(asdict(cfg)),
        "products": {"best": "best.pt", "metrics": "metrics.json"},
        "metrics_summary": {k: final.get(k) for k in
                            ("map50", "map50_95", "map_loose", "mar_loose",
                             "precision", "recall", "auroc", "exact_match",
                             "per_class_ap50") if k in final},
        "deterministic": bool(getattr(cfg, "deterministic", False)),
    }
    return manifest


def write_run_manifest(cfg, metrics: dict, output_dir) -> None:
    """训练收尾调用；失败仅告警（不阻塞收尾）。"""
    try:
        from ..io_utils import save_json

        manifest = build_run_manifest(cfg, metrics)
        save_json(Path(output_dir) / "run_manifest.json", manifest)
        log.info("run_manifest.json 已写入: %s", output_dir)
    except Exception:
        log.warning("run_manifest.json 写入失败（不阻塞训练收尾）", exc_info=True)

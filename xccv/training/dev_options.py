"""开发者选项：数据体检与训练前预检的阈值设置（训练平台齿轮入口）。

设置持久化在 platform_data/dev_options.json。按用户规则，所有检查
仅分「提示 / 警告」两档，不阻止训练；"疑似错标"检查暂不实现（D3）。
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ..app_paths import app_root
from ..engine.io_utils import read_json, save_json


@dataclass
class DevOptions:
    """体检/预检阈值（集中成常量便于调整，齿轮对话框可改）。"""

    # ---- 数据体检（默认阈值按用户 2026-09-16 回执保守化：宁少报不误报）----
    imbalance_share: float = 0.05  # 类别不均衡：最小类实例占整体比例 < 该值 → 警告（原 5× 比例口径，用户 Q3 改占比口径）
    min_cls_samples: int = 15      # 分类：单类样本 < 该值 → 警告
    min_det_instances: int = 30    # 检测/分割：单类实例 < 该值 → 警告
    dup_hamming: int = 4           # 疑似重复图：dHash 64 位汉明距离 ≤ 该值 → 提示
    dup_scan_cap: int = 2000       # 重复图检测的图片数上限（超出跳过该项，避免体检过慢）
    size_outlier_mad: float = 3.0  # 尺寸离群：偏离中位数 ±N×MAD → 提示
    # ---- 训练前预检 ----
    preflight_enabled: bool = True    # 开始训练前执行预检（体检 + 参数建议）
    preflight_info_popup: bool = True  # 仅有提示级问题时是否也弹预检窗（用户 A5 回执：默认提示一下）


def options_path() -> Path:
    return app_root() / "platform_data" / "dev_options.json"


def load_options() -> DevOptions:
    """读取开发者选项；文件缺失或字段损坏时逐项回退默认值。"""
    p = options_path()
    data = read_json(p) if p.is_file() else None
    if not isinstance(data, dict):
        return DevOptions()
    defaults = DevOptions()
    clean = {}
    for f in fields(DevOptions):
        if f.name not in data:
            continue
        v = data[f.name]
        try:
            if isinstance(getattr(defaults, f.name), bool):
                clean[f.name] = bool(v)
            elif isinstance(getattr(defaults, f.name), int):
                clean[f.name] = int(v)
            elif isinstance(getattr(defaults, f.name), float):
                clean[f.name] = float(v)
            else:
                clean[f.name] = v
        except (TypeError, ValueError):
            continue  # 手改 json 类型写错 → 用默认值，不让设置文件弄崩平台
    return DevOptions(**clean)


def save_options(options: DevOptions) -> None:
    save_json(options_path(), asdict(options))

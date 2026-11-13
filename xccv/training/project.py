def migrate_history(recs) -> list:
    """训练历史旧键名清理（社区版：历史记录无旧键，原样返回）。"""
    return recs or []


"""训练项目管理：项目信息与配置的持久化。

每个项目对应 platform_data/projects/<项目名>/project.json。
"""

import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from ..app_paths import app_root
from ..engine.dataset.base import TaskType
from ..engine.io_utils import read_json, save_json

# 平台数据根目录：随运行根（安装态=安装根 / 开发态=仓库根）
PLATFORM_ROOT = app_root() / "platform_data"
PROJECTS_DIR = PLATFORM_ROOT / "projects"

# 项目自有的剪枝预训练模型目录（模型工具"训练前剪枝"的输出，训练平台可读取）
PRUNED_DIR_NAME = "pruned"

_NAME_PATTERN = re.compile(r'^[^\\/:*?"<>|]{1,64}$')


@dataclass
class Project:
    """一个训练项目的全部设置。"""

    name: str
    dataset_dir: str
    model_dir: str                 # 训练模型储存路径（各次训练输出到其子目录）
    task: str                      # TaskType.value
    model_name: str = ""
    augment_ops: dict = field(default_factory=dict)  # resize 宽高即训练输入尺寸（归一化）
    augment_probs: dict = field(default_factory=dict)  # 每算子启用概率；旧项目缺失按 0.5
    val_ratio: float = 0.2
    train_ratio: float = 0.8     # 训练集占比（与 val_ratio 之和可 >1：重叠抽样）
    multiply_factor: int = 1
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    eval_interval: int = 1
    device: str = "auto"
    workers: int = 4              # 数据加载线程数
    use_pretrained: bool = True   # 是否使用预训练权重
    industrial_opt: bool = True   # 工业场景优化（注意力 + 改进损失）
    transfer_file: str = ""       # 迁移学习：指定起始权重文件（.pt；空=不迁移）
    patience: int = 10            # 早停耐心（0 关闭）
    # ---- 训练加速（默认关闭；GPU 工控机上按需开启）----
    freeze_backbone: bool = False  # 冻结预训练骨干（快速迁移模式）
    rpn_reduced: bool = False      # R-CNN 系列：削减 RPN 候选框（其他家族忽略）
    use_amp: bool = True           # AMP 混合精度（仅 GPU 生效，CPU 自动关闭）
    mask_head_light: bool = False  # Mask R-CNN：轻量掩码头（其他家族忽略）
    slim_head: bool = False        # T9-5 slim 头（仅 maskrcnn18）：FPN64 轻量结构
    anchor_small: bool = False     # 小目标锚框自适应（R-CNN 家族，默认关）
    # R-CNN 优化器（默认 AdamW 1e-4，可回退 SGD）
    rcnn_optimizer: str = "adamw"
    rcnn_learning_rate: float = 0.0   # 0=自动
    rcnn_weight_decay: float = 0.05
    train_input: int = 0           # T9-2 两段式缩放（R-CNN 家族 opt-in，0=跟随画布）
    tiling: bool = False           # T9-3 切图训练（大图小目标；推理需配分块推理）
    tile_size: int = 640           # 切图训练 tile 边长
    use_negatives: bool = True     # 负样本：OK/良品 等文件夹内未标注图作背景训练（压误检）
    p2_head: bool = False          # P2 小目标检测头（YOLO 家族，默认关；其他家族忽略）
    color_mode: str = "color"      # O5（V0.7）：训练颜色模式 color/gray（灰度防颜色干扰）
    deterministic: bool = False    # E3（V0.7）：确定性训练 opt-in（同机同种子可复现；有性能损失）
    tf32: bool = True              # GPU 提速：TF32 + cuDNN 自动调优（仅 GPU 生效）
    save_epoch_weights: bool = True  # 逐轮保存权重到共用 epoch_weights 文件夹（覆盖式）
    created_at: str = ""

    @property
    def task_type(self) -> TaskType:
        return TaskType(self.task)

    @property
    def project_dir(self) -> Path:
        return PROJECTS_DIR / self.name

    def save(self) -> None:
        save_json(self.project_dir / "project.json", asdict(self))


def validate_name(name: str) -> str:
    """校验项目名；不合法时抛 ValueError（信息面向用户）。"""
    name = (name or "").strip()
    if not _NAME_PATTERN.match(name):
        raise ValueError("项目名称不能为空，且不能包含 \\/:*?\"<>| 等字符（长度≤64）")
    return name


def _project_touch_time(d: Path) -> float:
    """项目最近更新时间：project.json 与各次训练 metrics.json 的最新 mtime。"""
    t = 0.0
    try:
        t = (d / "project.json").stat().st_mtime
    except OSError:
        pass
    models = d / "models"
    try:
        if models.is_dir():
            for sub in models.iterdir():
                m = sub / "metrics.json"
                if m.is_file():
                    t = max(t, m.stat().st_mtime)
    except OSError:
        pass
    return t


def list_projects() -> list[str]:
    """项目名列表，按最近更新时间倒序（更新越近越靠前，第三批要求）。"""
    if not PROJECTS_DIR.is_dir():
        return []
    dirs = [d for d in PROJECTS_DIR.iterdir()
            if d.is_dir() and (d / "project.json").is_file()]
    dirs.sort(key=lambda d: d.name)
    dirs.sort(key=_project_touch_time, reverse=True)
    return [d.name for d in dirs]


def load_project(name: str) -> Project:
    data = read_json(PROJECTS_DIR / name / "project.json")
    fields = Project.__dataclass_fields__
    if "image_size" in data:  # 旧版迁移：方形输入尺寸 → 增强策略 resize（宽高一致）
        old = data.pop("image_size") or 0
        if old > 0 and not data.get("augment_ops"):
            data["augment_ops"] = {"resize": {"width": int(old), "height": int(old)}}
    if "train_ratio" not in data:  # 旧版迁移：只有验证集比例 → 训练集 = 1 - 验证集
        data["train_ratio"] = round(1.0 - float(data.get("val_ratio", 0.2)), 4)
    # 兼容旧版本项目文件：缺失的新字段走 dataclass 默认值
    return Project(**{k: v for k, v in data.items() if k in fields})


def create_project(name: str, dataset_dir: str, model_dir: str, task: str) -> Project:
    """创建项目；数据集目录必须存在，项目名不允许重复。"""
    name = validate_name(name)
    if not Path(dataset_dir).is_dir():
        raise ValueError(f"数据集路径不存在：{dataset_dir}")
    if name in list_projects():
        raise ValueError(f"项目「{name}」已存在")
    project = Project(
        name=name,
        dataset_dir=str(Path(dataset_dir).resolve()),
        model_dir=str(Path(model_dir).resolve() if model_dir else PROJECTS_DIR / name / "models"),
        task=TaskType(task).value,
        # 图像增强的 resize 尺寸默认开启，同时作为训练输入尺寸（归一化）
        augment_ops={"resize": {"width": 640, "height": 640}},
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    project.save()
    return project


def save_project(project: Project, old_name: str | None = None) -> None:
    """保存项目修改；old_name 与当前名不同时视为重命名（迁移目录）。"""
    name = validate_name(project.name)
    old_dir = PROJECTS_DIR / old_name if old_name and old_name != name else None
    if old_dir is None and name not in list_projects() and old_name is None:
        raise ValueError(f"项目「{name}」不存在")
    project.save()
    if old_dir is not None and old_dir.is_dir():
        shutil.move(str(old_dir), str(project.project_dir))


def delete_project(name: str) -> None:
    shutil.rmtree(PROJECTS_DIR / name, ignore_errors=True)


def list_pruned_models(project: Project) -> list[Path]:
    """列出项目文件夹下可用的剪枝预训练模型（.pt，按名称排序）。"""
    d = Path(project.model_dir) / PRUNED_DIR_NAME
    return sorted(d.glob("*.pt")) if d.is_dir() else []

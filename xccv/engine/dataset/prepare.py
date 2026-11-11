"""训练前的统一数据准备：扫描 -> 划分 -> 数据倍增。全程只在内存进行。

设计原则：训练数据（图片与标注）零落盘。
- 不导出任何图片副本、标注文件或数据清单到输出目录（包括增强副本）；
- 分类：训练时直接读取原文件、在线增强（classifier._FolderDataset）；
- YOLO：内存构建标签、直接读原图（yolo_data.InMemoryYoloDataset）；
- R-CNN：内存构建 COCO 元数据与训练目标（rcnn.RcnnDataset）；
- 增强副本不再预生成：数据倍增为重复引用原图，增强在每次取样时在线随机生成，
  同一份数据每个轮次看到不同的随机增强，等价于传统的"烘焙副本"且零磁盘占用。
"""

import hashlib
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from .base import Sample, TaskType
from .converters import split_samples
from .reader import scan_classification_dataset, scan_labelme_dataset

log = logging.getLogger(__name__)

# 模型家族 -> 支持的任务
FAMILY_TASKS = {
    "folder": {TaskType.CLASSIFICATION},
    "yolo": {TaskType.DETECTION, TaskType.SEGMENTATION},
    "coco": {TaskType.DETECTION, TaskType.SEGMENTATION},
}


def dataset_fingerprint(dataset_dir: Path) -> str:
    """数据集内容指纹：sha1(排序后的 相对路径+大小+mtime)。

    只读元数据不解码图片，万张图秒级完成；训练完成时写入 metrics.json，
    对比视图据此提示"该训练完成后数据集已修改，指标可能与当前数据不对应"。
    """
    root = Path(dataset_dir)
    items = []
    for p in root.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            st = p.stat()
            items.append((p.relative_to(root).as_posix(), st.st_size, int(st.st_mtime)))
    digest = hashlib.sha1()
    for rel, size, mtime in sorted(items):
        digest.update(f"{rel}\n{size}\n{mtime}\n".encode("utf-8", "surrogateescape"))
    return digest.hexdigest()


@dataclass
class PreparedData:
    """一次数据准备结果：各模型按家族取用所需信息（全部为内存数据）。"""

    task: TaskType
    classes: list[str]
    train_count: int
    val_count: int
    family: str = ""
    dataset_dir: Path | None = None
    train_samples: list[Sample] = field(default_factory=list)  # 含倍增重复引用
    val_samples: list[Sample] = field(default_factory=list)
    augment_ops: dict = field(default_factory=dict)
    augment_probs: dict = field(default_factory=dict)  # 每算子启用概率，缺失按 0.5

    @property
    def data_yaml(self) -> Path:
        """兼容属性：内存化训练后不再产生 data.yaml 文件。"""
        raise RuntimeError("YOLO 训练已改为内存数据集，不再使用 data.yaml 文件")

    @property
    def train_json(self) -> Path:
        raise RuntimeError("R-CNN 训练已改为内存 COCO 元数据，不再生成 json 文件")

    @property
    def val_json(self) -> Path:
        raise RuntimeError("R-CNN 训练已改为内存 COCO 元数据，不再生成 json 文件")

    @property
    def train_dir(self) -> Path:
        raise RuntimeError("训练数据已零落盘，不再导出图片目录")

    @property
    def val_dir(self) -> Path:
        raise RuntimeError("训练数据已零落盘，不再导出图片目录")


def prepare(
    task: TaskType,
    family: str,
    dataset_dir: Path,
    val_ratio: float,
    seed: int = 42,
    multiply_factor: int = 1,
    augment_ops: dict | None = None,
    train_ratio: float | None = None,
    augment_probs: dict | None = None,
) -> PreparedData:
    """从平台固定结构的数据集目录做内存数据准备（不写任何文件）。

    multiply_factor>1 时训练集在一个 epoch 内被学习 factor 遍（每轮重复学习
    次数）；增强按随机概率启用，与该值无关；val_ratio=0 时验证集复用训练集
    保证指标曲线可用；train_ratio 语义见 split_samples（None=1-val_ratio，
    与 val_ratio 之和 >1 时允许重叠）。
    """
    task = TaskType(task)
    if task not in FAMILY_TASKS[family]:
        raise ValueError(f"模型家族 {family} 不支持任务 {task.label}")
    augment_ops = dict(augment_ops or {})
    dataset_dir = Path(dataset_dir)

    if task is TaskType.CLASSIFICATION:
        samples, classes = scan_classification_dataset(dataset_dir)
        train, val = split_samples(samples, val_ratio, seed, stratify=True,
                                   train_ratio=train_ratio)
    else:
        samples, classes, _ = scan_labelme_dataset(dataset_dir)
        train, val = split_samples(samples, val_ratio, seed,
                                   train_ratio=train_ratio, stratify_labels=True)
    train = _multiply(train, multiply_factor)

    if not classes:
        raise ValueError(f"数据集 {dataset_dir} 中未发现任何类别标注，请先标注或检查目录结构")

    def _counts(samples: list[Sample]) -> str:
        from collections import Counter

        counter = Counter(sh.label for s in samples for sh in s.shapes)
        return "、".join(f"{c}={counter.get(c, 0)}" for c in classes)

    log.info(
        "数据准备完成（零落盘）: %d 类, 训练 %d / 验证 %d, 倍增 x%d, 增强策略=%s",
        len(classes), len(train), len(val), multiply_factor,
        "、".join(augment_ops) or "未启用")
    log.info("类别实例分布 — 训练集: %s | 验证集: %s（不均衡时建议开启工业优化）",
             _counts(train), _counts(val))
    return PreparedData(task=task, classes=classes, train_count=len(train),
                        val_count=len(val), family=family, dataset_dir=dataset_dir,
                        train_samples=train, val_samples=val, augment_ops=augment_ops,
                        augment_probs=dict(augment_probs or {}))


def _multiply(samples: list[Sample], factor: int) -> list[Sample]:
    """每轮重复学习次数：训练集在一个 epoch 内被学习 factor 遍。

    实现为重复引用原图（不生成任何文件）；增强按随机概率启用（见
    transforms.maybe_apply_ops），与该值无关——factor=1 时启用增强策略
    也会随机增强。
    """
    if factor <= 1:
        return list(samples)
    log.info("每轮重复学习 x%d：训练集在 epoch 内重复 %d 遍（增强随机启用，零副本）",
             factor, factor)
    return list(samples) * factor

"""数据集统计：类别分布与图像尺寸分布，供训练平台可视化。"""

from dataclasses import dataclass, field
from pathlib import Path

from ..engine.dataset.base import TaskType
from ..engine.dataset.ignore import IGNORE_LABEL
from ..engine.dataset.reader import scan_classification_dataset, scan_labelme_dataset


@dataclass
class DatasetStats:
    classes: list[str] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    n_images: int = 0
    n_labeled: int = 0          # 分类任务等于 n_images
    n_ok: int = 0               # O6（V0.7）：OK 负样本图数（检测/分割）
    n_ignore: int = 0           # O8（V0.7）：`__ignore__` 忽略区域数
    widths: list[int] = field(default_factory=list)
    heights: list[int] = field(default_factory=list)


def collect_stats(task: TaskType | str, dataset_dir: Path) -> DatasetStats:
    """统计一个数据集的类别分布与尺寸分布。"""
    task = TaskType(task)
    dataset_dir = Path(dataset_dir)
    stats = DatasetStats()

    if task is TaskType.CLASSIFICATION:
        samples, classes = scan_classification_dataset(dataset_dir)
        stats.classes = classes
        stats.class_counts = {c: 0 for c in classes}
        for s in samples:
            stats.class_counts[s.label] += 1
            stats.widths.append(s.width)
            stats.heights.append(s.height)
        stats.n_images = len(samples)
        stats.n_labeled = len(samples)
        return stats

    samples, classes, from_file = scan_labelme_dataset(dataset_dir)
    stats.classes = classes
    stats.class_counts = {c: 0 for c in classes}
    for s in samples:
        stats.widths.append(s.width)
        stats.heights.append(s.height)
        for shape in s.shapes:
            if shape.label == IGNORE_LABEL:   # O8（V0.7）：忽略区域单独计数
                stats.n_ignore += 1
                continue
            if shape.label in stats.class_counts:
                stats.class_counts[shape.label] += 1
            else:
                stats.class_counts[shape.label] = 1
    stats.n_images = len(samples)
    stats.n_labeled = len(samples)
    # O6（V0.7）：OK 负样本数量与训练侧同源口径（NEGATIVE_DIR_NAMES）
    from ..engine.dataset.reader import scan_negative_images

    stats.n_ok = len(scan_negative_images(dataset_dir))
    return stats

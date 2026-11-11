"""数据集统计：类别分布与图像尺寸分布，供训练平台可视化。"""

from dataclasses import dataclass, field
from pathlib import Path

from ..engine.dataset.base import TaskType
from ..engine.dataset.reader import scan_classification_dataset, scan_labelme_dataset


@dataclass
class DatasetStats:
    classes: list[str] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    n_images: int = 0
    n_labeled: int = 0          # 分类任务等于 n_images
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
            if shape.label in stats.class_counts:
                stats.class_counts[shape.label] += 1
            else:
                stats.class_counts[shape.label] = 1
    stats.n_images = len(samples)
    stats.n_labeled = len(samples)
    return stats

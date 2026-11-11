"""把磁盘上的数据集扫描为内部标准格式。

- 检测/分割任务（labelme 结构）：图片与同名 .json 同目录，可选 classes.txt 声明类别。
- 分类任务（文件夹结构）：数据集目录下的子目录名即类别名。
"""

import json
from pathlib import Path

from ..io_utils import read_json
from .base import IMAGE_EXTS, Sample, Shape, image_size

# labelme 中参与本平台任务的形状，其余（line/point 等）忽略
SUPPORTED_SHAPE_TYPES = {"rectangle", "polygon"}


def read_labelme_json(json_path: Path) -> Sample:
    """读取单个 labelme json。"""
    data = read_json(json_path)
    shapes = [
        Shape(
            label=str(s["label"]),
            shape_type=str(s["shape_type"]),
            points=[[float(x), float(y)] for x, y in s["points"]],
        )
        for s in data.get("shapes", [])
        if s.get("shape_type") in SUPPORTED_SHAPE_TYPES and s.get("label")
    ]
    image_path = _resolve_image(json_path, str(data.get("imagePath", "")))
    width = int(data.get("imageWidth") or 0)
    height = int(data.get("imageHeight") or 0)
    if width <= 0 or height <= 0:
        width, height = image_size(image_path)
    return Sample(image_path=image_path, width=width, height=height, shapes=shapes)


def _resolve_image(json_path: Path, image_path_field: str) -> Path:
    """labelme 的 imagePath 通常是同目录文件名；缺失时按同名图片猜测。"""
    if image_path_field:
        p = json_path.parent / image_path_field
        if p.exists():
            return p
    for ext in (".jpg", ".png", ".jpeg", ".bmp", ".webp"):
        p = json_path.with_suffix(ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到 {json_path.name} 对应的图片文件")


def scan_labelme_dataset(dataset_dir: Path) -> tuple[list[Sample], list[str], bool]:
    """扫描检测/分割数据集（递归含子文件夹），返回 (样本列表, 类别列表, classes.txt是否已存在)。

    类别优先读数据集根目录的 classes.txt（每行一个类名）；文件不存在时由全部标注汇总。
    """
    dataset_dir = Path(dataset_dir)
    samples: list[Sample] = []
    for img in sorted(dataset_dir.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = img.relative_to(dataset_dir)
        if any(part.startswith(".") for part in rel.parts[:-1]):
            continue  # 跳过隐藏目录
        json_path = img.with_suffix(".json")
        if json_path.exists():
            sample = read_labelme_json(json_path)
            sample.image_path = img  # 以实际图片文件为准
            samples.append(sample)

    classes_file = dataset_dir / "classes.txt"
    if classes_file.exists():
        classes = [
            line.strip()
            for line in classes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        from_file = True
    else:
        classes = sorted({s.label for sample in samples for s in sample.shapes})
        from_file = False
    return samples, classes, from_file


def scan_classification_dataset(dataset_dir: Path) -> tuple[list[Sample], list[str]]:
    """扫描分类数据集，返回 (样本列表, 类别列表)。"""
    dataset_dir = Path(dataset_dir)
    classes = sorted(d.name for d in dataset_dir.iterdir() if d.is_dir())
    samples: list[Sample] = []
    for cls in classes:
        for img in sorted((dataset_dir / cls).iterdir()):
            if img.suffix.lower() in IMAGE_EXTS:
                w, h = image_size(img)
                samples.append(Sample(image_path=img, width=w, height=h, label=cls))
    return samples, classes

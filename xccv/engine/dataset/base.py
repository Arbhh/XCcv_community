"""数据集内部标准格式：平台各模块统一使用，训练导出时再转换为各模型所需格式。"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image

# 平台支持的图片格式
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_TASK_LABELS = {
    "classification": "图像分类",
    "detection": "目标检测",
    "segmentation": "实例分割",
}


class TaskType(str, Enum):
    """平台支持的三种任务类型。"""

    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"

    @property
    def label(self) -> str:
        return _TASK_LABELS[self.value]


@dataclass
class Shape:
    """单个标注对象。rectangle 存两个角点，polygon 存轮廓顶点，单位均为原图像素。"""

    label: str
    shape_type: str  # "rectangle" 或 "polygon"
    points: list[list[float]]

    def bbox(self) -> tuple[float, float, float, float]:
        """外接框 (x1, y1, x2, y2)。"""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class Sample:
    """一张图及其标注。

    image 不为 None 时表示内存中的增强副本（导出时直接写出，不再读原图）。
    label 仅分类任务使用；shapes 仅检测/分割任务使用。
    """

    image_path: Path
    width: int
    height: int
    shapes: list[Shape] = field(default_factory=list)
    label: str = ""
    image: np.ndarray | None = None

    @property
    def stem(self) -> str:
        return self.image_path.stem


def image_size(path: Path) -> tuple[int, int]:
    """读取图片实际尺寸 (宽, 高)，只解析文件头，开销极小。"""
    with Image.open(path) as im:
        w, h = im.size
    return w, h

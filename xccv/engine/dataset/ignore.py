"""O8（V0.7）：`__ignore__` 忽略区域（近似语义，纯逻辑、无第三方依赖）。

VIMO O8：界限模糊区域（反光、丝纹等）用特殊标签 `__ignore__` 标注
（labelme 画 rectangle/polygon 即可），训练时该区域不计损失。严格语义
（区域级损失屏蔽）需要改两家框架的损失函数，成本高；本模块实现约定的
**近似语义**：

1. `__ignore__` 形状不参与类别表（reader 聚合时排除），永不进入训练标签；
2. 与任一 ignore 区域重叠率 > OVERLAP_DROP_RATIO（按 GT 实例外接框面积计）
   的 GT 标注在数据准备层剔除——两头都不像（既不该当目标、也不该当背景）
   的实例直接不学；
3. 评估侧同口径：GT 匹配统计（batch_match）剔除 ignore 区域内 GT 与检出，
   不把模糊区误报计入误检；
4. 已知近似（文档声明）：YOLO 的无目标网格仍会把 ignore 区域当背景学习
   （严格屏蔽需损失手术）；多边形区域按外接框近似；区域内"少量负样本"
   与"零负样本"的差异不控制。

关键决策：过滤放在 prepare.prepare（三族模型 YOLO/R-CNN/OCR 的共同数据
咽喉）单点实现，而不是按 VIMO 原文分别写"YOLO 组装层 + R-CNN 采样层"
两份——内存数据管线里两层拿到的都是同一份 prepare 产物，单点过滤天然
覆盖且口径唯一；build_labels_from_shapes 另有防御性跳过（直接调用者兜底）。
"""

from __future__ import annotations

import numpy as np

from .base import Shape

IGNORE_LABEL = "__ignore__"
# GT 实例与 ignore 区域的重叠率（占 GT 外接框面积）超过该值即剔除
OVERLAP_DROP_RATIO = 0.5


def is_ignore_shape(shape: Shape) -> bool:
    """该形状是否为忽略区域标记。"""
    return str(shape.label).strip() == IGNORE_LABEL


def _inter_ratio(a: tuple, b: tuple) -> float:
    """两 xyxy 框交面积 / a 框面积（a 为"被考察对象"，如 GT/检出）。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    if area_a <= 0:
        return 0.0
    return (iw * ih) / area_a


def ignore_regions(shapes: list) -> list:
    """形状列表中的全部 ignore 区域（外接框 xyxy）。"""
    return [sh.bbox() for sh in shapes if is_ignore_shape(sh)]


def filter_ignore_shapes(shapes: list) -> tuple[list, int, int]:
    """按近似语义过滤标注形状。

    返回 (保留的 GT 形状, ignore 区域数, 剔除的 GT 数)：
    - `__ignore__` 形状本身从结果移除（不进类别表/训练标签）；
    - 与任一区域重叠率 > 0.5 的 GT 实例剔除；多边形区域按外接框近似。
    """
    regions = ignore_regions(shapes)
    if not regions:
        return list(shapes), 0, 0
    kept, dropped = [], 0
    for sh in shapes:
        if is_ignore_shape(sh):
            continue
        bb = sh.bbox()
        if any(_inter_ratio(bb, r) > OVERLAP_DROP_RATIO for r in regions):
            dropped += 1
            continue
        kept.append(sh)
    return kept, len(regions), dropped


def suppress_boxes(boxes, labels, regions, ratio: float = OVERLAP_DROP_RATIO):
    """评估侧：剔除落在 ignore 区域内的框（GT 或检出通用）。

    boxes 为 xyxy 列表；regions 为 ignore 区域外接框列表。返回保留的
    (boxes, labels)（浅拷贝列表，原元素不动）。
    """
    if not regions or not boxes:
        return list(boxes), list(labels)
    arr = np.asarray(boxes, dtype=float).reshape(-1, 4)
    keep = []
    for i in range(arr.shape[0]):
        bb = tuple(arr[i])
        if not any(_inter_ratio(bb, r) > ratio for r in regions):
            keep.append(i)
    return [boxes[i] for i in keep], [labels[i] for i in keep]

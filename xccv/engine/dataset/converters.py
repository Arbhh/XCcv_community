"""数据集划分与标注转换的纯函数工具（无任何磁盘写入）。

训练数据零落盘后，本模块只保留：
- split_samples：训练/验证划分（分层抽样支持）；
- YOLO 归一化标注行构造（yolo_data 内存标签使用）；
- COCO 目标构造（rcnn 内存元数据与训练目标使用）。
"""

import random

from .base import Sample, Shape


def split_samples(samples: list[Sample], val_ratio: float, seed: int = 42,
                  stratify: bool = False, train_ratio: float | None = None,
                  stratify_labels: bool = False):
    """随机划分训练/验证集。

    stratify=True 时按 sample.label 分层抽样，保证小数据集下每个类别都进验证集。
    stratify_labels=True（检测/分割，图含多类别标注）按「图的类别组合」分层——
    稀有类别所在图按同比例进验证集，避免验证集缺类导致该类 AP 恒为空
    （2026-11-06 R-CNN 收敛实测：随机划分下缩水类验证实例=0）。
    val_ratio=0 时验证集直接复用训练集，保证指标曲线可用。
    train_ratio：训练集占比（0~1）；None 时 = 1-val_ratio（旧口径，划分结果
    与历史版本完全一致）。train_ratio + val_ratio > 1 时允许重叠——验证集
    先从全集抽出，训练集再从全集抽出，支持稀缺类别场景的
    "训练集 100% + 验证集 20%"（验证集为训练集子集）。
    """
    rng = random.Random(seed)
    if train_ratio is None:
        train_ratio = 1.0 - val_ratio
    train_ratio = min(max(float(train_ratio), 0.0), 1.0)
    # 重叠模式：训练+验证占比超过全集；否则互斥（验证只从训练之外的剩余里取，
    # 剩余内抽取比例按 val_ratio/(1-train_ratio) 折算，保证验证量仍是全集的 val_ratio）
    overlap = train_ratio + val_ratio > 1.0 + 1e-9

    def _split_pool(items: list[Sample]) -> tuple[list[Sample], list[Sample]]:
        tr, va = [], []
        if overlap:
            n_val = round(len(items) * val_ratio) if val_ratio > 0 else 0
            va = items[:n_val]
            n_train = round(len(items) * train_ratio)
            tr = items[:n_train]
        else:
            n_train = round(len(items) * train_ratio)
            tr = items[:n_train]
            rest = items[n_train:]
            frac = val_ratio / (1.0 - train_ratio) if train_ratio < 1.0 else 0.0
            n_val = round(len(rest) * frac) if val_ratio > 0 else 0
            va = rest[:n_val]
        return tr, va

    if stratify:
        groups: dict[str, list[Sample]] = {}
        for s in samples:
            groups.setdefault(s.label, []).append(s)
        train, val = [], []
        for cls in sorted(groups):
            items = groups[cls][:]
            rng.shuffle(items)
            tr, va = _split_pool(items)
            train += tr
            val += va
    elif stratify_labels:
        groups: dict[frozenset, list[Sample]] = {}
        for s in samples:
            key = frozenset(sh.label for sh in s.shapes if sh.label) \
                or frozenset({""})
            groups.setdefault(key, []).append(s)
        train, val = [], []
        for key in sorted(groups, key=lambda k: sorted(k)):
            items = groups[key][:]
            rng.shuffle(items)
            tr, va = _split_pool(items)
            train += tr
            val += va
        # 稀有类兜底：某类别在验证集完全缺席（组内图太少、20% 取整为 0）
        # 时，把训练集中含该类的一张图移入验证集——否则该类 AP 恒为空，
        # 收敛指标失去参照（2026-11-06 D1 实测：缩水 12 实例验证=0）。
        val_cls = {sh.label for s in val for sh in s.shapes if sh.label}
        all_cls = set()
        for k in groups:
            all_cls |= set(k)
        for cls in sorted(all_cls):
            if not cls or cls in val_cls:
                continue
            for s in list(train):
                if cls in {sh.label for sh in s.shapes}:
                    train.remove(s)
                    val.append(s)
                    break
    else:
        idx = list(range(len(samples)))
        rng.shuffle(idx)
        shuffled = [samples[i] for i in idx]
        train, val = _split_pool(shuffled)
    if not val:
        val = list(train)
    return train, val


def _class_index(classes: list[str], label: str) -> int:
    return classes.index(label) if label in classes else -1


def _rect_corners(shape: Shape) -> list[list[float]]:
    x1, y1, x2, y2 = shape.bbox()
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _clipped_bbox(shape: Shape, w: int, h: int) -> tuple[float, float, float, float] | None:
    """裁剪到图幅内的外接框，退化（宽或高<=1px）返回 None。"""
    x1, y1, x2, y2 = shape.bbox()
    x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
    y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    return x1, y1, x2, y2


def _clip_points(shape: Shape, w: int, h: int) -> list[list[float]]:
    return [[min(max(x, 0), w), min(max(y, 0), h)] for x, y in shape.points]


def _seg_points(shape: Shape, w: int, h: int) -> list[list[float]] | None:
    """分割任务的轮廓顶点：多边形取顶点，矩形按矩形掩码转四角点。"""
    pts = _clip_points(shape, w, h)
    if shape.shape_type == "rectangle":
        pts = _clip_points(Shape(shape.label, "polygon", _rect_corners(shape)), w, h)
    return pts if len(pts) >= 3 else None


def _yolo_det_line(shape: Shape, w: int, h: int, cls: int) -> str | None:
    bbox = _clipped_bbox(shape, w, h)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    return f"{cls} {(x1 + bw / 2) / w:.6f} {(y1 + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def _yolo_seg_line(shape: Shape, w: int, h: int, cls: int) -> str | None:
    """分割任务：多边形取轮廓顶点；矩形按矩形掩码处理（转为四角点）。"""
    pts = _seg_points(shape, w, h)
    if pts is None:
        return None
    coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in pts)
    return f"{cls} {coords}"

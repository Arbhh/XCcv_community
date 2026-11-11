# -*- coding: utf-8 -*-
"""切图训练纯逻辑（P9/T9-3，opt-in；R-CNN 家族）。

大图小目标在整图训练口径下目标占比过小、检测头难学到；切图训练把每张
原图按滑窗切成 tile 训练（与 T8-3 分块推理配对）。设计要点（与方案 §四.2.2
一致）：

- **索引现场裁剪、不落盘**：仅预生成 `(样本序号, x, y)` 索引，训练时按索引
  从内存原图裁剪；
- **train/val 按原图划分**：先按原图划分训练/验证集，再只对训练集切图
  （切图只发生在 RcnnDataset 内部，划分逻辑零改动）；
- **验证集始终整图评估**（val 不切图，COCO 口径不变）；
- **标签归属**：GT 框中心点落在窗口内才算该 tile 的目标；中心同时落在
  多个（重叠）窗口时只归第一个扫描到的 tile——**重叠区去重**，同一目标
  全数据集只出现一次；
- **可见面积过滤**：裁剪后被截断的目标可见面积比 < min_visibility 丢弃；
- **背景 tile 保留率** bg_tile_keep：无目标 tile 按 seeded 随机保留一部分
  （负样本防刷屏，OK 图同理按比例保留）；
- **mix_full_view**：按比例把原图整图也混入训练集（全图视野记忆，防
  「只见局部」的分布偏移）。
"""

import random
from dataclasses import dataclass
from itertools import product


def _starts(total: int, tile: int, overlap: float) -> list:
    """滑动窗起点序列（步长=tile×(1-overlap)，保证覆盖到末边）。"""
    if total <= tile:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap))))
    s = list(range(0, total - tile + 1, step))
    if s[-1] != total - tile:
        s.append(total - tile)
    return s


@dataclass
class TileEntry:
    """一条训练条目：sample_idx 对应原图；x/y 为 None 表示整图（mix_full_view）。

    box_ids=归属本窗口的框下标（规划期按"首个含中心的窗口"唯一归属——
    重叠区去重的执行点；x/y 为 None 的整图条目为全部框）。
    """

    sample_idx: int
    x: int | None = None
    y: int | None = None
    box_ids: tuple = ()


def _centers_in_window(boxes, w: int, h: int, x: int, y: int,
                       tile: int) -> list:
    """落在窗口内的框下标（中心点判据；boxes 为原图坐标 [x1,y1,x2,y2]）。"""
    out = []
    for bi, (x1, y1, x2, y2) in enumerate(boxes):
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if x <= cx < min(x + tile, w) and y <= cy < min(y + tile, h):
            out.append(bi)
    return out


def plan_tile_entries(samples, tile: int = 640, overlap: float = 0.2,
                      bg_tile_keep: float = 0.15, mix_full_view: float = 0.1,
                      min_visibility: float = 0.3,
                      seed: int = 42) -> list:
    """为训练样本预生成切图索引（含背景 tile 保留与整图混入）。

    samples 元素需有 width/height 与 shapes（.shapes 元素可取
    [x1,y1,x2,y2] 或含 .bbox 属性皆可，取不到的形状按无坐标跳过）。
    返回 list[TileEntry]；bg_tile_keep/mix_full_view 用 seeded 随机
    （同一数据集多次构建结果一致，便于复现与测试）。
    """
    rng = random.Random(seed)
    entries: list = []
    for si, s in enumerate(samples):
        w = int(getattr(s, "width", 0) or 0)
        h = int(getattr(s, "height", 0) or 0)
        boxes = _boxes_of(s)
        xs = _starts(w, tile, overlap) if w > 0 else [0]
        ys = _starts(h, tile, overlap) if h > 0 else [0]
        windows = list(product(ys, xs))
        # 唯一归属：每个框只记入首个含其中心的窗口（重叠区去重的执行点）
        owners: dict = {}
        for bi, (x1, y1, x2, y2) in enumerate(boxes):
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            for (y, x) in windows:
                if x <= cx < min(x + tile, w) and y <= cy < min(y + tile, h):
                    owners.setdefault((y, x), set()).add(bi)
                    break
        for (y, x) in windows:
            ids = tuple(sorted(owners.get((y, x), ())))
            if ids or rng.random() < bg_tile_keep:
                entries.append(TileEntry(si, x, y, ids))
        if w <= 0 or h <= 0 or not boxes:
            continue
        if rng.random() < mix_full_view:
            entries.append(TileEntry(si, None, None,
                                     tuple(range(len(boxes)))))  # 整图混入
    return entries


def _boxes_of(s) -> list:
    """样本形状 → 原图坐标框列表（无坐标信息的形状跳过）。"""
    out = []
    for sh in (getattr(s, "shapes", None) or []):
        b = getattr(sh, "bbox", None)
        if callable(b):  # 真实 shape.bbox() 是方法；测试桩可能是属性
            try:
                b = b()
            except Exception:
                b = None
        if b is None and hasattr(sh, "points") and sh.points:
            xs = [p[0] for p in sh.points]
            ys = [p[1] for p in sh.points]
            b = (min(xs), min(ys), max(xs), max(ys))
        if not b:
            continue
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        if x2 > x1 and y2 > y1:
            out.append([x1, y1, x2, y2])
    return out


def crop_target_boxes(boxes, x: int, y: int, tile: int, w: int, h: int,
                      min_visibility: float = 0.3,
                      only_ids=None) -> list:
    """原图框 → tile 内坐标框；可见面积比不足的丢弃。

    boxes 为原图坐标；返回 [(原下标, [tx1,ty1,tx2,ty2])]。only_ids 给定时
    （切图训练条目，来自 TileEntry.box_ids）只处理归属框——中心判据由规划
    期完成，这里只做裁剪/过滤；缺省时保留中心在窗口内的框（直接调用语义）。
    """
    out = []
    wx2, wy2 = min(x + tile, w), min(y + tile, h)
    for bi, (x1, y1, x2, y2) in enumerate(boxes):
        if only_ids is not None and bi not in only_ids:
            continue
        if only_ids is None:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if not (x <= cx < wx2 and y <= cy < wy2):
                continue  # 中心不在本窗口（直接调用时的归属判据）
        tx1, ty1 = max(x1, x), max(y1, y)
        tx2, ty2 = min(x2, wx2), min(y2, wy2)
        area = (x2 - x1) * (y2 - y1)
        visible = (tx2 - tx1) * (ty2 - ty1)
        if area <= 0 or visible / area < min_visibility:
            continue
        out.append((bi, [tx1 - x, ty1 - y, tx2 - x, ty2 - y]))
    return out

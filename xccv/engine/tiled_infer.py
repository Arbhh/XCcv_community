# -*- coding: utf-8 -*-
"""4K 大图分块推理（P8/T8-3）。

问题：检测模型把整张 4K 图缩到 640 输入时，小目标（电表读数、焊点、
细小异物）尺寸低于可检下限而漏检。分块推理让每块保持模型原生输入口径，
小目标不再被整图缩水吞掉（与 P9/T9-3「切图训练」配对使用效果最佳）。

- 切片：tile×tile（默认 640）滑窗，重叠 overlap（默认 15%）防边缘截断；
- 逐块 model.predict（与整图同 conf/iou 参数）；
- 坐标还原：块内 xyxy + 块偏移 → 原图坐标；
- 合并：**同类贪心 NMS**（merge_iou 默认 0.5）去重跨块重复检出——相邻块
  重叠区同一目标会被两块各检出一次；跨类别不互斥（不同类别目标可重叠）；
- 分类任务不分块（无空间意义），直接原样 predict；
- 纯 numpy/CPU 逻辑，功能级验收用桩模型（不依赖真实权重）。
"""

import numpy as np

from .models.base import Prediction


def _starts(total: int, tile: int, step: int) -> list:
    """滑动窗起点序列（保证覆盖到末边：最后一块贴边）。"""
    if total <= tile:
        return [0]
    s = list(range(0, total - tile + 1, step))
    if s[-1] != total - tile:
        s.append(total - tile)
    return s


def _nms_merge(boxes, labels, scores, iou_thr: float) -> list:
    """同类贪心 NMS：返回按分数降序保留的下标列表。"""
    if not boxes:
        return []
    arr = np.asarray(boxes, dtype=float)
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    order = scores.argsort()[::-1]
    keep: list = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        same = labels[rest] == labels[i]
        rest_same = rest[same]
        rest_other = rest[~same]
        if rest_same.size:
            xx1 = np.maximum(arr[i, 0], arr[rest_same, 0])
            yy1 = np.maximum(arr[i, 1], arr[rest_same, 1])
            xx2 = np.minimum(arr[i, 2], arr[rest_same, 2])
            yy2 = np.minimum(arr[i, 3], arr[rest_same, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            a1 = (arr[i, 2] - arr[i, 0]) * (arr[i, 3] - arr[i, 1])
            a2 = (arr[rest_same, 2] - arr[rest_same, 0]) * \
                 (arr[rest_same, 3] - arr[rest_same, 1])
            iou = inter / np.clip(a1 + a2 - inter, 1e-9, None)
            rest_same = rest_same[iou <= iou_thr]
        order = np.concatenate([rest_same, rest_other])
    return keep


def tiled_predict(model, image: np.ndarray, tile: int = 640,
                  overlap: float = 0.15, conf: float = 0.25,
                  iou: float = 0.45, merge_iou: float = 0.5) -> Prediction:
    """大图分块推理：整图不大于 tile 时等价于 model.predict。"""
    h, w = image.shape[:2]
    if h <= tile and w <= tile:
        return model.predict(image, conf=conf, iou=iou)
    step = max(1, int(round(tile * (1.0 - overlap))))
    ys = _starts(h, tile, step)
    xs = _starts(w, tile, step)
    boxes: list = []
    labels: list = []
    scores: list = []
    for y0 in ys:
        for x0 in xs:
            y1i = min(y0 + tile, h)
            x1i = min(x0 + tile, w)
            pred = model.predict(image[y0:y1i, x0:x1i], conf=conf, iou=iou)
            pboxes = getattr(pred, "boxes", None) or []  # 判空不用真值（P150）
            for i, b in enumerate(pboxes):
                bx1, by1, bx2, by2 = (float(v) for v in b)
                boxes.append([bx1 + x0, by1 + y0, bx2 + x0, by2 + y0])
                labels.append(int(pred.labels[i]))
                scores.append(float(pred.scores[i]))
    keep = _nms_merge(boxes, labels, scores, merge_iou)
    return Prediction(
        labels=[labels[i] for i in keep],
        scores=[round(scores[i], 4) for i in keep],
        boxes=[[round(v, 2) for v in boxes[i]] for i in keep],
    )

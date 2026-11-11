"""检测/分割指标：IoU 匹配混淆矩阵（mAP 由各模型自身验证引擎给出）。"""

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两组 xyxy 框的 IoU 矩阵 [len(a), len(b)]。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def detection_confusion_matrix(per_image_gt: list, per_image_pred: list, n_classes: int, iou_thr: float = 0.5) -> np.ndarray:
    """基于 IoU 贪心匹配的混淆矩阵，尺寸 [n_classes+1, n_classes+1]。

    最后一行/列代表“背景”：漏检进最后一列，误检进最后一行。
    per_image_* 每个元素为 {"boxes": [K,4] xyxy, "labels": [K], "scores": [K]}。
    """
    m = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)
    for gt, pred in zip(per_image_gt, per_image_pred):
        gt_boxes = np.asarray(gt["boxes"], dtype=float).reshape(-1, 4)
        gt_labels = np.asarray(gt["labels"], dtype=int)
        pred_boxes = np.asarray(pred["boxes"], dtype=float).reshape(-1, 4)
        pred_labels = np.asarray(pred["labels"], dtype=int)
        pred_scores = np.asarray(pred.get("scores", [1.0] * len(pred_boxes)), dtype=float)

        matched = np.zeros(len(gt_boxes), dtype=bool)
        ious = iou_matrix(pred_boxes, gt_boxes)
        for pi in np.argsort(-pred_scores):
            candidates = [
                gi for gi in range(len(gt_boxes))
                if not matched[gi] and gt_labels[gi] == pred_labels[pi] and ious[pi, gi] >= iou_thr
            ]
            if candidates:
                best = max(candidates, key=lambda gi: ious[pi, gi])
                matched[best] = True
                m[gt_labels[best], pred_labels[pi]] += 1
            else:
                m[n_classes, pred_labels[pi]] += 1  # 误检：背景 -> 预测类
        for gi in range(len(gt_boxes)):
            if not matched[gi]:
                m[gt_labels[gi], n_classes] += 1  # 漏检：真实类 -> 背景
    return m

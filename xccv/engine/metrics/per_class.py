"""O1（V0.7）：逐类精确率/召回率表的数据聚合（纯逻辑，UI 与测试共用）。

对照 VIMO"类型详情表"：每类 GT 实例数/检出数/TP/FP/FN/精确率/召回率/AP@50。
数据源全部现成——IoU 匹配混淆矩阵（metrics.detection.detection_confusion_matrix，
训练收尾随 metrics.json 落盘）+ 各家族评估器的逐类 AP@50（additive 键
per_class_ap50：R-CNN 从 COCOeval precision 数组取，YOLO 从 ultralytics
Metric.ap50 取）。本模块不做任何推理，只做聚合。
"""

from __future__ import annotations

import numpy as np


def per_class_rows(confusion_matrix, classes: list[str],
                   ap50_by_class=None) -> list[dict]:
    """混淆矩阵 [n+1, n+1] → 逐类指标行（UI 表格与回归测试共用）。

    矩阵约定（detection_confusion_matrix）：
    - 对角线 = TP（同类 IoU≥阈值 贪心匹配）；
    - 最后一列 = 漏检 FN（真实类 → 背景）；
    - 最后一行 = 误检 FP（背景 → 预测类）。

    ap50_by_class：逐类 AP@50（按类下标对齐的 list；None/缺项 → None，
    UI 显示"-"）。precision/recall 无法定义（分母 0）时同为 None。
    """
    n = len(classes)
    cm = np.asarray(confusion_matrix, dtype=float)
    if cm.shape != (n + 1, n + 1):
        raise ValueError(f"混淆矩阵尺寸 {cm.shape} 与类别数 {n} 不匹配")
    ap = list(ap50_by_class) if ap50_by_class else []
    rows = []
    for c, name in enumerate(classes):
        tp = float(cm[c, c])
        fn = float(cm[c, n])
        fp = float(cm[n, c])
        gt = tp + fn
        pred = tp + fp
        rows.append({
            "label": name,
            "gt": int(gt),
            "pred": int(pred),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": round(tp / pred, 4) if pred > 0 else None,
            "recall": round(tp / gt, 4) if gt > 0 else None,
            "ap50": (round(float(ap[c]), 4)
                     if c < len(ap) and ap[c] is not None else None),
        })
    return rows


def coco_per_class_ap50(precision: np.ndarray) -> list[float]:
    """COCOeval precision 数组 [T,R,K,A,M] → 逐类 AP@50（area=all, maxdet=末档）。

    哨兵 -1（该类无有效评估组合）按 pycocotools 口径剔除后平均；整类
    无值 → 0.0（与平台"展示口径统一归零"约定一致）。T 轴下标 2 = IoU 0.50。
    """
    arr = np.asarray(precision, dtype=float)
    if arr.ndim != 5 or arr.shape[0] < 3:
        raise ValueError(f"precision 数组形状异常: {arr.shape}")
    out = []
    for k in range(arr.shape[2]):
        v = arr[2, :, k, 0, -1]
        v = v[v > -1]
        out.append(round(float(v.mean()), 4) if v.size else 0.0)
    return out


def ultralytics_per_class_ap50(metric) -> list[float] | None:
    """ultralytics DetMetrics.box（Metric）→ 逐类 AP@50（缺类记 0.0）。

    只返回验证集中出现过的类有值：ap50 与 ap_class_index 等长对齐；其余
    类（该次验证未出现）按 0.0 补齐到最大类下标。8.4.7 的 Metric 恒有
    ap50/ap_class_index，但防御性读取（异常/为空 → None，调用方跳过）。
    """
    try:
        ap50 = getattr(metric, "ap50", None)
        idx = getattr(metric, "ap_class_index", None)
        if ap50 is None or idx is None or len(ap50) == 0:
            return None
        by_idx = {int(i): float(a) for i, a in zip(idx, ap50)}
        size = max(by_idx) + 1
        return [round(by_idx.get(i, 0.0), 4) for i in range(size)]
    except Exception:
        return None

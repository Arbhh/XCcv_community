"""O2/O19/O20（V0.7）：批量推理的 GT 匹配统计（纯逻辑，不依赖 Qt）。

批量目录里与图片同名的 labelme .json 视为真值标注：
- 有 GT 的图：检出按"同类 + IoU≥阈值"贪心匹配——命中=绿（TP）、未命中=红（FP）；
- GT 为空（OK/良品图）：全部检出记为 OK 误检（红·OK）；
- 无同名 json 的图：不参与匹配统计（无法判定）。

产出 match_stats（随 摘要.json 落盘，additive 键）：
    gt_total        GT 实例总数（参与匹配的图）
    n_gt_images     有 GT 的图片数
    n_ok_images     GT 为空的 OK 图数
    n_unmatched     未被任何检出命中的 GT 实例数（漏检候选）
    green           TP 检出分数列表（升序）
    red_defect      有 GT 图上的 FP 分数列表
    red_ok          OK 图上的 FP 分数列表（OK 误检）
    per_image       [{path, gt, missed, fp, fp_ok, matched_all}]（复判建议用）

阈值 t 下的指标（threshold_metrics）：precision=TP/(TP+FP)、
recall=TP/gt_total、ok_fp_per_img=OK 误检框/OK 图数。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..dataset.ignore import OVERLAP_DROP_RATIO, _inter_ratio
from ..dataset.reader import is_negative_dir_name
from .detection import iou_matrix

# O2 扩展（V0.7 用户追加）：阈值面板多维度（score=conf 单独走既有通道）
_DIM_KEYS = ("area", "short", "long", "fg_ratio", "gray_diff")


def _pts_bbox(points) -> tuple:
    """点集外接框 xyxy。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _json_regions(shapes_raw) -> list:
    """labelme 原始 shapes → `__ignore__` 区域外接框列表（多边形按外接框近似）。"""
    out = []
    for sh in shapes_raw or []:
        if str(sh.get("label", "")).strip() != "__ignore__":
            continue
        pts = sh.get("points") or []
        if pts:
            out.append(_pts_bbox(pts))
    return out


def read_gt_full(json_path: Path) -> tuple[list, list, list]:
    """labelme json → (GT 框, GT 标签, `__ignore__` 区域外接框)。

    GT 已按 O8 近似语义过滤：ignore 标记不是 GT、与区域重叠率>50% 的 GT 剔除。
    """
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception:
        return [], [], []
    raw = data.get("shapes") or []
    regions = _json_regions(raw)
    boxes: list = []
    labels: list = []
    for sh in raw:
        pts = sh.get("points") or []
        st = sh.get("shape_type", "polygon")
        if not pts:
            continue
        if str(sh.get("label", "")).strip() == "__ignore__":
            continue  # 忽略区域标记不是真值实例
        if st not in ("rectangle", "polygon"):
            continue  # point/line/circle 等：本期不参与框匹配
        bb = _pts_bbox(pts)
        x1, y1, x2, y2 = bb
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(str(sh.get("label", "")))
    if regions:
        keep = [i for i, b in enumerate(boxes)
                if not any(_inter_ratio(tuple(b), r) > OVERLAP_DROP_RATIO
                           for r in regions)]
        boxes = [boxes[i] for i in keep]
        labels = [labels[i] for i in keep]
    return boxes, labels, regions


def read_gt_shapes(json_path: Path) -> tuple[list, list]:
    """labelme json → ([xyxy 框], [标签])（read_gt_full 的 GT 部分）。"""
    boxes, labels, _ = read_gt_full(json_path)
    return boxes, labels


def match_detections(det_boxes, det_labels, det_scores,
                     gt_boxes, gt_labels, iou_thr: float = 0.5) -> list[bool]:
    """逐检出是否命中 GT（同类 IoU≥iou_thr，按分数降序贪心，GT 一对一）。

    与 detection_confusion_matrix 同款匹配语义；返回与 det_boxes 等长的
    matched 列表（True=TP，False=FP）。
    """
    n_det, n_gt = len(det_boxes), len(gt_boxes)
    matched_gt = [False] * n_gt
    out = [False] * n_det
    if n_det == 0 or n_gt == 0:
        return out
    boxes_det = np.asarray(det_boxes, dtype=float).reshape(-1, 4)
    boxes_gt = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    ious = iou_matrix(boxes_det, boxes_gt)
    order = np.argsort(-np.asarray(det_scores, dtype=float))
    for pi in order:
        best, best_iou = -1, iou_thr
        for gi in range(n_gt):
            if matched_gt[gi] or gt_labels[gi] != det_labels[pi]:
                continue
            if ious[pi, gi] >= best_iou:
                best, best_iou = gi, float(ious[pi, gi])
        if best >= 0:
            matched_gt[best] = True
            out[int(pi)] = True
    return out


def _det_dim_values(det: dict) -> dict:
    """单检出的多维度值（O2 扩展）：面积/短长边由 bbox 现算，
    fg_ratio/gray_diff 取批量线程预计算的 det["_dims"]（缺则 None）。"""
    box = det.get("bbox")
    if not box:
        return {}
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    values = {"area": w * h, "short": min(w, h), "long": max(w, h)}
    pre = det.get("_dims") or {}
    values["fg_ratio"] = pre.get("fg_ratio")
    values["gray_diff"] = pre.get("gray_diff")
    return values


def _det_record(det: dict, matched: bool) -> list:
    """逐检出紧凑记录（复盘页阈值复用的数据源，列序见 review_logic）：
    [score, matched, area, short, long, fg_ratio, gray_diff]。"""
    v = _det_dim_values(det)
    return [float(det.get("score", 0.0)), bool(matched),
            v.get("area"), v.get("short"), v.get("long"),
            v.get("fg_ratio"), v.get("gray_diff")]


def collect_match_stats(results: list, iou_thr: float = 0.5) -> dict:
    """批量结果（含 path/detections）→ match_stats dict。

    results 每项需含 "path" 与 "detections"（label/score/bbox，批量线程
    既有产出）。无任何同名 GT json 时返回 {"gt_available": False}。

    O2 扩展（V0.7 用户追加）：同时收集多维度值表 stats["dims"]
    （{dim: {green, red_defect, red_ok}}）——阈值面板各维度直方图数据源；
    fg_ratio/gray_diff 取批量线程预计算的 det["_dims"]（无图维度为空表）。
    """
    stats = {"gt_total": 0, "n_gt_images": 0, "n_ok_images": 0,
             "n_unmatched": 0, "green": [], "red_defect": [], "red_ok": [],
             "per_image": [],
             "dims": {k: {"green": [], "red_defect": [], "red_ok": []}
                      for k in _DIM_KEYS}}

    def _push(values: dict, bucket: str):
        for k in _DIM_KEYS:
            v = values.get(k)
            if v is not None:
                stats["dims"][k][bucket].append(float(v))

    seen_gt = False
    for r in results:
        path = Path(str(r.get("path") or r.get("file") or ""))
        dets = r.get("detections") or []
        json_path = path.with_suffix(".json")
        gt_boxes, gt_labels, regions = read_gt_full(json_path)
        det_boxes = [d.get("bbox") for d in dets]
        det_labels = [str(d.get("label", "")) for d in dets]
        det_scores = [float(d.get("score", 0.0)) for d in dets]
        # O8（V0.7）：落在 ignore 区域内的检出不计 FP（模糊区不算误检，
        # 与训练侧同口径）；GT 侧的过滤已在 read_gt_full 内完成
        kept_dets = list(dets)
        if regions and det_boxes:
            keep = [i for i, b in enumerate(det_boxes)
                    if not any(_inter_ratio(tuple(b), rg) > OVERLAP_DROP_RATIO
                               for rg in regions)]
            det_boxes = [det_boxes[i] for i in keep]
            det_labels = [det_labels[i] for i in keep]
            det_scores = [det_scores[i] for i in keep]
            kept_dets = [dets[i] for i in keep]
        if not gt_boxes and not gt_labels:
            # 再勘误③（2026-11-13）：OK 图判定与训练平台同源——
            # NEGATIVE_DIR_NAMES 目录（OK/良品/正常/负样本/背景…）内**无同名
            # json** 的图 = OK 负样本，全部检出记 OK 误检；其余无 json 图 =
            # 未评估（生产图），空 json 图 = 已审无缺陷样本（按 GT 空评估）
            ok_dir = any(is_negative_dir_name(p.name) for p in path.parents)
            if json_path.exists():
                seen_gt = True
                stats["n_ok_images"] += 1
                stats["red_ok"].extend(det_scores)
                for d in kept_dets:
                    _push(_det_dim_values(d), "red_ok")
                stats["per_image"].append({
                    "path": str(path), "gt": 0, "gt_available": True,
                    "ok_dir": ok_dir, "missed": 0,
                    "fp": len(det_scores), "fp_ok": len(det_scores),
                    "matched_all": not det_scores,
                    "dets": [_det_record(d, False) for d in kept_dets]})
            elif ok_dir:
                seen_gt = True
                stats["n_ok_images"] += 1
                stats["red_ok"].extend(det_scores)
                for d in kept_dets:
                    _push(_det_dim_values(d), "red_ok")
                stats["per_image"].append({
                    "path": str(path), "gt": 0, "gt_available": True,
                    "ok_dir": True, "missed": 0,
                    "fp": len(det_scores), "fp_ok": len(det_scores),
                    "matched_all": not det_scores,
                    "dets": [_det_record(d, False) for d in kept_dets]})
            else:
                stats["per_image"].append({
                    "path": str(path), "gt": None, "gt_available": False,
                    "ok_dir": False, "missed": None, "fp": None,
                    "fp_ok": None, "matched_all": None, "dets": []})
            continue
        seen_gt = True
        matched = match_detections(det_boxes, det_labels, det_scores,
                                   gt_boxes, gt_labels, iou_thr)
        n_fp = sum(1 for m in matched if not m)
        n_missed = len(gt_boxes) - sum(1 for m in matched if m)
        stats["n_gt_images"] += 1
        stats["gt_total"] += len(gt_boxes)
        stats["n_unmatched"] += n_missed
        stats["green"].extend(s for s, m in zip(det_scores, matched) if m)
        stats["red_defect"].extend(s for s, m in zip(det_scores, matched) if not m)
        for d, m in zip(kept_dets, matched):
            _push(_det_dim_values(d), "green" if m else "red_defect")
        stats["per_image"].append({
            "path": str(path), "gt": len(gt_boxes), "gt_available": True,
            "ok_dir": any(is_negative_dir_name(p.name) for p in path.parents),
            "missed": n_missed,
            "fp": n_fp, "fp_ok": 0, "matched_all": n_fp == 0 and n_missed == 0,
            "dets": [_det_record(d, m) for d, m in zip(kept_dets, matched)]})
    # 再勘误④（2026-11-13）：即使批量目录完全没有 GT/OK json（纯生产图），
    # 也返回完整 per_image/dims——复盘页的「未评估」组要展示每张生产图的
    # 检出供肉眼确认；gt_available 仅表示"是否出现过可评估的 GT"
    stats["gt_available"] = seen_gt
    return stats


def threshold_metrics_for(stats: dict, dim: str, thr: float) -> dict:
    """多维度指标现算（O2 扩展）：与 threshold_metrics 同口径，但取
    stats["dims"][dim] 的红绿表按维度阈值过滤（score 维走 threshold_metrics）。"""
    if dim == "score":
        return threshold_metrics(stats, thr)
    d = (stats.get("dims") or {}).get(dim) or {}
    green = [v for v in d.get("green", []) if v >= thr]
    fp_def = [v for v in d.get("red_defect", []) if v >= thr]
    fp_ok = [v for v in d.get("red_ok", []) if v >= thr]
    tp = len(green)
    fp_all = len(fp_def) + len(fp_ok)
    gt_total = int(stats.get("gt_total") or 0)
    n_ok = int(stats.get("n_ok_images") or 0)
    return {"tp": tp, "fp": fp_all,
            "precision": (round(tp / (tp + fp_all), 4) if tp + fp_all else None),
            "recall": (round(tp / gt_total, 4) if gt_total else None),
            "ok_fp_per_img": (round(len(fp_ok) / n_ok, 3) if n_ok else None),
            "ok_fp_images": None}


def threshold_metrics(stats: dict, conf: float) -> dict:
    """在阈值 conf 下从 match_stats 现算指标（拖线实时刷新用，零重推理）。"""
    green = [s for s in stats.get("green", []) if s >= conf]
    fp_def = [s for s in stats.get("red_defect", []) if s >= conf]
    fp_ok = [s for s in stats.get("red_ok", []) if s >= conf]
    gt_total = int(stats.get("gt_total") or 0)
    n_ok = int(stats.get("n_ok_images") or 0)
    tp = len(green)
    fp_all = len(fp_def) + len(fp_ok)
    out = {"tp": tp, "fp": fp_all,
           "precision": (round(tp / (tp + fp_all), 4) if tp + fp_all else None),
           "recall": (round(tp / gt_total, 4) if gt_total else None),
           "ok_fp_per_img": (round(len(fp_ok) / n_ok, 3) if n_ok else None),
           "ok_fp_images": None}
    if n_ok:
        # OK 误检率：按 fp_ok 归属图无法从分数列表还原——汇总口径用
        # per_image 在该阈值的现算（调用方需要时另取），此处给框/张口径
        out["ok_fp_images"] = None
    return out

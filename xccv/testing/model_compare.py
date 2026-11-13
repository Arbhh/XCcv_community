"""O21（V0.7 评审项补全）：模型 A/B 批量比对（纯逻辑，不依赖 Qt）。

对同一批图片，把「已完成的模型 A 批量结果」与「模型 B 复推结果」逐图
对比：两模型分别与同名 labelme GT 匹配（batch_match 同款贪心一对一、
ignore 区域同口径），按图给出判定：

- 有 GT 图：优势分 = 命中数 − 误检数；差值 >0 → A 优 / <0 → B 优 / =0 → 一致；
- OK 图（GT 空）：误检少者优，相等为一致；
- 无 GT：未评估（只列两模型检出数）。

产出 payload 随批量落盘 `检出结果/比对.json`（additive，不影响
result_schema 契约——比对是批量任务的伴生文件，与 复判.json 同位）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..engine.metrics.batch_match import read_gt_full, match_detections


def score_dets(dets: list, gt_boxes: list, gt_labels: list,
               iou_thr: float = 0.5) -> dict:
    """单图单模型计分：{"n": 检出数, "hit": 命中 GT 数, "fp": 误检数}。"""
    det_boxes = [d.get("bbox") for d in dets]
    det_labels = [str(d.get("label", "")) for d in dets]
    det_scores = [float(d.get("score", 0.0)) for d in dets]
    matched = match_detections(det_boxes, det_labels, det_scores,
                               gt_boxes, gt_labels, iou_thr)
    n_hit = sum(1 for m in matched if m)
    return {"n": len(dets), "hit": n_hit, "fp": len(dets) - n_hit}


def compare_image(a_dets: list, b_dets: list, gt_boxes: list, gt_labels: list,
                  iou_thr: float = 0.5, gt_available: bool = True) -> dict:
    """单图 A/B 对比记录（dict：gt / a / b / verdict）。verdict ∈
    {"一致", "A优", "B优", "未评估"}。gt_available=False（无同名 json）
    时记"未评估"，两模型仍给出检出数/误检数供参考。"""
    if gt_boxes is None:
        gt_boxes, gt_labels = [], []
    a = score_dets(a_dets, gt_boxes, gt_labels or [], iou_thr)
    b = score_dets(b_dets, gt_boxes, gt_labels or [], iou_thr)
    if not gt_available:
        verdict = "未评估"
    elif gt_boxes:
        adv = (a["hit"] - a["fp"]) - (b["hit"] - b["fp"])
        verdict = "A优" if adv > 0 else ("B优" if adv < 0 else "一致")
    else:
        # OK 图（GT 空）：误检少者优
        verdict = ("一致" if a["fp"] == b["fp"]
                   else ("A优" if a["fp"] < b["fp"] else "B优"))
    return {"gt": len(gt_boxes), "a": a, "b": b, "verdict": verdict,
            "gt_available": bool(gt_available)}


def compare_batch(results_a: list, predictions_b: dict, iou_thr: float = 0.5) -> list:
    """整批对比。results_a：模型 A 批量结果（含 path/detections）；
    predictions_b：{path: [det...]}——模型 B 逐图检出（label/score/bbox）。
    返回 records 列表 [{path, ...compare_image 字段}]。"""
    records = []
    for r in results_a:
        path = str(r.get("path") or r.get("file") or "")
        jp = Path(path).with_suffix(".json")
        gt_boxes, gt_labels, _regions = read_gt_full(jp)
        rec = compare_image(r.get("detections") or [],
                            predictions_b.get(path) or [],
                            gt_boxes, gt_labels, iou_thr,
                            gt_available=jp.is_file())
        rec["path"] = path
        rec["file"] = r.get("file") or Path(path).name
        records.append(rec)
    return records


def compare_summary(records: list, cap: int = 50) -> dict:
    """按判定聚合：计数 + 每类样例文件名（cap 截断）。"""
    counts = {"一致": 0, "A优": 0, "B优": 0, "未评估": 0}
    samples: dict = {k: [] for k in counts}
    for rec in records:
        v = rec.get("verdict") or "未评估"
        counts[v] = counts.get(v, 0) + 1
        if len(samples[v]) < cap:
            samples[v].append(rec.get("file") or Path(str(rec.get("path"))).name)
    return {"counts": counts, "samples": samples}


def build_compare_payload(model_a_name: str, model_b_name: str, conf: float,
                          iou: float, records: list) -> dict:
    """落盘 payload（比对.json；schema 1，additive 演进同 result_schema 纪律）。"""
    return {
        "schema": 1,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_a": model_a_name,
        "model_b": model_b_name,
        "conf": round(float(conf), 4),
        "iou": round(float(iou), 4),
        "summary": compare_summary(records),
        "records": records,
    }


def save_compare_json(path, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

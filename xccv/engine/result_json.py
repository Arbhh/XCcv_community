"""统一检出结果 JSON（schema_version=1）——用户回执 Q1/Q2/Q5 的对外契约。

单一出口 build_result_json()：测试平台批量、流程图平台页批量、后续
C# 接口/独立推理服务共用同一份 schema。原则：

- 全 ASCII snake_case 键、UTF-8 落盘、纯 JSON 基元类型（C# 可直接反序列化）；
- detections / classifications 数组恒存在（无数据为空数组），消费端零分支；
- 新任务类型（如 OCR）只新增 results 内数组字段，不改既有字段（向后兼容）；
- bbox=[x1, y1, x2, y2] 像素；面积单位像素/像素²；score 保留 4 位小数；
- 每张图片一个同名 .json（文件名 = 图片名），错误图片 error 字段说明原因。

字段说明同时维护在 docs/result_schema.md（对外契约文档）。
"""

import json
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = 1
_TOPK = 5  # 分类导出的 Top-K（与测试平台"识别结果"面板一致）


def _round(v: float, nd: int = 4) -> float:
    return round(float(v), nd)


def classification_topk(prediction, classes: list, k: int = _TOPK) -> list:
    """分类预测 → [{label, label_id, score}]（按得分降序 Top-K）。"""
    probs = list(getattr(prediction, "probs", None) or [])
    ranked = sorted(enumerate(probs), key=lambda kv: -kv[1])[:k]
    out = []
    for idx, prob in ranked:
        name = classes[idx] if classes and idx < len(classes) else str(idx)
        out.append({"label": name, "label_id": int(idx), "score": _round(prob)})
    return out


def detection_rows(prediction, classes: list) -> list:
    """Prediction → detections 数组（build_result_json 与流程出口/批量收集器
    共用的单一实现，保证各链路 schema 一致；含掩码面积）。"""
    classes = list(classes or [])
    masks = getattr(prediction, "masks", None)
    rows = []
    boxes = getattr(prediction, "boxes", None)
    boxes = [] if boxes is None else boxes   # 判空不用真值（ndarray 歧义，P150）
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (float(v) for v in box)
        label_id = int(prediction.labels[i]) \
            if i < len(getattr(prediction, "labels", [])) else -1
        label = classes[label_id] if 0 <= label_id < len(classes) else str(label_id)
        row = {"label": label, "label_id": label_id,
               "score": _round(prediction.scores[i])
               if i < len(getattr(prediction, "scores", [])) else 0.0,
               "bbox": [_round(v, 2) for v in (x1, y1, x2, y2)],
               "bbox_area": int(round(abs(x2 - x1) * abs(y2 - y1)))}
        if masks is not None and i < len(masks):
            row["mask_area"] = int(masks[i].sum())
        rows.append(row)
    return rows


def build_result_json(*, task, source: dict, model: dict | None = None,
                      inference: dict | None = None, prediction=None,
                      classes: list | None = None, flow_info: dict | None = None,
                      error: str | None = None,
                      verdict: dict | None = None) -> dict:
    """组装一份统一检出结果。

    - task：TaskType 或其 value（"detection"/"segmentation"/"classification"，
      新任务自由扩展，序列化为 str）；
    - source：{file, path, width, height}（width/height 可为 None）；
    - model：{file, name}（可缺省）；
    - inference：{conf, iou}（可选；flow 相关字段由 flow_info 合并生成）；
    - prediction：引擎 Prediction（可 None——纯错误条目）；
    - flow_info：apply_flow 的 info dict（removed / count_check）。
    """
    from .dataset.base import TaskType

    task_v = getattr(task, "value", task) or ""
    classes = list(classes or [])
    flow_info = dict(flow_info or {})
    inference = dict(inference or {})

    detections: list = []
    classifications: list = []
    if prediction is not None:
        if task_v == "classification":
            classifications = classification_topk(prediction, classes)
        else:
            detections = detection_rows(prediction, classes)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": str(task_v),
        "source": {
            "file": str(source.get("file", "")),
            "path": str(source.get("path", "")),
            "width": source.get("width"),
            "height": source.get("height"),
        },
        "model": {"file": str((model or {}).get("file", "")),
                  "name": str((model or {}).get("name", ""))},
        "inference": {
            "conf": inference.get("conf"),
            "iou": inference.get("iou"),
            "flow_applied": bool(flow_info or inference.get("flow_applied")),
            "flow_removed": int(flow_info.get("removed", 0) or 0),
            "count_check": flow_info.get("count_check"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "detections": detections,
        "classifications": classifications,
        # v0.5 契约增补：顶层判定（key 恒存在；flow_info 带数量判定时自动填充）
        "verdict": verdict or verdict_from_count_check(flow_info),
        "error": error,
    }
    return payload


def build_flow_result_json(*, source: dict, detections: list | None = None,
                           classifications: list | None = None,
                           geometries: list | None = None,
                           measurements: list | None = None,
                           flow_name: str = "", error: str | None = None,
                           verdict: dict | None = None) -> dict:
    """传统算法/流程图的统一结果 JSON（用户指令：各类型模型输出保持一致、
    剔除无效信息）。

    与 build_result_json 同一 schema_version 契约（全 ASCII snake_case、
    UTF-8、纯基元、C# 直读），任务无关字段与 AI 检出完全一致；传统算子
    特有的几何/测量结果放扩展数组（只增不改）：

    - detections     与 AI 检出同构 [{label, label_id, score, bbox, ...}]；
    - classifications 与分类任务同构；
    - geometries     [{kind, data, subpixel?}]（Geometry.to_json，剔除
      无效的 subpixel=false）；
    - measurements   [{value?/values?, unit?, passed?}]（None/NaN 剔除）；
    - model 恒为 {"file": "", "name": ""}（无模型，字段保留保证结构一致）；
    - 所有取值为 None / NaN 的字段直接剔除（不写键），不再输出占位噪声。
    """
    import math

    def _clean(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    def _strip(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            v = _clean(v)
            if v is None or v == "":
                continue
            if isinstance(v, dict):
                v = _strip(v)
            elif isinstance(v, (list, tuple)):
                v = [_clean(x) for x in v
                     if not (isinstance(x, float) and math.isnan(x))]
            out[k] = v
        return out

    geoms = [_strip(g.to_json() if hasattr(g, "to_json") else dict(g))
             for g in (geometries or [])]
    # 剔除无效条目：没有 data（几何）或没有任何实质数值（测量）的空壳不输出
    geoms = [g for g in geoms if g.get("data")]
    for g in geoms:
        if g.get("subpixel") is not True:
            g.pop("subpixel", None)      # 默认 False 属噪声
    meas = [_strip(m.to_json() if hasattr(m, "to_json") else dict(m))
            for m in (measurements or [])]
    meas = [m for m in meas if m.get("value") is not None or m.get("values")]
    for m in meas:
        if m.get("unit") == "px":
            m.pop("unit", None)          # 默认单位 px 属噪声（消费端默认 px）
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": "flow",
        "source": {
            "file": str(source.get("file", "")),
            "path": str(source.get("path", "")),
            "width": source.get("width"),
            "height": source.get("height"),
        },
        "model": {"file": "", "name": ""},
        "inference": {
            "flow_applied": True,
            "flow_name": str(flow_name or ""),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "detections": list(detections or []),
        "classifications": list(classifications or []),
        "geometries": geoms,
        "measurements": meas,
        # v0.5 契约增补：顶层判定（key 恒存在，无判定 = null；None 不被 _strip 剔除）
        "verdict": verdict,
        "error": error,
    }
    return payload


# ---- P8/T8-1 导出格式扩展（纯函数，批量线程与测试共用）----

LABELME_VERSION = "5.8.1"  # 与随包 labelme 版本一致


def to_labelme(payload: dict) -> dict:
    """统一检出 JSON → labelme 可加载格式（标注平台直接打开复核）。

    - detections → rectangle shapes（labelme 标准四点顺时针）；score 写入
      shape flags（labelme 画布不显示、导出保留，复核时可对照置信度）；
    - classifications → 图级 flags 的 top1；无检出/无类别时键恒存在；
    - imageData=None（不内嵌图像）；imagePath 优先用**绝对路径**——批量导出
      的 json 落在独立输出目录，相对路径会让 labelme 在 json 旁边找图失败
      （labelme 按 imagePath 相对 json 目录解析，绝对路径直接采用）。
    """
    src = payload.get("source", {})
    shapes = []
    for d in payload.get("detections", []):
        x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
        shapes.append({
            "label": str(d.get("label", "")),
            "points": [[float(x1), float(y1)], [float(x2), float(y1)],
                       [float(x2), float(y2)], [float(x1), float(y2)]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {"score": float(d.get("score", 0.0))},
        })
    top = payload.get("classifications") or []
    return {
        "version": LABELME_VERSION,
        "flags": {"top1": str(top[0]["label"])} if top else {},
        "shapes": shapes,
        "imagePath": str(src.get("path") or src.get("file", "")),
        "imageData": None,
        "imageHeight": int(src.get("height") or 0),
        "imageWidth": int(src.get("width") or 0),
    }


def to_coco(payloads: list, classes: list | None = None,
            name: str = "XCcv batch export") -> dict:
    """多张统一检出 JSON → 合并 COCO dict（第三方工具生态通用格式）。

    categories 完整性：传入 classes 时全量包含（含零检出类别，验收点）；
    未传时取 payloads 中出现过的 label 并集。category_id=classes 下标+1；
    bbox 转 COCO 的 [x, y, w, h]；score 作为扩展字段保留。
    """
    name_to_id: dict = {}
    if classes:
        for i, n in enumerate(classes):
            name_to_id[str(n)] = i + 1
    for p in payloads:
        for d in p.get("detections", []):
            n = str(d.get("label", ""))
            if n and n not in name_to_id:
                name_to_id[n] = len(name_to_id) + 1
    cats = [{"id": cid, "name": n} for n, cid
            in sorted(name_to_id.items(), key=lambda kv: kv[1])]
    images, anns = [], []
    ann_id = 1
    for img_id, p in enumerate(payloads, start=1):
        src = p.get("source", {})
        images.append({"id": img_id, "file_name": str(src.get("file", "")),
                       "width": int(src.get("width") or 0),
                       "height": int(src.get("height") or 0)})
        for d in p.get("detections", []):
            x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
            ww, hh = float(x2) - float(x1), float(y2) - float(y1)
            anns.append({
                "id": ann_id, "image_id": img_id,
                "category_id": name_to_id.get(str(d.get("label", "")), 1),
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(ww, 2), round(hh, 2)],
                "area": round(abs(ww * hh), 2),
                "iscrowd": 0,
                "score": float(d.get("score", 0.0)),
            })
            ann_id += 1
    return {"info": {"description": name, "schema": "XCcv export",
                     "date_created": datetime.now().strftime(
                         "%Y-%m-%d %H:%M:%S")},
            "licenses": [], "images": images, "annotations": anns,
            "categories": cats}


# ---- P8/T8-2 批量统计摘要 ----

def batch_summary(results: list, costs_ms: list | None = None) -> dict:
    """批量运行统计摘要：图片/检出总数、每类检出数与平均置信度、耗时 p50/max。

    costs_ms 与 results 顺序无关（按值统计）；costs 缺省时耗时字段为 None。
    """
    import statistics

    per_class: dict = {}
    total = 0
    for r in results:
        for d in r.get("detections", []):
            lab = str(d.get("label", "?"))
            e = per_class.setdefault(lab, {"count": 0, "score_sum": 0.0})
            e["count"] += 1
            e["score_sum"] += float(d.get("score", 0.0))
            total += 1
    for e in per_class.values():
        s = e.pop("score_sum")
        e["avg_score"] = round(s / e["count"], 4) if e["count"] else 0.0
    costs = sorted(float(v) for v in (costs_ms or []))
    latency = None
    if costs:
        latency = {"p50_ms": round(statistics.median(costs), 1),
                   "max_ms": round(costs[-1], 1)}
    return {"images": len(results), "detections": total,
            "per_class": per_class, "latency_ms": latency}


def save_result_json(path, payload: dict) -> None:
    """UTF-8 落盘（ensure_ascii=False，中文类别名直接可读）。"""
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                          encoding="utf-8")


# ---- 顶层判定 verdict（v0.5 契约增补：只增不改，key 恒存在，无判定 = null）----

def verdict_from_count_check(flow_info: dict | None) -> dict | None:
    """AI 检出链路的数量判定（apply_flow info）→ verdict 结构。

    返回 {"passed": bool, "source": "count_check", "detail": str}；无判定 = None。
    """
    cc = (flow_info or {}).get("count_check")
    if cc not in ("OK", "NG"):
        return None
    return {"passed": cc == "OK", "source": "count_check",
            "detail": f"数量判定 {cc}"}


def verdict_from_node_results(results: dict) -> dict | None:
    """流程执行器节点结果 → verdict（扫描全部 ok 节点的判定输出）。

    修复循环 3 起判定节点统一输出 result 端口（值=布尔）：
    - compare_value / bool_op / result_merge：outputs["result"] 为 bool；
    - filter_count_check：结果对象 extra["verdict"] 为 bool（检出本体继续
      向下传递）；旧图的 "passed" 端口仍兼容扫描。
    多判定并存时取 AND（任一 NG 即整体不通过）。
    """
    flags: list[tuple[str, bool]] = []
    judges = ("compare_value", "bool_op", "result_merge", "filter_count_check")
    for nid in sorted(results):
        r = results[nid]
        if not getattr(r, "ok", False):
            continue
        name = str(getattr(r, "name", ""))
        for key, v in (getattr(r, "outputs", None) or {}).items():
            if key == "passed" and isinstance(v, bool):
                flags.append((name, v))
            elif key == "result" and isinstance(v, bool) and name in judges:
                flags.append((name, v))
            else:
                ev = getattr(v, "extra", None)
                if isinstance(ev, dict) and isinstance(ev.get("verdict"), bool):
                    flags.append((name, ev["verdict"]))
    if not flags:
        return None
    passed = all(v for _n, v in flags)
    if passed:
        detail = "全部判定通过（" + "、".join(dict.fromkeys(n for n, _ in flags)) + "）"
    else:
        detail = "未通过：" + "、".join(dict.fromkeys(n for n, v in flags if not v))
    source = "count_check" if any(n == "filter_count_check" for n, _ in flags) \
        else "logic"
    return {"passed": passed, "source": source, "detail": detail}

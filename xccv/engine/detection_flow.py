"""检出过滤流程引擎：结果过滤与计数语义。

流程 = 有序过滤节点链，对模型检出的 Prediction 依次应用：
- confidence   置信度下限
- class_filter 仅保留指定类别
- bbox_size    bbox 宽/高范围（像素）
- bbox_area    bbox 面积范围
- mask_area    mask 面积范围（实例分割）
- region_block 屏蔽区域：区域内的目标被过滤（用户在图像上框选；坐标归一化）
- region_keep  保留区域：仅保留区域内的目标
- count_check  数量判定：整图检出数超出 [min,max] 时在 info 中给出判定

流程以 JSON 存于项目文件夹（检出流程.json），测试平台自动加载应用。
"""

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # torch 延迟导入（V0.6 T1-4）：models.base 模块级 import torch，
    from .models.base import Prediction  # 运行期仅 apply_flow 构造结果时局部导入

FLOW_VERSION = 1

_LOG = logging.getLogger("xccv.engine.flow.legacy")  # 传统过滤链日志（节点级定位）


class FlowNode:
    """节点定义：op 名称、中文标题、参数 schema（编辑器据此生成表单）。"""

    def __init__(self, op: str, title: str, params: dict, description: str = ""):
        self.op = op
        self.title = title
        self.params = params          # {参数名: 默认值}
        self.description = description


NODE_LIBRARY: list[FlowNode] = [
    FlowNode("confidence", "置信度过滤", {"min": 0.25},
             "过滤得分低于下限的检出"),
    FlowNode("class_filter", "类别过滤", {"keep": ""},
             "仅保留指定类别（逗号分隔；留空=全部保留）"),
    FlowNode("bbox_size", "bbox 宽高过滤",
             {"w_min": 0, "w_max": 0, "h_min": 0, "h_max": 0},
             "0 表示不限制；单位像素"),
    FlowNode("bbox_area", "bbox 面积过滤", {"min": 0, "max": 0},
             "外接框面积（宽×高，像素²）；0 表示不限制"),
    FlowNode("mask_area", "mask 面积过滤", {"min": 0, "max": 0},
             "实例分割掩码面积（像素²）；0 表示不限制"),
    FlowNode("region_block", "屏蔽区域",
             {"regions": []},
             "屏蔽区域内的目标被过滤（区域在预览图上框选，坐标已归一化）"),
    FlowNode("region_keep", "保留区域",
             {"regions": []},
             "仅保留区域内的目标；可多区域取并集"),
    FlowNode("count_check", "数量判定",
             {"min": 0, "max": 0},
             "过滤后整图数量判定：min=0 不限下限，max=0 不限上限；"
             "超限在结果中标记 NG"),
]

_NODE_BY_OP = {n.op: n for n in NODE_LIBRARY}
_INT_OPS = {"bbox_size", "bbox_area", "mask_area", "count_check"}


def new_flow() -> dict:
    return {"version": FLOW_VERSION, "nodes": []}


def normalize_flow(flow: dict) -> dict:
    """补全缺失字段并对齐节点默认值（旧文件兼容）。"""
    out = {"version": flow.get("version", FLOW_VERSION), "nodes": []}
    for node in flow.get("nodes", []):
        op = node.get("op")
        if op not in _NODE_BY_OP:
            continue
        lib = _NODE_BY_OP[op]
        params = {**lib.params, **(node.get("params") or {})}
        out["nodes"].append({"op": op, "params": params})
    return out


def _box_from_mask(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _mask_of(pred: "Prediction", i: int, w: int, h: int) -> np.ndarray | None:
    if pred.masks is None or i >= len(pred.masks):
        return None
    m = pred.masks[i]
    if m is None or m.shape[:2] != (h, w):
        return None
    return np.asarray(m)


def _parse_regions(value, w: int, h: int) -> list:
    """【旧接口，仅矩形】把 region 参数解析成归一化 [x1,y1,x2,y2] 列表。

    V0.6 T3-1 起运行期判定改用 ``flow.regions``（矩形/圆/多边形三形状，
    旧串逐位兼容）；本函数保留供外部旧调用方，带形状前缀的条目在此跳过。

    接受两种形态：
    - 列表 [[x1,y1,x2,y2], ...]（程序内生成，归一化坐标）；
    - 字符串 "x1,y1,x2,y2;x1,y1,x2,y2"（编辑器文本输入）。
    坐标 >1 视为像素坐标，按图像尺寸归一化。
    """
    out = []
    if isinstance(value, str):
        entries = [s for s in value.split(";") if s.strip()]
    elif isinstance(value, (list, tuple)):
        entries = list(value)
    else:
        return out
    for e in entries:
        try:
            if isinstance(e, str):
                nums = [float(v) for v in e.replace("，", ",").split(",") if v.strip()]
            elif isinstance(e, (list, tuple)):
                nums = [float(v) for v in e]
            else:
                continue
            if len(nums) != 4:
                continue
            x1, y1, x2, y2 = nums
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.0 + 1e-9:
                x1, x2 = x1 / w, x2 / w
                y1, y2 = y1 / h, y2 / h
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            out.append([x1, y1, x2, y2])
        except (TypeError, ValueError):
            continue  # 单条区域格式损坏只影响该条，不让整条流程崩溃
    return out


def _in_any_region(cx: float, cy: float, regions, w: int, h: int) -> bool:
    """检出中心点是否落在任一区域（矩形/圆/多边形，见 flow.regions；T3-1）。"""
    from .flow.regions import any_contains

    return any_contains(regions, cx, cy, w, h)


def apply_flow(pred: "Prediction", image_size: tuple[int, int], flow: dict,
               classes: list[str] | None = None) -> tuple["Prediction", dict]:
    """对检出结果应用过滤流程，返回 (过滤后 Prediction, 统计信息)。

    classes 为模型的类别名表（class_filter 节点按名称匹配时需要）。
    统计信息: {"steps": [{op, before, after}...], "count_check": "OK"/"NG"/None,
    "removed": 移除数, "removed_prediction": 被滤掉的检出 Prediction（V0.6 T3-2）}
    """
    w, h = int(image_size[0]), int(image_size[1])
    n = len(pred.boxes)
    keep = np.ones(n, dtype=bool)
    steps = []

    # 预计算掩码面积 / 外接框（掩码可能未提供）
    mask_areas = np.zeros(n)
    for i in range(n):
        m = _mask_of(pred, i, w, h)
        if m is not None:
            mask_areas[i] = float(m.sum())

    for node in flow.get("nodes", []):
        op, p = node["op"], node.get("params") or {}
        before = int(keep.sum())
        idx = np.where(keep)[0]
        try:
            if op == "confidence":
                for i in idx:
                    if float(pred.scores[i]) < float(p.get("min") or 0):
                        keep[i] = False
            elif op == "class_filter":
                names = [s.strip() for s in str(p.get("keep") or "").split(",") if s.strip()]
                if names:
                    for i in idx:
                        label = int(pred.labels[i])
                        cls = classes[label] if classes and label < len(classes) else str(label)
                        if cls not in names:
                            keep[i] = False
            elif op == "bbox_size":
                for i in idx:
                    x1, y1, x2, y2 = pred.boxes[i]
                    bw, bh = x2 - x1, y2 - y1
                    if (p.get("w_min") and bw < p["w_min"]) or (p.get("w_max") and bw > p["w_max"]) \
                            or (p.get("h_min") and bh < p["h_min"]) or (p.get("h_max") and bh > p["h_max"]):
                        keep[i] = False
            elif op == "bbox_area":
                lo, hi = float(p.get("min") or 0), float(p.get("max") or 0)
                for i in idx:
                    x1, y1, x2, y2 = pred.boxes[i]
                    area = (x2 - x1) * (y2 - y1)
                    if (lo and area < lo) or (hi and area > hi):
                        keep[i] = False
            elif op == "mask_area":
                lo, hi = float(p.get("min") or 0), float(p.get("max") or 0)
                for i in idx:
                    area = mask_areas[i]
                    if (lo and area < lo) or (hi and area > hi):
                        keep[i] = False
            elif op in ("region_block", "region_keep"):
                from .flow.regions import parse_regions

                regions = parse_regions(p.get("regions") or [])  # 每步解析一次（非逐检出）
                for i in idx:
                    x1, y1, x2, y2 = pred.boxes[i]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    inside = _in_any_region(cx, cy, regions, w, h)
                    if op == "region_block" and inside:
                        keep[i] = False
                    if op == "region_keep" and not inside:
                        keep[i] = False
        except Exception:
            # 日志捕捉：精确定位到具体过滤算子后原样抛出（行为不变）
            _LOG.error("过滤算子 %s 执行失败（参数 %s）", op, p, exc_info=True)
            raise
        _LOG.debug("过滤步骤 %s：%d → %d", op, before, int(keep.sum()))
        steps.append({"op": op, "before": before, "after": int(keep.sum())})

    labels, scores, boxes = [], [], []
    masks = None if pred.masks is None else []
    for i in np.where(keep)[0]:
        labels.append(int(pred.labels[i]))
        scores.append(float(pred.scores[i]))
        boxes.append([float(v) for v in pred.boxes[i]])
        if pred.masks is not None:
            masks.append(pred.masks[i])

    # 数量判定（基于过滤后数量）
    verdict = None
    for node in flow.get("nodes", []):
        if node["op"] == "count_check":
            p = node.get("params") or {}
            cnt = len(labels)
            lo, hi = int(p.get("min") or 0), int(p.get("max") or 0)
            ok = (cnt >= lo if lo else True) and (cnt <= hi if hi else True)
            verdict = "OK" if ok else "NG"

    from .models.base import Prediction  # 局部导入：见模块头 TYPE_CHECKING 注释

    # V0.6 T3-2：被滤掉的检出（removed 端口；additive，原 info 键与返回值不变）
    removed_idx = np.where(~keep)[0]
    r_masks = None
    if pred.masks is not None:
        r_masks = [pred.masks[i] if i < len(pred.masks) else None for i in removed_idx]
    removed_pred = Prediction(
        labels=[int(pred.labels[i]) for i in removed_idx],
        scores=[float(pred.scores[i]) for i in removed_idx],
        boxes=[[float(v) for v in pred.boxes[i]] for i in removed_idx],
        masks=r_masks)
    info = {"steps": steps, "count_check": verdict,
            "removed": int(n - len(labels)), "removed_prediction": removed_pred}
    _LOG.info("检出过滤链完成：%d → %d（移除 %d，count_check=%s）",
              n, len(labels), info["removed"], verdict)
    return Prediction(labels=labels, scores=scores, boxes=boxes, masks=masks), info


def flow_class_mismatch(flow: dict, classes: list[str]) -> list[str]:
    """流程中类别过滤引用了、但不在模型类别表里的类别名（匹配提示用）。

    返回排序后的缺失类别名列表；流程无 class_filter 节点或模型类别为空
    时返回空列表（空类别表视为无法判断，不误报）。
    """
    names: set = set()
    for node in flow.get("nodes", []):
        if node.get("op") == "class_filter":
            names.update(s.strip() for s in
                         str((node.get("params") or {}).get("keep") or "").split(",")
                         if s.strip())
    if not names or not classes:
        return []
    known = set(classes)
    return sorted(n for n in names if n not in known)


def region_param_problems(value, context: str = "") -> list[str]:
    """区域参数合法性检查（.xvap 打包前校验用）：返回问题描述列表。

    与运行期的宽容解析不同，本函数用于打包把关：每条区域必须形状合法
    （矩形 4 数字且宽高>0 / 圆半径>0 / 多边形≥3 顶点且面积>0），否则该
    过滤规则在产线上不会按预期生效。V0.6 T3-1 起委托 ``flow.regions``
    （矩形条目文案与旧版逐字一致）。
    """
    from .flow.regions import region_problems

    return region_problems(value, context)


def load_flow(path) -> dict | None:
    """读取项目流程文件；不存在或为空流程返回 None。

    兼容三种落盘格式（统一读取入口，嗅探见 flow.legacy_adapter.sniff_format）：
    - 链式（save_flow 的输出，节点带 op/params）；
    - 流程图（流程图编辑器保存的 JSON/.xvap 内嵌图，节点带 type/properties
      与 links，旧画布 model/post 类型）——先经 graph_to_chain 转成执行链
      再规范化。此前测试平台直接按链式解析图格式文件，所有节点因缺 op
      被静默丢弃，后处理整体失效；
    - 新 IR 图格式（节点 type 为 flow 注册表算子名）——**不是过滤链**，
      由新引擎 flow.Executor 运行，此处返回 None（测试平台视为无流程降级，
      绝不误跑）。
    """
    from pathlib import Path

    from .flow.legacy_adapter import sniff_format

    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    fmt = sniff_format(data)
    if fmt == "xcv_graph":
        _LOG.warning("流程文件为新 IR 图格式（%s），不走过滤链引擎", p.name)
        return None
    nodes = data.get("nodes") or []
    if nodes and isinstance(nodes[0], dict) \
            and "op" not in nodes[0] and "type" in nodes[0]:
        data = graph_to_chain(data)
    flow = normalize_flow(data)
    return flow or None


def save_flow(path, flow: dict) -> None:
    from pathlib import Path

    Path(path).write_text(
        json.dumps(normalize_flow(flow), ensure_ascii=False, indent=1),
        encoding="utf-8")


def graph_to_chain(flow_graph: dict) -> dict:
    """流程图（LiteGraph 序列化格式）→ 检测流程链 {"nodes":[{op,params}...]}。

    沿连线从模型节点拓扑遍历收集后处理节点；无模型节点时按声明顺序
    全量收集。节点参数兼容 properties（LiteGraph）与 params（旧版）。
    """
    nodes = {n["id"]: n for n in flow_graph.get("nodes", [])}
    outgoing: dict[int, list[int]] = {}
    for lk in flow_graph.get("links", []):
        # 兼容两种链接格式：旧 5 元 [id, src, 0, dst, 0] 与
        # LiteGraph serialize 的 6 元 [id, src, slot, dst, slot, type]
        src, dst = lk[1], lk[3]
        outgoing.setdefault(src, []).append(dst)
    # 模型起点：旧画布模型节点（v0.4.2 起新流程用 xccv2/ 模型节点走新引擎，
    # 旧链仅服务旧流程文件，不再为新节点做旧链兼容——用户批注 Q5）
    start = next((nid for nid, n in nodes.items()
                  if str(n["type"]).startswith(("model/", "xccv/model"))), None)
    chain: list[dict] = []
    visited = set()
    queue = [start] if start is not None else [n["id"] for n in flow_graph.get("nodes", [])]
    while queue:
        nid = queue.pop(0)
        if nid in visited or nid not in nodes:
            continue
        visited.add(nid)
        n = nodes[nid]
        t = str(n["type"])
        params = n.get("properties") or n.get("params") or {}
        if t.startswith("post/"):
            chain.append({"op": t.split("/", 1)[1], "params": dict(params)})
        elif t.startswith("xccv/post_"):
            chain.append({"op": t.split("xccv/post_", 1)[1], "params": dict(params)})
        queue.extend(outgoing.get(nid, []))
    return {"nodes": chain}

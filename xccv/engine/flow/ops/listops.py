"""多图批处理算子族（V0.6 F2-2 建立；修复循环 3 统一端口；修复循环 4 取消
image_list 端口类型）。

修复循环 4（用户批注：图像/图像列表之分"极其不方便"，应统一为图像端口并按
并行输入设计）：图像端口可承载单张或多张（多张=ImageList 集合，逐张带原图
坐标）；除本模块三个原生集合节点（HANDLES_LIST=True）外，**任意节点**收到
多张时由执行器逐张自动映射（executor._map_over）——「按检出全部裁图」的
输出可直接接任意节点（预处理/定位/AI 检出…），无需再经「逐图推理/逐图执行
算子」中转。原「取列表第k张」节点已按用户批注删除（并行输入下不再需要）。

本模块节点（端口约定；合并=成对（图像, 结果）输入，连上后自动出现下一对）：
- crop_by_detection_all「按检出全部裁图」：image + result → image（多张）；
- grid_tile_all「网格全部分块」：image（结果可选）→ image（多张）；
- model_on_list「逐图推理」：image（多张）→ result（值=逐图结果列表）；
- op_on_list「逐图执行算子」：image（多张）→ result（值=逐图结果列表）；
- merge_results「结果合并」：(image, result) ×N 对 → image + result
  （结果统一换算回原图坐标；N 最大 8，画布连上一对自动出现下一对）；
- mask_to_box「掩码转外接框」：image（可选）+ result → image + result。

统一度量衡（ResultItem，ir.result_kind）：每条结果 = 种类 + 原值（该图自身像素坐标）
+ roi_rect（该图在原图中的位置）。合并时一律换算到**原图像素坐标**：
- 检出：框平移、掩码回贴原图尺寸；分类：以 roi_rect 为框、最高类为标签；
- 几何：点/线/圆/旋转矩形/多边形平移到原图，转成"外接框 + 可选掩码"的检出
  （标签=参数「几何结果类别」），精确几何另存 extra["merged_items"]；
- 测量 / 数值 / 布尔：不成框，逐条记入 extra["merged_items"]（带 roi_rect）。
合并输出是平台统一的检出结果（保存结果/过滤/表格节点原样可用）。
"""

import json

import numpy as np

from ..ir import Geometry, ImageData, ImageList, ResultItem, ResultList, result_kind
from ..node_base import ParamSpec, Port, VisionNode
from ..registry import get_node_class, register_node
from .crop import (
    _MARGIN_PX, _save_dir, _seq, _source_stem, _write, expand_box,
    grid_rects,
)

# 结果合并的成对输入上限（图像, 结果 × 8；画布按连线自动增减可见对数）
_MERGE_PAIRS = 8


def _merge_inputs() -> tuple:
    ports = []
    for k in range(1, _MERGE_PAIRS + 1):
        suf = "" if k == 1 else str(k)
        ports.append(Port(f"image{suf}", "image", required=False,
                          label=f"图像{suf}"))
        ports.append(Port(f"result{suf}", "result", required=False,
                          label=f"结果{suf}"))
    return tuple(ports)


def _item(img: ImageData, rect: list, extra: dict) -> ImageData:
    x1, y1, x2, y2 = rect
    meta = dict(img.meta)
    meta.update({"roi_rect": [int(x1), int(y1), int(x2), int(y2)],
                 "source_size": [int(img.mat.shape[1]), int(img.mat.shape[0])]})
    meta.update(extra)
    return ImageData(mat=img.mat[y1:y2, x1:x2].copy(), meta=meta)


@register_node
class CropByDetectionAll(VisionNode):
    """按检出全部裁图：每个检出框一张裁图（外扩百分比/像素、补方、最小边同
    「按检出裁图」），输出图像列表（逐张带原图矩形 roi_rect、类别序号、得分）。

    「检出」不限于目标检测模型：任何输出检出的节点都可接入——实例分割（可先经
    「掩码转外接框」得到紧致框）、斑点分析、各类检出过滤、结果合并等。
    需要按区域筛选检出时，先接「保留区域/屏蔽区域」过滤节点。
    彩色/灰度均可；最多张数=0 不限（按检出顺序截取）；完全在图外的框跳过。
    """

    NAME = "crop_by_detection_all"
    TITLE = "按检出全部裁图"
    CATEGORY = "通用"
    INPUTS = (Port("image", "image", label="图像"),
              Port("result", "result", label="结果"))
    OUTPUTS = (Port("images", "image", label="图像"),)
    PARAMS = {
        "margin_pct": ParamSpec("margin_pct", "外扩边距%", "float", 0.0,
                                range=(0.0, 200.0), tooltip="每侧按框宽/高的百分比外扩"),
        "margin_px": _MARGIN_PX,
        "square": ParamSpec("square", "补成正方形", "bool", False),
        "min_side": ParamSpec("min_side", "最小边长", "int", 0, range=(0, 8192),
                              tooltip="裁图宽/高不足时以框中心外扩到该像素"),
        "max_items": ParamSpec("max_items", "最多张数", "int", 0, range=(0, 4096),
                               tooltip="0 = 不限（按检出顺序截取）"),
        "save_dir": ParamSpec("save_dir", "落盘目录", "path", "", folder=True,
                              tooltip="留空不落盘；设置后同时保存全部裁图"),
        "name_prefix": ParamSpec("name_prefix", "文件名前缀", "text", "crop"),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        det = inputs["result"]
        h, w = img.mat.shape[:2]
        boxes = [] if det is None else _seq(det.boxes)
        labels = _seq(getattr(det, "labels", None))   # 判空不用真值（P150）
        scores = _seq(getattr(det, "scores", None))
        keep = list(range(len(boxes)))   # 区域筛选已移交「保留/屏蔽区域」过滤节点
        limit = int(params["max_items"])
        if limit > 0:
            keep = keep[:limit]
        out_dir = _save_dir(params, ctx, "crops")
        stem = _source_stem(ctx)
        prefix = str(params.get("name_prefix") or "crop")
        items = []
        for i in keep:
            b = boxes[i]
            bx1, by1, bx2, by2 = (float(v) for v in b)
            if bx2 <= 0 or by2 <= 0 or bx1 >= w or by1 >= h:
                # expand_box 会把贴边窗口平移回图内——完全在图外的框不能借此"裁"出无关内容
                ctx.warn(f"按检出全部裁图：第 {i} 个检出框在图外，已跳过")
                continue
            x1, y1, x2, y2 = expand_box(b, w, h, params["margin_pct"], params["square"],
                                        params["min_side"], params.get("margin_px", 0))
            if x2 - x1 < 1 or y2 - y1 < 1:
                ctx.warn(f"按检出全部裁图：第 {i} 个检出框在图外，已跳过")
                continue
            extra = {"index": len(items), "det_index": i,
                     "label_id": int(labels[i]) if i < len(labels) else None,
                     "score": float(scores[i]) if i < len(scores) else None}
            if out_dir is not None:
                path = out_dir / f"{prefix}_{stem}_{i:02d}.png"
                if _write(path, img.mat[y1:y2, x1:x2]):
                    extra["file"] = str(path)
                else:
                    ctx.warn(f"按检出全部裁图：写入失败 {path}")
            items.append(_item(img, [x1, y1, x2, y2], extra))
        if not items:
            ctx.warn("按检出全部裁图：没有可裁的检出框，输出空列表")
        return {"images": ImageList(items=items, meta={"source_size": [int(w), int(h)]})}


@register_node
class GridTileAll(VisionNode):
    """网格全部分块：rows×cols 均分（行优先），每块四周外扩重叠像素，输出
    图像列表（逐张带原图矩形 roi_rect）。大图分块推理的入口，可选同时落盘。
    """

    NAME = "grid_tile_all"
    TITLE = "网格全部分块"
    CATEGORY = "通用"
    INPUTS = (Port("image", "image", label="图像"),
              Port("result", "result", required=False, label="结果"))
    OUTPUTS = (Port("images", "image", label="图像"),)
    PARAMS = {
        "rows": ParamSpec("rows", "行数", "int", 2, range=(1, 32)),
        "cols": ParamSpec("cols", "列数", "int", 2, range=(1, 32)),
        "overlap_px": ParamSpec("overlap_px", "重叠像素", "int", 0, range=(0, 4096),
                                tooltip="分块推理建议设为目标尺寸的一半，避免目标被切断"),
        "save_dir": ParamSpec("save_dir", "落盘目录", "path", "", folder=True,
                              tooltip="留空不落盘；设置后同时保存全部分块"),
        "name_prefix": ParamSpec("name_prefix", "文件名前缀", "text", "tile"),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        h, w = img.mat.shape[:2]
        cols = max(int(params["cols"]), 1)
        rects = grid_rects(w, h, params["rows"], cols, params["overlap_px"])
        out_dir = _save_dir(params, ctx, "tiles")
        stem = _source_stem(ctx)
        prefix = str(params.get("name_prefix") or "tile")
        items = []
        for i, (x1, y1, x2, y2) in enumerate(rects):
            if x2 - x1 < 1 or y2 - y1 < 1:
                ctx.warn(f"网格全部分块：第 {i} 块为空（图像过小），已跳过")
                continue
            extra = {"index": i, "row": i // cols, "col": i % cols}
            if out_dir is not None:
                path = out_dir / f"{prefix}_{stem}_r{i // cols}c{i % cols}.png"
                if _write(path, img.mat[y1:y2, x1:x2]):
                    extra["file"] = str(path)
                else:
                    ctx.warn(f"网格全部分块：写入失败 {path}")
            items.append(_item(img, [x1, y1, x2, y2], extra))
        return {"images": ImageList(items=items, meta={"source_size": [int(w), int(h)]})}


def _roi(item: ImageData):
    r = item.meta.get("roi_rect")
    return [int(v) for v in r] if r else None


def _as_list(v):
    """图像端口值 → ImageList（修复循环 4：端口类型统一后单张也允许接入）。"""
    if isinstance(v, ImageList):
        return v
    return ImageList(items=[v])


@register_node
class ModelOnList(VisionNode):
    """逐图推理：对图像列表的每一张执行同一个 AI 模型，输出逐图结果列表。

    模型按路径会话缓存——整个列表（以及同一流程的多次运行）只加载一次。任务
    类型不限（检测/分割/分类）。单张失败记警告、该项为空检出，不中断其余各张。
    输出接「结果合并」换算回原图坐标。
    """

    NAME = "model_on_list"
    TITLE = "逐图推理"
    CATEGORY = "AI"
    HANDLES_LIST = True   # 原生消费多张集合（关闭执行器自动映射）
    INPUTS = (Port("images", "image", label="图像"),)
    OUTPUTS = (Port("result", "result", label="结果"),)
    PARAMS = {
        "model_path": ParamSpec("model_path", "模型文件", "path", "",
                                tooltip=".xva / .pt / .xcs 权重（检测/分割/分类均可）"),
        "conf": ParamSpec("conf", "置信度", "float", 0.25, range=(0.0, 1.0)),
        "iou": ParamSpec("iou", "IOU", "float", 0.45, range=(0.0, 1.0)),
        "device": ParamSpec("device", "设备", "enum", "auto",
                            options=["auto", "cpu", "gpu"]),
    }

    def run(self, inputs, params, ctx):
        from ...models.base import Prediction   # 惰性：牵引 torch（P154）
        from .ai_bridge import _load_model_session

        path = str(params.get("model_path") or "")
        if not path:
            raise ValueError("未设置模型文件（逐图推理）")
        lst = _as_list(inputs["images"])
        model = _load_model_session(ctx, path, device=str(params.get("device") or "auto"))
        if getattr(model, "classes", None):
            ctx.set("model_classes", list(model.classes))
        items = []
        for k, img in enumerate(lst.items):
            try:
                pred = model.predict(img.mat, conf=float(params["conf"]),
                                     iou=float(params["iou"]))
            except Exception as e:  # noqa: BLE001  单张失败不拖垮整列
                ctx.warn(f"逐图推理：第 {k} 张推理失败（{e}），该项记为空检出")
                pred = Prediction()
            items.append(ResultItem(kind=result_kind(pred), value=pred, roi_rect=_roi(img),
                                    meta={"index": k, "source": "model"}))
        meta = dict(lst.meta)
        meta["task"] = getattr(getattr(model, "task", None), "value", "")
        return {"result": ResultList(items=items, meta=meta)}


@register_node
class OpOnList(VisionNode):
    """逐图执行算子：对图像列表的每一张执行同一个传统算子（如 find_circle、
    measure 类），收集指定输出为逐图结果列表——接「结果合并」换算回原图坐标。

    算子须以图像为唯一必需输入；参数用 JSON 填写（未填的用该算子默认值，非法
    参数执行前报错）。单张失败记警告并跳过该张，不中断其余各张。
    """

    NAME = "op_on_list"
    TITLE = "逐图执行算子"
    CATEGORY = "通用"
    HANDLES_LIST = True   # 原生消费多张集合（关闭执行器自动映射）
    INPUTS = (Port("images", "image", label="图像"),)
    OUTPUTS = (Port("result", "result", label="结果"),)
    PARAMS = {
        "op": ParamSpec("op", "算子名", "text", "find_circle",
                        tooltip="注册名，如 find_circle / blob_analysis / find_line / "
                                "fit_rectangle（算子库列表里的英文名）"),
        "op_params": ParamSpec("op_params", "算子参数(JSON)", "text", "{}",
                               tooltip='如 {"polarity": "bright", "num_rays": 72}；'
                                       "留空 {} = 全部默认值"),
        "output": ParamSpec("output", "收集输出端口", "text", "",
                            tooltip="留空 = 该算子第一个 result/数值类输出"
                                    "（如 result / score / distance）"),
    }

    def _resolve(self, params):
        name = str(params.get("op") or "").strip()
        cls = get_node_class(name)
        if cls is None:
            raise ValueError(f"逐图执行算子：未找到算子「{name}」（请填写算子库中的英文注册名）")
        img_slots = [p for p in cls.INPUTS if p.required and p.type == "image"]
        other_required = [p for p in cls.INPUTS if p.required and p.type != "image"]
        if len(img_slots) != 1 or other_required:
            raise ValueError(f"逐图执行算子：「{name}」需要除一张图像外的其它必需输入，"
                             "暂不支持逐图执行")
        try:
            user = json.loads(str(params.get("op_params") or "{}") or "{}")
        except ValueError as e:
            raise ValueError(f"逐图执行算子：参数 JSON 格式错误（{e}）") from None
        if not isinstance(user, dict):
            raise ValueError("逐图执行算子：参数须为 JSON 对象，如 {\"num_rays\": 72}")
        full = {k: s.default for k, s in cls.PARAMS.items()}
        full.update(user)
        errs = cls().validate(full)
        if errs:
            raise ValueError(f"逐图执行算子「{name}」参数无效：" + "；".join(errs))
        out_name = str(params.get("output") or "").strip()
        names = [p.name for p in cls.OUTPUTS]
        if out_name and out_name not in names:
            raise ValueError(f"逐图执行算子：「{name}」没有输出端口「{out_name}」"
                             f"（可选：{', '.join(names)}）")
        if not out_name:
            out_name = next((p.name for p in cls.OUTPUTS if p.type != "image"), names[0])
        return cls, img_slots[0].name, full, out_name

    def run(self, inputs, params, ctx):
        cls, in_name, op_params, out_name = self._resolve(params)
        lst = _as_list(inputs["images"])
        items = []
        for k, img in enumerate(lst.items):
            try:
                out = cls().run({in_name: img}, dict(op_params), ctx)
            except Exception as e:  # noqa: BLE001  单张失败不拖垮整列
                ctx.warn(f"逐图执行算子：{cls.NAME} 第 {k} 张执行失败（{e}），已跳过")
                continue
            value = out.get(out_name)
            items.append(ResultItem(kind=result_kind(value), value=value, roi_rect=_roi(img),
                                    meta={"index": k, "source": cls.NAME, "port": out_name}))
        return {"result": ResultList(items=items, meta=dict(lst.meta))}


@register_node
class MaskToBox(VisionNode):
    """掩码转外接框：用每个目标的掩码重新计算紧致外接框（如实例分割结果 → 精确裁图）。

    输入可来自任意输出检出的节点（检测/分割模型、过滤、斑点分析、结果合并…）；
    掩码/类别/得分原样保留，只替换框。无掩码的目标按参数保留原框或丢弃；
    掩码面积小于下限的目标丢弃（去噪点）。属于"修改类"：图像与结果原样向下传。
    """

    NAME = "mask_to_box"
    TITLE = "掩码转外接框"
    CATEGORY = "检出过滤"
    INPUTS = (Port("image", "image", required=False, label="图像"),
              Port("result", "result", label="结果"))
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"))
    PARAMS = {
        "no_mask": ParamSpec("no_mask", "无掩码目标", "enum", "keep",
                             options=["keep", "drop"],
                             tooltip="keep=保留原框；drop=丢弃"),
        "min_area": ParamSpec("min_area", "最小掩码面积", "int", 0, range=(0, 10 ** 9),
                              tooltip="像素数；小于此值的目标丢弃（0 = 不限）"),
    }

    def run(self, inputs, params, ctx):
        from ...models.base import Prediction   # 惰性：牵引 torch（P154）
        from .ai_bridge import DetectionPlus

        det = inputs["result"]
        boxes = _seq(getattr(det, "boxes", None))
        labels = _seq(getattr(det, "labels", None))
        scores = _seq(getattr(det, "scores", None))
        masks = getattr(det, "masks", None)
        masks = [] if masks is None else list(masks)   # P150：不用真值判空
        min_area = int(params["min_area"])
        out_b, out_l, out_s, out_m = [], [], [], []
        dropped = 0
        for j, b in enumerate(boxes):
            m = masks[j] if j < len(masks) else None
            if m is not None and np.asarray(m).size:
                ys, xs = np.nonzero(np.asarray(m))
                if len(xs) == 0 or len(xs) < min_area:
                    dropped += 1
                    continue
                box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            elif params["no_mask"] == "drop":
                dropped += 1
                continue
            else:
                box = [float(v) for v in b]
            out_b.append(box)
            out_l.append(int(labels[j]) if j < len(labels) else 0)
            out_s.append(float(scores[j]) if j < len(scores) else 0.0)
            out_m.append(m)
        if dropped:
            ctx.warn(f"掩码转外接框：丢弃 {dropped} 个目标（无掩码/空掩码/面积不足）")
        has_mask = any(m is not None for m in out_m)
        pred = Prediction(labels=out_l, scores=out_s, boxes=out_b,
                          masks=out_m if has_mask else None,
                          probs=getattr(det, "probs", None))
        wrapped = DetectionPlus(pred=pred, extra=det.plus_extra) \
            if isinstance(det, DetectionPlus) else pred
        out = {"result": wrapped}
        img = inputs.get("image")
        if img is not None:
            out["image"] = img
        return out


def nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_thr: float,
                labels: np.ndarray | None = None) -> list:
    """贪心 NMS（类内）：返回保留下标（按得分降序）。iou_thr<=0 不去重。"""
    n = len(boxes)
    if n == 0:
        return []
    order = list(np.argsort(-scores, kind="stable"))
    if iou_thr <= 0:
        return order
    keep = []
    area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    suppressed = np.zeros(n, bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[:, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[:, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[:, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[:, 3])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / np.maximum(area[i] + area - inter, 1e-9)
        same = (labels == labels[i]) if labels is not None else np.ones(n, bool)
        suppressed |= (iou > iou_thr) & same
        suppressed[i] = True
    return keep


def shift_geometry(geo: Geometry, ox: float, oy: float) -> Geometry:
    """几何平移到原图坐标（点/线/多边形逐点；圆/旋转矩形平移中心）。"""
    d = np.asarray(geo.data, dtype=float).copy()
    if geo.kind in ("point", "line", "polygon"):
        d = d.reshape(-1, 2) + np.array([ox, oy])
        if geo.kind == "line":
            d = d.reshape(2, 2)
    elif geo.kind in ("circle", "rotated_rect"):
        d[0] += ox
        d[1] += oy
    return Geometry(kind=geo.kind, data=d, subpixel=getattr(geo, "subpixel", False))


def geometry_box_mask(geo: Geometry, W: int, H: int, want_mask: bool):
    """原图坐标几何 → (外接框 [x1,y1,x2,y2], 掩码 or None)。点/线不成掩码。"""
    import cv2

    d = np.asarray(geo.data, dtype=float)
    mask = None
    if geo.kind == "circle":
        cx, cy, r = (float(v) for v in d[:3])
        box = [cx - r, cy - r, cx + r, cy + r]
        if want_mask and W and H:
            mask = np.zeros((H, W), np.uint8)
            cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 1, -1)
    elif geo.kind in ("rotated_rect", "polygon"):
        if geo.kind == "rotated_rect":
            pts = cv2.boxPoints(((d[0], d[1]), (max(d[2], 1.0), max(d[3], 1.0)), d[4]))
        else:
            pts = d.reshape(-1, 2)
        box = [float(pts[:, 0].min()), float(pts[:, 1].min()),
               float(pts[:, 0].max()), float(pts[:, 1].max())]
        if want_mask and W and H:
            mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    else:   # point / line：外接框（至少 2px），无掩码
        pts = d.reshape(-1, 2)
        box = [float(pts[:, 0].min()) - 1, float(pts[:, 1].min()) - 1,
               float(pts[:, 0].max()) + 1, float(pts[:, 1].max()) + 1]
    return box, (mask.astype(bool) if mask is not None else None)


@register_node
class MergeResults(VisionNode):
    """结果合并：多路（图像, 结果）对 → 原图坐标下的统一检出。

    输入为成对端口（图像1, 结果1）（图像2, 结果2）…（用户批注/示意图）：
    在画布上连上一对后自动出现下一对空输入。每对中：
    - 图像：原图（结果已是原图坐标）或裁图列表（ImageList，逐张带 roi_rect
      ——结果按 roi_rect 换算回原图）；结果为逐图结果列表（ResultList）时与
      图像列表逐张配对；
    - 结果：检出/分类/几何/测量/数值均可（统一结果）。
    合并规则：检出框平移+掩码回贴；分类以 roi_rect 为框；几何转"外接框 +
    掩码（圆/旋转矩形/多边形）"，标签=「几何结果类别」；测量/数值逐条记入
    extra["merged_items"]。图像输出 = 第一个图像输入（原图直通）。
    """

    NAME = "merge_results"
    TITLE = "结果合并"
    CATEGORY = "通用"
    HANDLES_LIST = True   # 原生消费多张集合（关闭执行器自动映射）
    INPUTS = _merge_inputs()
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"))
    PARAMS = {
        "nms_iou": ParamSpec("nms_iou", "去重 IOU", "float", 0.5, range=(0.0, 1.0),
                             tooltip="0 = 不去重；分块重叠推理建议 0.5"),
        "class_aware": ParamSpec("class_aware", "按类别去重", "bool", True,
                                 tooltip="否 = 不同类别的重叠框也只留一个"),
        "geometry_label": ParamSpec("geometry_label", "几何结果类别", "int", 0,
                                    range=(0, 9999),
                                    tooltip="传统算子几何结果合并成检出时使用的类别序号"),
        "geometry_mask": ParamSpec("geometry_mask", "几何生成掩码", "bool", True,
                                   tooltip="圆/旋转矩形/多边形结果生成原图尺寸掩码"),
    }

    def run(self, inputs, params, ctx):
        from ...models.base import Prediction   # 惰性：牵引 torch（P154）
        from ..ir import NumberVal
        from .ai_bridge import DetectionPlus

        # ---- 收集全部成对输入：(roi 或 None, 结果值) ----
        src_size = None
        base_img = None
        entries = []          # [(roi_rect 或 None, value), ...]
        for k in range(1, _MERGE_PAIRS + 1):
            suf = "" if k == 1 else str(k)
            img = inputs.get(f"image{suf}")
            res = inputs.get(f"result{suf}")
            if img is None and res is None:
                continue
            if img is not None and base_img is None and isinstance(img, ImageData):
                base_img = img
            if isinstance(img, ImageList):
                size = img.meta.get("source_size")
                if size and not src_size:
                    src_size = (int(size[0]), int(size[1]))
                rois = [_roi(it) for it in img.items]
                if isinstance(res, ResultList):
                    # 逐图结果与图像列表按 index 配对（坐标以图像列表为准）
                    for it in res.items:
                        k_idx = it.meta.get("index")
                        roi = rois[k_idx] if isinstance(k_idx, int) and k_idx < len(rois) \
                            else it.roi_rect
                        entries.append((roi, it.value))
                else:
                    for roi in rois:
                        entries.append((roi, res))
                continue
            # 单图或无图输入：ResultList 逐项自带 roi_rect（逐图处理链）
            if isinstance(res, ResultList):
                for it in res.items:
                    entries.append((it.roi_rect, it.value))
            elif res is not None:
                roi = _roi(img) if isinstance(img, ImageData) else None
                entries.append((roi, res))
            elif isinstance(img, ImageList):
                pass
        if base_img is not None:
            H, W = int(base_img.mat.shape[0]), int(base_img.mat.shape[1])
        elif src_size:
            W, H = int(src_size[0]), int(src_size[1])
        else:
            H = W = 0

        boxes, scores, labels, masks = [], [], [], []
        records = []
        any_mask = False
        geo_label = int(params["geometry_label"])
        for k, (roi, v) in enumerate(entries):
            ox, oy = (float(roi[0]), float(roi[1])) if roi else (0.0, 0.0)
            kind = result_kind(v)
            if kind == "classification":
                probs = _seq(getattr(v, "probs", None))
                if roi and probs:
                    top = int(np.argmax(probs))
                    boxes.append([float(x) for x in roi])
                    scores.append(float(probs[top]))
                    labels.append(top)
                    masks.append(None)
                continue
            if kind == "detection":
                p_boxes = _seq(getattr(v, "boxes", None))
                p_scores = _seq(getattr(v, "scores", None))
                p_labels = _seq(getattr(v, "labels", None))
                p_masks = getattr(v, "masks", None)
                p_masks = [] if p_masks is None else list(p_masks)   # P150：不用真值判空
                for j, b in enumerate(p_boxes):
                    x1, y1, x2, y2 = (float(t) for t in b)
                    boxes.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy])
                    scores.append(float(p_scores[j]) if j < len(p_scores) else 0.0)
                    labels.append(int(p_labels[j]) if j < len(p_labels) else 0)
                    m = p_masks[j] if j < len(p_masks) else None
                    if m is not None and W and H:
                        any_mask = True
                        masks.append(_paste_mask(np.asarray(m), int(round(ox)),
                                                 int(round(oy)), W, H))
                    else:
                        masks.append(None)
                continue
            if kind == "geometry" and isinstance(v, Geometry):
                g = shift_geometry(v, ox, oy)
                box, m = geometry_box_mask(g, W, H, bool(params["geometry_mask"]))
                boxes.append(box)
                scores.append(1.0)
                labels.append(geo_label)
                masks.append(m)
                any_mask = any_mask or m is not None
                records.append({"index": k, "kind": "geometry", "roi_rect": roi,
                                "value": g.to_json()})
                continue
            if kind in ("measure", "number", "bool"):
                val = v.to_json() if hasattr(v, "to_json") else (bool(v) if kind == "bool" else v)
                records.append({"index": k, "kind": kind, "roi_rect": roi, "value": val})
        n_in = len(boxes)
        if boxes:
            b = np.asarray(boxes, float)
            s = np.asarray(scores, float)
            lab = np.asarray(labels, int)
            keep = nms_indices(b, s, float(params["nms_iou"]),
                               lab if bool(params["class_aware"]) else None)
            if len(keep) < n_in:
                ctx.warn(f"结果合并：去重移除 {n_in - len(keep)} 个重叠结果")
            out_masks = None
            if any_mask:
                out_masks = [masks[i] if masks[i] is not None else np.zeros((H, W), bool)
                             for i in keep]
            pred = Prediction(labels=[int(lab[i]) for i in keep],
                              scores=[float(s[i]) for i in keep],
                              boxes=[[float(t) for t in b[i]] for i in keep],
                              masks=out_masks)
        else:
            pred = Prediction()
        out = DetectionPlus(pred=pred, extra={"merged_items": records}) if records else pred
        outv = {"result": out}
        if base_img is not None:
            outv["image"] = base_img
        return outv


def _res_values(res):
    """结果端口值 → 逐条原值列表（ResultList 展开为 items.value）。"""
    if isinstance(res, ResultList):
        return [it.value for it in res.items]
    return [res]


def _paste_mask(m: np.ndarray, ox: int, oy: int, W: int, H: int) -> np.ndarray:
    """裁图坐标系掩码 → 原图尺寸 bool 掩码（越界部分裁掉）。"""
    canvas = np.zeros((H, W), bool)
    m = m.astype(bool) if m.dtype != bool else m
    if m.ndim != 2 or not m.size:
        return canvas
    mh, mw = m.shape
    x1, y1 = max(ox, 0), max(oy, 0)
    x2, y2 = min(ox + mw, W), min(oy + mh, H)
    if x2 > x1 and y2 > y1:
        canvas[y1:y2, x1:x2] = m[y1 - oy:y2 - oy, x1 - ox:x2 - ox]
    return canvas

"""AI 桥接算子（批次十八建立，v0.4.2 按用户批注 Q5 重构）。

- DetectionPlus：检出扩展字段的**组合式包装**（用户决策 D3：Prediction
  保持冻结）。__getattr__ 委托内部 Prediction，下游过滤节点/apply_flow
  可直接消费；经旧链过滤后 extra 会丢失（展平为普通 Prediction，属预期）。
- 模型节点**按任务拆分为三个**（Q5：像旧版一样分模型暴露流程步骤）：
  model_detection / model_instance_seg / model_classification，共用
  AiModelBase 加载链路（engine.models.registry.load_model 唯一入口，
  D5 不另造加载链路），加载后校验任务类型与节点一致。
- roi_batch_classify：检出框裁剪 → 分类模型批量复判 → 结果挂回检出
  （DetectionPlus.extra["roi_classifications"]）。
- detection_to_measure：检出几何量（宽/高/面积/得分）→ 测量结果。
- geometry_to_affine：定位几何 → 6 元仿射矩阵（供 affine_transform 归正）。

模型会话缓存：RunContext 生命周期内按解析路径缓存已加载模型（键
ai_model:<path>），同一会话内不重复加载。加载/推理全部惰性导入，
保持 engine.flow 轻导入纪律。
"""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..ir import Geometry, MeasureResult, NumberVal
from ..node_base import ParamSpec, Port, VisionNode
from ..registry import register_node


@dataclass
class DetectionPlus:
    """Prediction + 扩展字段（组合式包装，D3）。

    __getattr__ 委托内部 Prediction：下游按 Prediction 消费的代码
    （过滤节点、apply_flow、result_json）无需感知本类。
    """

    pred: object
    extra: dict = field(default_factory=dict)

    def __getattr__(self, name):
        return getattr(self.__dict__["pred"], name)

    @property
    def plus_extra(self) -> dict:
        return dict(self.extra)


def _load_model_session(ctx, path: str, device: str = "cpu"):
    """会话内模型缓存：同一路径只加载一次（RunContext 生命周期）。"""
    from ...models.registry import load_model  # 惰性：torch 链路

    key = f"ai_model:{path}"
    model = ctx.get(key)
    if model is None:
        model = load_model(path, device=device)
        ctx.set(key, model)
    return model


class AiModelBase(VisionNode):
    """AI 模型节点基类（按任务拆分为三个节点，Q5：流程步骤对外暴露清晰）。

    模型加载走 engine.models.registry.load_model 唯一入口（D5），支持
    .xva/.pt/.xcs；加载后校验任务类型与节点一致；模型按路径会话缓存
    （ctx 键 ai_model:<path>）；类别名表写入 ctx.model_classes。
    """

    NAME = ""
    CATEGORY = "AI"
    TASK = ""                    # 期望的模型任务（detection/segmentation/classification）
    TASK_LABEL = ""              # 人话任务名（报错用）
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"))
    PARAMS = {
        "model_path": ParamSpec("model_path", "模型文件", "path", "",
                                tooltip=".xva / .pt / .xcs 权重"),
        "conf": ParamSpec("conf", "置信度", "float", 0.25, range=(0.0, 1.0)),
        "iou": ParamSpec("iou", "IOU", "float", 0.45, range=(0.0, 1.0)),
        "device": ParamSpec("device", "设备", "enum", "auto",
                            options=["auto", "cpu", "gpu"]),
    }

    def run(self, inputs, params, ctx):
        path = str(params.get("model_path") or "")
        if not path:
            raise ValueError(f"未设置模型文件（{self.TITLE}）")
        model = _load_model_session(ctx, path, device=str(params.get("device") or "auto"))
        task = getattr(getattr(model, "task", None), "value", "")
        if self.TASK and task != self.TASK:
            raise ValueError(f"{self.TITLE} 需要任务类型为「{self.TASK_LABEL}」的模型，"
                             f"所选文件的实际任务为「{task}」")
        img = inputs["image"]
        pred = model.predict(img.mat, conf=float(params["conf"]), iou=float(params["iou"]))
        if model.classes:
            ctx.set("model_classes", list(model.classes))
        return {"result": pred, "image": img}


@register_node
class ModelDetection(AiModelBase):
    """目标检测模型：载入检测权重输出检出框（image → detection）。"""

    NAME = "model_detection"
    TITLE = "目标检测模型"
    TASK = "detection"
    TASK_LABEL = "目标检测"


@register_node
class ModelInstanceSeg(AiModelBase):
    """实例分割模型：载入分割权重输出检出框与掩码（image → detection）。"""

    NAME = "model_instance_seg"
    TITLE = "实例分割模型"
    TASK = "segmentation"
    TASK_LABEL = "实例分割"


@register_node
class ModelClassification(AiModelBase):
    """分类模型：载入分类权重输出整图类别与置信度（image → detection，
    Prediction.probs 携带类别概率，检出过滤链自动直通）。"""

    NAME = "model_classification"
    TITLE = "分类模型"
    TASK = "classification"
    TASK_LABEL = "分类"


@register_node
class RoiBatchClassify(VisionNode):
    """检出框二级复判：逐框裁剪（含边缘比）→ 分类模型批量复判。

    输出 DetectionPlus：extra["roi_classifications"] = [
        {"index": 框序号, "label": 类别名, "label_id": 序号, "score": 得分}, ...]
    ；裁剪失败的框（越界/退化）记 ctx 警告并跳过。属于"修改类"节点：
    图像与结果原样向下传（复判信息挂在结果扩展字段上）。
    """

    NAME = "roi_batch_classify"
    TITLE = "二级分类复判"
    CATEGORY = "AI"
    INPUTS = (Port("image", "image", label="图像"),
              Port("result", "result", label="结果"))
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"))
    PARAMS = {
        "model_path": ParamSpec("model_path", "分类模型", "path", "",
                                tooltip="分类任务权重（.xva/.pt）"),
        "conf": ParamSpec("conf", "置信度", "float", 0.5, range=(0.0, 1.0),
                          tooltip="复判得分的展示口径，不二次过滤"),
        "margin": ParamSpec("margin", "边缘比", "float", 0.1, range=(0.0, 0.5),
                            tooltip="框四周外扩比例（保留上下文）"),
        "device": ParamSpec("device", "设备", "enum", "auto",
                            options=["auto", "cpu", "gpu"]),
    }

    def run(self, inputs, params, ctx):
        path = str(params.get("model_path") or "")
        if not path:
            raise ValueError("未设置分类模型（roi_batch_classify）")
        model = _load_model_session(ctx, path, device=str(params.get("device") or "auto"))
        if getattr(model, "classes", None):
            ctx.set("model_classes", list(model.classes))
        det = inputs["result"]
        pred = det.pred if isinstance(det, DetectionPlus) else det
        img = inputs["image"]
        mat = img.mat
        h, w = mat.shape[:2]
        margin = float(params["margin"])
        results = []
        for i, box in enumerate(pred.boxes):
            x1, y1, x2, y2 = (float(v) for v in box)
            mx, my = (x2 - x1) * margin, (y2 - y1) * margin
            cx1 = max(0, int(math.floor(x1 - mx)))
            cy1 = max(0, int(math.floor(y1 - my)))
            cx2 = min(w, int(math.ceil(x2 + mx)))
            cy2 = min(h, int(math.ceil(y2 + my)))
            if cx2 - cx1 < 2 or cy2 - cy1 < 2:
                ctx.warn(f"roi_batch_classify：第 {i} 个检出框越界/退化，跳过复判")
                continue
            crop = mat[cy1:cy2, cx1:cx2]
            sub = model.predict(crop, conf=float(params["conf"]), iou=0.45)
            if not len(sub.labels):
                results.append({"index": i, "label": "", "label_id": -1,
                                "score": 0.0})
                continue
            top = int(np.argmax(sub.scores))
            label_id = int(sub.labels[top])
            results.append({
                "index": i,
                "label": (model.classes[label_id]
                          if model.classes and label_id < len(model.classes)
                          else str(label_id)),
                "label_id": label_id,
                "score": round(float(sub.scores[top]), 4),
            })
        extra = {}
        if isinstance(det, DetectionPlus):
            extra.update(det.plus_extra)
        extra["roi_classifications"] = results
        return {"result": DetectionPlus(pred=pred, extra=extra), "image": img}


@register_node
class DetectionToMeasure(VisionNode):
    """检出转测量：把检出框几何量转成测量结果（可带容差判定）。

    metric = width/height/area/score；多检出输出 values 序列，passed =
    全部落在容差内（tol_min/tol_max 均 0 = 不判定，passed=None）。
    输出：图像 + 结果（MeasureResult）+ 数值（检出数量）。
    """

    NAME = "detection_to_measure"
    TITLE = "检出转测量"
    CATEGORY = "测量"
    INPUTS = (Port("image", "image", required=False, label="图像"),
              Port("result", "result", label="结果"))
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"),
               Port("count", "number", label="数量"))
    PARAMS = {
        "metric": ParamSpec("metric", "几何量", "enum", "area",
                            options=["width", "height", "area", "score"]),
        "tol_min": ParamSpec("tol_min", "容差下限", "float", 0.0,
                             tooltip="0 = 不限下限"),
        "tol_max": ParamSpec("tol_max", "容差上限", "float", 0.0,
                             tooltip="0 = 不限上限"),
    }

    def run(self, inputs, params, ctx):
        det = inputs["result"]
        pred = det.pred if isinstance(det, DetectionPlus) else det
        metric = params["metric"]
        vals = []
        for box, score in zip(pred.boxes, pred.scores):
            x1, y1, x2, y2 = (float(v) for v in box)
            if metric == "width":
                vals.append(x2 - x1)
            elif metric == "height":
                vals.append(y2 - y1)
            elif metric == "score":
                vals.append(float(score))
            else:
                vals.append((x2 - x1) * (y2 - y1))
        lo, hi = float(params["tol_min"]), float(params["tol_max"])
        if lo == 0.0 and hi == 0.0:
            passed = None
        elif not vals:
            # 空检出不允许"全称量化恒真"判通过（P90：没检出 ≠ 合格）
            passed = False
            ctx.warn("检出转测量：未检出目标，测量无值，判定不通过")
        else:
            passed = all((v >= lo if lo else True) and (v <= hi if hi else True)
                         for v in vals)
        out = {"result": MeasureResult(values=vals, passed=passed),
               "count": NumberVal(value=float(len(vals)), unit="个")}
        img = inputs.get("image")
        if img is not None:
            out["image"] = img
        return out


@register_node
class GeometryToAffine(VisionNode):
    """定位几何 → 6 元仿射矩阵（a,b,tx,c,d,ty，行优先）。

    - rotate_level：以 line 几何的方向角把图转平（绕图像中心；无 image
      端口时绕原点）；
    - translate_point：把 point 几何第一个点平移到图像中心（无 image
      端口时平移到原点）。
    输出 result（值=6 元矩阵 MeasureResult），接 affine_transform 的
    matrix_in 端口；angle 输出取到的角度（度，display 用）。
    """

    NAME = "geometry_to_affine"
    TITLE = "几何转归正矩阵"
    CATEGORY = "测量"
    INPUTS = (Port("image", "image", required=False, label="图像"),
              Port("result", "result", label="结果"))
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="归正矩阵"),
               Port("angle", "number", label="角度"))
    PARAMS = {
        "mode": ParamSpec("mode", "模式", "enum", "rotate_level",
                          options=["rotate_level", "translate_point"]),
    }

    def run(self, inputs, params, ctx):
        geo = inputs["result"]
        if not isinstance(geo, Geometry):
            raise ValueError("几何转归正矩阵：结果输入须为几何（定位类节点的结果）")
        data = np.asarray(geo.data, dtype=float)
        img = inputs.get("image")
        if params["mode"] == "rotate_level":
            if geo.kind != "line" or data.shape != (2, 2):
                raise ValueError("rotate_level 需要 line 几何输入（两端点）")
            dx, dy = data[1] - data[0]
            angle = math.degrees(math.atan2(dy, dx))
            if img is not None:
                h, w = img.mat.shape[:2]
                cx, cy = w / 2.0, h / 2.0
            else:
                cx = cy = 0.0
            ctx.warn(f"geometry_to_affine：按线方向 {angle:.2f}° 转平")
            rad = math.radians(-angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            # R(-a) 绕 (cx,cy)：T(c)·R·T(-c)
            a = cos_a
            b = -sin_a
            c = sin_a
            d = cos_a
            tx = cx - a * cx - b * cy
            ty = cy - c * cx - d * cy
            values = [round(a, 6), round(b, 6), round(tx, 3),
                      round(c, 6), round(d, 6), round(ty, 3)]
            out = {"result": MeasureResult(values=values),
                   "angle": NumberVal(value=round(angle, 3), unit="deg")}
            if img is not None:
                out["image"] = img
            return out
        # translate_point
        if geo.kind != "point" or data.size < 2:
            raise ValueError("translate_point 需要 point 几何输入")
        px, py = float(data.reshape(-1, 2)[0][0]), float(data.reshape(-1, 2)[0][1])
        if img is not None:
            h, w = img.mat.shape[:2]
            tx, ty = w / 2.0 - px, h / 2.0 - py
        else:
            tx, ty = -px, -py
        values = [1.0, 0.0, round(tx, 3), 0.0, 1.0, round(ty, 3)]
        out = {"result": MeasureResult(values=values),
               "angle": NumberVal(value=0.0, unit="deg")}
        if img is not None:
            out["image"] = img
        return out


def parse_relabel_mapping(text) -> list:
    """映射表文本 → [(旧, 新), ...]；格式错误抛 ValueError（中文原因）。

    语法：`旧:新` 以逗号/分号分隔；中文冒号/逗号/分号可用；旧可为类别名或序号。
    """
    out = []
    raw = str(text or "").replace("：", ":").replace("，", ",").replace("；", ",")
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"映射「{part}」缺少冒号（应为 旧:新）")
        old, _, new = part.partition(":")
        old, new = old.strip(), new.strip()
        if not old or not new:
            raise ValueError(f"映射「{part}」的旧名或新名为空")
        out.append((old, new))
    return out


@register_node
class RelabelMap(VisionNode):
    """标签修改：按映射表改检出类别名（未映射的可保留或删除），原标签留痕。

    mapping 如 `scratch:划痕,1:脏污`（旧可写类别名或类别序号）。新名字
    追加到类别表末尾——原序号的名字不变（同一模型的其它分支不受影响），
    并写回会话类别表，下游过滤/保存结果按新名字工作；每个检出的原类别名
    写入 extra["original_label"]，统一结果 JSON 的 label 即为新名字。
    """

    NAME = "relabel_map"
    TITLE = "标签修改"
    CATEGORY = "检出过滤"
    INPUTS = (Port("image", "image", required=False, label="图像"),
              Port("result", "result", label="结果"))
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"))
    PARAMS = {
        "mapping": ParamSpec("mapping", "映射表", "text", "",
                             tooltip="旧:新，逗号分隔；旧可写类别名或序号，如 scratch:划痕,1:脏污"),
        "default_action": ParamSpec("default_action", "未映射类别", "enum", "keep",
                                    options=["keep", "drop"],
                                    tooltip="keep=保留原标签；drop=删除该检出"),
    }

    def validate(self, params):
        errs = super().validate(params)
        try:
            parse_relabel_mapping(params.get("mapping"))
        except ValueError as e:
            errs.append(str(e))
        return errs

    def run(self, inputs, params, ctx):
        from ...models.base import Prediction  # 惰性：torch 链路

        det = inputs["result"]
        key = "flow_class_names" if (not ctx.get("model_classes")
                                     and ctx.get("flow_class_names")) else "model_classes"
        classes = list(ctx.get(key) or [])
        name_of = (lambda i: classes[i] if 0 <= i < len(classes) else str(i))
        rename = {}
        for old, new in parse_relabel_mapping(params["mapping"]):
            if old not in classes and old.lstrip("-").isdigit():
                old = name_of(int(old))   # 序号 → 类别名
            rename[old] = new
        new_classes = list(classes)

        def _id(name: str) -> int:
            if name not in new_classes:
                new_classes.append(name)
            return new_classes.index(name)

        labels = [] if det.labels is None else list(det.labels)
        boxes = [] if det.boxes is None else list(det.boxes)
        scores = [] if det.scores is None else list(det.scores)
        masks = getattr(det, "masks", None)
        keep_idx, new_labels, originals = [], [], []
        for i, lid in enumerate(labels):
            name = name_of(int(lid))
            if name in rename:
                new_labels.append(_id(rename[name]))
            elif params["default_action"] == "keep":
                new_labels.append(int(lid))
            else:
                continue
            keep_idx.append(i)
            originals.append(name)
        dropped = len(labels) - len(keep_idx)
        pred = Prediction(
            labels=new_labels,
            scores=[float(scores[i]) for i in keep_idx if i < len(scores)],
            boxes=[[float(v) for v in boxes[i]] for i in keep_idx if i < len(boxes)],
            masks=None if masks is None else [masks[i] for i in keep_idx if i < len(masks)])
        extra = {}
        old_extra = getattr(det, "extra", None)
        if isinstance(old_extra, dict):
            if dropped:
                ctx.warn("标签修改：删除了检出，上游扩展字段（按检出对齐）已丢弃")
            else:
                extra.update(old_extra)
        extra.update({"original_label": originals, "classes": list(new_classes)})
        ctx.set(key, new_classes)   # 下游过滤/保存结果按新名字工作（原序号名字不变）
        if dropped:
            ctx.warn(f"标签修改：删除未映射类别的检出 {dropped} 个")
        out = {"result": DetectionPlus(pred=pred, extra=extra)}
        img = inputs.get("image")
        if img is not None:
            out["image"] = img
        return out


# ---------------------------------------------------------------- V0.6 T4-3


def _subset(det, idx) -> "object":
    """按序号取检出子集（掩码随检出对齐；输出普通 Prediction）。"""
    from ...models.base import Prediction  # 惰性：torch 链路

    labels = [] if det.labels is None else list(det.labels)
    scores = [] if det.scores is None else list(det.scores)
    boxes = [] if det.boxes is None else list(det.boxes)
    masks = getattr(det, "masks", None)
    return Prediction(labels=[int(labels[i]) for i in idx],
                      scores=[float(scores[i]) for i in idx],
                      boxes=[[float(v) for v in boxes[i]] for i in idx],
                      masks=None if masks is None else
                      [masks[i] for i in idx if i < len(masks)])


def _box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _rank_key(det, by: str):
    boxes = [] if det.boxes is None else [[float(v) for v in b] for b in det.boxes]
    if by == "area":
        return [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    return [float(s) for s in ([] if det.scores is None else det.scores)]


@register_node
class FilterOverlap(VisionNode):
    """重叠过滤：框重叠（IoU 超阈值）时只保留一个（贪心 NMS）。

    keep=score 保留分数高者，area 保留面积大者；类别无关开启时跨类别也去重。
    多模型/分块结果合并后去重常用。
    """

    NAME = "filter_overlap"
    TITLE = "重叠过滤"
    CATEGORY = "检出过滤"
    INPUTS = (Port("image", "image", required=False, label="图像"),
              Port("result", "result", label="结果"))
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"),
               Port("removed", "result", label="被滤除"))
    PARAMS = {
        "iou": ParamSpec("iou", "IoU 阈值", "float", 0.5, range=(0.0, 1.0),
                         tooltip="重叠度超过该值视为重复"),
        "keep": ParamSpec("keep", "保留", "enum", "score", options=["score", "area"]),
        "class_agnostic": ParamSpec("class_agnostic", "跨类别去重", "bool", False),
    }

    def run(self, inputs, params, ctx):
        det = inputs["result"]
        boxes = [] if det.boxes is None else [[float(v) for v in b] for b in det.boxes]
        labels = [] if det.labels is None else [int(v) for v in det.labels]
        key = _rank_key(det, params["keep"])
        order = sorted(range(len(boxes)), key=lambda i: -key[i] if i < len(key) else 0.0)
        thr = float(params["iou"])
        kept: list = []
        for i in order:
            if all(not (params["class_agnostic"] or labels[i] == labels[j])
                   or _box_iou(boxes[i], boxes[j]) <= thr for j in kept):
                kept.append(i)
        kept.sort()
        removed = [i for i in range(len(boxes)) if i not in set(kept)]
        if removed:
            ctx.warn(f"重叠过滤：去除重复检出 {len(removed)} 个")
        out = {"result": _subset(det, kept), "removed": _subset(det, removed)}
        img = inputs.get("image")
        if img is not None:
            out["image"] = img
        return out


@register_node
class FilterTopK(VisionNode):
    """TopK 过滤：只保留分数（或面积）最高的 K 个检出，可按类别分别取。"""

    NAME = "filter_topk"
    TITLE = "TopK 过滤"
    CATEGORY = "检出过滤"
    INPUTS = (Port("image", "image", required=False, label="图像"),
              Port("result", "result", label="结果"))
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"),
               Port("removed", "result", label="被滤除"))
    PARAMS = {
        "k": ParamSpec("k", "保留个数 K", "int", 5, range=(1, 10000)),
        "by": ParamSpec("by", "排序依据", "enum", "score", options=["score", "area"]),
        "per_class": ParamSpec("per_class", "按类别分别取", "bool", False),
    }

    def run(self, inputs, params, ctx):
        det = inputs["result"]
        n = 0 if det.boxes is None else len(det.boxes)
        labels = [] if det.labels is None else [int(v) for v in det.labels]
        key = _rank_key(det, params["by"])
        k = int(params["k"])
        order = sorted(range(n), key=lambda i: -key[i] if i < len(key) else 0.0)
        if params["per_class"]:
            seen: dict = {}
            kept = []
            for i in order:
                c = labels[i] if i < len(labels) else -1
                if seen.get(c, 0) < k:
                    kept.append(i)
                    seen[c] = seen.get(c, 0) + 1
        else:
            kept = order[:k]
        kept.sort()
        removed = [i for i in range(n) if i not in set(kept)]
        out = {"result": _subset(det, kept), "removed": _subset(det, removed)}
        img = inputs.get("image")
        if img is not None:
            out["image"] = img
        return out

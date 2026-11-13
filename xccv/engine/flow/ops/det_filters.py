"""O2 扩展（V0.7 用户追加）：检出过滤算子——短长边 / 前景占比 / 灰度差。

对应阈值拖线面板的多维度过滤（用户在面板拖定阈值后"生成流程图模板"
一键成图）：阈值语义与面板直方图的横轴一一对应——

- filter_bbox_side    外接框短边/长边（像素）min（0=不限）；
- filter_foreground   前景占比：框内 |灰度-背景中位| ≥ gray_thr 的像素占比
                      ≥ min_ratio 才保留（"前景/背景阈值"）；
- filter_gray_diff    灰度差：框内灰度均值与框外环形背景中位差的绝对值
                      ≥ min_diff 才保留。
- （面积过滤沿用既有 legacy 薄委托节点 filter_bbox_area，不在此重复注册。）

背景参考 = 框外扩 15%（≥8px）的环形区域灰度中位数（亮度稳健，不受框内
目标污染）；两张阈值算子（前景占比/灰度差）需要图像输入，缺图时告警并
整图放行（与 region 系算子的缺参语义一致）。
"""

import cv2
import numpy as np

from ..node_base import ParamSpec, Port, VisionNode
from ..registry import register_node


def _subset(det, idx: list):
    from ...models.base import Prediction  # 惰性：torch 链路

    labels = [] if det.labels is None else list(det.labels)
    scores = [] if det.scores is None else list(det.scores)
    boxes = [] if det.boxes is None else list(det.boxes)
    masks = getattr(det, "masks", None)
    return Prediction(labels=[int(labels[i]) for i in idx if i < len(labels)],
                      scores=[float(scores[i]) for i in idx if i < len(scores)],
                      boxes=[[float(v) for v in boxes[i]]
                             for i in idx if i < len(boxes)],
                      masks=None if masks is None else
                      [masks[i] for i in idx if i < len(masks)])


def _bg_ring_median(gray: np.ndarray, box) -> float | None:
    """框外扩环形区域的灰度中位数（背景参考；无有效环形像素返回 None）。"""
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    mx = max(8.0, bw * 0.15)
    my = max(8.0, bh * 0.15)
    ex1, ey1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    ex2, ey2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    ix1, iy1 = max(0, int(x1)), max(0, int(y1))
    ix2, iy2 = min(w, int(x2)), min(h, int(y2))
    if ex2 - ex1 <= 0 or ey2 - ey1 <= 0:
        return None
    region = gray[ey1:ey2, ex1:ex2]
    ring_mask = np.ones(region.shape[:2], dtype=bool)
    iy1e, iy2e = max(0, iy1 - ey1), max(0, iy2 - ey1)
    ix1e, ix2e = max(0, ix1 - ex1), max(0, ix2 - ex1)
    if iy2e > iy1e and ix2e > ix1e:
        ring_mask[iy1e:iy2e, ix1e:ix2e] = False   # 抠掉框内
    vals = region[ring_mask]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def box_dims(box) -> dict:
    """单框几何维度（面积/短边/长边；与批量线程 _dims 同一口径）。"""
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    return {"area": w * h, "short": min(w, h), "long": max(w, h)}


def box_image_dims(gray: np.ndarray, box) -> dict:
    """单框含像素维度（fg_ratio/gray_diff；缺背景参考时两项为 None）。"""
    out = {"fg_ratio": None, "gray_diff": None}
    bg = _bg_ring_median(gray, box)
    if bg is None:
        return out
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = (int(max(0, v)) for v in
                      (box[0], box[1], box[2], box[3]))
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return out
    crop = gray[y1:y2, x1:x2].astype(np.float32)
    out["gray_diff"] = round(abs(float(crop.mean()) - bg), 2)
    out["fg_ratio"] = round(float((np.abs(crop - bg) >= 20).mean()), 4)
    return out


class _ResultFilterBase(VisionNode):
    """检出过滤节点公共基类：image(可选)+result → image+result+removed。"""

    INPUTS = (Port("image", "image", required=False, label="图像"),
              Port("result", "result", label="结果"))
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"),
               Port("removed", "result", label="被滤除"))

    def _run(self, inputs, ctx, decide):
        """decide(box, gray_or_None) -> bool（True=保留）。"""
        det = inputs["result"]
        boxes = [] if det.boxes is None else [[float(v) for v in b]
                                              for b in det.boxes]
        gray = None
        img = inputs.get("image")
        if img is not None:
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kept = []
        for i, box in enumerate(boxes):
            if decide(box, gray):
                kept.append(i)
        removed = [i for i in range(len(boxes)) if i not in set(kept)]
        if removed:
            ctx.warn(f"{self.TITLE}：过滤检出 {len(removed)} 个")
        out = {"result": _subset(det, kept), "removed": _subset(det, removed)}
        if img is not None:
            out["image"] = img
        return out


@register_node
class FilterBboxSide(_ResultFilterBase):
    """短边/长边过滤：外接框短边、长边任一低于下限即过滤（像素，0=不限）。

    与 bbox_size（宽/高分别限）不同：短边口径与目标朝向无关，适合"小目标
    一刀切"（VIMO 高质量标注技巧：极小目标建议增大输入尺寸或切图训练）。
    """

    NAME = "filter_bbox_side"
    TITLE = "短长边过滤"
    CATEGORY = "检出过滤"
    PARAMS = {
        "short_min": ParamSpec("short_min", "短边下限", "float", 0.0,
                               range=(0.0, 1e6),
                               tooltip="短边（px）低于该值的检出被过滤；0=不限"),
        "long_min": ParamSpec("long_min", "长边下限", "float", 0.0,
                              range=(0.0, 1e6),
                              tooltip="长边（px）低于该值的检出被过滤；0=不限"),
    }

    def run(self, inputs, params, ctx):
        smin, lmin = float(params["short_min"]), float(params["long_min"])

        def decide(box, _g):
            d = box_dims(box)
            return (d["short"] >= smin or smin <= 0) and \
                   (d["long"] >= lmin or lmin <= 0)

        return self._run(inputs, ctx, decide)


@register_node
class FilterForeground(_ResultFilterBase):
    """前景占比过滤：框内 |灰度-背景中位| ≥ gray_thr 的像素占比 ≥ min_ratio
    才保留（"前景/背景阈值"；灰度差不足的疑似误检被过滤）。需要图像输入。"""

    NAME = "filter_foreground"
    TITLE = "前景占比过滤"
    CATEGORY = "检出过滤"
    PARAMS = {
        "min_ratio": ParamSpec("min_ratio", "最小前景占比", "float", 0.05,
                               range=(0.0, 1.0),
                               tooltip="框内前景像素占比低于该值的检出被过滤"),
        "gray_thr": ParamSpec("gray_thr", "前景灰度差阈值", "int", 20,
                              range=(1, 255),
                              tooltip="像素与背景中位灰度差 ≥ 该值记为前景"),
    }

    def run(self, inputs, params, ctx):
        if inputs.get("image") is None:
            ctx.warn("前景占比过滤：未接图像输入，本节点放行全部检出")
            return self._run(inputs, ctx, lambda *_: True)
        thr = int(params["gray_thr"])
        lo = float(params["min_ratio"])

        def decide(box, gray):
            bg = _bg_ring_median(gray, box)
            if bg is None:
                return True
            h, w = gray.shape[:2]
            x1, y1 = int(max(0, box[0])), int(max(0, box[1]))
            x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                return True
            crop = gray[y1:y2, x1:x2].astype(np.float32)
            return float((np.abs(crop - bg) >= thr).mean()) >= lo

        return self._run(inputs, ctx, decide)


@register_node
class FilterGrayDiff(_ResultFilterBase):
    """灰度差过滤：框内灰度均值与框外环形背景中位差的绝对值 ≥ min_diff
    才保留（整框"灰度差阈值"；与前景占比互补——前者看比例，本节点看整体）。"""

    NAME = "filter_gray_diff"
    TITLE = "灰度差过滤"
    CATEGORY = "检出过滤"
    PARAMS = {
        "min_diff": ParamSpec("min_diff", "最小灰度差", "float", 15.0,
                              range=(0.0, 255.0),
                              tooltip="框均值与背景中位灰度差低于该值的检出被过滤"),
    }

    def run(self, inputs, params, ctx):
        if inputs.get("image") is None:
            ctx.warn("灰度差过滤：未接图像输入，本节点放行全部检出")
            return self._run(inputs, ctx, lambda *_: True)
        lo = float(params["min_diff"])

        def decide(box, gray):
            bg = _bg_ring_median(gray, box)
            if bg is None:
                return True
            h, w = gray.shape[:2]
            x1, y1 = int(max(0, box[0])), int(max(0, box[1]))
            x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                return True
            crop = gray[y1:y2, x1:x2].astype(np.float32)
            return abs(float(crop.mean()) - bg) >= lo

        return self._run(inputs, ctx, decide)

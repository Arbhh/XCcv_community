"""裁图算子族（V0.6 T3-3 建立；修复循环 3 统一端口；修复循环 4 取消 image_list）。

端口约定（用户批注：裁图节点归通用算子；输入 image+result，输出图像；
算子内部不再带区域屏蔽/保留——筛选请先接「保留区域/屏蔽区域」过滤节点）：
- crop_by_detection「按检出裁图」：image + result → image（单张）；
- crop_by_detection_all「按检出全部裁图」：image + result → image（多张集合）；
- grid_tile「网格分块」：image（结果可选）→ image；
- grid_tile_all「网格全部分块」：image（结果可选）→ image（多张集合）。
多张集合与单张同用 image 端口（修复循环 4），下游任意节点逐张自动映射。

- 按检出裁图：选一个检出框（首个/最大/指定序号）裁出子图，可外扩边距、
  补成正方形、保证最小边长；可选把**全部**检出的裁图落盘（中文路径安全）。
  裁图清单写入输出图像 meta["crops"]，所选框写 meta["roi_rect"]（下游可据此
  把坐标映射回原图）。
- 网格分块：rows×cols 均分（可设重叠像素），输出所选分块；可选全部分块落盘；
  清单写 meta["tiles"]。

空检出 / 序号越界：输出 1×1 黑图（meta["empty"]=True）并人话警告——不输出
0×0 数组（下游 cv2/模型遇空数组会直接崩溃）。
"""

import math
from pathlib import Path

import numpy as np

from ..ir import ImageData
from ..node_base import ParamSpec, Port, VisionNode
from ..registry import register_node


def _seq(x) -> list:
    """可能为 None / list / ndarray 的序列 → list（不用真值判空，P150）。"""
    return [] if x is None else list(x)


def _empty_image(img: ImageData, reason: str, ctx) -> ImageData:
    ctx.warn(reason)
    channels = img.mat.shape[2] if img.mat.ndim == 3 else None
    shape = (1, 1, channels) if channels else (1, 1)
    meta = dict(img.meta)
    meta["empty"] = True
    meta["empty_reason"] = reason
    return ImageData(mat=np.zeros(shape, dtype=img.mat.dtype), meta=meta)


def _save_dir(params, ctx, sub: str) -> Path | None:
    out = str(params.get("save_dir") or "").strip()
    return Path(out) if out else None


def _source_stem(ctx) -> str:
    name = str(ctx.get("source_name") or "")
    return name.rsplit(".", 1)[0] if name else "image"


def _write(path: Path, mat) -> bool:
    from ...io_utils import imwrite_unicode

    return imwrite_unicode(path, mat)


# 外扩像素参数（F2-2 用户建议：外扩可按百分比或像素，两种叠加）
_MARGIN_PX = ParamSpec("margin_px", "外扩像素", "int", 0, range=(0, 4096),
                       tooltip="每侧额外外扩的像素（与外扩百分比叠加）")


def expand_box(box, w: int, h: int, margin_pct: float = 0.0,
               square: bool = False, min_side: int = 0, margin_px: int = 0) -> list:
    """检出框 → 裁剪矩形（像素、整数、在图内）：外扩（百分比 + 像素）→补方→最小边。

    先定整数尺寸再定位（避免半像素中心经 floor/ceil 多出一行）；贴边时
    把窗口**平移回图内**而不是截短——补方/最小边的尺寸要求得以保持
    （图像本身不够大时才截短）。margin_px 为每侧额外外扩像素（与百分比叠加）。
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    m = max(float(margin_pct), 0.0) / 100.0
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    px = 2.0 * max(int(margin_px), 0)
    bw, bh = bw * (1 + 2 * m) + px, bh * (1 + 2 * m) + px
    if square:
        bw = bh = max(bw, bh)
    side = float(max(int(min_side), 0))
    sw = int(math.ceil(max(bw, side) - 1e-9))
    sh = int(math.ceil(max(bh, side) - 1e-9))

    def _place(center: float, size: int, limit: int) -> tuple:
        a = int(math.floor(center - size / 2.0 + 0.5))
        b = a + size
        if a < 0:
            a, b = 0, b - a
        if b > limit:
            a, b = a - (b - limit), limit
        return max(0, a), min(limit, b)

    nx1, nx2 = _place(cx, sw, w)
    ny1, ny2 = _place(cy, sh, h)
    return [nx1, ny1, nx2, ny2]


@register_node
class CropByDetection(VisionNode):
    """按检出裁图：选一个检出框裁出子图（外扩/补方/最小边），可全部落盘。

    彩色/灰度均可（保持输入通道）。select=largest 按框面积取最大；
    index 按检出顺序（0 起）。save_dir 留空不落盘；设置后把**每个**检出
    的裁图写为 `<前缀>_<来源图名>_<序号>.png`，清单在 meta["crops"]。
    需要按区域筛选检出时，先接「保留区域/屏蔽区域」过滤节点。
    """

    NAME = "crop_by_detection"
    TITLE = "按检出裁图"
    CATEGORY = "通用"
    INPUTS = (Port("image", "image", label="图像"),
              Port("result", "result", label="结果"))
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "select": ParamSpec("select", "选取", "enum", "largest",
                            options=["first", "largest", "index"],
                            tooltip="first=首个检出；largest=面积最大；index=按序号"),
        "index": ParamSpec("index", "序号", "int", 0, range=(0, 9999),
                           tooltip="仅 select=index 时生效（0 起）"),
        "margin_pct": ParamSpec("margin_pct", "外扩边距%", "float", 0.0,
                                range=(0.0, 200.0), tooltip="每侧按框宽/高的百分比外扩"),
        "margin_px": _MARGIN_PX,
        "square": ParamSpec("square", "补成正方形", "bool", False),
        "min_side": ParamSpec("min_side", "最小边长", "int", 0, range=(0, 8192),
                              tooltip="裁图宽/高不足时以框中心外扩到该像素"),
        "save_dir": ParamSpec("save_dir", "落盘目录", "path", "", folder=True,
                              tooltip="留空不落盘；设置后保存全部检出的裁图"),
        "name_prefix": ParamSpec("name_prefix", "文件名前缀", "text", "crop"),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        det = inputs["result"]
        h, w = img.mat.shape[:2]
        boxes = [] if det is None else _seq(det.boxes)
        all_labels = _seq(getattr(det, "labels", None))  # 判空不用真值（P150）
        all_scores = _seq(getattr(det, "scores", None))
        keep = list(range(len(boxes)))   # 区域筛选已移交「保留/屏蔽区域」过滤节点
        if not boxes:
            return {"image": _empty_image(img, "按检出裁图：没有检出框，输出空图", ctx)}
        rects = [expand_box(b, w, h, params["margin_pct"], params["square"],
                            params["min_side"], params.get("margin_px", 0)) for b in boxes]
        sel = params["select"]
        if sel == "first":
            idx = 0
        elif sel == "largest":
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in
                     ([float(v) for v in bb] for bb in boxes)]
            idx = int(np.argmax(areas))
        else:
            idx = int(params["index"])
        crops = []
        out_dir = _save_dir(params, ctx, "crops")
        stem = _source_stem(ctx)
        prefix = str(params.get("name_prefix") or "crop")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            oi = keep[i]   # 原检出序号
            item = {"index": i, "det_index": oi, "rect": [x1, y1, x2, y2],
                    "label_id": int(all_labels[oi]) if oi < len(all_labels) else None,
                    "score": float(all_scores[oi]) if oi < len(all_scores) else None,
                    "file": ""}
            if out_dir is not None and x2 > x1 and y2 > y1:
                path = out_dir / f"{prefix}_{stem}_{i:02d}.png"
                if _write(path, img.mat[y1:y2, x1:x2]):
                    item["file"] = str(path)
                else:
                    ctx.warn(f"按检出裁图：写入失败 {path}")
            crops.append(item)
        if not 0 <= idx < len(rects):
            return {"image": _empty_image(
                img, f"按检出裁图：序号 {idx} 越界（共 {len(rects)} 个检出），输出空图", ctx)}
        x1, y1, x2, y2 = rects[idx]
        if x2 - x1 < 1 or y2 - y1 < 1:
            return {"image": _empty_image(img, "按检出裁图：检出框在图外，输出空图", ctx)}
        meta = dict(img.meta)
        meta.update({"roi_rect": [x1, y1, x2, y2], "crops": crops,
                     "crop_index": idx})
        return {"image": ImageData(mat=img.mat[y1:y2, x1:x2].copy(), meta=meta)}


def grid_rects(w: int, h: int, rows: int, cols: int, overlap_px: int = 0) -> list:
    """rows×cols 均分网格（行优先），每块四周外扩 overlap_px 并夹到图内。"""
    rows, cols = max(int(rows), 1), max(int(cols), 1)
    ov = max(int(overlap_px), 0)
    xs = np.linspace(0, w, cols + 1).round().astype(int)
    ys = np.linspace(0, h, rows + 1).round().astype(int)
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append([max(0, int(xs[c]) - ov), max(0, int(ys[r]) - ov),
                        min(w, int(xs[c + 1]) + ov), min(h, int(ys[r + 1]) + ov)])
    return out


@register_node
class GridTile(VisionNode):
    """网格分块：rows×cols 均分（行优先序号），输出所选分块，可全部落盘。

    彩色/灰度均可。overlap_px 为每块四周外扩的重叠像素（夹到图内），
    用于大图分块检测时避免目标被切断。清单在 meta["tiles"]。
    """

    NAME = "grid_tile"
    TITLE = "网格分块"
    CATEGORY = "通用"
    INPUTS = (Port("image", "image", label="图像"),
              Port("result", "result", required=False, label="结果"))
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "rows": ParamSpec("rows", "行数", "int", 2, range=(1, 32)),
        "cols": ParamSpec("cols", "列数", "int", 2, range=(1, 32)),
        "overlap_px": ParamSpec("overlap_px", "重叠像素", "int", 0, range=(0, 4096)),
        "index": ParamSpec("index", "输出分块序号", "int", 0, range=(0, 1023),
                           tooltip="行优先，0 起"),
        "save_dir": ParamSpec("save_dir", "落盘目录", "path", "", folder=True,
                              tooltip="留空不落盘；设置后保存全部分块"),
        "name_prefix": ParamSpec("name_prefix", "文件名前缀", "text", "tile"),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        h, w = img.mat.shape[:2]
        rects = grid_rects(w, h, params["rows"], params["cols"], params["overlap_px"])
        out_dir = _save_dir(params, ctx, "tiles")
        stem = _source_stem(ctx)
        prefix = str(params.get("name_prefix") or "tile")
        tiles = []
        for i, (x1, y1, x2, y2) in enumerate(rects):
            item = {"index": i, "rect": [x1, y1, x2, y2], "file": ""}
            if out_dir is not None and x2 > x1 and y2 > y1:
                path = out_dir / f"{prefix}_{stem}_r{i // max(int(params['cols']), 1)}" \
                                 f"c{i % max(int(params['cols']), 1)}.png"
                if _write(path, img.mat[y1:y2, x1:x2]):
                    item["file"] = str(path)
                else:
                    ctx.warn(f"网格分块：写入失败 {path}")
            tiles.append(item)
        idx = int(params["index"])
        if not 0 <= idx < len(rects):
            return {"image": _empty_image(
                img, f"网格分块：序号 {idx} 越界（共 {len(rects)} 块），输出空图", ctx)}
        x1, y1, x2, y2 = rects[idx]
        if x2 - x1 < 1 or y2 - y1 < 1:
            return {"image": _empty_image(img, "网格分块：分块为空（图像过小），输出空图", ctx)}
        meta = dict(img.meta)
        meta.update({"roi_rect": [x1, y1, x2, y2], "tiles": tiles, "tile_index": idx})
        return {"image": ImageData(mat=img.mat[y1:y2, x1:x2].copy(), meta=meta)}

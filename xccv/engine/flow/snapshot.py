"""节点调试快照：会话内存 + 降采样存储（用户决策 D6）。

- 快照只存副本：降采样（长边 ≤ max_side）后的 ndarray，原图引用零改动；
- 落盘不属于本类——磁盘导出仅限冒烟脚本（runs/_dbg_flow_smoke.py），
  避免多节点 8K 图快照把用户目录撑爆；
- overlays 与 images 同构存放（键=端口名/图层名），查看器按需取用。
"""

import base64

import cv2
import numpy as np

DEFAULT_MAX_SIDE = 1600


def downscale(img: np.ndarray, max_side: int = DEFAULT_MAX_SIDE) -> np.ndarray:
    """长边超过 max_side 时等比缩小（快照专用副本；输入不改）。"""
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return img
    scale = max_side / float(long_side)
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def to_png_base64(img: np.ndarray) -> str:
    """快照图 → PNG base64（前端 <img> 直用）。"""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def overlay_result(mat: np.ndarray, outputs: dict) -> np.ndarray:
    """把节点输出以"蒙版 + 描边"叠加到输入图副本（用户批注 ToC4/ToE2：
    细描边几乎不可见，改为半透明填充蒙版，类似分割掩码的绘制方式）。

    - Geometry：point 画大圆点、line 画粗线 + 端点、circle/rotated_rect
      画半透明填充蒙版 + 2px 描边；
    - Prediction / DetectionPlus（有 boxes 属性）：每个框画半透明填充
      蒙版 + 2px 描边；
    - 数值/测量结果不画图（已在日志与结果区展示）。
    """
    import cv2 as _cv

    from .ir import Geometry

    disp = mat if mat.ndim == 3 else _cv.cvtColor(mat, _cv.COLOR_GRAY2BGR)
    vis = disp.copy()
    green, orange = (0, 220, 60), (0, 140, 255)
    alpha = 0.38          # 蒙版不透明度（用户要求透明度较高的蒙版）

    def _blend(mask: np.ndarray, color) -> None:
        nonlocal vis
        region = vis[mask]
        if region.size:
            fill = np.empty_like(region)
            fill[:] = color
            vis[mask] = (region.astype(np.float32) * (1 - alpha)
                         + fill.astype(np.float32) * alpha).astype(np.uint8)

    h, w = vis.shape[:2]
    # 大图（长边 >1000）线宽/点径按比例放大：缩小显示时描边仍可见；小图保持原值
    s = max(1.0, max(h, w) / 1000.0)
    t_edge, t_line = max(2, round(2 * s)), max(3, round(3 * s))
    r_pt, r_dot, r_end = round(6 * s), max(2, round(2 * s)), round(5 * s)
    yy, xx = np.ogrid[:h, :w]
    values = outputs.values() if hasattr(outputs, "values") else outputs
    for v in values:
        if isinstance(v, Geometry):
            data = np.asarray(v.data, dtype=float)
            if v.kind == "point" and data.size:
                for x, y in data.reshape(-1, 2):
                    _cv.circle(vis, (int(x), int(y)), r_pt, green, t_edge, _cv.LINE_AA)
                    _cv.circle(vis, (int(x), int(y)), r_dot, green, -1, _cv.LINE_AA)
            elif v.kind == "line" and data.shape == (2, 2):
                _cv.line(vis, (int(data[0][0]), int(data[0][1])),
                         (int(data[1][0]), int(data[1][1])), green, t_line, _cv.LINE_AA)
                for px, py in data:
                    _cv.circle(vis, (int(px), int(py)), r_end, green, -1, _cv.LINE_AA)
            elif v.kind == "circle" and data.shape == (3,):
                r = max(1.0, data[2])
                mask = (xx - data[0]) ** 2 + (yy - data[1]) ** 2 <= r ** 2
                _blend(mask, green)
                _cv.circle(vis, (int(data[0]), int(data[1])), int(r),
                           green, t_edge, _cv.LINE_AA)
            elif v.kind == "rotated_rect" and data.shape == (5,):
                box = _cv.boxPoints(((data[0], data[1]),
                                     (max(1.0, data[2]), max(1.0, data[3])), data[4]))
                boxi = box.astype(np.int32)
                mask_u8 = np.zeros((h, w), np.uint8)
                _cv.fillPoly(mask_u8, [boxi], 1)
                _blend(mask_u8.astype(bool), green)
                _cv.drawContours(vis, [boxi], 0, green, t_edge, _cv.LINE_AA)
        elif getattr(v, "boxes", None) is not None and len(v.boxes) > 0:
            for b in v.boxes:
                x1, y1, x2, y2 = (int(round(float(t))) for t in b)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                roi = vis[y1:y2 + 1, x1:x2 + 1]   # 矩形蒙版就地混合（与 _blend 同式，免整图掩码）
                roi[:] = (roi.astype(np.float32) * (1 - alpha)
                          + np.asarray(orange, np.float32) * alpha).astype(np.uint8)
                _cv.rectangle(vis, (x1, y1), (x2, y2), orange, t_edge, _cv.LINE_AA)
    return vis


def mosaic(images: list, max_side: int = DEFAULT_MAX_SIDE, max_items: int = 16) -> np.ndarray:
    """多张图 → 一张拼图（列表端口的快照/预览，V0.6 F2-2）。

    近似方阵排布，每格等比缩放居中、左上角标序号；超过 max_items 只拼前
    max_items 张并在末格标注剩余数。空列表返回 1×1 黑图（不返回 0×0）。
    """
    import math

    import cv2 as _cv

    imgs = [np.asarray(m) for m in images if m is not None and np.asarray(m).size]
    if not imgs:
        return np.zeros((1, 1, 3), np.uint8)
    total = len(imgs)
    imgs = imgs[:max_items]
    n = len(imgs)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    cell = max(32, int(max_side) // max(cols, rows))
    pad = 4
    canvas = np.full((rows * cell, cols * cell, 3), 40, np.uint8)
    for k, m in enumerate(imgs):
        if m.dtype != np.uint8:
            m = _cv.normalize(m.astype(np.float32), None, 0, 255, _cv.NORM_MINMAX).astype(np.uint8)
        if m.ndim == 2:
            m = _cv.cvtColor(m, _cv.COLOR_GRAY2BGR)
        elif m.shape[2] == 4:
            m = _cv.cvtColor(m, _cv.COLOR_BGRA2BGR)
        h, w = m.shape[:2]
        s = (cell - 2 * pad) / float(max(h, w))
        tw, th = max(1, int(w * s)), max(1, int(h * s))
        thumb = _cv.resize(m, (tw, th), interpolation=_cv.INTER_AREA if s < 1 else _cv.INTER_LINEAR)
        r, c = divmod(k, cols)
        y0 = r * cell + (cell - th) // 2
        x0 = c * cell + (cell - tw) // 2
        canvas[y0:y0 + th, x0:x0 + tw] = thumb
        label = f"#{k}" if not (k == n - 1 and total > n) else f"#{k} (+{total - n})"
        fs = max(0.4, cell / 260.0)
        _cv.putText(canvas, label, (c * cell + 6, r * cell + int(22 * fs) + 4),
                    _cv.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, _cv.LINE_AA)
        _cv.putText(canvas, label, (c * cell + 6, r * cell + int(22 * fs) + 4),
                    _cv.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, _cv.LINE_AA)
    return canvas


class SnapshotStore:
    """按 node_id 存最近一次快照（输入/输出图 + 叠加层），会话内存态。"""

    def __init__(self, max_side: int = DEFAULT_MAX_SIDE):
        self._items: dict = {}
        self._max_side = int(max_side)

    def save(self, node_id: int, images: dict, overlays: dict | None = None) -> None:
        """images: {名称: ndarray}；overlays: {名称: [叠加元素 dict]}（可选）。

        快照必须存副本：未触发降采样时 downscale 返回原引用，这里补拷贝，
        保证快照与运行数据隔离（D6 契约）。
        """
        item: dict = {"images": {}, "overlays": dict(overlays or {})}
        for k, v in (images or {}).items():
            if v is None:
                continue
            arr = np.asarray(v)
            snap = downscale(arr, self._max_side)
            if snap is arr:
                snap = arr.copy()
            item["images"][k] = snap
        self._items[int(node_id)] = item

    def get(self, node_id: int):
        return self._items.get(int(node_id))

    def node_ids(self) -> list:
        return sorted(self._items)

    def clear(self) -> None:
        self._items.clear()

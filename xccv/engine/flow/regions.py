"""检出过滤区域（V0.6 T3-1）：矩形 / 圆 / 多边形三种形状，旧串零改动兼容。

区域串语法（多条区域以 ``;`` 分隔，可混合形状；中文逗号自动当英文逗号）：

- ``x1,y1,x2,y2``                 旧格式（无前缀）＝ rect——历史流程文件零改动
- ``rect:x1,y1,x2,y2``            矩形
- ``circle:cx,cy,r``              圆；**归一化半径按短边换算**：r_px = r × min(W, H)
- ``polygon:x1,y1,x2,y2,x3,y3…``  多边形（≥3 个顶点，按顺序连成闭合轮廓）

列表形态：``[[x1,y1,x2,y2], ...]``（程序/框选生成的矩形，旧行为）亦可，
列表元素也可以是上述字符串。

坐标规则（沿用旧链）：一条区域内任一数值 >1 视为**像素坐标**，否则为
**0~1 归一化坐标**（按图像宽高换算）。

命中判定（以检出框中心点为准）：rect 为闭区间包含（与旧链逐位一致）；
circle 为到圆心距离 ≤ 半径；polygon 为 ``cv2.pointPolygonTest ≥ 0``
（边界算命中）。

两套入口：``parse_regions`` 运行期宽容解析（坏条目跳过，不让整条流程
崩溃——旧链语义）；``region_problems`` 打包期严格校验（每条给出中文
问题描述，用于 .xvap 打包把关）。本模块不依赖 Qt / torch。
"""

from dataclasses import dataclass

import numpy as np

SHAPES = ("rect", "circle", "polygon")
_EPS = 1e-9


@dataclass(frozen=True)
class Region:
    """一条区域：shape + 原始数值（像素或归一化，pixel 标记口径）。"""

    shape: str
    values: tuple
    pixel: bool

    def to_pixels(self, w: int, h: int):
        """换算到像素坐标：rect→(x1,y1,x2,y2)；circle→(cx,cy,r)；polygon→(N,2)。"""
        v = self.values
        if self.shape == "rect":
            x1, y1, x2, y2 = v if self.pixel else (v[0] * w, v[1] * h, v[2] * w, v[3] * h)
            return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        if self.shape == "circle":
            if self.pixel:
                return v[0], v[1], abs(v[2])
            return v[0] * w, v[1] * h, abs(v[2]) * min(w, h)
        pts = np.asarray(v, dtype=np.float32).reshape(-1, 2)
        if not self.pixel:
            pts = pts * np.array([w, h], dtype=np.float32)
        return pts

    def contains(self, x: float, y: float, w: int, h: int) -> bool:
        """点 (x, y)（像素）是否落在区域内（边界算命中）。"""
        if self.shape == "rect":
            # 与旧链 `_parse_regions`+`_in_any_region` 逐位同式：像素坐标先除以
            # 宽高归一化，再以 `x1*w <= cx <= x2*w` 判定（浮点舍入口径一致）
            nx1, ny1, nx2, ny2 = self.values
            if self.pixel:
                nx1, nx2, ny1, ny2 = nx1 / w, nx2 / w, ny1 / h, ny2 / h
            nx1, nx2 = min(nx1, nx2), max(nx1, nx2)
            ny1, ny2 = min(ny1, ny2), max(ny1, ny2)
            return nx1 * w <= x <= nx2 * w and ny1 * h <= y <= ny2 * h
        if self.shape == "circle":
            cx, cy, r = self.to_pixels(w, h)
            return (x - cx) ** 2 + (y - cy) ** 2 <= r * r
        import cv2

        pts = self.to_pixels(w, h)
        return cv2.pointPolygonTest(pts.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0


def _split_entries(value) -> list:
    if isinstance(value, str):
        return [s.strip() for s in value.split(";") if s.strip()]
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _parse_entry(entry):
    """单条 → (shape, numbers) ；格式错误抛 ValueError（中文原因）。"""
    if isinstance(entry, str):
        text = entry.strip().replace("，", ",").replace("：", ":")
        shape = "rect"
        if ":" in text:
            prefix, _, text = text.partition(":")
            shape = prefix.strip().lower()
            if shape not in SHAPES:
                raise ValueError(f"未知区域形状「{prefix.strip()}」（支持 rect/circle/polygon）")
        try:
            nums = [float(t) for t in text.split(",") if t.strip()]
        except ValueError:
            raise ValueError("不是有效的数字") from None
        return shape, nums
    if isinstance(entry, (list, tuple)):
        try:
            return "rect", [float(t) for t in entry]
        except (TypeError, ValueError):
            raise ValueError("不是有效的数字") from None
    raise ValueError("格式无效")


def _check_shape(shape: str, nums: list) -> str | None:
    """形状数值合法性：返回中文问题描述或 None。"""
    if shape == "rect":
        if len(nums) != 4:
            return "不是 4 个数字（应为 x1,y1,x2,y2）"
        x1, y1, x2, y2 = nums
        if x2 <= x1 or y2 <= y1:
            return "宽或高为零/为负"
        return None
    if shape == "circle":
        if len(nums) != 3:
            return "圆应为 3 个数字（circle:cx,cy,r）"
        if nums[2] <= 0:
            return "圆半径须大于零"
        return None
    if len(nums) < 6 or len(nums) % 2:
        return "多边形至少 3 个顶点且坐标成对（polygon:x1,y1,x2,y2,x3,y3…）"
    pts = np.asarray(nums, dtype=np.float64).reshape(-1, 2)
    area = 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1))
                     - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
    if area <= 0:
        return "多边形面积为零（顶点共线或重合）"
    return None


def _make_region(shape: str, nums: list) -> Region:
    pixel = max(abs(v) for v in nums) > 1.0 + _EPS
    if shape == "rect":
        x1, y1, x2, y2 = nums
        nums = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return Region(shape, tuple(nums), pixel)


def parse_regions(value) -> list:
    """运行期宽容解析：坏条目跳过（单条损坏只影响该条，旧链语义）。"""
    out = []
    for entry in _split_entries(value):
        try:
            shape, nums = _parse_entry(entry)
        except ValueError:
            continue
        if shape == "rect":
            if len(nums) != 4:
                continue  # 旧链：零宽高的矩形照常参与（命中退化为线/点）
        elif _check_shape(shape, nums) is not None:
            continue
        out.append(_make_region(shape, nums))
    return out


def any_contains(regions, x: float, y: float, w: int, h: int) -> bool:
    """点是否落在任一区域内；regions 可为原始参数或已解析的 Region 列表。"""
    if not (isinstance(regions, list) and all(isinstance(r, Region) for r in regions)):
        regions = parse_regions(regions)
    return any(r.contains(x, y, w, h) for r in regions)


def region_problems(value, context: str = "") -> list:
    """打包期严格校验：返回问题描述列表（空=合法）。

    与旧 ``region_param_problems`` 文案逐字兼容（矩形条目），并覆盖新形状。
    """
    if not isinstance(value, (str, list, tuple)):
        return [f"{context}区域参数类型无效"] if value else []  # 旧链口径：假值视为未设置
    problems = []
    for i, entry in enumerate(_split_entries(value), start=1):
        where = f"{context}第 {i} 条区域"
        try:
            shape, nums = _parse_entry(entry)
        except ValueError as e:
            problems.append(f"{where}{e}")
            continue
        msg = _check_shape(shape, nums)
        if msg:
            problems.append(f"{where}{msg}")
    return problems

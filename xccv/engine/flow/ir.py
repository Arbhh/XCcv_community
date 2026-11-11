"""数据中间表示（IR）：类型化端口的数据契约 + 端口类型表。

用户批注 Q1/Q5 的地基：几何/测量/数值结果可序列化、可被 C# 等外部
接口稳定消费；图像本体不进 JSON（快照走 base64 PNG 旁路）。

约定（本包纪律，测试有断言）：
- 图像（ImageData.mat）一律 ndarray 引用传递，执行器与节点实现不得对
  全图 .copy()——"零拷贝"约束执行器与节点作者，算法输出天然是新数组；
- Geometry.data 形状契约见类 docstring，全库算子必须遵守。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# 端口类型表：前后端共享的唯一事实源（前端经桥接下发，JS 不得重复维护一份）
#
# V0.6 修复循环 3（用户批注：统一"度量衡"）：端口收敛为四类；
# V0.6 修复循环 4（用户批注：图像/图像列表之分"极其不方便"）：**取消
# image_list 端口类型**，端口只余三类——
# - image：图像。一个图像端口可承载**单张或多张**（ImageList 集合，裁图/
#   分块族生产、逐张带原图坐标）；下游任意节点收到多张时由执行器**逐张
#   自动映射**（并行输入，节点实现无需感知"列表"，见 executor._map_over）；
# - result：统一结果（检出/分类/几何/测量/数值/布尔/逐图结果列表全部走它，
#   节点间不再按 detection/geometry/… 逐点连线）；
# - number：数值（测量节点的长度、计数节点的数量等"附加类型"，
#   判定/逻辑节点的统一输入）。
# 各节点族端口约定见 ops 各模块 docstring（检出=image→image+result，
# 过滤=image+result→image+result+被滤除，修改=image+result→image+result，
# 裁图=image+result→image，预处理=image→image）。
PORT_TYPES = {
    "image":      {"color": "#4a9eff", "desc": "图像"},
    "result":     {"color": "#f5a623",
                   "desc": "统一结果（检出/分类/几何/测量/数值/布尔，格式统一）"},
    "number":     {"color": "#9b9b9b",
                   "desc": "数值（测量长度/计数数量；判定与逻辑节点的统一输入）"},
}


@dataclass
class Geometry:
    """几何结果。

    data 形状契约（kind → np.ndarray 形状，全库算子必须遵守）：
    - point:        (N, 2) float —— N 个点 [x, y]
    - line:         (2, 2) float —— 两端点 [x, y]
    - circle:       (3,)   float —— [cx, cy, r]
    - polygon:      (N, 2) float —— 顶点序列（不重复首点，闭合由绘制侧处理）
    - rotated_rect: (5,)   float —— [cx, cy, w, h, angle_deg]（角度为度）
    """

    KINDS = ("point", "line", "circle", "polygon", "rotated_rect")

    kind: str
    data: np.ndarray
    subpixel: bool = False

    def __post_init__(self):
        if self.kind not in self.KINDS:
            raise ValueError(f"未知几何类型 {self.kind!r}，允许：{self.KINDS}")

    def to_json(self) -> dict:
        return {"kind": self.kind, "data": np.asarray(self.data).tolist(),
                "subpixel": bool(self.subpixel)}

    @classmethod
    def from_json(cls, d: dict) -> "Geometry":
        return cls(kind=str(d["kind"]),
                   data=np.asarray(d["data"], dtype=float),
                   subpixel=bool(d.get("subpixel", False)))


@dataclass
class MeasureResult:
    """测量结果：单点值 value 与多点序列 values 二选一（勿混用）。

    unit 常用 "px" | "mm"（mm 需标定，二期）；角度测量为 "deg"（V0.6 T4-2）；
    tolerance=(下限, 上限) 时 passed 给出判定，无 tolerance 时 passed 为 None。
    """

    value: Optional[float] = None
    values: Optional[list] = None          # list[float]
    unit: str = "px"
    tolerance: Optional[tuple] = None
    passed: Optional[bool] = None

    def to_json(self) -> dict:
        return {"value": self.value, "values": list(self.values or []),
                "unit": self.unit,
                "tolerance": list(self.tolerance) if self.tolerance else None,
                "passed": self.passed}

    @classmethod
    def from_json(cls, d: dict) -> "MeasureResult":
        tol = d.get("tolerance")
        return cls(value=d.get("value"), values=list(d.get("values") or []),
                   unit=str(d.get("unit", "px")),
                   tolerance=(float(tol[0]), float(tol[1])) if tol else None,
                   passed=d.get("passed"))


@dataclass
class NumberVal:
    """带单位的标量（逻辑/测量节点间的数值通行证）。"""

    value: float
    unit: str = ""

    def to_json(self) -> dict:
        return {"value": float(self.value), "unit": self.unit}

    @classmethod
    def from_json(cls, d: dict) -> "NumberVal":
        return cls(value=float(d["value"]), unit=str(d.get("unit", "")))


@dataclass
class ImageData:
    """流程中的图像：BGR/GRAY uint8 ndarray 引用传递（禁止 float 传递）。

    meta 携带跨节点元信息（scale mm/px、channel、source_node 等）；
    roi 为可选的关联几何（ROI 提取类节点填写）。
    """

    mat: np.ndarray
    roi: Optional[Geometry] = None
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        """序列化仅覆盖 meta/roi/shape——图像本体不进 JSON（契约）。"""
        return {"shape": [int(v) for v in self.mat.shape],
                "roi": self.roi.to_json() if self.roi is not None else None,
                "meta": dict(self.meta)}

    @classmethod
    def from_json(cls, d: dict) -> "ImageData":
        """反序列化只重建 meta/roi 外壳；mat 为空数组（图像不来自 JSON）。"""
        roi = Geometry.from_json(d["roi"]) if d.get("roi") else None
        return cls(mat=np.empty(0, dtype=np.uint8), roi=roi,
                   meta=dict(d.get("meta") or {}))


@dataclass
class ImageList:
    """多张图像（V0.6 F2-2，裁图/分块族生产）。

    修复循环 4 起不再是端口类型：ImageList 作为 image 端口的**数据载体**
    在图上传递（单张=ImageData，多张=ImageList）；下游节点由执行器逐张
    自动映射（并行输入）。每个 item 是 ImageData，其 meta["roi_rect"]=
    [x1,y1,x2,y2] 为该图在**原图**中的像素矩形（合并回原图坐标的唯一
    依据）；meta 可带 source_size=[W,H]。
    """

    items: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)

    def to_json(self) -> dict:
        return {"count": len(self.items),
                "items": [it.to_json() for it in self.items],
                "meta": dict(self.meta)}


def result_kind(value) -> str:
    """统一结果"度量衡"的种类判定（V0.6 F2-2，用户建议：各节点输出统一格式）。

    detection（AI 检出：框/掩码）、classification（AI 分类：无框有类别概率）、
    geometry、measure、number、bool、image；其余 unknown。
    """
    if isinstance(value, Geometry):
        return "geometry"
    if isinstance(value, MeasureResult):
        return "measure"
    if isinstance(value, NumberVal):
        return "number"
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, ImageData):
        return "image"
    boxes = getattr(value, "boxes", None)
    if boxes is not None:
        probs = getattr(value, "probs", None)
        if len(boxes) == 0 and probs is not None and len(probs) > 0:
            return "classification"
        return "detection"
    return "unknown"


@dataclass
class ResultItem:
    """统一结果记录：一张图（裁图/分块）上一个节点的一个输出。

    value 保持各自 IR 类型（Prediction/Geometry/MeasureResult/NumberVal/bool），
    坐标一律是**该图自身**的像素坐标；roi_rect=[x1,y1,x2,y2] 记录该图在原图中
    的位置（None=本就是原图）——「结果合并」据此统一换算到原图像素坐标。
    """

    kind: str
    value: object
    roi_rect: Optional[list] = None
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        v = self.value
        if hasattr(v, "to_json"):
            vj = v.to_json()
        elif self.kind in ("detection", "classification"):
            vj = {"count": len(getattr(v, "boxes", None) or [])}
        else:
            vj = v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)
        return {"kind": self.kind, "roi_rect": list(self.roi_rect) if self.roi_rect else None,
                "value": vj, "meta": dict(self.meta)}


@dataclass
class ResultList:
    """结果列表（与 ImageList 逐项对应，V0.6 F2-2）：items 为 ResultItem。

    meta 可带 source_size=[W,H]（掩码回贴原图尺寸用）。AI 逐图推理与传统算子
    逐图执行都产出本类型——「结果合并」一个节点统一合并（用户建议 result_list）。
    """

    items: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def values(self) -> list:
        return [it.value for it in self.items]

    def to_json(self) -> dict:
        return {"count": len(self.items), "items": [it.to_json() for it in self.items],
                "meta": dict(self.meta)}

"""旧流程体系适配层（批次十七）。

- sniff_format：流程 JSON 三格式嗅探（链式 / 旧图格式 / 新 IR 图格式）；
- chain_to_xcv_graph：过滤链 → 新 IR 图（8 个薄委托过滤节点串行）；
- 8 个过滤节点：run() 内**薄委托** df.apply_flow 的单步语义，零逻辑复制
  （detection_flow.apply_flow 本体不改）；新旧引擎同链输出的 Prediction
  完全一致由等价性测试锁定；
- apply_flow_safe：apply_flow 的分类安全包装——Q2 修复：分类 Prediction
  （无框、有 probs）不再经过 apply_flow（它只回传 labels/scores/boxes，
  probs 会丢失导致分类流程图无结果输出），原样直通。

导入纪律：detection_flow 牵引 torch（经 models.base），必须全部惰性导入，
保持 engine.flow 的轻导入（五平台冷启动不受影响）。
"""

from .ir import NumberVal
from .node_base import ParamSpec, Port, VisionNode
from .registry import register_node


def sniff_format(flow: dict) -> str:
    """识别流程 JSON 格式：'chain'（op/params 链）/ 'legacy_graph'
    （旧画布 model·post 节点图）/ 'xcv_graph'（新 IR 注册节点图）。"""
    nodes = (flow or {}).get("nodes") or []
    if not nodes:
        return "chain"
    first = nodes[0]
    if not isinstance(first, dict):
        return "chain"
    if "op" in first:
        return "chain"
    t = str(first.get("type", ""))
    if t.startswith(("xccv2/", "model/", "xccv/model", "post/", "xccv/post_")):
        return "xcv_graph" if t.startswith("xccv2/") else "legacy_graph"
    if t:
        from .registry import get_node_class

        if get_node_class(t) is not None:
            return "xcv_graph"
    return "chain"


def chain_to_xcv_graph(chain: dict, start_id: int = 1,
                       with_source: bool = True) -> dict:
    """过滤链 {"nodes":[{op,params}...]} → 新 IR 图（过滤节点串行）。

    with_source 时最前面挂一个 detection_source 源节点（真实流程中检出
    来自 AI 模型节点；链式迁移/等价性测试/工作台组合测试用它供检出）。
    旧图格式先经 detection_flow.graph_to_chain 转链再进来（读取入口
    统一走 df.load_flow 的既有纪律不变）。
    """
    from .registry import get_node_class

    def _slot(node_type: str, out: bool) -> int:
        """result 端口槽位（修复循环 4 重排后按名解析，不再写死 0）。"""
        cls = get_node_class(node_type)
        ports = (cls.OUTPUTS if out else cls.INPUTS)
        for i, p in enumerate(ports):
            if p.name == "result":
                return i
        return 0

    nodes, links, prev = [], [], None
    prev_type = ""
    nid = start_id
    if with_source and chain.get("nodes"):
        nodes.append({"id": nid, "type": "detection_source", "properties": {}})
        prev, prev_type = nid, "detection_source"
        nid += 1
    for n in chain.get("nodes", []):
        node_type = f"filter_{n.get('op')}"
        if get_node_class(node_type) is None:
            continue  # 未知过滤节点跳过（与 normalize_flow 的宽容口径一致）
        nodes.append({"id": nid, "type": node_type,
                      "properties": dict(n.get("params") or {})})
        if prev is not None:
            links.append([len(links) + 1, prev, _slot(prev_type, out=True),
                          nid, _slot(node_type, out=False)])
        prev, prev_type = nid, node_type
        nid += 1
    return {"nodes": nodes, "links": links}


def _image_size(pred, img) -> tuple:
    """过滤节点用的图幅：(宽, 高)。优先 image 端口，其次掩码尺寸，
    最后 bbox 外扩（仅影响像素坐标区域的归一化口径）。"""
    if img is not None and getattr(img.mat, "ndim", 0) >= 2:
        h, w = img.mat.shape[:2]
        return int(w), int(h)
    if pred.masks is not None and len(pred.masks) > 0:  # 不用真值（ndarray 歧义，P150）
        h, w = pred.masks[0].shape[:2]
        return int(w), int(h)
    if len(pred.boxes):
        xs = [b[2] for b in pred.boxes]
        ys = [b[3] for b in pred.boxes]
        return int(max(xs)) + 1, int(max(ys)) + 1
    return (1, 1)


class _FilterNodeBase(VisionNode):
    """旧 8 节点的薄委托基类：单步调用 df.apply_flow（零逻辑复制）。

    端口约定（修复循环 3 统一；修复循环 4 图像在上）：输入 = 图像（可选）
    + 结果；输出 = 图像 + 结果 + 被滤除（图像原样直通，供预览/落盘）。
    """

    OP = ""                       # 对应 detection_flow 的 op 名
    PARAM_KEYS: tuple = ()        # 透传给 apply_flow 的参数键

    def _flow(self, params: dict) -> dict:
        return {"nodes": [{"op": self.OP,
                           "params": {k: params[k] for k in self.PARAM_KEYS}}]}

    def run(self, inputs, params, ctx):
        from ..detection_flow import apply_flow  # 惰性：牵引 torch 的链路

        pred = inputs["result"]
        classes = ctx.get("model_classes")
        new_pred, info = apply_flow(pred, _image_size(pred, inputs.get("image")),
                                    self._flow(params), classes=classes)
        if info.get("removed"):
            ctx.warn(f"{self.OP}：移除 {info['removed']} 个检出")
        return self._outputs(new_pred, info, params, inputs)

    def _outputs(self, pred, info, params, inputs) -> dict:
        out = {"result": pred, "removed": info.get("removed_prediction")}
        img = inputs.get("image")
        if img is not None:
            out["image"] = img
        return out


# V0.6 T3-2：过滤算子 removed 端口（被滤掉的检出，additive）；修复循环 3 追加
# 图像直通输出；修复循环 4 统一「图像在上、结果在下」（旧图经迁移表重排槽位）
_REMOVED_PORT = Port("removed", "result", label="被滤除")
_FILTER_OUTPUTS = (Port("image", "image", label="图像"),
                   Port("result", "result", label="结果"),
                   _REMOVED_PORT)


def _filter_ports():
    return (Port("image", "image", required=False, label="图像"),
            Port("result", "result", label="结果"))


@register_node
class FilterConfidence(_FilterNodeBase):
    """置信度过滤（旧链 confidence 的薄委托）。"""

    NAME = "filter_confidence"
    TITLE = "置信度过滤"
    CATEGORY = "检出过滤"
    INPUTS = _filter_ports()
    OUTPUTS = _FILTER_OUTPUTS
    OP = "confidence"
    PARAM_KEYS = ("min",)
    PARAMS = {
        "min": ParamSpec("min", "置信度下限", "float", 0.25, range=(0.0, 1.0)),
    }


@register_node
class FilterClass(_FilterNodeBase):
    """类别过滤（旧链 class_filter 的薄委托）。

    类别名表优先取 ctx 共享键 model_classes（AI 模型节点会写入），
    没有时按标签序号匹配名称失败=不过滤（与旧链口径一致）。
    """

    NAME = "filter_class"
    TITLE = "类别过滤"
    CATEGORY = "检出过滤"
    INPUTS = _filter_ports()
    OUTPUTS = _FILTER_OUTPUTS
    OP = "class_filter"
    PARAM_KEYS = ("keep",)
    PARAMS = {
        "keep": ParamSpec("keep", "保留类别", "text", "",
                          tooltip="逗号分隔的类别名；留空=全部保留"),
    }


@register_node
class FilterBboxSize(_FilterNodeBase):
    """bbox 宽高过滤（旧链 bbox_size 的薄委托）；0 = 不限制。"""

    NAME = "filter_bbox_size"
    TITLE = "宽高过滤"
    CATEGORY = "检出过滤"
    INPUTS = _filter_ports()
    OUTPUTS = _FILTER_OUTPUTS
    OP = "bbox_size"
    PARAM_KEYS = ("w_min", "w_max", "h_min", "h_max")
    PARAMS = {
        "w_min": ParamSpec("w_min", "宽下限", "int", 0, range=(0, 10 ** 6)),
        "w_max": ParamSpec("w_max", "宽上限", "int", 0, range=(0, 10 ** 6)),
        "h_min": ParamSpec("h_min", "高下限", "int", 0, range=(0, 10 ** 6)),
        "h_max": ParamSpec("h_max", "高上限", "int", 0, range=(0, 10 ** 6)),
    }


@register_node
class FilterBboxArea(_FilterNodeBase):
    """bbox 面积过滤（旧链 bbox_area 的薄委托）；0 = 不限制。"""

    NAME = "filter_bbox_area"
    TITLE = "面积过滤"
    CATEGORY = "检出过滤"
    INPUTS = _filter_ports()
    OUTPUTS = _FILTER_OUTPUTS
    OP = "bbox_area"
    PARAM_KEYS = ("min", "max")
    PARAMS = {
        "min": ParamSpec("min", "面积下限", "int", 0, range=(0, 10 ** 9)),
        "max": ParamSpec("max", "面积上限", "int", 0, range=(0, 10 ** 9)),
    }


@register_node
class FilterMaskArea(_FilterNodeBase):
    """mask 面积过滤（旧链 mask_area 的薄委托）；无掩码时不过滤。"""

    NAME = "filter_mask_area"
    TITLE = "掩码面积过滤"
    CATEGORY = "检出过滤"
    INPUTS = _filter_ports()
    OUTPUTS = _FILTER_OUTPUTS
    OP = "mask_area"
    PARAM_KEYS = ("min", "max")
    PARAMS = {
        "min": ParamSpec("min", "面积下限", "int", 0, range=(0, 10 ** 9)),
        "max": ParamSpec("max", "面积上限", "int", 0, range=(0, 10 ** 9)),
    }


@register_node
class FilterCountCheck(_FilterNodeBase):
    """数量判定（旧链 count_check 的薄委托）：不过滤，输出判定与数量。

    判定随统一结果 JSON 的 verdict 字段对外可读（OK/NG）；检出本体照常
    向下传（result 端口），count 数值端口供判定/逻辑节点使用。
    """

    NAME = "filter_count_check"
    TITLE = "数量判定"
    CATEGORY = "检出过滤"
    INPUTS = _filter_ports()
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"),
               Port("count", "number", label="数量"),
               _REMOVED_PORT)
    OP = "count_check"
    PARAM_KEYS = ("min", "max")
    PARAMS = {
        "min": ParamSpec("min", "数量下限", "int", 0, range=(0, 10 ** 6),
                         tooltip="0 = 不限下限"),
        "max": ParamSpec("max", "数量上限", "int", 0, range=(0, 10 ** 6),
                         tooltip="0 = 不限上限"),
    }

    def validate(self, params):
        errs = super().validate(params)
        lo, hi = int(params.get("min") or 0), int(params.get("max") or 0)
        if lo and hi and lo > hi:
            errs.append(f"数量判定下限 {lo} 大于上限 {hi}")
        return errs

    def _outputs(self, pred, info, params, inputs) -> dict:
        from .ops.ai_bridge import DetectionPlus

        verdict = info.get("count_check") != "NG"
        wrapped = DetectionPlus(pred=pred, extra={"verdict": verdict})
        out = {"result": wrapped,
               "count": NumberVal(value=float(len(getattr(pred, "boxes", []) or [])),
                                  unit="个"),
               "removed": info.get("removed_prediction")}
        img = inputs.get("image")
        if img is not None:
            out["image"] = img
        return out


class _RegionFilterBase(_FilterNodeBase):
    """区域类过滤基类：区域参数按形状严格校验（打包口径；矩形/圆/多边形，见 flow.regions）。"""

    def validate(self, params):
        errs = super().validate(params)
        from ..detection_flow import region_param_problems  # 惰性：牵引 torch

        errs += region_param_problems(params.get("regions"))
        return errs


@register_node
class FilterRegionBlock(_RegionFilterBase):
    """屏蔽区域（旧链 region_block 的薄委托）：区域内目标被过滤。

    区域多区并集；坐标支持归一化列表或像素串（口径与旧链一致）。
    V0.6 T3-1：支持 rect:/circle:/polygon: 前缀（无前缀=矩形，旧串零改动）。
    image 端口（可选）提供像素坐标归一化所需的图幅。
    """

    NAME = "filter_region_block"
    TITLE = "屏蔽区域"
    CATEGORY = "检出过滤"
    INPUTS = _filter_ports()
    OUTPUTS = _FILTER_OUTPUTS
    OP = "region_block"
    PARAM_KEYS = ("regions",)
    PARAMS = {
        "regions": ParamSpec("regions", "屏蔽区域", "text", "",
                             tooltip="多区域以 ; 分隔：x1,y1,x2,y2（矩形，可加 rect: 前缀）／circle:cx,cy,r／polygon:x1,y1,x2,y2,x3,y3…；数值>1 为像素，否则 0~1 归一化（圆半径按短边）"),
    }


@register_node
class FilterRegionKeep(_RegionFilterBase):
    """保留区域（旧链 region_keep 的薄委托）：仅保留区域内目标。"""

    NAME = "filter_region_keep"
    TITLE = "保留区域"
    CATEGORY = "检出过滤"
    INPUTS = _filter_ports()
    OUTPUTS = _FILTER_OUTPUTS
    OP = "region_keep"
    PARAM_KEYS = ("regions",)
    PARAMS = {
        "regions": ParamSpec("regions", "保留区域", "text", "",
                             tooltip="多区域以 ; 分隔：x1,y1,x2,y2（矩形，可加 rect: 前缀）／circle:cx,cy,r／polygon:x1,y1,x2,y2,x3,y3…；数值>1 为像素，否则 0~1 归一化（圆半径按短边）"),
    }


@register_node
class DetectionSource(VisionNode):
    """检出源（无输入 → 结果）：供过滤链调试 / 等价性测试 / 组合测试。

    - ctx 共享键 inject_detection 存在时**零拷贝直通**该对象（程序注入）；
    - 否则从统一检出结果 JSON（build_result_json 导出物，schema_version=1）
      读取 detections 重建 Prediction（掩码不含在 JSON 中，为 None）。
    """

    NAME = "detection_source"
    TITLE = "检出源"
    CATEGORY = "图像源"
    INPUTS = ()
    OUTPUTS = (Port("result", "result", label="结果"),)
    PARAMS = {
        "path": ParamSpec("path", "检出结果 JSON", "path", "",
                          tooltip="统一检出结果 JSON 路径；留空时须由程序注入"
                                  "（ctx: inject_detection）"),
    }

    def run(self, inputs, params, ctx):
        import json
        from pathlib import Path

        from ..models.base import Prediction  # 惰性：牵引 torch 的链路

        inj = ctx.get("inject_detection")
        if inj is not None:
            return {"result": inj}
        path = str(params.get("path") or "")
        if not path:
            raise ValueError("未提供检出：请填写检出结果 JSON 路径，或由程序注入")
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ValueError(f"无法读取检出结果 JSON：{path}（{e}）") from e
        dets = data.get("detections") or []
        return {"result": Prediction(
            labels=[int(d.get("label_id", 0)) for d in dets],
            scores=[float(d.get("score", 0.0)) for d in dets],
            boxes=[[float(v) for v in d.get("bbox", [0, 0, 0, 0])] for d in dets])}


def apply_flow_safe(pred, image_size, flow, classes=None):
    """apply_flow 的分类安全包装（Q2 修复）。

    分类 Prediction（无框 + 有 probs）原样直通（零拷贝），info 标记
    classification=True；检测/分割照常委托 df.apply_flow。
    """
    if getattr(pred, "probs", None) is not None and len(pred.boxes) == 0:
        return pred, {"steps": [], "count_check": None, "removed": 0,
                      "classification": True}
    from ..detection_flow import apply_flow  # 惰性：牵引 torch 的链路

    return apply_flow(pred, image_size, flow, classes=classes)

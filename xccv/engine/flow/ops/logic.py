"""逻辑类算子（4 个，批次十五；修复循环 3 端口统一）。

- compare_value：数值比较判定（支持回差，状态存 RunContext 会话）；
  输入=数值（与测量/计数节点的数值输出对应），输出=统一结果（值=布尔判定，
  结果 JSON 的 verdict 可读出 OK/NG）；
- math_op：数值运算；除零输出 NaN 并在 ctx 记人话警告（不静默放行，
  避免下游 compare 恰好判过的产线事故）；
- condition_branch：布尔选择两路图像之一（零拷贝直通）；
- result_merge：多路测量结果聚合（全过/任过/NG 计数/均值/最大/最小）；
- bool_op：布尔运算（输入为携带布尔的统一结果）。
"""

import math

from ..ir import MeasureResult, NumberVal
from ..node_base import ParamSpec, Port, RunContext, VisionNode
from ..registry import register_node


@register_node
class CompareValue(VisionNode):
    """数值比较判定：NaN 输入恒判 False 并警告（防除零链路误通过）。"""

    NAME = "compare_value"
    TITLE = "数值判定"
    CATEGORY = "逻辑"
    INPUTS = (Port("number", "number", label="数值"),)
    OUTPUTS = (Port("result", "result", label="判定"),)
    PARAMS = {
        "op": ParamSpec("op", "比较", "enum", "gt",
                        options=["gt", "lt", "ge", "le", "eq", "ne"]),
        "threshold": ParamSpec("threshold", "阈值", "float", 0.0),
        "hysteresis": ParamSpec("hysteresis", "回差", "float", 0.0,
                                range=(0.0, 1e9),
                                tooltip=">0 时启用滞回：触发需超阈+回差，"
                                        "保持只需不跌破阈-回差（防抖动）"),
    }

    _CMP = {"gt": lambda a, b: a > b, "lt": lambda a, b: a < b,
            "ge": lambda a, b: a >= b, "le": lambda a, b: a <= b,
            "eq": lambda a, b: a == b, "ne": lambda a, b: a != b}

    def run(self, inputs, params, ctx):
        num = inputs["number"]
        value = float(num.value) if num is not None else float("nan")
        if math.isnan(value):
            ctx.warn("compare_value 收到 NaN（上游可能除零），判定为不通过")
            return {"result": False}
        op = self._CMP[params["op"]]
        th = float(params["threshold"])
        hys = float(params.get("hysteresis") or 0.0)
        if hys > 0 and params["op"] in ("gt", "ge"):
            key = f"hysteresis:{self.NAME}"
            prev = bool(ctx.get(key, False))
            eff = th + hys if not prev else th - hys
            passed = op(value, eff)
            ctx.set(key, passed)  # 会话内状态（跨运行不持久，见 node_base 纪律）
            return {"result": passed}
        return {"result": bool(op(value, th))}


@register_node
class MathOp(VisionNode):
    """数值运算：除零输出 NaN 并警告（下游 compare_value 判 False）。"""

    NAME = "math_op"
    CATEGORY = "逻辑"
    INPUTS = (Port("a", "number", label="数值a"),
              Port("b", "number", required=False, label="数值b"))
    OUTPUTS = (Port("number", "number", label="数值"),)
    PARAMS = {
        "op": ParamSpec("op", "运算", "enum", "sub",
                        options=["add", "sub", "mul", "div", "abs",
                                 "min", "max"]),
    }

    def run(self, inputs, params, ctx):
        a = inputs.get("a")
        b = inputs.get("b")
        av = float(a.value) if a is not None else float("nan")
        bv = float(b.value) if b is not None else 0.0
        op = params["op"]
        if op == "add":
            out = av + bv
        elif op == "sub":
            out = av - bv
        elif op == "mul":
            out = av * bv
        elif op == "div":
            if bv == 0:
                ctx.warn("math_op 除零：除数为 0，输出 NaN（下游判定按不通过处理）")
                out = float("nan")
            else:
                out = av / bv
        elif op == "abs":
            out = abs(av)
        elif op == "min":
            out = min(av, bv)
        else:
            out = max(av, bv)
        return {"number": NumberVal(value=float(out), unit=a.unit if a else "")}


@register_node
class ConditionBranch(VisionNode):
    """条件选路：按判定选择两路图像之一（零拷贝直通，不复制图像）。

    判定输入接判定类节点的统一结果（值=布尔）；True 选「真路图像」。
    """

    NAME = "condition_branch"
    TITLE = "条件选路"
    CATEGORY = "逻辑"
    INPUTS = (Port("true_input", "image", label="真路图像"),
              Port("false_input", "image", label="假路图像"),
              Port("cond", "result", label="判定"))
    OUTPUTS = (Port("image", "image", label="图像"),)

    def run(self, inputs, params, ctx):
        chosen = inputs["true_input"] if inputs["cond"] else inputs["false_input"]
        return {"image": chosen}  # 直通引用，禁止复制（零拷贝纪律）


@register_node
class ResultMerge(VisionNode):
    """多路测量聚合：allpass / anypass / count_ng / mean / max / min。

    输出 number（数值）与 result（判定，值=布尔；mean/max/min 无判定语义时
    为 False——判定语义看快照/日志）。未连接的可选测量口按"无数据"跳过；
    全部为空时输出 NaN。
    """

    NAME = "result_merge"
    TITLE = "多路判定"
    CATEGORY = "逻辑"
    INPUTS = (Port("m1", "result", label="测量1"),
              Port("m2", "result", required=False, label="测量2"),
              Port("m3", "result", required=False, label="测量3"),
              Port("m4", "result", required=False, label="测量4"))
    OUTPUTS = (Port("number", "number", label="数值"),
               Port("result", "result", label="判定"))
    PARAMS = {
        "op": ParamSpec("op", "聚合", "enum", "allpass",
                        options=["allpass", "anypass", "count_ng",
                                 "mean", "max", "min"]),
    }

    @staticmethod
    def _values(results):
        vals = []
        for r in results:
            if r.value is not None:
                vals.append(float(r.value))
            elif r.values:
                vals.extend(float(v) for v in r.values)
        return vals

    def run(self, inputs, params, ctx):
        results = [inputs[k] for k in ("m1", "m2", "m3", "m4")
                   if isinstance(inputs.get(k), MeasureResult)]
        op = params["op"]
        if op in ("allpass", "anypass", "count_ng"):
            flags = [r.passed for r in results if r.passed is not None]
            if op == "allpass":
                passed = bool(flags) and all(flags)
                value = 1.0 if passed else 0.0
            elif op == "anypass":
                passed = any(flags)
                value = 1.0 if passed else 0.0
            else:
                ng = sum(1 for f in flags if not f)
                passed = ng == 0
                value = float(ng)
            return {"number": NumberVal(value=value), "result": passed}
        vals = self._values(results)
        if not vals:
            ctx.warn("result_merge 未收集到任何测量数值")
            return {"number": NumberVal(value=float("nan")), "result": False}
        if op == "mean":
            value = sum(vals) / len(vals)
        elif op == "max":
            value = max(vals)
        else:
            value = min(vals)
        return {"number": NumberVal(value=value), "result": False}


@register_node
class BoolOp(VisionNode):
    """布尔运算：and / or / not / nor（not 只用 a 输入）。

    输入接判定类节点的统一结果（值=布尔）。未连接 b 时 and/or/nor 报错
    （避免缺线被静默当 False 误判）；上游给出 None（未产出）按 False 处理并警告。
    """

    NAME = "bool_op"
    TITLE = "布尔运算"
    CATEGORY = "逻辑"
    INPUTS = (Port("a", "result", label="判定a"),
              Port("b", "result", required=False, label="判定b"))
    OUTPUTS = (Port("result", "result", label="判定"),)
    PARAMS = {
        "op": ParamSpec("op", "运算", "enum", "and", options=["and", "or", "not", "nor"],
                        tooltip="and=全真；or=任一真；not=取反 a；nor=全假为真"),
    }

    @staticmethod
    def _as_bool(v, name: str, ctx) -> bool:
        if v is None:
            ctx.warn(f"布尔运算：输入 {name} 无值，按 False 处理")
            return False
        return bool(v)

    def run(self, inputs, params, ctx):
        op = params["op"]
        a = self._as_bool(inputs.get("a"), "a", ctx)
        if op == "not":
            return {"result": not a}
        if "b" not in inputs:  # 执行器约定：未连线的可选端口不出现在 inputs
            raise ValueError(f"布尔运算「{op}」需要连接 b 输入")
        b = self._as_bool(inputs.get("b"), "b", ctx)
        if op == "and":
            return {"result": a and b}
        if op == "or":
            return {"result": a or b}
        return {"result": not (a or b)}   # nor

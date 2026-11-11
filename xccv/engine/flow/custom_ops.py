"""自定义算子运行时（V0.6 T4-4）：`custom/` 前缀、复合 / 脚本两种实现。

存放：``platform_data/operators/<id>.op.json``（每个算子一个文件，utf-8）。
注册名 ``custom/<id>``（画布类型 ``xccv2/custom/<id>``，与内置算子隔离），
分类「自定义」。节点库懒加载纪律不变：只在 ops.load_all()（画布节点库 /
工作台初始化 / 流程会话）时首次读盘；之后靠「刷新」显式热重载。

op.json（schema=1）::

    {"schema": 1, "id": "my_op", "title": "我的算子", "description": "…",
     "kind": "composite" | "script",
     "inputs":  [{"name": "image", "type": "image", "required": true}],
     "outputs": [{"name": "image", "type": "image"}],
     "params":  {"k": {"label": "核数", "type": "int", "default": 5,
                       "range": [1, 31], "options": null, "tooltip": ""}},
     "composite": {"graph": {...子流程 JSON...},
                   "input_map":  {"image": [[子节点id, 输入槽], ...]},
                   "output_map": {"image": [子节点id, "输出端口名"]}},
     "script": {"code": "def run(inputs, params, ctx):\\n    ...\\n    return {...}"}}

- composite：子流程节点 properties 中的 ``"${参数名}"`` 整串替换为参数值
  （保留类型），嵌在长串里则按文本插值；子流程不得再引用 custom/ 算子
  （防递归）；输入/输出端口类型须与子流程对应端口一致。
- script：受控片段——顶层只允许 import / 函数定义 / 常量赋值；import 仅限
  numpy / cv2 / math / statistics；禁用 eval / exec / open / __import__ /
  compile / globals / getattr 等与任何双下划线属性；compile 预检；以受限
  builtins 执行。**这是防误用的受控环境，不是对抗恶意代码的安全沙箱**——
  算子文件来自本机用户自己（或其信任的来源）。
- ``.xcvop``：zip 包，仅含 ``op.json``（导入时校验成员名与大小）。
"""

import ast
import copy
import json
import re
import zipfile
from pathlib import Path

from .ir import PORT_TYPES, Geometry, ImageData, MeasureResult, NumberVal
from .node_base import PARAM_TYPES, ParamSpec, Port, VisionNode
from .registry import get_node_class, register_node, registered_names, unregister_node

PREFIX = "custom/"
CATEGORY = "自定义"
SCHEMA_VERSION = 1
ALLOWED_IMPORTS = ("numpy", "cv2", "math", "statistics")
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "open", "__import__", "compile", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "help", "exit", "quit",
    "memoryview", "__builtins__", "__loader__", "__spec__"})
_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_MAX_XCVOP_BYTES = 2 * 1024 * 1024

_loaded_once = False


# ------------------------------------------------------------------ 存放位置
def operators_dir(root: Path | None = None) -> Path:
    """算子目录（默认 app_root/platform_data/operators；测试可传临时目录）。"""
    if root is not None:
        return Path(root)
    from ...app_paths import app_root

    return app_root() / "platform_data" / "operators"


# ------------------------------------------------------------------ 脚本安检
def check_script(code: str) -> list:
    """脚本静态检查：返回中文问题列表（空=通过，且已 compile 预检）。"""
    try:
        tree = ast.parse(code or "", filename="<custom>")
    except SyntaxError as e:
        return [f"脚本语法错误：第 {e.lineno} 行 {e.msg}"]
    errs = []
    run_ok = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            continue
        if isinstance(node, ast.FunctionDef):
            if node.name == "run":
                a = node.args
                n_pos = len(a.posonlyargs) + len(a.args)
                run_ok = n_pos == 3 and not a.vararg and not a.kwarg and not a.kwonlyargs
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue   # 文档字符串
        errs.append(f"第 {node.lineno} 行：顶层只允许 import / 函数定义 / 常量赋值")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    errs.append(f"第 {node.lineno} 行：不允许 import {alias.name}"
                                f"（仅限 {', '.join(ALLOWED_IMPORTS)}）")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS or node.level:
                errs.append(f"第 {node.lineno} 行：不允许 from {node.module or '.'} import"
                            f"（仅限 {', '.join(ALLOWED_IMPORTS)}）")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errs.append(f"第 {node.lineno} 行：禁止使用 {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errs.append(f"第 {node.lineno} 行：禁止访问双下划线属性 {node.attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            errs.append(f"第 {node.lineno} 行：禁止 global / nonlocal")
    if not run_ok:
        errs.append("脚本须定义 def run(inputs, params, ctx)（恰好 3 个参数）")
    if not errs:
        try:
            compile(tree, "<custom>", "exec")
        except Exception as e:  # noqa: BLE001  compile 预检
            errs.append(f"脚本编译失败：{e}")
    return errs


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"自定义脚本不允许 import {name}")
    return __import__(name, globals, locals, fromlist, level)


_SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                  for n in ("abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
                            "float", "int", "isinstance", "len", "list", "map", "max", "min",
                            "pow", "range", "reversed", "round", "set", "sorted", "str", "sum",
                            "tuple", "zip", "Exception", "ValueError", "TypeError", "KeyError",
                            "IndexError", "ZeroDivisionError", "RuntimeError", "NotImplementedError")}
_SAFE_BUILTINS["__import__"] = _safe_import


def _compile_script(op_id: str, code: str):
    errs = check_script(code)
    if errs:
        raise ValueError("；".join(errs))
    import math

    import cv2
    import numpy as np

    glb = {"__builtins__": dict(_SAFE_BUILTINS), "__name__": f"custom_{op_id}",
           "np": np, "cv2": cv2, "math": math, "ImageData": ImageData,
           "Geometry": Geometry, "MeasureResult": MeasureResult, "NumberVal": NumberVal}
    exec(compile(code, f"<custom/{op_id}>", "exec"), glb)  # noqa: S102  已经 AST 安检
    return glb["run"]


# ------------------------------------------------------------------ 定义校验
def _param_spec(key: str, d: dict) -> ParamSpec:
    rng = d.get("range")
    return ParamSpec(key, str(d.get("label") or key), str(d.get("type")), d.get("default"),
                     range=tuple(rng) if rng else None,
                     options=list(d["options"]) if d.get("options") else None,
                     tooltip=str(d.get("tooltip") or ""), folder=bool(d.get("folder")))


def _ports(items, kind: str, errs: list) -> list:
    out, seen = [], set()
    if not isinstance(items, list):
        errs.append(f"{kind}应为列表")
        return out
    for i, p in enumerate(items):
        name = str((p or {}).get("name") or "")
        typ = str((p or {}).get("type") or "")
        if typ == "image_list":
            typ = "image"   # 修复循环 4：image_list 端口类型取消，旧定义平移
        if not _NAME_RE.match(name):
            errs.append(f"{kind}第 {i + 1} 个端口名「{name}」非法（字母/数字/下划线）")
            continue
        if name in seen:
            errs.append(f"{kind}端口名重复：{name}")
        if typ not in PORT_TYPES:
            errs.append(f"{kind}端口 {name} 类型「{typ}」非法（可选：{', '.join(PORT_TYPES)}）")
            continue
        seen.add(name)
        out.append(Port(name, typ, bool(p.get("required", True))))
    return out


def _subst(value, params: dict):
    """属性值里的 ${参数} 替换：整串占位保留类型，嵌入则文本插值。"""
    if isinstance(value, str):
        m = _PLACEHOLDER.fullmatch(value.strip())
        if m:
            return params[m.group(1)]
        return _PLACEHOLDER.sub(lambda mm: str(params[mm.group(1)]), value)
    if isinstance(value, list):
        return [_subst(v, params) for v in value]
    if isinstance(value, dict):
        return {k: _subst(v, params) for k, v in value.items()}
    return value


def _placeholders(obj) -> set:
    found = set()
    if isinstance(obj, str):
        found.update(_PLACEHOLDER.findall(obj))
    elif isinstance(obj, list):
        for v in obj:
            found |= _placeholders(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            found |= _placeholders(v)
    return found


def _norm_type(t: str) -> str:
    return t[6:] if t.startswith("xccv2/") else t


def validate_op_def(d) -> list:
    """校验 op.json 定义：返回中文问题列表（空=合法）。"""
    errs: list = []
    if not isinstance(d, dict):
        return ["算子定义应为 JSON 对象"]
    if d.get("schema") != SCHEMA_VERSION:
        errs.append(f"schema 应为 {SCHEMA_VERSION}")
    op_id = str(d.get("id") or "")
    if not _ID_RE.match(op_id):
        errs.append("id 须以小写字母开头，仅含小写字母/数字/下划线（2~41 位）")
    if not str(d.get("title") or "").strip():
        errs.append("缺少显示名 title")
    kind = d.get("kind")
    if kind not in ("composite", "script"):
        errs.append("kind 须为 composite（复合）或 script（脚本）")
    inputs = _ports(d.get("inputs", []), "输入", errs)
    outputs = _ports(d.get("outputs", []), "输出", errs)
    if not outputs:
        errs.append("至少需要一个输出端口")
    params: dict = {}
    raw_params = d.get("params") or {}
    if not isinstance(raw_params, dict):
        errs.append("params 应为对象")
        raw_params = {}
    for key, pd in raw_params.items():
        if not _NAME_RE.match(str(key)):
            errs.append(f"参数名「{key}」非法")
            continue
        try:
            spec = _param_spec(key, pd or {})
        except Exception as e:  # noqa: BLE001
            errs.append(f"参数 {key}：{e}（类型可选 {', '.join(PARAM_TYPES)}）")
            continue
        e2 = spec.validate(spec.default)
        if e2:
            errs.append(f"参数 {key} 默认值非法：{e2}")
        params[key] = spec
    if kind == "script":
        code = str((d.get("script") or {}).get("code") or "")
        errs += check_script(code)
    elif kind == "composite":
        errs += _validate_composite(d.get("composite") or {}, inputs, outputs, params)
    return errs


def _validate_composite(comp: dict, inputs, outputs, params) -> list:
    from .executor import Executor, FlowError

    errs = []
    graph = comp.get("graph")
    if not isinstance(graph, dict) or not graph.get("nodes"):
        return ["复合算子缺少子流程 graph（含 nodes）"]
    nodes = {int(n["id"]): n for n in graph["nodes"]}
    for n in graph["nodes"]:
        if _norm_type(str(n.get("type", ""))).startswith(PREFIX):
            errs.append(f"子流程节点 {n['id']} 引用了自定义算子（不允许嵌套）")
    unknown = _placeholders([n.get("properties") or {} for n in graph["nodes"]]) - set(params)
    if unknown:
        errs.append(f"子流程引用了未定义参数：{', '.join(sorted(unknown))}")
    norm = _normalize_graph(graph)
    try:
        Executor(norm).validate()
    except FlowError as e:
        errs.append(f"子流程结构错误：{e}")
        return errs
    imap = comp.get("input_map") or {}
    for p in inputs:
        targets = imap.get(p.name)
        if not targets:
            errs.append(f"输入 {p.name} 未映射到子流程（input_map）")
            continue
        for t in targets:
            nid, slot = int(t[0]), int(t[1])
            cls = get_node_class(_norm_type(str(nodes.get(nid, {}).get("type", ""))))
            if cls is None or not 0 <= slot < len(cls.INPUTS):
                errs.append(f"输入 {p.name} 映射的子节点 {nid} 槽 {slot} 不存在")
            elif cls.INPUTS[slot].type != p.type:
                errs.append(f"输入 {p.name}（{p.type}）与子节点 {nid} 的"
                            f"「{cls.INPUTS[slot].name}」（{cls.INPUTS[slot].type}）类型不一致")
    omap = comp.get("output_map") or {}
    for p in outputs:
        t = omap.get(p.name)
        if not t:
            errs.append(f"输出 {p.name} 未映射（output_map）")
            continue
        nid, port = int(t[0]), str(t[1])
        cls = get_node_class(_norm_type(str(nodes.get(nid, {}).get("type", ""))))
        out_port = next((o for o in (cls.OUTPUTS if cls else ()) if o.name == port), None)
        if out_port is None:
            errs.append(f"输出 {p.name} 映射的子节点 {nid} 无输出端口「{port}」")
        elif out_port.type != p.type:
            errs.append(f"输出 {p.name}（{p.type}）与子节点 {nid}「{port}」（{out_port.type}）类型不一致")
    return errs


def _normalize_graph(graph: dict) -> dict:
    g = copy.deepcopy(graph)
    for n in g.get("nodes", []):
        n["type"] = _norm_type(str(n.get("type", "")))
    return g


# ------------------------------------------------------------------ 节点类
class _CustomNodeBase(VisionNode):
    _DEF: dict = {}

    def run(self, inputs, params, ctx):
        raise NotImplementedError


class _CompositeNode(_CustomNodeBase):
    def run(self, inputs, params, ctx):
        from .executor import Executor

        comp = self._DEF["composite"]
        graph = _normalize_graph(comp["graph"])
        for n in graph["nodes"]:
            n["properties"] = _subst(n.get("properties") or {}, params)
        external = {}
        for name, targets in (comp.get("input_map") or {}).items():
            if name in inputs:
                for t in targets:
                    external[(int(t[0]), int(t[1]))] = inputs[name]
        sub = ctx.child()
        res = Executor(graph, sub, external=external).run()
        failed = [f"子节点 {nid}（{r.name}）：{r.error}" for nid, r in res.items()
                  if not r.ok and not r.skipped]
        if failed:
            raise RuntimeError(f"{self.TITLE} 子流程执行失败——" + "；".join(failed[:3]))
        out = {}
        for name, (nid, port) in (comp.get("output_map") or {}).items():
            r = res.get(int(nid))
            if r is None or not r.ok:
                raise RuntimeError(f"{self.TITLE}：输出 {name} 的子节点 {nid} 未产出结果")
            out[name] = r.outputs.get(str(port))
        return out


class _ScriptNode(_CustomNodeBase):
    _FN = None

    def run(self, inputs, params, ctx):
        cls = type(self)
        if cls._FN is None:
            cls._FN = _compile_script(self._DEF["id"], self._DEF["script"]["code"])
        out = cls._FN(dict(inputs), dict(params), ctx)
        if not isinstance(out, dict):
            raise TypeError(f"{self.TITLE}：run() 应返回 dict（输出端口名→数据）")
        return out


def build_node_class(d: dict) -> type:
    """op.json 定义 → 节点类（须先通过 validate_op_def）。"""
    errs = []
    inputs = tuple(_ports(d.get("inputs", []), "输入", errs))
    outputs = tuple(_ports(d.get("outputs", []), "输出", errs))
    params = {k: _param_spec(k, v or {}) for k, v in (d.get("params") or {}).items()}
    base = _CompositeNode if d["kind"] == "composite" else _ScriptNode
    attrs = {"NAME": PREFIX + d["id"], "TITLE": str(d["title"]), "CATEGORY": CATEGORY,
             "__doc__": str(d.get("description") or d["title"]),
             "INPUTS": inputs, "OUTPUTS": outputs, "PARAMS": params, "_DEF": copy.deepcopy(d)}
    return type(f"Custom_{d['id']}", (base,), attrs)


# ------------------------------------------------------------------ 加载 / 存取
def load_custom_ops(root: Path | None = None) -> tuple:
    """（热）重载：注销全部 custom/ 节点后按目录重新注册。返回 (已加载名单, 错误列表)。"""
    global _loaded_once
    _loaded_once = True
    for name in registered_names(PREFIX):
        unregister_node(name)
    folder = operators_dir(root)
    loaded, errors = [], []
    if not folder.is_dir():
        return loaded, errors
    for f in sorted(folder.glob("*.op.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{f.name}：JSON 解析失败（{e}）")
            continue
        errs = validate_op_def(d)
        if errs:
            errors.append(f"{f.name}：" + "；".join(errs))
            continue
        if get_node_class(PREFIX + d["id"]) is not None:
            errors.append(f"{f.name}：算子 id {d['id']} 重复")
            continue
        register_node(build_node_class(d))
        loaded.append(PREFIX + d["id"])
    return loaded, errors


def ensure_loaded(root: Path | None = None) -> None:
    """首次需要节点库时读盘一次（懒加载；之后靠显式 reload）。"""
    if not _loaded_once:
        load_custom_ops(root)


def save_op(d: dict, root: Path | None = None, overwrite: bool = True) -> Path:
    """校验并保存 op.json，随即热重载（保存即可试跑）。"""
    errs = validate_op_def(d)
    if errs:
        raise ValueError("；".join(errs))
    folder = operators_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{d['id']}.op.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"算子 {d['id']} 已存在")
    path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    load_custom_ops(root)
    return path


def read_op(op_id: str, root: Path | None = None) -> dict:
    return json.loads((operators_dir(root) / f"{op_id}.op.json").read_text(encoding="utf-8"))


def list_custom_ops(root: Path | None = None) -> list:
    folder = operators_dir(root)
    return sorted(f.name[:-len(".op.json")] for f in folder.glob("*.op.json")) \
        if folder.is_dir() else []


def delete_op(op_id: str, root: Path | None = None) -> bool:
    path = operators_dir(root) / f"{op_id}.op.json"
    if not path.exists():
        return False
    path.unlink()
    load_custom_ops(root)
    return True


def export_xcvop(op_id: str, dest, root: Path | None = None) -> Path:
    """导出 .xcvop（zip，仅含 op.json）。"""
    d = read_op(op_id, root)
    dest = Path(dest)
    if dest.suffix.lower() != ".xcvop":
        dest = dest.with_suffix(".xcvop")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("op.json", json.dumps(d, ensure_ascii=False, indent=1))
    return dest


def import_xcvop(path, root: Path | None = None, overwrite: bool = False) -> str:
    """导入 .xcvop：仅接受根目录 op.json 单成员、≤2MB；校验通过即保存并热重载。"""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if names != ["op.json"]:
            raise ValueError(f".xcvop 内容非法：应只含 op.json（实际 {names[:5]}）")
        info = z.getinfo("op.json")
        if info.file_size > _MAX_XCVOP_BYTES:
            raise ValueError(".xcvop 内 op.json 过大（>2MB）")
        d = json.loads(z.read("op.json").decode("utf-8"))
    save_op(d, root, overwrite=overwrite)
    return d["id"]


def new_op_template(kind: str = "script") -> dict:
    """编辑器"新建"用的起始定义（可直接保存试跑）。"""
    base = {"schema": SCHEMA_VERSION, "id": "my_op", "title": "我的算子",
            "description": "自定义算子：请在此描述功能", "kind": kind,
            "inputs": [{"name": "image", "type": "image", "required": True}],
            "outputs": [{"name": "image", "type": "image"}],
            "params": {"k": {"label": "核数", "type": "int", "default": 5, "range": [1, 31]}}}
    if kind == "script":
        base["script"] = {"code": (
            "def run(inputs, params, ctx):\n"
            "    img = inputs[\"image\"]\n"
            "    k = int(params[\"k\"]) | 1\n"
            "    out = cv2.GaussianBlur(img.mat, (k, k), 0)\n"
            "    return {\"image\": ImageData(mat=out, meta=dict(img.meta))}\n")}
    else:
        base["composite"] = {
            "graph": {"nodes": [{"id": 1, "type": "xccv2/gaussian_blur",
                                 "properties": {"ksize": "${k}", "sigma": 0.0}}],
                      "links": []},
            "input_map": {"image": [[1, 0]]},
            "output_map": {"image": [1, "image"]}}
    return base

"""节点注册表：NAME → 节点类。算子经 @register_node 完成注册。

注册时做完整性检查（端口类型合法、参数默认值过 schema），保证
list_nodes() 下发给前端的节点库数据自洽（批次十九的 schema 面板数据源）。
"""

from .ir import PORT_TYPES
from .node_base import VisionNode

_REGISTRY: dict = {}


def register_node(cls):
    """类装饰器：注册节点类。NAME 重复 / 缺 CATEGORY / 定义不完整即抛错。"""
    name = getattr(cls, "NAME", "")
    if not name or not isinstance(name, str):
        raise ValueError(f"{cls.__name__} 缺少合法的 NAME")
    if name in _REGISTRY:
        raise ValueError(f"节点注册名重复：{name}（{_REGISTRY[name].__name__}）")
    if not getattr(cls, "CATEGORY", ""):
        raise ValueError(f"节点 {name} 缺少 CATEGORY")
    # 定义完整性：端口类型合法、参数默认值能通过自身 schema 校验
    for ports in (cls.INPUTS, cls.OUTPUTS):
        for p in ports:
            if p.type not in PORT_TYPES:
                raise ValueError(f"节点 {name} 端口 {p.name!r} 类型非法")
    for key, spec in getattr(cls, "PARAMS", {}).items():
        err = spec.validate(spec.default)
        if err:
            raise ValueError(f"节点 {name} 参数 {key} 默认值非法：{err}")
    _REGISTRY[name] = cls
    return cls


def get_node_class(name: str):
    """按注册名取节点类；未注册返回 None。"""
    return _REGISTRY.get(name)


def unregister_node(name: str) -> bool:
    """注销节点（自定义算子热重载用，V0.6 T4-4）；内置算子不应调用。"""
    return _REGISTRY.pop(name, None) is not None


def registered_names(prefix: str = "") -> list:
    """已注册名（可按前缀过滤，如 "custom/"）。"""
    return sorted(n for n in _REGISTRY if n.startswith(prefix))


def node_title(cls) -> str:
    """节点中文显示名（节点库/画布标题条/属性面板用）。

    优先取类上的 TITLE 属性；否则从 docstring 首行提取第一段：
    在 "："/"（"/"；"/"。"/";" 处截断（如"置信度过滤（旧链 confidence 的
    薄委托）"→"置信度过滤"、"bbox 宽高过滤（…）；0 = 不限制。"→"bbox 宽高过滤"）。
    """
    import re

    explicit = getattr(cls, "TITLE", "")
    if explicit:
        return str(explicit)
    lines = (cls.__doc__ or "").strip().splitlines()
    line = lines[0].strip() if lines else ""
    s = re.split(r"[：；。;]|（", line)[0].strip()
    return s or cls.NAME


def list_nodes(category=None) -> list:
    """节点库数据（前端 LIST_NODES 下发用），按名称排序。

    每项：{name, title, category, desc, inputs, outputs, params}；
    params 为 {键: {label,type,default,range,options,tooltip}} schema。
    """
    out = []
    for name in sorted(_REGISTRY):
        cls = _REGISTRY[name]
        if category is not None and cls.CATEGORY != category:
            continue
        doc = (cls.__doc__ or "").strip().splitlines()
        out.append({
            "name": name,
            "title": node_title(cls),
            "category": cls.CATEGORY,
            "desc": doc[0].strip() if doc else "",
            "inputs": [{"name": p.name, "type": p.type, "required": p.required,
                        "label": p.label or p.name,
                        "accepts": list(p.accepts)}
                       for p in cls.INPUTS],
            "outputs": [{"name": p.name, "type": p.type, "label": p.label or p.name}
                        for p in cls.OUTPUTS],
            "params": {key: {"label": s.label, "type": s.type,
                             "default": s.default, "range": list(s.range) if s.range else None,
                             "options": list(s.options) if s.options else None,
                             "tooltip": s.tooltip, "folder": bool(s.folder)}
                       for key, s in cls.PARAMS.items()},
        })
    return out


def all_categories() -> list:
    """已注册节点的分类清单（节点库分组用，排序去重）。"""
    return sorted({cls.CATEGORY for cls in _REGISTRY.values()})

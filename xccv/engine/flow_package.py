"""流程图打包格式 .xvap（社区版：明文容器）：流程图（节点/连线/参数，含所选
模型引用）+ 引用的模型文件本体，打包为单一 .xvap 文件。

payload 布局（未加密态）：
    b"XVAP" + <I json 长度> + <I 模型个数>
    + JSON（流程图：nodes/links/参数；模型节点记录 model_file 文件名）
    + 每个模型: <H 文件名长> 文件名 <I 数据长> 数据（模型文件原字节）
"""

import json
import struct
from pathlib import Path


_MAGIC = b"XVAP"

# 模型引用参数键（预留扩展接口）：旧链模型节点用 model_file，新引擎 AI
# 节点（xccv2/model_*）用 model_path——收集/解包改写一律遍历本表，新增
# 引用键时只改这里。
MODEL_PARAM_KEYS = ("model_file", "model_path")


def _model_refs(node: dict) -> list[tuple[str, str]]:
    """节点里出现的模型引用 [(参数键, 值)]（按 MODEL_PARAM_KEYS 顺序去空）。"""
    props = node.get("properties") or node.get("params") or {}
    return [(k, str(props.get(k) or "").strip()) for k in MODEL_PARAM_KEYS
            if str(props.get(k) or "").strip()]


def pack_xvap(flow: dict, models: dict[str, bytes]) -> bytes:
    """流程图 + 模型文件字节 → .xvap 明文 payload（调用方负责加密落盘）。"""
    flow_b = json.dumps(flow, ensure_ascii=False).encode("utf-8")
    parts = [_MAGIC, struct.pack("<II", len(flow_b), len(models)), flow_b]
    for name, data in models.items():
        name_b = name.encode("utf-8")
        parts.append(struct.pack("<H", len(name_b)) + name_b
                     + struct.pack("<I", len(data)) + data)
    return b"".join(parts)


def unpack_xvap(payload: bytes) -> tuple[dict, dict[str, bytes]]:
    """解析 .xvap 明文 payload → (流程图, {文件名: 模型字节})。"""
    if not payload.startswith(_MAGIC):
        raise ValueError("不是有效的 .xvap 流程包（魔数不匹配）")
    n_json, n_models = struct.unpack_from("<II", payload, 4)
    off = 12
    flow = json.loads(payload[off:off + n_json].decode("utf-8"))
    off += n_json
    models: dict[str, bytes] = {}
    for _ in range(n_models):
        (n_name,) = struct.unpack_from("<H", payload, off)
        off += 2
        name = payload[off:off + n_name].decode("utf-8")
        off += n_name
        (n_data,) = struct.unpack_from("<I", payload, off)
        off += 4
        models[name] = payload[off:off + n_data]
        off += n_data
    return flow, models


def save_xvap(path, flow: dict, models: dict[str, bytes]) -> None:
    """打包落盘为 .xvap（社区版明文容器）。"""
    payload = pack_xvap(flow, models)
    Path(path).write_bytes(payload)


def load_xvap(path) -> tuple[dict, dict[str, bytes]]:
    """读取 .xvap → (流程图, {文件名: 模型字节})。"""
    return unpack_xvap(Path(path).read_bytes())


def collect_model_files(flow: dict) -> dict[str, bytes]:
    """扫描流程中模型节点引用的模型文件，读取字节（用于打包）。

    同时识别旧链 model_file 与新引擎 model_path（P87：此前只认 model_file，
    新引擎 AI 流程打包会静默丢权重）。
    """
    models: dict[str, bytes] = {}
    for node in flow.get("nodes", []):
        for _key, name in _model_refs(node):
            if name in models or Path(name).name in models:
                continue
            p = Path(name)
            if not p.is_file():
                raise FileNotFoundError(f"模型节点引用的文件不存在: {name}")
            models[p.name] = p.read_bytes()
    return models


def unpack_to_dir(path) -> tuple[dict, "Path"]:
    """解包 .xvap 到同级：内嵌模型写入「流程模型/」，节点模型引用键改写为
    该目录下的绝对路径（画布编辑/运行/打包可直接使用）。

    返回 (流程 dict, 模型目录)。do_open / _materialize_xvap 共用，
    模型引用键的识别与改写以 MODEL_PARAM_KEYS 为准。
    """
    flow, models = load_xvap(path)
    out_dir = Path(path).parent / "流程模型"
    out_dir.mkdir(exist_ok=True)
    for name, data in models.items():
        (out_dir / name).write_bytes(data)
    for n in flow.get("nodes", []):
        props = n.setdefault("properties", {})
        for key, name in _model_refs(n):
            props[key] = str(out_dir / Path(name).name)
    return flow, out_dir

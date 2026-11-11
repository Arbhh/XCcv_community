# -*- coding: utf-8 -*-
"""流程图文件工具与打包校验（社区版）。

从旧「流程图编辑器」窗口抽出的纯逻辑函数：空白流程构造、流程 JSON 读写、
.xvap 解包落盘，以及打包前的结构/模型引用校验。不包含任何窗口类——
主界面流程图页（flow_platform.FlowPage）与 Web 画布直接复用这些函数。
"""

import json
import logging
from pathlib import Path

from PyQt5.QtWidgets import QMessageBox

from ..engine import detection_flow as df
from ..engine import flow_package as fp

log = logging.getLogger(__name__)

_EMPTY_FLOW = {"last_node_id": 0, "last_link_id": 0, "nodes": [], "links": []}
_MODEL_FILE_DLG = "模型权重 (*.pt);;所有文件 (*)"


def _write_flow_json(path: Path, flow: dict) -> None:
    path.write_text(json.dumps(flow, ensure_ascii=False, indent=1),
                    encoding="utf-8")


_MAGIC_XCVA = b"XCVA"


def _model_file_problems(model_file: str) -> list[str]:
    """模型文件校验：存在、非空、头信息可读（.pt 的 zip 头）。"""
    p = Path(model_file)
    if not p.is_file():
        return [f"模型节点引用的权重文件不存在：{model_file}"]
    with p.open("rb") as f:
        head = f.read(8)
    if len(head) < 4:
        return [f"模型文件为空或损坏：{p.name}"]
    suffix = p.suffix.lower()
    if suffix == ".pt":
        if not head.startswith(b"PK"):
            return [f"模型文件不是有效的 torch 权重（缺 zip 头）：{p.name}"]
    return []


_XCV_MODEL_NODES = ("model_detection", "model_instance_seg", "model_classification")


def _xcv_pack_problems(flow: dict) -> list[str]:
    """新算子流程（xccv2/ 图）的打包前检查（用户批注 ToC1）。

    - 未知/混入旧版节点 → 问题；
    - 每个节点参数按注册 schema 校验（VisionNode.validate，含数量判定
      上下限、区域格式等节点级 validate）；
    - 模型节点（目标检测/实例分割/分类）必须已选权重且文件头可读；
    - 连通性/端口类型/有环/多输入 → Executor 静态校验。
    """
    from ..engine.detection_flow import region_param_problems
    from ..engine.flow import Executor, FlowError
    from ..engine.flow.node_base import RunContext
    from ..engine.flow.port_migrate import migrate_ports
    from ..engine.flow.registry import get_node_class, node_title
    from ..engine.flow.ops import load_all

    import xccv.engine.flow.legacy_adapter  # noqa: F401  过滤节点注册

    load_all()
    flow = migrate_ports(flow)   # 旧端口布局连线 → 当前布局（修复循环 3）
    problems: list[str] = []
    norm = []
    for i, n in enumerate(flow.get("nodes") or [], start=1):
        t = str(n.get("type", ""))
        if t.startswith("xccv2/"):
            t = t[6:]
        if get_node_class(t) is None:
            problems.append(f"节点 {i}：类型 {n.get('type')!r} 未注册"
                            "（可能混入了旧版节点，请改用「通用算子」面板节点）")
            continue
        props = dict(n.get("properties") or n.get("params") or {})
        cls = get_node_class(t)
        errs = cls().validate(props)
        for e in errs:
            problems.append(f"节点 {i}（{node_title(cls)}）：{e}")
        if t in _XCV_MODEL_NODES:
            mp = str(props.get("model_path") or "")
            if not mp:
                problems.append(f"节点 {i}（{node_title(cls)}）：尚未选择权重文件")
            else:
                problems += _model_file_problems(mp)
        norm.append({**n, "type": t})
    try:
        Executor({"nodes": norm, "links": flow.get("links") or []},
                 RunContext()).validate()
    except FlowError as e:
        problems.append(f"流程结构问题：{e}")
    # 必需输入缺连线（Executor.validate 不查必填，运行期才会报——打包期提前拦）
    linked_in = {(int(lk[3]), int(lk[4])) for lk in flow.get("links") or []}
    for i, n in enumerate(norm, start=1):
        cls = get_node_class(str(n.get("type", "")))
        if cls is None:
            continue
        for slot, port in enumerate(cls.INPUTS):
            if port.required and (int(n.get("id")), slot) not in linked_in:
                problems.append(f"节点 {i}（{node_title(cls)}）：必需输入"
                                f"「{port.name}」缺少连线")
    # 区域参数用打包口径严查（节点 validate 为宽容口径，这里补严）
    for i, n in enumerate(norm, start=1):
        cls = get_node_class(str(n.get("type", "")))
        if cls is not None and getattr(cls, "OP", "") in ("region_block", "region_keep"):
            props = dict(n.get("properties") or {})
            problems += region_param_problems(
                props.get("regions"), f"节点 {i}（{node_title(cls)}）：")
    return problems


def validate_flow_for_pack(flow: dict) -> list[str]:
    """流程包（.xvap）打包前校验（用户批注 To3.3）。返回问题清单，空=通过。

    检查：模型节点已选权重且文件可读、区域参数格式合法、节点连接完整
    （图格式下后处理节点必须与模型节点连通才会被执行）、数量判定上下限
    合理。不通过时调用方列出问题清单并中止打包，不产出坏包。
    """
    problems: list[str] = []
    nodes = flow.get("nodes") or []
    if not nodes:
        return ["流程为空：请先添加模型节点与后处理节点"]

    is_graph = isinstance(nodes[0], dict) and "type" in nodes[0]
    if is_graph:
        # 图格式（画布保存的流程）。链式旧格式（检出流程.json）本就没有内嵌
        # 模型引用，打包时由调用方的既有提示兜底。
        has_legacy = any(str(n.get("type", "")).startswith(
            ("model/", "xccv/model", "post/", "xccv/post_")) for n in nodes)
        if not has_legacy:
            # 纯新算子流程：不再强制 AI 模型节点（允许纯传统算子链打包），
            # 但放坏包检查全部保留：参数 schema / 数量判定 / 区域格式 /
            # 连通性 / 模型节点权重头（用户批注 ToC1：校验必须作用到新节点）
            problems += _xcv_pack_problems(flow)
            return problems
        # 旧链流程（旧版流程文件）：校验模型节点与连接完整性
        model_nodes = [n for n in nodes
                       if str(n.get("type", "")).startswith(("model/", "xccv/model"))]
        if not model_nodes:
            problems.append("流程中没有模型节点（推理无从开始）")
        elif len(model_nodes) > 1:
            problems.append(f"流程中有 {len(model_nodes)} 个模型节点，运行时只会使用"
                            "第一个（请删掉多余模型节点）")
        for n in model_nodes:
            mf = (n.get("properties") or n.get("params") or {}).get("model_file")
            if not mf:
                problems.append("模型节点尚未选择权重文件（右键节点 → 选择模型文件）")
            else:
                problems += _model_file_problems(str(mf))
        # 连接完整性：从模型节点拓扑可达的后处理节点才会被执行
        chain = df.graph_to_chain(flow)
        declared = sum(1 for n in nodes
                       if str(n.get("type", "")).startswith(("post/", "xccv/post_")))
        reached = len(chain.get("nodes", []))
        if declared > reached:
            problems.append(
                f"有 {declared - reached} 个后处理节点未与模型节点连通"
                "（其参数不会生效），请检查连线")
    else:
        chain = flow

    # 参数合法性：区域、数量判定（链式与图格式统一按链检查）
    for i, node in enumerate(chain.get("nodes", []), start=1):
        op = node.get("op")
        params = node.get("params") or {}
        ctx = f"节点 {i}（{op}）："
        if op in ("region_block", "region_keep"):
            problems += df.region_param_problems(params.get("regions"), ctx)
        if op == "count_check":
            lo, hi = int(params.get("min") or 0), int(params.get("max") or 0)
            if lo and hi and lo > hi:
                problems.append(f"{ctx}数量判定下限 {lo} 大于上限 {hi}")
    return problems


def _show_pack_problems(parent, problems: list[str]) -> None:
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle("无法打包：请先解决以下问题")
    box.setText("打包前校验未通过（不产出坏包）：\n\n" + "\n".join(
        f"· {p}" for p in problems))
    box.exec_()



def _materialize_xvap(path: Path) -> tuple[Path, dict]:
    """把 .xvap 解到同级：内嵌模型写入 流程模型/，流程图写为 <同名>.json。

    返回 (流程 json 路径, 流程 dict)；模型引用键（model_file/model_path）
    的解包与绝对路径改写由 flow_package.unpack_to_dir 统一处理。
    """
    flow, _model_dir = fp.unpack_to_dir(path)
    flow_json = path.with_suffix(".json")
    _write_flow_json(flow_json, flow)
    return flow_json, flow

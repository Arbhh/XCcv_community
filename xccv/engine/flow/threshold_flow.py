"""O2 扩展（V0.7 用户追加）：阈值调优 → 流程图模板一键生成（纯逻辑）。

用户在阈值拖线面板各维度拖定阈值后，本模块把所选维度编排成新 IR 流程图
（schema:3，与模板库同款格式）：

    image_loader → model_detection(conf) → [面积] → [短长边]
                 → [前景占比] → [灰度差] → result_output

- conf 维度写进 model_detection 的 conf 参数（不单独建节点）；
- 短边/长边同时启用时合并为一个 filter_bbox_side 节点；
- model_path 写入当前权重绝对路径——生成即可直接运行；
- 落盘位置：`<模型存放文件夹>/flow/<时间戳>_<模型名>.json`（用户约定），
  流程图平台「打开流程」选取即可使用。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

_CONF = "conf"
_AREA = "area"
_SHORT = "short"
_LONG = "long"
_FG = "fg_ratio"
_GRAY = "gray_diff"

DIM_TITLES = {
    _CONF: "置信度",
    _AREA: "面积",
    _SHORT: "短边",
    _LONG: "长边",
    _FG: "前景占比",
    _GRAY: "灰度差",
}

# 参与流程图生成的维度（conf 由 model_detection.conf 承载，不单独建节点）
_FLOW_DIMS = (_AREA, _SHORT, _LONG, _FG, _GRAY, _CONF)


def build_threshold_graph(model_path: str, model_name: str, dims: dict,
                          task: str = "detection") -> dict:
    """维度阈值 dict → 新 IR 流程图（schema:3）。dims 取 _FLOW_DIMS 子集
    （值>0 才建过滤节点；conf 恒写入模型节点，缺省 0.25）。"""
    dims = {k: float(v) for k, v in (dims or {}).items() if float(v or 0) > 0}
    conf = dims.get(_CONF, 0.25)
    nodes: list = []
    links: list = []

    def add_node(ntype: str, props: dict, pos_x: int, n_in: int,
                 n_out: int) -> int:
        nid = len(nodes) + 1
        nodes.append({
            "id": nid, "type": ntype, "pos": [pos_x, 200],
            "properties": props,
            "inputs": [{"name": "image" if i == 0 else "result",
                        "type": "image" if i == 0 else "result", "link": None}
                       for i in range(n_in)],
            "outputs": [{"name": "image" if i == 0 else "result",
                         "type": "image" if i == 0 else "result", "links": []}
                        for i in range(n_out)]})
        return nid

    def connect(src_id: int, src_slot: int, dst_id: int, dst_slot: int) -> None:
        lid = len(links) + 1
        links.append([lid, src_id, src_slot, dst_id, dst_slot])
        nodes[src_id - 1]["outputs"][src_slot]["links"].append(lid)
        nodes[dst_id - 1]["inputs"][dst_slot]["link"] = lid

    # 1) 载图 → 2) 模型（conf 写参数）
    loader_id = add_node("xccv2/image_loader",
                         {"path": "", "is_grayscale": False}, 60, 0, 1)
    model_id = add_node("xccv2/model_detection",
                        {"model_path": str(model_path or ""),
                         "conf": round(conf, 2), "iou": 0.45,
                         "device": "auto"}, 340, 1, 2)
    connect(loader_id, 0, model_id, 0)

    # 3) 过滤节点（按用户拖定维度依次串联；image/result 双通道）
    img_src, res_src = model_id, model_id

    def add_filter(ntype: str, props: dict) -> None:
        nonlocal img_src, res_src
        nid = add_node(ntype, props, 340 + 280 * (len(nodes) - 1), 2, 3)
        connect(img_src, 0, nid, 0)
        connect(res_src, 1, nid, 1)
        img_src, res_src = nid, nid

    if _AREA in dims:
        # 面积过滤用既有 legacy 薄委托节点 filter_bbox_area（int 参数口径）
        add_filter("xccv2/filter_bbox_area",
                   {"min": int(dims[_AREA]), "max": 0})
    if _SHORT in dims or _LONG in dims:
        add_filter("xccv2/filter_bbox_side",
                   {"short_min": dims.get(_SHORT, 0.0),
                    "long_min": dims.get(_LONG, 0.0)})
    if _FG in dims:
        add_filter("xccv2/filter_foreground",
                   {"min_ratio": dims[_FG], "gray_thr": 20})
    if _GRAY in dims:
        add_filter("xccv2/filter_gray_diff",
                   {"min_diff": dims[_GRAY]})

    # 4) 结果输出
    out_id = add_node("xccv2/result_output", {"out_dir": ""},
                      340 + 280 * len(nodes), 2, 0)
    connect(img_src, 0, out_id, 0)
    connect(res_src, 1, out_id, 1)

    enabled = "、".join(DIM_TITLES[k] for k in _FLOW_DIMS if k in dims) or "无"
    return {
        "template": "threshold_tuned",
        "schema": 3,
        "title": f"阈值调优流程（{model_name}）",
        "description": (
            f"由测试平台阈值面板一键生成（{time.strftime('%Y-%m-%d %H:%M')}）："
            f"模型 {model_name}，启用过滤维度：{enabled}。模型路径已填当前"
            "权重，可直接运行；各节点参数可在属性面板继续微调。"),
        "last_node_id": len(nodes),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
    }


def flow_filename(model_name: str) -> str:
    """流程文件名：`<时间戳>_<模型名>.json`（用户约定；模型名做文件名净化）。"""
    stem = re.sub(r'[\\/:*?"<>|]+', "_", str(model_name or "model")).strip() \
        or "model"
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{stem}.json"


def save_threshold_flow(model_dir, model_name: str, dims: dict,
                        model_path: str, task: str = "detection") -> Path:
    """生成并落盘到 `<model_dir>/flow/<时间戳>_<模型名>.json`，返回文件路径。"""
    out_dir = Path(model_dir) / "flow"
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = build_threshold_graph(model_path, model_name, dims, task)
    path = out_dir / flow_filename(model_name)
    path.write_text(json.dumps(graph, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return path

"""通用视觉流程子系统（V0.4 批次十四起）：类型化端口的节点图引擎。

- ir.py        数据中间表示（ImageData/Geometry/MeasureResult/NumberVal）+ 端口类型表
- node_base.py VisionNode 基类 / Port / ParamSpec / RunContext
- registry.py  节点注册表（@register_node），同时反哺前端节点库
- executor.py  Kahn 拓扑执行 + 环检测 + 类型检查 + 会话缓存 + 单节点重跑
- snapshot.py  节点调试快照（会话内存 + 降采样）

包纪律：本包禁止 import 任何 Qt 模块（保证无头可测，测试有导入断言）；
图像引用传递零拷贝；算法逻辑全部在 Python 侧（前端只做图编辑与展示）。
"""

from .ir import (
    PORT_TYPES, Geometry, ImageData, MeasureResult, NumberVal,
)
from .node_base import ParamSpec, Port, RunContext, VisionNode
from .registry import get_node_class, list_nodes, register_node
from .executor import Executor, FlowError, NodeResult
from .snapshot import SnapshotStore

__all__ = [
    "PORT_TYPES", "Geometry", "ImageData", "MeasureResult", "NumberVal",
    "ParamSpec", "Port", "RunContext", "VisionNode",
    "get_node_class", "list_nodes", "register_node",
    "Executor", "FlowError", "NodeResult", "SnapshotStore",
]

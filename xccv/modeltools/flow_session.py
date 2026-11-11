"""新引擎画布运行会话（批次十九）：运行到此节点 / 单节点重跑 / 快照 / 日志。

- RunContext 跨调用持有：AI 模型缓存（ai_model:<path>）、会话警告、
  节点日志与快照存储在"同一画布会话"内保留——单节点重跑命中缓存 O(1)；
- 图 JSON 变化（重新序列化）时重建 Executor（拓扑/参数重新解析）；
- 汇总结果供桥接下发：{focus, errors, logs, snapshot_ids, status}。
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .. import app_paths
from ..engine.flow import Executor, RunContext, SnapshotStore
from ..engine.flow.executor import format_session_logs
from ..engine.flow.snapshot import to_png_base64

_TEMPLATE_DIR = Path(__file__).resolve().parent / "flow_templates"
_RUN_LOG_DIR = Path(app_paths.app_root()) / "platform_data" / "logs" / "流程运行日志"
_RUN_LOG_KEEP = 50    # 保留最近 N 份运行日志，超出自动清理（用户批注 Q3：50 份即可）


class FlowErrorMixed(Exception):
    """新旧节点混搭（画布图里同时出现新算子与旧版节点）。"""


def ensure_ops_registered() -> None:
    """确保全部算子（含旧链薄委托节点）已注册（幂等）。

    新引擎的所有入口（画布会话/流程图页/批量线程/测试平台）都会经
    FlowSession——在此统一注册，调用方无需各自处理（v0.4.4 修复：
    此前只有画布节点库下发路径会注册，其他入口在生产环境首次运行
    会把全部节点误判为"旧版节点"）。
    """
    import importlib

    importlib.import_module("..engine.flow.legacy_adapter",
                            package=__package__)
    from ..engine.flow.ops import load_all

    load_all()


def list_templates() -> list:
    """官方流程模板清单：[{title, description, template, path, flow}]。

    模板标记（template 字段等）存流程 JSON 内部字段——.xvap 容器字节
    布局零改动（计划书 §5.5）。
    """
    out = []
    if _TEMPLATE_DIR.is_dir():
        for p in sorted(_TEMPLATE_DIR.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({"title": str(data.get("title") or p.stem),
                        "description": str(data.get("description") or ""),
                        "template": str(data.get("template") or p.stem),
                        "path": str(p), "flow": data})
    return out


def node_library_payload() -> dict:
    """节点库 + 端口类型表下发数据（Python 唯一事实源，Q3/计划书 §5.6）。

    无 Qt 依赖：画布桥接与「传统平台」算子工作台共用（本函数首次调用时
    完成全部算子注册，幂等）。
    """
    from ..engine.flow.ir import PORT_TYPES
    from ..engine.flow.ops import load_all
    from ..engine.flow.registry import list_nodes
    from ..engine.flow import legacy_adapter as _la  # noqa: F401  过滤节点注册

    load_all()
    return {"event": "node_library", "port_types": PORT_TYPES,
            "nodes": list_nodes()}


def _graph_key(flow: dict) -> str:
    import json

    return hashlib.sha1(json.dumps(flow, sort_keys=True, ensure_ascii=False)
                        .encode("utf-8")).hexdigest()


def _normalize_types(flow: dict) -> dict:
    """画布节点类型 "xccv2/<注册名>" → 注册表键 "<注册名>"（浅拷贝，不改原图）。

    旧 xccv/ 类型不归新引擎管（由检测过滤链引擎处理），保持原样——
    Executor 会对未注册类型报错，正好把混入旧节点的图挡在新引擎之外。
    同时做旧端口布局迁移（修复循环 3 端口统一，port_migrate.migrate_ports）。
    """
    from ..engine.flow.port_migrate import migrate_ports

    flow = migrate_ports(flow)
    nodes = flow.get("nodes") or []
    out = dict(flow)
    out["nodes"] = [
        {**n, "type": str(n.get("type", ""))[6:]}
        if str(n.get("type", "")).startswith("xccv2/") else n
        for n in nodes
    ]
    return out


class FlowSession:
    """一个画布窗口一份：桥接层持有，run / rerun / snapshot 三组能力。"""

    def __init__(self):
        ensure_ops_registered()
        self.ctx = RunContext(snapshot_store=SnapshotStore())
        self.executor: Executor | None = None
        self._key: str | None = None

    def _build(self, flow: dict) -> Executor:
        norm = _normalize_types(flow)
        self._check_mixed(norm)
        key = _graph_key(norm)
        if self.executor is None or key != self._key:
            self.ctx.logs.clear()
            self.ctx.cache_clear()   # 图已变化：节点输出缓存整体失效（模型加载等共享键缓存保留）
            self.executor = Executor(norm, self.ctx)
            self._key = key
        return self.executor

    @staticmethod
    def _check_mixed(norm: dict) -> None:
        """新旧节点混搭给出人话报错（否则执行器只报"未注册"，难定位）。"""
        from ..engine.flow.registry import get_node_class

        legacy = [str(n.get("type", "")) for n in norm.get("nodes") or []
                  if get_node_class(str(n.get("type", ""))) is None]
        if legacy:
            raise FlowErrorMixed(
                "流程混用了旧版节点：" + "、".join(sorted(set(legacy))
                ) + "；请在「传统算子」面板改用对应新节点，"
                "或仅用旧版节点走「运行预览」旧链")

    def run(self, flow: dict, to_node: int | None = None, label: str = "",
            flow_dir: str | Path | None = None, source_name: str = "",
            flow_name: str = "") -> dict:
        """全图运行；to_node 给定时只执行其上游闭包（运行到此节点）。

        flow_dir/source_name/flow_name 写入会话共享键（输出类节点落盘
        定位与结果 JSON 的 source/flow_name 用）。
        """
        ex = self._build(flow)
        if self._key != _graph_key(_normalize_types(flow)) or to_node is None:
            self.ctx.snapshot_store.clear()
        if flow_dir is not None:
            self.ctx.set("flow_dir", str(flow_dir))
        if source_name:
            self.ctx.set("source_name", source_name)
        if flow_name:
            self.ctx.set("flow_name", flow_name)
        results = ex.run(to_node=to_node)
        summary = self._summarize(results, focus=to_node)
        self.write_run_log(summary, label=label)
        return summary

    def rerun(self, node_id: int, label: str = "") -> dict:
        """仅重跑单节点（上游取上次输出，命中会话缓存时 O(1)）。"""
        if self.executor is None:
            raise RuntimeError("尚未运行过流程，无法单节点重跑")
        res = self.executor.rerun_node(node_id)
        summary = self._summarize({node_id: res}, focus=node_id)
        self.write_run_log(summary, label=f"重跑 #{node_id} {label}".strip())
        return summary

    @staticmethod
    def write_run_log(summary: dict, label: str = "") -> None:
        """每次运行落一份日志到 platform_data/logs/流程运行日志/（Q4 用户决策）。

        只保留最近 _RUN_LOG_KEEP 份；写日志失败不影响运行（best-effort）。
        """
        try:
            _RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            head = [f"流程运行日志  {stamp}",
                    f"来源：{label or '画布运行'}",
                    f"结果：{summary['ok_count']}/{summary['total']} 节点成功"
                    + (f"，{len(summary['errors'])} 个失败" if summary["errors"] else "")]
            for nid, err in (summary.get("errors") or {}).items():
                head.append(f"  ✘ 节点#{nid}：{err}")
            body = summary.get("logs") or "（无节点日志）"
            (_RUN_LOG_DIR / f"flow_{stamp}.txt").write_text(
                "\n".join(head) + "\n" + "-" * 46 + "\n" + body + "\n",
                encoding="utf-8")
            olds = sorted(_RUN_LOG_DIR.glob("flow_*.txt"), key=lambda p: p.stat().st_mtime)
            for p in olds[:-_RUN_LOG_KEEP] if len(olds) > _RUN_LOG_KEEP else []:
                p.unlink(missing_ok=True)
        except OSError:
            pass

    def _summarize(self, results: dict, focus) -> dict:
        from ..engine.result_json import verdict_from_node_results

        errors = {str(r.node_id): r.error for r in results.values() if not r.ok}
        ok_n = sum(1 for r in results.values() if r.ok)
        return {"focus": focus, "errors": errors, "ok_count": ok_n,
                "total": len(results),
                "logs": format_session_logs(self.ctx.logs),
                "warnings": list(self.ctx.warnings)[-5:],
                "snapshot_ids": self.ctx.snapshot_store.node_ids(),
                "verdict": verdict_from_node_results(results)}

    def snapshot_payload(self, node_id: int) -> dict:
        """节点快照 → base64 PNG 字典（前端 <img> 直用）。"""
        item = self.ctx.snapshot_store.get(node_id)
        if not item:
            return {}
        return {name: to_png_base64(img)
                for name, img in item["images"].items()}

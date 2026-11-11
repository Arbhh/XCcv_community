"""拓扑执行引擎：Kahn 排序 + 环检测（报环路径）+ 端口类型检查 + 会话缓存。

- 输入图 JSON 与画布 serialize 同源（LiteGraph 兼容 nodes/links；
  链接兼容 5 元与 6 元格式，见坑 17/第五批）；
- 端口按槽序对齐注册类的 INPUTS/OUTPUTS；一个输入端口只允许一条入线；
- 缓存 = 会话内记忆化：键 (node_id, params_hash, 各输入对象 id)，
  生命周期与 RunContext 相同——"单节点重跑"是调试语义，不做跨运行缓存；
- 错误隔离：单节点异常记入 NodeResult，其下游标记 skipped，独立分支继续；
- **并行输入**（修复循环 4，用户批注：统一图像端口、按并行输入设计）：
  任一输入收到多张集合（ImageList/ResultList）且节点非 HANDLES_LIST 时，
  节点按集合逐张自动映射（_map_over）——标量输入广播，输出按端口类型
  重新集合成 ImageList / ResultList；节点实现只面向单张编程。
"""

import hashlib
import logging
import time
import traceback
from dataclasses import dataclass, field

from .ir import (PORT_TYPES, Geometry, ImageData, ImageList, ResultItem,
                 ResultList, result_kind)
from .node_base import RunContext
from .registry import get_node_class

_LOG = logging.getLogger("xccv.engine.flow")   # 平台日志系统统一捕获


class FlowError(Exception):
    """图结构错误（未知节点类型 / 端口类型不匹配 / 有环 / 多重输入）。"""


@dataclass
class NodeResult:
    """单节点执行结果（UI 错误标红与快照查看的数据源）。"""

    node_id: int
    name: str
    ok: bool
    error: str = ""
    outputs: dict = field(default_factory=dict)
    cached: bool = False
    skipped: bool = False


def _params_hash(params: dict) -> str:
    text = repr(sorted(params.items(), key=lambda kv: kv[0]))
    return hashlib.sha1(text.encode("utf-8", "surrogateescape")).hexdigest()[:12]


def _node_params(spec: dict) -> dict:
    """画布节点 properties（兼容 params）里的用户参数。"""
    return dict(spec.get("properties") or spec.get("params") or {})


class Executor:
    """一次图执行的会话：validate → run(to_node=) / rerun_node()。"""

    def __init__(self, graph: dict, ctx: RunContext | None = None,
                 external: dict | None = None):
        """external：{(节点id, 输入槽): 数据}——无连线的输入槽从这里取值
        （自定义复合算子把外部输入注入子流程，V0.6 T4-4；默认 None 行为不变）。"""
        self.ctx = ctx or RunContext()
        self._external = dict(external or {})
        self._specs: dict = {}
        for n in graph.get("nodes", []):
            nid = int(n["id"])
            if nid in self._specs:
                raise FlowError(f"节点 id 重复：{nid}")
            self._specs[nid] = n
        self._in_links: dict = {}    # (dst_id, dst_slot) -> (src_id, src_slot)
        self._out_links: dict = {}   # (src_id, src_slot) -> [(dst_id, dst_slot)]
        for lk in graph.get("links", []) or []:
            src, dst = int(lk[1]), int(lk[3])
            sslot, dslot = int(lk[2]), int(lk[4])
            if (dst, dslot) in self._in_links:
                dst_name = self._port_name(dst, dslot, output=False)
                raise FlowError(f"节点 {dst} 的输入端口 {dst_name} 存在多条输入连线")
            self._in_links[(dst, dslot)] = (src, sslot)
            self._out_links.setdefault((src, sslot), []).append((dst, dslot))
        self._outputs: dict = {}     # node_id -> {输出端口名: 数据}（rerun 用）
        self._params: dict = {}      # node_id -> 合并默认后的参数
        self._results: dict = {}     # node_id -> NodeResult

    # ---------- 结构校验 ----------
    def _class_of(self, node_id: int):
        t = str(self._specs[node_id].get("type", ""))
        cls = get_node_class(t)
        if cls is None:
            raise FlowError(f"节点 {node_id} 的类型 {t!r} 未注册")
        return t, cls

    def _port_name(self, node_id: int, slot: int, output: bool) -> str:
        _t, cls = self._class_of(node_id)
        ports = cls.OUTPUTS if output else cls.INPUTS
        if 0 <= slot < len(ports):
            return ports[slot].name
        return f"#{slot}"

    def validate(self) -> None:
        """类型/结构静态校验：未知类型、槽越界、类型不匹配、有环。"""
        for nid in self._specs:          # 无连线的节点也要做类型注册校验
            self._class_of(nid)
        deps: dict = {nid: set() for nid in self._specs}   # dst -> src 集合
        for (dst, dslot), (src, sslot) in self._in_links.items():
            if src not in self._specs or dst not in self._specs:
                raise FlowError(f"连线引用了不存在的节点：{src}→{dst}")
            t_src, cls_src = self._class_of(src)
            t_dst, cls_dst = self._class_of(dst)
            if not (0 <= sslot < len(cls_src.OUTPUTS)):
                raise FlowError(f"节点 {src}（{t_src}）输出槽 {sslot} 越界")
            if not (0 <= dslot < len(cls_dst.INPUTS)):
                raise FlowError(f"节点 {dst}（{t_dst}）输入槽 {dslot} 越界")
            pt_src = cls_src.OUTPUTS[sslot].type
            port_dst = cls_dst.INPUTS[dslot]
            pt_dst = port_dst.type
            if pt_src != pt_dst and pt_src not in port_dst.accepts:
                name = PORT_TYPES.get(pt_src, {}).get("desc", pt_src)
                name2 = PORT_TYPES.get(pt_dst, {}).get("desc", pt_dst)
                raise FlowError(
                    f"端口类型不匹配：{src}:{cls_src.OUTPUTS[sslot].name}"
                    f"（{name}）→ {dst}:{port_dst.name}（{name2}）")
            deps[dst].add(src)
        self._check_cycle(deps)

    def _check_cycle(self, deps: dict) -> None:
        """Kahn 拓扑；有剩余节点即存在环，DFS 还原一条环路径用于报错。"""
        indeg = {n: len(s) for n, s in deps.items()}
        succ: dict = {}
        for dst, srcs in deps.items():
            for s in srcs:
                succ.setdefault(s, []).append(dst)
        queue = [n for n, d in indeg.items() if d == 0]
        seen = 0
        while queue:
            n = queue.pop()
            seen += 1
            for m in succ.get(n, []):
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        if seen == len(deps):
            return
        left = {n for n, d in indeg.items() if d > 0}

        def dfs(n, path):
            if n in path:
                start = path.index(n)
                chain = [self._specs[i].get("type", i) for i in path[start:]]
                return " → ".join(chain + [chain[0]])
            for m in succ.get(n, []):
                if m in left:
                    got = dfs(m, path + [n])
                    if got:
                        return got
            return None

        raise FlowError(f"流程图存在环：{dfs(sorted(left)[0], [])}")

    # ---------- 拓扑与执行 ----------
    def topo_order(self, to_node: int | None = None) -> list:
        """Kahn 序；to_node 给定时只返回该节点的上游闭包（含自身）。"""
        self.validate()
        deps = {nid: set() for nid in self._specs}
        succ: dict = {}
        for (dst, _ds), (src, _ss) in self._in_links.items():
            deps[dst].add(src)
            succ.setdefault(src, []).append(dst)
        if to_node is not None:
            keep = set()
            stack = [to_node]
            while stack:
                n = stack.pop()
                if n in keep:
                    continue
                keep.add(n)
                stack.extend(deps[n])
            deps = {n: {s for s in srcs if s in keep}
                    for n, srcs in deps.items() if n in keep}
            succ = {n: [m for m in ms if m in keep] for n, ms in succ.items()
                    if n in keep}
        indeg = {n: len(s) for n, s in deps.items()}
        order, queue = [], sorted(n for n, d in indeg.items() if d == 0)
        while queue:
            n = queue.pop(0)
            order.append(n)
            for m in sorted(succ.get(n, [])):
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        return order

    def _gather_inputs(self, node_id: int):
        _t, cls = self._class_of(node_id)
        inputs = {}
        for slot, port in enumerate(cls.INPUTS):
            link = self._in_links.get((node_id, slot))
            if link is None:
                if (node_id, slot) in self._external:
                    inputs[port.name] = self._external[(node_id, slot)]
                    continue
                if port.required:
                    raise RuntimeError(f"缺少必需输入「{port.name}」")
                continue
            src, sslot = link
            src_out = self._outputs.get(src)
            if src_out is None:
                raise RuntimeError(f"上游节点 {src} 尚未产出结果")
            src_cls = self._class_of(src)[1]
            oname = src_cls.OUTPUTS[sslot].name
            if oname not in src_out:
                raise RuntimeError(f"上游节点 {src}（{src_cls.NAME}）本次运行未产出"
                                   f"「{oname}」输出（该节点本次无此输出）")
            inputs[port.name] = src_out[oname]
        return inputs

    def _fill_params(self, node_id: int) -> dict:
        _t, cls = self._class_of(node_id)
        merged = {k: s.default for k, s in cls.PARAMS.items()}
        merged.update(_node_params(self._specs[node_id]))
        return merged

    # ---------- 并行输入：多张集合的逐张自动映射（修复循环 4） ----------
    @staticmethod
    def _bundles(inputs: dict) -> dict:
        """输入中的多张集合（长度 ≠1 的 ImageList/ResultList；0 张也算集合）。"""
        return {k: v for k, v in inputs.items()
                if isinstance(v, (ImageList, ResultList)) and len(v) != 1}

    @staticmethod
    def _in_roi(inputs: dict, k: int) -> list | None:
        """第 k 张对应的原图矩形（优先取图像集合项的 roi_rect）。"""
        for v in inputs.values():
            if isinstance(v, ImageList) and len(v) > k:
                r = v.items[k].meta.get("roi_rect")
                return [int(t) for t in r] if r else None
        for v in inputs.values():
            if isinstance(v, ResultList) and len(v) > k:
                return v.items[k].roi_rect
        return None

    @staticmethod
    def _unwrap_at(v, k: int):
        """集合输入的第 k 张原值：图像集合→ImageData；结果集合→ResultItem.value
        （节点面向单张原值编程，不感知 ResultItem 壳）。"""
        if isinstance(v, ImageList):
            return v.items[k]
        if isinstance(v, ResultList):
            return v.items[k].value
        return v

    def _invoke_node(self, cls, inputs: dict, params: dict) -> dict:
        """节点调用入口：需要时先做多张集合的逐张映射（并行输入语义）。"""
        if not getattr(cls, "HANDLES_LIST", False):
            bundles = self._bundles(inputs)
            if bundles:
                return self._map_over(cls, inputs, params, bundles)
        return cls().run(inputs, params, self.ctx)

    def _map_over(self, cls, inputs: dict, params: dict, bundles: dict) -> dict:
        """逐张映射：标量输入广播、集合输入按 k 取第 k 张，输出重新集合。

        - 图像输出 → ImageList（每张的 meta 继承输入项 meta——roi_rect/
          source_size 得以沿链传递，结果合并换算原图坐标依赖它）；
        - 其余输出（result/number）→ ResultList（ResultItem 带 roi_rect
          与 index）；映射内层已产出集合（ImageList/ResultList）时直接
          摊平（如整条链都是逐张节点）。
        - 任一张失败即整节点失败（错误信息带张序号，与单节点错误隔离口径
          一致：下游 skipped，独立分支继续）。
        """
        lengths = {k: len(v) for k, v in bundles.items()}
        n = next(iter(lengths.values()))
        if any(v != n for v in lengths.values()):
            raise ValueError(
                "并行输入的张数不一致（" + "、".join(
                    f"{k}={v}" for k, v in lengths.items())
                + "）；请检查上游集合是否来自同一来源")
        collected: dict = {}
        for k in range(n):
            sub = {name: self._unwrap_at(v, k) if isinstance(v, (ImageList, ResultList))
                   else v for name, v in inputs.items()}
            try:
                out = cls().run(sub, params, self.ctx)
            except Exception as e:
                raise ValueError(f"第 {k} 张执行失败：{e}") from None
            for name, v in out.items():
                is_image = any(p.name == name and p.type == "image"
                               for p in cls.OUTPUTS)
                bucket = collected.setdefault(name, [])
                if is_image and isinstance(v, ImageData):
                    base = {}
                    for iv in sub.values():
                        if isinstance(iv, ImageData):
                            base = dict(iv.meta)
                            break
                    bucket.append(ImageData(mat=v.mat, meta={**base, **dict(v.meta)}))
                elif isinstance(v, (ImageList, ResultList)):
                    bucket.extend(v.items)
                else:
                    bucket.append(ResultItem(kind=result_kind(v), value=v,
                                             roi_rect=self._in_roi(inputs, k),
                                             meta={"index": len(bucket)}))
        merged: dict = {}
        for name, items in collected.items():
            if not items:
                continue
            if isinstance(items[0], ImageData):
                merged[name] = ImageList(items=items)
            else:
                merged[name] = ResultList(items=items)
        return merged

    def _run_one(self, node_id: int, use_cache: bool = True) -> NodeResult:
        t, cls = self._class_of(node_id)
        params = self._fill_params(node_id)
        t0 = time.perf_counter()
        try:
            inputs = self._gather_inputs(node_id)
        except Exception as e:
            res = NodeResult(node_id, t, ok=False, error=str(e), skipped=True)
            self._results[node_id] = res
            self.ctx.node_log(node_id, t, f"跳过：{e}", level="WARN")
            _LOG.warning("节点 %s(#%s) 跳过：%s", t, node_id, e)
            return res
        key = (node_id, _params_hash(params), self.ctx.revision,
               tuple(sorted((k, id(v)) for k, v in inputs.items())))
        if use_cache:
            cached = self.ctx.cache_get(key)
            if cached is not None:
                res = NodeResult(node_id, t, ok=True, outputs=cached, cached=True)
                self._outputs[node_id] = cached
                self._results[node_id] = res
                self._params[node_id] = params
                self.ctx.node_log(node_id, t, "命中会话缓存（未重新计算）")
                _LOG.debug("节点 %s(#%s) 命中缓存", t, node_id)
                return res
        try:
            errs = cls().validate(params)
            if errs:
                raise ValueError("；".join(errs))
            out = self._invoke_node(cls, inputs, params)
            if not isinstance(out, dict):
                raise TypeError(f"run() 应返回 dict，得到 {type(out).__name__}")
            allowed = {p.name for p in cls.OUTPUTS}
            bad = set(out) - allowed
            if bad:
                raise ValueError(f"run() 返回了未声明的输出端口：{sorted(bad)}")
            self.ctx.cache_set(key, out)
            res = NodeResult(node_id, t, ok=True, outputs=out)
            if self.ctx.snapshot_store is not None:
                self._save_snapshot(node_id, inputs, out)
            dur = (time.perf_counter() - t0) * 1000.0
            self.ctx.node_log(node_id, t, f"执行成功（{dur:.1f} ms）")
            _LOG.info("节点 %s(#%s) 执行成功（%.1f ms）", t, node_id, dur)
        except Exception as e:  # 错误隔离：记录后由下游标记 skipped
            tb = traceback.format_exc(limit=3)
            res = NodeResult(node_id, t, ok=False, error=f"{e}")
            self.ctx.node_log(node_id, t,
                              f"执行失败：{e}（{tb.splitlines()[-1] if tb else ''}）",
                              level="ERROR")
            _LOG.error("节点 %s(#%s) 执行失败：%s", t, node_id, e)
        self._outputs[node_id] = res.outputs if res.ok else None
        self._results[node_id] = res
        self._params[node_id] = params
        return res

    def _save_snapshot(self, node_id: int, inputs: dict, out: dict) -> None:
        """节点成功后自动捕获调试快照。

        - 有 image 输出：存输出图（降采样副本），geometry 存叠加要素；
        - 无 image 输出（定位/逻辑/过滤类节点）：把结果叠加到首个 image
          输入上生成"输入+结果叠加"快照（blob/find_* 等节点可见结果）。
        """
        from .ir import ImageList, ResultList
        from .snapshot import mosaic, overlay_result

        images: dict = {}
        overlays: dict = {}
        for name, v in out.items():
            if isinstance(v, ImageData):
                images[str(name)] = v.mat
            elif isinstance(v, ImageList):   # 列表端口：拼图快照（F2-2）
                images[f"{name}（{len(v)} 张拼图）"] = mosaic([it.mat for it in v.items])
            elif isinstance(v, ResultList):   # 逐图结果：叠加到对应输入图后拼图
                src = next((iv for iv in inputs.values() if isinstance(iv, ImageList)), None)
                if src is not None and len(src):
                    try:
                        by_index = {it.meta.get("index", i): it for i, it in enumerate(v.items)}
                        images[f"{name}（逐图叠加拼图）"] = mosaic(
                            [overlay_result(img.mat, {"r": by_index[k].value})
                             if k in by_index else img.mat
                             for k, img in enumerate(src.items)])
                    except Exception:
                        pass
            elif isinstance(v, Geometry):
                overlays[str(name)] = v.to_json()
        if not images:
            for v in inputs.values():
                if isinstance(v, ImageData) and v.mat.size:
                    try:
                        images["输入+结果叠加"] = overlay_result(v.mat, out)
                    except Exception:
                        pass  # 叠加绘制失败不影响运行
                    break
        if images or overlays:
            self.ctx.snapshot_store.save(node_id, images, overlays)

    def run(self, to_node: int | None = None) -> dict:
        """按拓扑序执行全图（或 to_node 的上游前缀）。返回 {node_id: NodeResult}。"""
        self._outputs.clear()
        self._results.clear()
        for nid in self.topo_order(to_node):
            self._run_one(nid)
        return {k: v for k, v in self._results.items()}

    def rerun_node(self, node_id: int) -> NodeResult:
        """仅重跑单节点：上游取上一次运行的输出（命中缓存时 O(1)）。"""
        if node_id not in self._specs:
            raise FlowError(f"节点 {node_id} 不存在")
        return self._run_one(node_id, use_cache=True)

    @property
    def results(self) -> dict:
        return self._results

    @property
    def session_logs(self) -> list:
        """本次会话的节点级结构化日志（ctx.logs 的引用）。"""
        return self.ctx.logs


def format_session_logs(logs: list) -> str:
    """结构化节点日志 → 人读文本（UI 日志面板 / 日志文件落盘用）。"""
    lines = []
    for e in logs:
        lines.append(f"[{e['level']}] {e['node']}(#{e['node_id']})：{e['message']}")
    return "\n".join(lines)

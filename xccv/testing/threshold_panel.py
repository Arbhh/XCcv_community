"""O2（V0.7）+ O2 扩展（V0.7 用户追加）：批量分数/维度分布 + 拖线阈值面板。

批量完成后，把 GT 匹配统计（engine.metrics.batch_match 的 match_stats）
画成红绿堆叠直方图：绿=命中 GT（TP）、红=未命中（FP，其中 OK 图上的
误检单独一色）。拖动竖线即在零推理成本下实时刷新该阈值下的精确率 /
召回率 / OK 误检（框/张）。

用户追加扩展——多维度阈值：
- 置信度 conf / 面积 px² / 短边 px / 长边 px / 前景占比 / 灰度差 六个维度
  可切换；每个维度独立记忆阈值与"参与过滤"开关；
- 拖线自动启用该维度；"一键生成流程图模板"把所有启用维度编排成新 IR
  流程图（threshold_flow），落在 <模型目录>/flow/<时间>_<模型名>.json；
- "应用为项目默认检出流程 conf" 维持既有语义（只写 conf 维）。

纯展示组件：数据经 set_stats 注入；applyRequested/generateFlowRequested
由宿主窗口接线落盘。
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGridLayout,
                             QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                             QWidget)

from ..engine.metrics.batch_match import threshold_metrics, threshold_metrics_for
from ..engine.flow.threshold_flow import DIM_TITLES
from ..ui_style import token

_N_BINS = 25
_GREEN = "#2E9E5B"   # TP（命中 GT）
_RED = "#D64545"     # FP（有 GT 图上的误检）
_AMBER = "#E08A1E"   # FP·OK 图（OK 误检，独立图例）

# 维度 key → (显示名, 单位)；score=conf 走既有 0~1 通道
_DIMS = (("score", "置信度 conf", ""),
         ("area", "面积", "px²"),
         ("short", "短边", "px"),
         ("long", "长边", "px"),
         ("fg_ratio", "前景占比", ""),
         ("gray_diff", "灰度差", ""))
_DIM_DEFAULTS = {"area": 0.0, "short": 0.0, "long": 0.0,
                 "fg_ratio": 0.05, "gray_diff": 15.0}


def _dim_label(key: str) -> str:
    for k, name, unit in _DIMS:
        if k == key:
            return f"{name}（{unit}）" if unit else name
    return key


class ThresholdPanel(QWidget):
    """分数/维度分布直方图 + 可拖阈值线 + 实时指标 + 应用/生成按钮。"""

    thresholdChanged = pyqtSignal(float)
    applyRequested = pyqtSignal(float)
    generateFlowRequested = pyqtSignal(dict)
    # 项目复盘页（用户勘误）：任一维度阈值/启用状态变化 → {dim: thr}（启用集）
    thresholdsChanged = pyqtSignal(dict)

    def __init__(self, parent=None, compact: bool = False):
        """compact=True：上下布局（直方图整幅在上、指标/按钮在下）。

        项目复盘页左列较窄，横排布局会把直方图挤成窄条（再勘误 ②）；
        测试平台旧内嵌处维持 compact=False 的横排原样。
        """
        super().__init__(parent)
        self._stats: dict = {}
        self._current = "score"
        # 每维度独立状态：{key: {"thr": float, "enabled": bool}}
        self._dim_state: dict = {"score": {"thr": 0.25, "enabled": True}}
        lay = (QVBoxLayout(self) if compact else QHBoxLayout(self))
        lay.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(None)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "置信度分数（IoU 匹配口径）")
        self.plot.setLabel("left", "检出数")
        self.plot.addLegend(offset=(8, 8))
        self._line = pg.InfiniteLine(
            pos=0.25, angle=90, movable=True,
            pen=pg.mkPen(token("primary"), width=2, style=Qt.DashLine),
            label="conf")
        self.plot.addItem(self._line)
        self._line.sigPositionChanged.connect(self._on_line_moved)
        if compact:
            self.plot.setMinimumHeight(190)
            self.plot.setMaximumHeight(230)
            lay.addWidget(self.plot, stretch=1)
            right = QGridLayout()
            right.setHorizontalSpacing(14)
            right.setVerticalSpacing(6)
        else:
            lay.addWidget(self.plot, stretch=3)
            right = QFormLayout()

        self.cmb_dim = QComboBox()
        for key, name, unit in _DIMS:
            self.cmb_dim.addItem(name, key)
        self.cmb_dim.currentIndexChanged.connect(self._on_dim_changed)
        self.lbl_thr = QLabel("—")
        self.lbl_thr.setStyleSheet("font-weight: bold;")
        self.cb_enable = QCheckBox("该维度参与流程图过滤")
        self.cb_enable.stateChanged.connect(self._on_enable_toggled)
        self.lbl_precision = QLabel("—")
        self.lbl_recall = QLabel("—")
        self.lbl_ok = QLabel("—")
        if compact:
            # 行1：维度 + 阈值 + 参与过滤；行2：三指标；行3：两按钮
            right.addWidget(QLabel("维度"), 0, 0)
            right.addWidget(self.cmb_dim, 0, 1)
            right.addWidget(QLabel("当前阈值"), 0, 2)
            right.addWidget(self.lbl_thr, 0, 3)
            right.addWidget(self.cb_enable, 0, 4, 1, 2)
            right.addWidget(QLabel("精确率 ≥"), 1, 0)
            right.addWidget(self.lbl_precision, 1, 1)
            right.addWidget(QLabel("召回率 ≥"), 1, 2)
            right.addWidget(self.lbl_recall, 1, 3)
            right.addWidget(QLabel("OK 误检 框/张"), 1, 4)
            right.addWidget(self.lbl_ok, 1, 5)
            right.setColumnStretch(6, 1)
        else:
            for row, w in (("维度", self.cmb_dim),
                           ("当前阈值", self.lbl_thr),
                           ("", self.cb_enable),
                           ("精确率（≥阈值）", self.lbl_precision),
                           ("召回率（≥阈值）", self.lbl_recall),
                           ("OK 误检（框/张）", self.lbl_ok)):
                if row:
                    right.addRow(row, w)
                else:
                    right.addRow(w)
        lay.addLayout(right)

        self.lbl_hint = QLabel("")
        self.lbl_hint.setWordWrap(True)
        lay.addWidget(self.lbl_hint)

        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton("应用为默认 conf")
        self.btn_apply.setToolTip(
            "写入该项目「检出流程.json」的置信度过滤节点（含变更留痕备注）；\n"
            "测试平台与部署服务此后按该阈值过滤检出")
        self.btn_apply.clicked.connect(self._emit_apply_conf)
        btn_row.addWidget(self.btn_apply, 1)
        self.btn_flow = QPushButton("生成流程图模板")
        self.btn_flow.setToolTip(
            "在模型存放文件夹下新建 flow/ 子文件夹，按当前启用的维度阈值"
            "生成流程图模板：<时间>_<模型名>.json（模型路径已填当前权重，"
            "生成即可在流程图平台打开运行）")
        self.btn_flow.clicked.connect(self._emit_generate_flow)
        btn_row.addWidget(self.btn_flow, 1)
        lay.addLayout(btn_row)

        # O18 上线检查单（与阈值调参同处提示；VIMO 产线过漏杀五因）
        self.lbl_warn = QLabel(
            "⚠ 阈值在本批测试集上调节有效，不代表产线全局——"
            "建议用新一批产线图复验后再上线")
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setStyleSheet(f"color: {token('warn')};")
        lay.addWidget(self.lbl_warn)

    # ---- 数据注入与刷新 ----
    def set_stats(self, stats: dict, conf: float):
        """注入 match_stats；按维度可用性重建下拉（无数据维度不出现）。"""
        self._stats = stats or {}
        self._dim_state = {"score": {"thr": float(conf), "enabled": True}}
        dims = self._stats.get("dims") or {}
        for key, _name, _unit in _DIMS[1:]:
            d = dims.get(key) or {}
            has = any(d.get(b) for b in ("green", "red_defect", "red_ok"))
            if has:
                self._dim_state[key] = {
                    "thr": float(_DIM_DEFAULTS.get(key, 0.0)), "enabled": False}
        combo = self.cmb_dim
        combo.blockSignals(True)
        combo.clear()
        for key, name, unit in _DIMS:
            if key == "score" or key in self._dim_state:
                combo.addItem(_dim_label(key), key)
        combo.blockSignals(False)
        self._current = "score"
        combo.setCurrentIndex(0)
        self._line.blockSignals(True)
        self._line.setValue(float(conf))
        self._line.blockSignals(False)
        self._sync_axis_label()
        self._redraw()
        if not self._stats.get("gt_available"):
            self.lbl_hint.setText(
                "批量目录未找到同名标注 JSON（GT）：分数分布与误检/漏检"
                "统计需要真值标注；请对测试目录补标注或改用数据集目录批量。")
            self.btn_apply.setEnabled(True)
            self.btn_flow.setEnabled(True)
            return
        self.lbl_hint.setText("")
        self.btn_apply.setEnabled(True)
        self.btn_flow.setEnabled(True)

    def current_threshold(self) -> float:
        st = self._dim_state.get(self._current)
        return float(st["thr"]) if st else 0.25

    def enabled_dims(self) -> dict:
        """启用且阈值>0 的维度 → {dim: thr}（生成流程图模板用）。"""
        return {k: float(v["thr"]) for k, v in self._dim_state.items()
                if v.get("enabled") and float(v["thr"]) > 0}

    def _emit_thresholds_changed(self):
        self.thresholdsChanged.emit(self.enabled_dims())

    # ---- 交互 ----
    def _emit_apply_conf(self):
        # 语义锁定 conf 维（其他维度走"生成流程图模板"）
        self.applyRequested.emit(float(self._dim_state["score"]["thr"]))

    def _emit_generate_flow(self):
        dims = self.enabled_dims()
        if not dims:
            self.lbl_hint.setText("请先拖动阈值线启用至少一个维度（拖线即启用）")
            return
        self.generateFlowRequested.emit(dims)

    def _on_dim_changed(self, idx: int):
        key = self.cmb_dim.itemData(idx)
        if not key or key == self._current:
            return
        self._current = str(key)
        st = self._dim_state.get(self._current)
        self._line.blockSignals(True)
        self._line.setValue(float(st["thr"]) if st else 0.0)
        self._line.blockSignals(False)
        self.cb_enable.blockSignals(True)
        self.cb_enable.setChecked(bool(st and st.get("enabled")))
        self.cb_enable.blockSignals(False)
        self._sync_axis_label()
        self._redraw()

    def _on_enable_toggled(self, state: int):
        st = self._dim_state.get(self._current)
        if st is not None:
            st["enabled"] = bool(state)
        self._refresh_numbers()
        self._emit_thresholds_changed()

    def _on_line_moved(self):
        # 真机实测（2026-11-13）：键盘/拖拽可把线拖出数据范围（conf=-0.26）
        # → 阈值钳回有效区间（score∈[0,1]，其余维度 ≥0）后再入状态
        v = float(self._line.value())
        v = (min(max(v, 0.0), 1.0) if self._current == "score"
             else max(v, 0.0))
        if abs(v - float(self._line.value())) > 1e-12:
            self._line.blockSignals(True)
            self._line.setValue(v)
            self._line.blockSignals(False)
        st = self._dim_state.setdefault(
            self._current, {"thr": 0.0, "enabled": False})
        st["thr"] = v
        if not st.get("enabled"):     # 拖线即启用（用户约定：拖完一键成图）
            st["enabled"] = True
            self.cb_enable.blockSignals(True)
            self.cb_enable.setChecked(True)
            self.cb_enable.blockSignals(False)
        self._refresh_numbers()
        self.thresholdChanged.emit(self.current_threshold())
        self._emit_thresholds_changed()

    # ---- 绘制 ----
    def _sync_axis_label(self):
        key = self._current
        unit = next((u for k, _n, u in _DIMS if k == key), "")
        if key == "score":
            self.plot.setLabel("bottom", "置信度分数（IoU 匹配口径）")
        else:
            self.plot.setLabel(
                "bottom", f"{_dim_label(key)} 分布（IoU 匹配口径）"
                + (f"，单位 {unit}" if unit else ""))

    def _bins_for(self, key: str):
        if key == "score":
            return np.linspace(0.0, 1.0, _N_BINS + 1)
        d = (self._stats.get("dims") or {}).get(key) or {}
        vals = [v for b in ("green", "red_defect", "red_ok") for v in d.get(b, [])]
        if not vals:
            return np.linspace(0.0, 1.0, _N_BINS + 1)
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if hi - lo < 1e-9:
            lo, hi = lo - 1.0, hi + 1.0
        pad = (hi - lo) * 0.05
        return np.linspace(lo - pad, hi + pad, _N_BINS + 1)

    def _redraw(self):
        self.plot.clear()
        self.plot.addItem(self._line)
        stats = self._stats
        key = self._current
        if not stats.get("gt_available") and key != "score":
            self._refresh_numbers()
            return
        if not stats.get("gt_available"):
            self._refresh_numbers()
            return
        bins = self._bins_for(key)
        centers = (bins[:-1] + bins[1:]) / 2
        width = (bins[1] - bins[0]) * 0.4
        if key == "score":
            series = [
                (_GREEN, "TP（命中 GT）", stats.get("green") or []),
                (_RED, "FP（误检·缺陷图）", stats.get("red_defect") or []),
                (_AMBER, "FP·OK 图（OK 误检）", stats.get("red_ok") or []),
            ]
            clip = (lambda v: np.clip(np.asarray(v, dtype=float), 0.0, 1.0))
        else:
            d = (stats.get("dims") or {}).get(key) or {}
            series = [
                (_GREEN, "TP（命中 GT）", d.get("green") or []),
                (_RED, "FP（误检·缺陷图）", d.get("red_defect") or []),
                (_AMBER, "FP·OK 图（OK 误检）", d.get("red_ok") or []),
            ]
            clip = (lambda v: np.asarray(v, dtype=float))
        for color, name, values in series:
            if not values:
                continue
            counts, _ = np.histogram(clip(values), bins=bins)
            if not counts.any():
                continue
            bar = pg.BarGraphItem(
                x=centers, height=counts, width=width,
                brush=pg.mkBrush(color), pen=pg.mkPen(None), name=name)
            self.plot.addItem(bar)
        self._refresh_numbers()

    def _refresh_numbers(self):
        key = self._current
        t = self.current_threshold()
        if key == "score":
            m = threshold_metrics(self._stats, t)
        else:
            m = threshold_metrics_for(self._stats, key, t)
        self.lbl_thr.setText(
            f"{t:g}（{_dim_label(key)}）" + ("  ✓参与过滤" if
                                             self._dim_state.get(key, {}).get("enabled") else ""))
        self.lbl_precision.setText(
            f"{m['precision']:.3f}" if m["precision"] is not None else "—")
        self.lbl_recall.setText(
            f"{m['recall']:.3f}" if m["recall"] is not None else "—")
        self.lbl_ok.setText(
            f"{m['ok_fp_per_img']:.2f}" if m["ok_fp_per_img"] is not None else "—")

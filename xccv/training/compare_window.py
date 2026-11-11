"""实验对比视图（新窗口）：2~3 条训练记录的参数 / 指标 / 曲线 / 混淆矩阵对比。

用户批注：对比视图必须新拉一个独立窗口（To2.5）、限 2~3 条（D6）、
数据集指纹不一致的提示必须明确出现在对比视图中（To2.6）。
曲线用 matplotlib 多线叠加（项目既有依赖，不引入 pyqtgraph）；
指标口径与训练平台可视化页一致（mAP/mAR 固定两条、分类 Top-1）。
"""

from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..engine.dataset.prepare import dataset_fingerprint
from ..ui_style import ACCENT
from .project import migrate_history
from .runs import RunRecord, format_duration

# 每条记录一种颜色（曲线与混淆矩阵标题共用）
RUN_COLORS = ["#11809c", "#e8853d", "#2e8b57"]

# 参数快照展示顺序（metrics.json config 的键）
_PARAM_LABELS = [
    ("model_name", "训练模型"), ("epochs", "训练轮次"), ("batch_size", "批量大小"),
    ("learning_rate", "学习率"), ("weight_decay", "权重衰减"),
    ("train_ratio", "训练集比例"), ("val_ratio", "验证集比例"),
    ("multiply_factor", "数据倍增"), ("patience", "早停耐心"),
    ("eval_interval", "评估间隔"), ("device", "训练设备"), ("seed", "随机种子"),
    ("industrial_opt", "工业优化"), ("freeze_backbone", "冻结骨干"),
    ("rpn_reduced", "RPN 削减"), ("mask_head_light", "轻量掩码头"),
    ("use_amp", "AMP 混合精度"), ("tf32", "TF32 提速"),
    ("save_epoch_weights", "轮次权重"),
    # 注意：增强算子与增强概率不再单列两行，合并为「增强算子（概率）」
    # （用户回执 E2），见 param_rows 内的 _fmt_augment
]

_AUGMENT_ROW_LABEL = "增强算子（概率）"


def _fmt_augment(cfg: dict) -> str:
    """增强算子与每算子概率合并展示：random_rescale(0.8)、resize(1.0)。"""
    ops = cfg.get("augment_ops") or {}
    probs = cfg.get("augment_probs") or {}
    if not ops:
        return "未启用"
    return "、".join(f"{op}({probs.get(op, 0.5):g})" for op in ops)

_METRIC_LABELS = [
    ("final_val_top1", "最终验证 Top-1", "pct"),
    ("best_val_top1", "最优验证 Top-1", "pct"),
    ("map_loose", "mAP(宽松)", "f3"), ("map50", "mAP@50", "f3"),
    ("map50_95", "mAP@50-95", "f3"), ("mar_loose", "mAR(宽松)", "f3"),
    ("precision", "精确率", "f3"), ("recall", "召回率", "f3"),
]


def _fmt_value(v) -> str:
    if isinstance(v, dict):
        return "、".join(str(k) for k in v) or "未启用"
    if isinstance(v, bool):
        return "开启" if v else "关闭"
    if isinstance(v, float):
        return f"{v:g}"
    if v is None:
        return "—"
    return str(v)


def param_rows(records: list) -> list:
    """参数对比表：(标签, [各记录值的展示串], 是否有差异)。"""
    out = []
    for key, label in _PARAM_LABELS:
        values = [(r.metrics.get("config") or {}).get(key) for r in records]
        texts = [_fmt_value(v) for v in values]
        out.append((label, texts, len(set(texts)) > 1))
    aug = [(r.metrics.get("config") or {}) for r in records]
    texts = [_fmt_augment(cfg) for cfg in aug]
    out.append((_AUGMENT_ROW_LABEL, texts, len(set(texts)) > 1))
    return out


def metric_rows(records: list) -> list:
    """最终指标对比表：(标签, [各记录值的展示串], 是否有差异)。

    Top-1 在 metrics 顶层；mAP/mAR 等在 metrics["final"] 子字典（与
    训练可视化页 _summarize_metrics 的读取路径一致）。
    """
    out = []
    for key, label, kind in _METRIC_LABELS:
        texts = []
        for r in records:
            v = r.metrics.get(key)
            if v is None and key not in ("final_val_top1", "best_val_top1"):
                v = (r.metrics.get("final") or {}).get(key)
            texts.append("—" if v is None else
                         (f"{v:.1%}" if kind == "pct" else f"{v:.3f}"))
        out.append((label, texts, len(set(texts)) > 1))
    extras = [
        ("完成轮次", [str(r.epochs_done) for r in records]),
        ("训练时长", [format_duration(r.duration_sec) for r in records]),
        ("状态", [r.status_text for r in records]),
    ]
    for label, texts in extras:
        out.append((label, texts, len(set(texts)) > 1))
    return out


def history_series(history: list) -> tuple:
    """历史 → (损失曲线对, 指标曲线对)；口径与训练可视化页一致。"""
    history = migrate_history(history)
    xs = [h["epoch"] for h in history]
    loss = []
    for key, label in (("train_loss", "训练损失"), ("val_loss", "验证损失")):
        if any(key in h for h in history):
            loss.append((label, xs, [h.get(key) for h in history]))
    if any("val_top1" in h for h in history):
        pairs = (("val_top1", "Top-1"),)
    elif any("map_loose" in h for h in history):
        pairs = (("map_loose", "mAP"), ("mar_loose", "mAR"))
    elif any("map50" in h for h in history):
        pairs = (("map50", "mAP"), ("recall", "mAR"))
    else:
        pairs = ()
    metric = [(label, xs, [h.get(k) for h in history])
              for k, label in pairs if any(k in h for h in history)]
    return loss, metric


def fingerprint_warnings(records: list, current_fp: str | None) -> list:
    """数据集指纹提示（用户要求：明确出现在对比视图中）。"""
    warnings = []
    fps = {}
    for r in records:
        fp = r.metrics.get("dataset_fingerprint")
        if fp:
            fps[r.run_dir.name] = fp
        if fp and current_fp and fp != current_fp:
            warnings.append(
                f"⚠ 记录「{r.start_time or r.run_dir.name}」训练完成后数据集已修改，"
                "其指标可能与当前数据不对应。")
    if len(set(fps.values())) > 1:
        warnings.append(
            "⚠ 所选记录使用了不同版本的数据集，曲线与指标对比仅供参考。")
    return warnings


def _run_caption(rec: RunRecord) -> str:
    model = rec.metrics.get("config", {}).get("model_name") or rec.model_name
    return f"{rec.start_time or rec.run_dir.name[:15]} {model}".strip()


class _AdaptiveFigure(FigureCanvasQTAgg):
    """随窗口宽度 1:1 重设 matplotlib 画布尺寸。

    固定 figsize 的画布放进滚动区会被 Qt 拉伸（字号/线宽比例失真，全屏时
    严重变形、非全屏时被裁剪不可见——用户回执 E2/ToE4）。此处监听
    resizeEvent，把 figure 的英寸尺寸同步为控件实际像素，渲染与显示一致。
    """

    def __init__(self, fig: Figure, min_height: int = 320):
        super().__init__(fig)
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        dpi = self.figure.dpi
        w = max(self.width(), 480) / dpi
        h = max(self.height(), 200) / dpi
        if abs(self.figure.get_figwidth() - w) > 0.01 or \
                abs(self.figure.get_figheight() - h) > 0.01:
            self.figure.set_size_inches(w, h)
            try:
                self.figure.tight_layout()
            except Exception:
                pass  # 布局计算在极端尺寸下可能失败，下一轮 resize 会再试
            self.draw_idle()


class CompareWindow(QMainWindow):
    """对比窗口（独立顶层窗口，一次可对比 2~3 条记录）。"""

    def __init__(self, project, records: list):
        super().__init__()
        self.records = list(records)
        self.setWindowTitle(f"实验对比（{len(self.records)} 条记录） - XCcv 训练平台")
        self.resize(1200, 860)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(10)
        title = QLabel("实验对比")
        title.setStyleSheet("font-size: 19px; font-weight: bold;")
        lay.addWidget(title)
        legend = QLabel("　".join(
            f"<span style='color:{RUN_COLORS[i % len(RUN_COLORS)]};font-weight:bold;'>"
            f"■ {_run_caption(r)}</span>" for i, r in enumerate(self.records)))
        legend.setTextFormat(Qt.RichText)
        legend.setWordWrap(True)
        lay.addWidget(legend)

        # 数据集指纹提示（To2.6：明确在对比视图中提示）
        current_fp = None
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            current_fp = dataset_fingerprint(Path(project.dataset_dir))
        except Exception:
            current_fp = None
        finally:
            QApplication.restoreOverrideCursor()
        for w in fingerprint_warnings(self.records, current_fp):
            banner = QFrame()
            banner.setStyleSheet(
                "QFrame { background: #fdf6ec; border: 1px solid #f0dcc0;"
                " border-radius: 8px; }")
            bv = QVBoxLayout(banner)
            bv.setContentsMargins(12, 8, 12, 8)
            lab = QLabel(w)
            lab.setWordWrap(True)
            lab.setStyleSheet("color: #b25e1a; border: none; background: transparent;")
            bv.addWidget(lab)
            lay.addWidget(banner)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        body = QVBoxLayout(host)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(12)

        body.addWidget(self._make_table("参数对比", param_rows(self.records)))
        body.addWidget(self._make_table("最终指标对比", metric_rows(self.records)))
        body.addWidget(self._make_curves())
        body.addWidget(self._make_confusions())
        body.addStretch(1)
        scroll.setWidget(host)
        lay.addWidget(scroll, stretch=1)
        self.setCentralWidget(central)

    # ---- 构建 ----
    def _make_table(self, title: str, rows: list) -> QFrame:
        box = QFrame()
        box.setStyleSheet(
            "QFrame { background: #ffffff; border: 1px solid #e3ebf5;"
            " border-radius: 10px; }")
        v = QVBoxLayout(box)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(6)
        cap = QLabel(title)
        cap.setStyleSheet("font-weight: bold; border: none; background: transparent;")
        v.addWidget(cap)
        table = QTableWidget(len(rows), len(self.records) + 1)
        table.setHorizontalHeaderLabels(
            ["参数"] + [_run_caption(r) for r in self.records])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        # 列宽策略统一：标签列固定宽，记录列均分剩余宽度——两张表左右
        # 对称且互相同步（用户回执 ToE2"最终指标表左右不对称"）
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        table.setColumnWidth(0, 150)
        for i, (label, texts, diff) in enumerate(rows):
            lab_item = QTableWidgetItem(label)
            lab_item.setForeground(QColor("#64748b"))
            table.setItem(i, 0, lab_item)
            for j, text in enumerate(texts):
                item = QTableWidgetItem(text)
                if diff:  # 差异行高亮
                    item.setForeground(QColor(ACCENT))
                    item.setFont(self._bold_font())
                table.setItem(i, j + 1, item)
        table.setMinimumHeight(min(720, 40 + len(rows) * 30))
        v.addWidget(table)
        return box

    @staticmethod
    def _bold_font():
        from PyQt5.QtGui import QFont

        f = QFont()
        f.setBold(True)
        return f

    def _make_curves(self) -> QFrame:
        box = QFrame()
        box.setStyleSheet(
            "QFrame { background: #ffffff; border: 1px solid #e3ebf5;"
            " border-radius: 10px; }")
        v = QVBoxLayout(box)
        v.setContentsMargins(12, 10, 12, 10)
        cap = QLabel("曲线对比（左：损失；右：评估指标）")
        cap.setStyleSheet("font-weight: bold; border: none; background: transparent;")
        v.addWidget(cap)
        fig = Figure(dpi=100)  # 尺寸由 _AdaptiveFigure 随窗口同步
        canvas = _AdaptiveFigure(fig, min_height=340)
        ax_loss, ax_metric = fig.subplots(1, 2)
        any_loss = any_metric = False
        for i, rec in enumerate(self.records):
            color = RUN_COLORS[i % len(RUN_COLORS)]
            loss, metric = history_series(rec.metrics.get("history") or [])
            for k, (label, xs, ys) in enumerate(loss):
                style = "-" if k == 0 else "--"  # 实线=训练损失，虚线=验证损失
                ax_loss.plot(xs, ys, style, color=color, linewidth=2,
                             label=f"{_run_caption(rec)} {label}")
                any_loss = any_loss or any(
                    isinstance(y, (int, float)) and np.isfinite(y) for y in ys)
            for label, xs, ys in metric:
                ax_metric.plot(xs, ys, "-", color=color, linewidth=2,
                               label=f"{_run_caption(rec)} {label}")
                any_metric = any_metric or any(
                    isinstance(y, (int, float)) and np.isfinite(y) for y in ys)
        ax_loss.set_title("损失曲线")
        ax_loss.set_xlabel("轮次")
        ax_loss.grid(alpha=0.3)
        if any_loss:
            ax_loss.legend(fontsize=8)
        else:
            ax_loss.text(0.5, 0.5, "无损失数据", ha="center", va="center",
                         color="#94a3b8")
        ax_metric.set_title("评估指标")
        ax_metric.set_xlabel("轮次")
        ax_metric.set_ylim(0, 1)
        ax_metric.grid(alpha=0.3)
        if any_metric:
            ax_metric.legend(fontsize=8)
        else:
            ax_metric.text(0.5, 0.5, "无指标数据", ha="center", va="center",
                           color="#94a3b8")
        fig.tight_layout()
        v.addWidget(canvas)
        return box

    def _make_confusions(self) -> QFrame:
        box = QFrame()
        box.setStyleSheet(
            "QFrame { background: #ffffff; border: 1px solid #e3ebf5;"
            " border-radius: 10px; }")
        v = QVBoxLayout(box)
        v.setContentsMargins(12, 10, 12, 10)
        cap = QLabel("混淆矩阵并排（归一化）")
        cap.setStyleSheet("font-weight: bold; border: none; background: transparent;")
        v.addWidget(cap)
        n = len(self.records)
        fig = Figure(dpi=100)  # 尺寸由 _AdaptiveFigure 随窗口同步
        canvas = _AdaptiveFigure(fig, min_height=360)
        axes = fig.subplots(1, n) if n > 1 else [fig.add_subplot(111)]
        drawn = False
        for i, (rec, ax) in enumerate(zip(self.records, axes)):
            matrix = rec.metrics.get("confusion_matrix")
            classes = rec.metrics.get("classes") or []
            if not matrix:
                ax.text(0.5, 0.5, "无混淆矩阵", ha="center", va="center",
                        color="#94a3b8")
                ax.set_axis_off()
                continue
            drawn = True
            arr = np.array(matrix, dtype=float)
            sums = arr.sum(axis=1, keepdims=True)
            norm = np.divide(arr, sums, out=np.zeros_like(arr), where=sums > 0)
            im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            step = max(1, len(classes) // 12)
            ticks = list(range(0, len(classes), step))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([classes[t] for t in ticks],
                               rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels([classes[t] for t in ticks], fontsize=8)
            ax.set_title(_run_caption(rec), fontsize=10,
                         color=RUN_COLORS[i % len(RUN_COLORS)])
        if drawn:
            fig.tight_layout()
        v.addWidget(canvas)
        return box


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402

apply_auto_slot_module(globals())

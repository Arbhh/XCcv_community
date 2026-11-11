"""训练平台通用对话框：折叠分区、体检/预检报告、开发者选项。

HealthReportDialog 同时服务两个入口（用户规则：少弹窗、两级档位）：
- mode="report"    数据体检报告（「去标注平台处理」闭环）；
- mode="preflight" 训练前预检（应用建议并开始 / 直接开始 / 取消），
  底部折叠区展示「一键应用」修改过的参数。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget,
)

from ..ui_style import ACCENT, FG
from .data_health import HealthIssue, LEVEL_WARN
from .dev_options import DevOptions, load_options, save_options
from .preflight import Finding

_WARN_COLOR = "#e8853d"
_INFO_COLOR = ACCENT


class CollapsibleSection(QWidget):
    """可折叠分区：标题行点击展开/收起内容（参数表单分层用）。"""

    def __init__(self, title: str, expanded: bool = False, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.btn = QPushButton(("▾  " if expanded else "▸  ") + title)
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setStyleSheet(
            f"QPushButton {{ text-align: left; font-weight: bold; color: {ACCENT};"
            " border: none; background: transparent; padding: 4px 2px;"
            " font-size: 15px; }"
            f"QPushButton:hover {{ color: {FG}; }}")
        self.btn.clicked.connect(self.toggle)
        lay.addWidget(self.btn)
        self.content = QWidget()
        self._content_lay = QVBoxLayout(self.content)
        self._content_lay.setContentsMargins(6, 0, 0, 0)
        self._content_lay.setSpacing(6)
        lay.addWidget(self.content)
        self.content.setVisible(expanded)
        self._title = title

    def toggle(self):
        # 用 isHidden（显式隐藏标志）而非 isVisible——后者依赖父链可见性，
        # 在窗口未显示（如 offscreen 测试）时恒为 False，会导致状态翻转失效
        expanded = self.content.isHidden()
        # 批量布局期间挂起重绘：避免逐控件回流造成的卡顿与滚动条闪动
        # 引发的窗口瞬间变形（用户回执 D1/H2）
        win = self.window()
        frozen = win is not None and win.isVisible()
        if frozen:
            win.setUpdatesEnabled(False)
        try:
            self.content.setVisible(expanded)
            self.btn.setText(("▾  " if expanded else "▸  ") + self._title)
        finally:
            if frozen:
                win.setUpdatesEnabled(True)

    def set_expanded(self, expanded: bool):
        # 仅当当前隐藏态与目标不一致时翻转（isHidden 与 expanded 语义相反）
        if self.content.isHidden() == expanded:
            self.toggle()

    def add_layout(self, layout):
        self._content_lay.addLayout(layout)

    def add_widget(self, widget):
        self._content_lay.addWidget(widget)


class ReportRow:
    """报告对话框的一行条目（体检 HealthIssue 或预检 Finding 的统一视图）。"""

    def __init__(self, level: str, title: str, detail: str = "",
                 suggestion: str = "", apply_field: str = "",
                 apply_value=None, apply_text: str = ""):
        self.level = level
        self.title = title
        self.detail = detail
        self.suggestion = suggestion
        self.apply_field = apply_field   # 一键应用的目标参数键（空=无按钮）
        self.apply_value = apply_value
        self.apply_text = apply_text or "一键应用"

    @classmethod
    def from_issue(cls, issue: HealthIssue):
        return cls(issue.level, issue.title, issue.detail, issue.suggestion)

    @classmethod
    def from_finding(cls, f: Finding):
        return cls(f.level, f.message, "", "",
                   apply_field=f.field, apply_value=f.value,
                   apply_text=f.apply_text)


class HealthReportDialog(QDialog):
    """体检 / 预检报告（两级档位 + 可折叠的已应用参数区）。"""

    def __init__(self, parent, title: str, rows: list, mode: str = "report",
                 summary: str = "", on_annotate=None, on_apply=None):
        super().__init__(parent)
        # rows: list[ReportRow]；on_apply(field, value)：一键应用写回训练表单
        self.mode = mode
        self.applied: list = []          # [(field, value)] 已应用的参数
        self._on_apply = on_apply
        self._row_specs = []
        for r in rows:
            if isinstance(r, HealthIssue):
                r = ReportRow.from_issue(r)
            if isinstance(r, Finding):
                r = ReportRow.from_finding(r)
            self._row_specs.append(r)
        self._on_annotate = on_annotate  # 「去标注平台处理」回调（report 模式）

        self.setWindowTitle(title)
        self.setMinimumWidth(560)
        self.setStyleSheet(self.parent().styleSheet() if parent else "")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(10)

        n_warn = sum(1 for r in self._row_specs if r.level == LEVEL_WARN)
        n_info = len(self._row_specs) - n_warn
        chips = []
        if n_warn:
            chips.append(f"<span style='color:{_WARN_COLOR};font-weight:bold;'>"
                         f"⚠ {n_warn} 项警告</span>")
        if n_info:
            chips.append(f"<span style='color:{_INFO_COLOR};font-weight:bold;'>"
                         f"· {n_info} 项提示</span>")
        head = QLabel("　".join(chips) if chips else "未发现问题")
        head.setTextFormat(Qt.RichText)
        lay.addWidget(head)
        if summary:
            s = QLabel(summary)
            s.setObjectName("hint")
            s.setWordWrap(True)
            lay.addWidget(s)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        host_lay = QVBoxLayout(host)
        host_lay.setContentsMargins(0, 0, 0, 0)
        host_lay.setSpacing(8)
        self._buttons = []
        for r in self._row_specs:
            host_lay.addWidget(self._build_row(r))
        host_lay.addStretch(1)
        scroll.setWidget(host)
        lay.addWidget(scroll, stretch=1)

        # 折叠区：一键应用修改过的参数（用户要求可折叠查看）
        self.applied_section = CollapsibleSection("已应用的参数修改", expanded=False)
        self.applied_label = QLabel("（暂无——点击条目中的「一键应用」后在此记录）")
        self.applied_label.setObjectName("hint")
        self.applied_label.setWordWrap(True)
        self.applied_section.add_widget(self.applied_label)
        lay.addWidget(self.applied_section)

        btns = QHBoxLayout()
        if mode == "preflight":
            self._btn_apply_all = QPushButton("应用全部建议并开始训练")
            self._btn_apply_all.setObjectName("accent")
            has_apply = any(r.apply_field for r in self._row_specs)
            self._btn_apply_all.setVisible(has_apply)
            self._btn_apply_all.clicked.connect(self._apply_all)
            btn_start = QPushButton("直接开始训练")
            btn_start.clicked.connect(self.accept)
            btn_cancel = QPushButton("取消")
            btn_cancel.clicked.connect(self.reject)
            btns.addWidget(self._btn_apply_all)
            btns.addWidget(btn_start)
            btns.addWidget(btn_cancel)
        else:
            if on_annotate is not None:
                btn_anno = QPushButton("去标注平台处理")
                btn_anno.clicked.connect(self._goto_annotate)
                btns.addWidget(btn_anno)
            btn_close = QPushButton("关闭")
            btn_close.clicked.connect(self.accept)
            btns.addStretch(1)
            btns.addWidget(btn_close)
        btns.addStretch(1)
        lay.addLayout(btns)

    # ---- 构建 ----
    def _build_row(self, r: ReportRow) -> QFrame:
        color = _WARN_COLOR if r.level == LEVEL_WARN else _INFO_COLOR
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: #ffffff; border: 1px solid #e3ebf5;"
            f" border-left: 3px solid {color}; border-radius: 8px; }}")
        v = QVBoxLayout(frame)
        v.setContentsMargins(12, 8, 12, 8)
        v.setSpacing(4)
        cap = QLabel(f"<b style='color:{color}'>{r.level}</b>　{r.title}")
        cap.setTextFormat(Qt.RichText)
        cap.setWordWrap(True)
        v.addWidget(cap)
        if r.detail:
            d = QLabel(r.detail)
            d.setObjectName("hint")
            d.setWordWrap(True)
            v.addWidget(d)
        if r.suggestion:
            s = QLabel("建议：" + r.suggestion)
            s.setStyleSheet(f"color: {ACCENT};")
            s.setWordWrap(True)
            v.addWidget(s)
        if r.apply_field:
            btn = QPushButton(r.apply_text)
            btn.setFixedHeight(30)

            def _do(_=False, key=r.apply_field, val=r.apply_value, btn=btn):
                if self._on_apply is not None:
                    self._on_apply(key, val)
                if (key, val) not in self.applied:
                    self.applied.append((key, val))
                btn.setEnabled(False)
                btn.setText("已应用")
                self._refresh_applied_label()

            btn.clicked.connect(_do)
            self._buttons.append(btn)
            v.addWidget(btn, alignment=Qt.AlignLeft)
        return frame

    def _refresh_applied_label(self):
        names = {
            "patience": "早停耐心", "learning_rate": "学习率",
            "batch_size": "批量大小", "device": "训练设备", "resize": "调整图像大小",
        }
        lines = [f"{names.get(k, k)} → {v}" for k, v in self.applied] or \
                ["（暂无——点击条目中的「一键应用」后在此记录）"]
        self.applied_label.setText("\n".join(str(x) for x in lines))

    # ---- 动作 ----
    def _apply_all(self):
        for btn in self._buttons:
            if btn.isEnabled():
                btn.click()
        # 用户回执 B4：「应用全部建议并开始训练」应用后需关闭对话框继续流程
        self.accept()

    def _goto_annotate(self):
        self.accept()
        if self._on_annotate is not None:
            self._on_annotate()


class DevOptionsDialog(QDialog):
    """开发者选项（训练平台齿轮入口）：各检测阈值 + 预检开关。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("开发者选项（体检 / 预检阈值）")
        self.setMinimumWidth(460)
        self.setStyleSheet(self.parent().styleSheet() if parent else "")
        opt = load_options()
        self._defaults = DevOptions()

        form = QFormLayout(self)
        form.setContentsMargins(18, 16, 18, 14)
        form.setSpacing(10)

        self.sp_imbalance = QDoubleSpinBox()
        self.sp_imbalance.setRange(1.0, 50.0)
        self.sp_imbalance.setSingleStep(1.0)
        self.sp_imbalance.setSuffix(" %")
        self.sp_imbalance.setValue(opt.imbalance_share * 100)
        self.sp_imbalance.setToolTip("类别不均衡判定：最小类别的实例数占整体比例低于该值时警告"
                                     "（默认 5%，用户回执 Q3 保守化口径）")
        self.sp_cls = QSpinBox()
        self.sp_cls.setRange(1, 1000)
        self.sp_cls.setValue(opt.min_cls_samples)
        self.sp_cls.setToolTip("分类任务：单类样本数低于该值警告")
        self.sp_det = QSpinBox()
        self.sp_det.setRange(1, 5000)
        self.sp_det.setValue(opt.min_det_instances)
        self.sp_det.setToolTip("检测/分割任务：单类实例数低于该值警告")
        self.sp_dup = QSpinBox()
        self.sp_dup.setRange(0, 32)
        self.sp_dup.setValue(opt.dup_hamming)
        self.sp_dup.setToolTip("疑似重复图判定：dHash 汉明距离 ≤ 该值视为高度相似")
        self.sp_cap = QSpinBox()
        self.sp_cap.setRange(100, 100000)
        self.sp_cap.setSingleStep(100)
        self.sp_cap.setValue(opt.dup_scan_cap)
        self.sp_cap.setToolTip("重复图检测最多处理的图片数（大集合跳过该项保证体检速度）")
        self.sp_mad = QDoubleSpinBox()
        self.sp_mad.setRange(1.0, 10.0)
        self.sp_mad.setSingleStep(0.5)
        self.sp_mad.setValue(opt.size_outlier_mad)
        self.sp_mad.setToolTip("尺寸离群判定：面积偏离中位数超过 N×MAD 时提示")

        form.addRow("类别不均衡：最小类占比 <", self.sp_imbalance)
        form.addRow("分类单类样本 <", self.sp_cls)
        form.addRow("检测/分割单类实例 <", self.sp_det)
        form.addRow("重复图汉明距离 ≤", self.sp_dup)
        form.addRow("重复图检测上限（张）", self.sp_cap)
        form.addRow("尺寸离群 MAD 倍数", self.sp_mad)
        self.ck_preflight = QCheckBox("开始训练前执行预检（推荐开启；仅有警告时才弹窗）")
        self.ck_preflight.setChecked(opt.preflight_enabled)
        form.addRow(self.ck_preflight)
        self.ck_info_popup = QCheckBox("预检仅发现提示级问题时也弹窗提示（用户回执 A5）")
        self.ck_info_popup.setChecked(opt.preflight_info_popup)
        form.addRow(self.ck_info_popup)

        note = QLabel("所有检查仅分「提示 / 警告」两档，不会阻止训练。")
        note.setObjectName("hint")
        form.addRow(note)

        btns = QHBoxLayout()
        btn_reset = QPushButton("恢复默认")
        btn_reset.clicked.connect(self._reset_defaults)
        btns.addWidget(btn_reset)
        btns.addStretch(1)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.button(QDialogButtonBox.Ok).setText("保存")
        box.button(QDialogButtonBox.Cancel).setText("取消")
        box.accepted.connect(self._save)
        box.rejected.connect(self.reject)
        btns.addWidget(box)
        form.addRow(btns)

    def _reset_defaults(self):
        d = self._defaults
        self.sp_imbalance.setValue(d.imbalance_share * 100)
        self.sp_cls.setValue(d.min_cls_samples)
        self.sp_det.setValue(d.min_det_instances)
        self.sp_dup.setValue(d.dup_hamming)
        self.sp_cap.setValue(d.dup_scan_cap)
        self.sp_mad.setValue(d.size_outlier_mad)
        self.ck_preflight.setChecked(d.preflight_enabled)
        self.ck_info_popup.setChecked(d.preflight_info_popup)

    def _collect(self) -> DevOptions:
        return DevOptions(
            imbalance_share=self.sp_imbalance.value() / 100.0,
            min_cls_samples=self.sp_cls.value(),
            min_det_instances=self.sp_det.value(),
            dup_hamming=self.sp_dup.value(),
            dup_scan_cap=self.sp_cap.value(),
            size_outlier_mad=self.sp_mad.value(),
            preflight_enabled=self.ck_preflight.isChecked(),
            preflight_info_popup=self.ck_info_popup.isChecked(),
        )

    def _save(self):
        save_options(self._collect())
        self.accept()


def rows_from_issues(issues) -> list:
    return [ReportRow.from_issue(i) for i in issues]


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402

apply_auto_slot_module(globals())

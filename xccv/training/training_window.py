"""训练平台窗口：
- TrainingWindow：项目列表窗口（项目卡片墙：创建 / 打开 / 删除）
- ProjectWindow：项目窗口（项目信息 + 数据可视化 + 增强策略 + 训练参数 + 训练可视化），
  每个项目一个独立窗口，支持多开；重复打开同一项目只激活已有窗口
- CreateProjectDialog：创建项目对话框

运行：python -m xccv.training.training_window
"""

if __name__ == "__main__":  # V0.6 T1-3：-m 直启先亮加载中界面，再做下方重型模块级导入
    from xccv.platform_entry import early_splash

    early_splash("training", "训练平台", heavy=True)

import math
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import pyqtgraph as pg
from PyQt5.QtCore import QPointF, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QIcon, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..engine.dataset import IMAGE_EXTS
from ..engine.dataset.base import TaskType
from ..engine.dataset.reader import scan_classification_dataset, scan_labelme_dataset
from ..engine.io_utils import imread_unicode
from ..gui_utils import auto_cleanup_thread, ensure_console_streams
from .charts import ConfusionMatrixCanvas, DistributionCanvas, SplitDonutCanvas
from .data_health import LEVEL_WARN, run_health_check
from .dataset_stats import collect_stats
from .dialogs import CollapsibleSection, DevOptionsDialog, HealthReportDialog, ReportRow
from .dev_options import load_options
from .preflight import dataset_facts, param_findings, recommend_settings
from .project import (
    Project, create_project, delete_project, list_projects,
    load_project, save_project,
)
from .runs import RunRecord, format_duration, scan_runs, STATUS_COLOR
from ..ui_style import APP_QSS, ACCENT, bind_style, icon_path, token  # noqa: F401  APP_QSS/ACCENT 保留兼容
from ..ui_widgets import (
    Badge, HoverCard, load_prefs, relative_time, save_prefs, toast,
)

# 三类模型的专属图标（v0.5.4 任务二；打开项目/项目卡随任务类型切换）
_TASK_MODEL_ICON = {
    TaskType.CLASSIFICATION: "model_classification.png",
    TaskType.DETECTION: "model_detection.png",
    TaskType.SEGMENTATION: "model_segmentation.png",
    TaskType.ANOMALY: "model_classification.png",  # A1：无监督复用分类图标
    TaskType.OCR: "model_classification.png",      # A2：OCR 复用分类图标
}
from .trainer_thread import TrainThread

ensure_console_streams()  # pythonw 下 stdout/stderr 为 None，先兜底

pg.setConfigOptions(antialias=True, background=token("panel", "#ffffff"),
                    foreground=token("text_2", "#3a4656"))


def _cuda_available() -> bool:
    """torch 延迟导入（T1-4）：本模块在启动闪屏显示之前导入，
    torch/引擎链必须后置到窗口构建期，否则加载中界面被 torch 拖慢 2s+。"""
    import torch

    return torch.cuda.is_available()

# 各模型的建议默认值（选择模型时自动带入，仍可修改）
DEFAULT_LR = {
    "resnet18": 0.001, "resnet50": 0.001, "mobilenet_v3_small": 0.001,
    "yolo11n": 0.01, "yolo11s": 0.01,
    "yolo11n-seg": 0.01, "yolo11s-seg": 0.01,
    "fasterrcnn": 0.005, "maskrcnn": 0.005, "maskrcnn18": 0.005,
}

# 图像增强策略行：(算子名, 中文名, [(参数名, 默认值, 最小, 最大, 是否整数)])
AUGMENT_ROWS = [
    ("resize", "调整图像大小（拉伸至宽×高）", [
        ("width", 640, 2, 8192, True), ("height", 640, 2, 8192, True)]),
    ("random_rescale", "随机比例缩放", [
        ("min_scale", 0.8, 0.1, 2.0, False), ("max_scale", 1.2, 0.1, 3.0, False)]),
    ("random_flip_h", "随机水平翻转", []),
    ("random_flip_v", "随机垂直翻转", []),
    ("rotate90", "随机旋转 90°/270°（工件朝向；见下方守卫说明）", []),
    ("random_crop", "随机裁剪（对齐目标宽高比，旋转算子的兜底）", [
        ("ratio", 1.0, 0.1, 10.0, False), ("min_cover", 0.7, 0.1, 1.0, False)]),
    ("random_brightness", "随机调整亮度（±max_delta 内随机偏移）", [
        ("max_delta", 20, 1, 100, True)]),
    ("random_contrast", "随机调整对比度", [
        ("low", 0.8, 0.1, 3.0, False), ("high", 1.2, 0.1, 3.0, False)]),
    ("random_hsv_shift", "随机 HSV 色彩扰动（光照/色温变化）", [
        ("h_delta", 6, 1, 90, True), ("s_delta", 0.15, 0.05, 1.0, False),
        ("v_delta", 0.15, 0.05, 1.0, False)]),
    ("gaussian_blur", "随机高斯模糊（模拟失焦）", [
        ("ksize", 3, 1, 15, True), ("sigma", 0.6, 0.1, 5.0, False)]),
    ("gaussian_noise", "随机高斯噪声（传感器噪声）", [
        ("sigma", 3, 1, 60, True)]),
    ("random_erase", "随机矩形擦除（模拟遮挡/反光）", [
        ("max_ratio", 0.1, 0.05, 0.9, False)]),
]

_CURVE_COLORS = ["#4c8bf5", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]

_TASK_BADGES = {
    TaskType.CLASSIFICATION: ("badge", "图像分类"),
    TaskType.DETECTION: ("badgeDet", "目标检测"),
    TaskType.SEGMENTATION: ("badgeSeg", "实例分割"),
    TaskType.ANOMALY: ("badge", "无监督异常检测"),  # A1（V0.7）
    TaskType.OCR: ("badge", "OCR 字符识别"),        # A2（V0.7）
}

_ILLEGAL_NAME_CHARS = re.compile(r'[\\/:*?"<>|]')


def _safe_run_tag(model_name: str) -> str:
    """把模型名转成合法目录名片段。

    模型引用形如 "pruned:xxx.pt" / "run:记录/best.pt"，其中的冒号、斜杠
    在 Windows 目录名中非法（曾导致训练无法启动）；这里统一替换并去掉
    权重后缀。
    """
    tag = _ILLEGAL_NAME_CHARS.sub("_", model_name or "model")
    for suffix in (".pt",):
        if tag.lower().endswith(suffix):
            tag = tag[: -len(suffix)]
            break
    return tag.strip("_") or "model"


def _apply_theme(widget):
    """全局样式套用 + 登记主题实时切换（V0.6 F2-5）。"""
    from ..ui_style import apply_app_theme

    apply_app_theme(widget)


# 预设档位（D4②）：一键套用 + 联动 recommend_settings；
# 参数键映射见 ProjectWindow._apply_param_kv
_PRESET_DEFS = [
    ("fast", "快速验证", "30 轮 · 冻结骨干\n小样本快速出结果",
     {"epochs": 30, "freeze_backbone": True, "patience": 10, "use_amp": True},
     "#b45309"),
    ("balanced", "均衡训练", "100 轮 · 工业优化\n精度与速度兼顾",
     {"epochs": 100, "freeze_backbone": False, "industrial_opt": True,
      "use_amp": True}, "#11809c"),
    ("quality", "高质量", "300 轮 · 全量加速\n追求最佳精度，耗时更长",
     {"epochs": 300, "freeze_backbone": False, "use_amp": True,
      "tf32": True, "patience": 20}, "#1d8a53"),
]


def _t2() -> str:
    """主题感知次要文字色（暗色主题下自动切换）。"""
    from ..ui_style import token

    return token("text_2", "#5f7086")


def _t3() -> str:
    from ..ui_style import token

    return token("text_3", "#93a3b5")


def _project_mtime(name: str) -> float:
    """项目最近更新时间：project.json 与各次训练 metrics.json 取最新
    （与项目列表排序同一口径——卡片排序按更新时间倒序，第三批要求）。"""
    from .project import PROJECTS_DIR, _project_touch_time

    return _project_touch_time(PROJECTS_DIR / name)


def _status_dot_icon(color: str, d: int = 12) -> QPixmap:
    """训练历史列表条目的状态圆点图标。"""
    pm = QPixmap(d + 2, d + 2)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    p.drawEllipse(1, 1, d, d)
    p.end()
    return pm


def _make_badge(task: TaskType) -> QLabel:
    obj, text = _TASK_BADGES[TaskType(task)]
    badge = QLabel(text)
    badge.setObjectName(obj)
    badge.setAlignment(Qt.AlignCenter)
    return badge


class _IntAxis(pg.AxisItem):
    """轮次轴：只显示整数刻度，避免出现 0.5、1.5 之类的小数。"""

    def tickStrings(self, values, scale, spacing):
        return [str(int(v)) if float(v).is_integer() else "" for v in values]


class _NoWheel:
    """滚轮不改变数值（事件透传给父级滚动区），防止浏览页面时误触参数。"""

    def wheelEvent(self, ev):
        ev.ignore()


class _Spin(_NoWheel, QSpinBox):
    pass


class _DoubleSpin(_NoWheel, QDoubleSpinBox):
    """小数参数框：最小位数显示（0.5 而非 0.50）+ 聚焦全选。

    用户反馈：默认两位小数补零后，改值必须先删尾零；最小表示 + 全选让
    点击后直接输入即可整体覆盖。
    """

    def textFromValue(self, val: float) -> str:
        return f"{val:g}"

    def valueFromText(self, text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return super().valueFromText(text)

    def focusInEvent(self, ev):
        super().focusInEvent(ev)
        self.lineEdit().selectAll()


class _Combo(_NoWheel, QComboBox):
    pass


class _PagePlot(pg.PlotWidget):
    """曲线图：滚轮事件透传给父级滚动区（鼠标缩放已禁用，滚轮只滚页面）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_leave = None
        self._xccv_title = ""
        self.refresh_theme()

    def refresh_theme(self):
        """主题切换（F2-5）：底色/坐标轴/标题随 token 重设（pyqtgraph 不吃 QSS）。"""
        bg, fg, title_c = token("panel", "#ffffff"), token("text_2", "#3a4656"), \
            token("text_1", "#1f2937")
        pg.setConfigOptions(background=bg, foreground=fg)   # 之后新建的图表同色
        self.setBackground(bg)
        pi = self.getPlotItem()
        for ax in ("bottom", "left"):
            axis = pi.getAxis(ax)
            axis.setPen(fg)
            axis.setTextPen(fg)
        if self._xccv_title:
            pi.setTitle(self._xccv_title, color=title_c)

    def wheelEvent(self, ev):
        ev.ignore()

    def leaveEvent(self, ev):
        if self._on_leave is not None:
            self._on_leave()
        super().leaveEvent(ev)


_TICK_FONT = QFont("Microsoft YaHei", 11)


def _style_plot(plot: _PagePlot, title: str):
    """统一图表排版：大号标题（HTML 渲染）+ 可读的轴刻度字体。"""
    pi = plot.getPlotItem()
    plot._xccv_title = f'<span style="font-size: 12pt; font-weight: 600;">{title}</span>'
    pi.setTitle(plot._xccv_title, color=token("text_1", "#1f2937"))
    for ax in ("bottom", "left"):
        pi.getAxis(ax).setTickFont(_TICK_FONT)


def _add_legend(plot: _PagePlot):
    """图例字号放大（旧版 pyqtgraph 无 labelTextSize 参数时回退默认）。"""
    try:
        return plot.addLegend(offset=(8, 8), labelTextSize="10pt")
    except TypeError:
        return plot.addLegend(offset=(8, 8))


class _PlotHover:
    """曲线图悬停提示：跟随鼠标显示最近采样点的数值（三位小数）。

    逐批平滑曲线按亚轮次取值，逐轮曲线按整轮取值；标签自动出现在
    鼠标所在半区的另一侧，避免遮挡光标。
    """

    def __init__(self, plot: _PagePlot, get_series):
        self.plot = plot
        self.get_series = get_series
        self.vline = pg.InfiniteLine(angle=90, movable=False,
                                     pen=pg.mkPen("#94a3b8", width=1, style=Qt.DashLine))
        self.label = pg.TextItem("", anchor=(0, 1), color="#1f2937")
        self.label.setZValue(60)
        self.vline.setZValue(50)
        plot._on_leave = self.hide_tip
        plot.scene().sigMouseMoved.connect(self._on_move)

    def install(self):
        """plot.clear() 会移除图元，每次重绘后重新挂载。"""
        pi = self.plot.getPlotItem()
        pi.addItem(self.vline, ignoreBounds=True)
        pi.addItem(self.label, ignoreBounds=True)
        self.hide_tip()

    def hide_tip(self):
        self.vline.hide()
        self.label.hide()

    @staticmethod
    def _fmt_epoch(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else f"{v:.2f}"

    def _on_move(self, pos):
        vb = self.plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(pos):  # 必须用 ViewBox 的矩形判断
            self.hide_tip()
            return
        series = [(k, xs, ys) for k, xs, ys in self.get_series() if xs]
        if not series:
            self.hide_tip()
            return
        view = vb.mapSceneToView(pos)
        first_xs = series[0][1]
        epoch = min(first_xs, key=lambda x: abs(x - view.x()))
        lines = [f"第 {self._fmt_epoch(epoch)} 轮"]
        for key, kxs, kys in series:
            j = min(range(len(kxs)), key=lambda i: abs(kxs[i] - epoch))
            if abs(kxs[j] - epoch) <= 0.5 + 1e-9:
                v = kys[j]
                if v is not None:
                    lines.append(f"{key} = {v:.3f}")
        x0, x1 = vb.viewRange()[0]
        mid = (x0 + x1) / 2
        # 锚点取文字顶边（y=0）：标签从视图顶部向下展开，留在可见区内；
        # 若锚在底边（y=1）文字会向上展开到 ViewBox 之外被整体裁剪（只剩竖线可见）
        anchor = QPointF(1, 0) if view.x() > mid else QPointF(0, 0)
        self.label.setAnchor(anchor)
        top = vb.viewRange()[1][1]
        self.vline.setPos(float(epoch))
        self.vline.show()
        self.label.setPos(float(epoch), top)
        self.label.setPlainText("\n".join(lines))
        self.label.show()


# ============================================================
# 项目卡片
# ============================================================
class _ProjectCard(HoverCard):
    clicked = pyqtSignal(str)
    deleteRequested = pyqtSignal(str)

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project_name = project.name
        self.setFixedSize(340, 160)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 14, 12)
        lay.setSpacing(7)

        head = QHBoxLayout()
        model_icon = QLabel()
        model_icon.setPixmap(QPixmap(str(icon_path(
            _TASK_MODEL_ICON.get(project.task_type, "training.png")))).scaled(
            34, 34, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        model_icon.setFixedSize(34, 34)
        head.addWidget(model_icon)
        title = QLabel(project.name)
        title.setObjectName("cardTitle")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(_make_badge(project.task_type))
        lay.addLayout(head)

        updated = QLabel(f"更新于 {relative_time(_project_mtime(project.name))}")
        updated.setObjectName("muted")
        created = QLabel(f"创建于 {project.created_at or '—'}")
        created.setObjectName("muted")
        model = QLabel(f"模型：{project.model_name or '未选择'}")
        model.setObjectName("muted")
        lay.addWidget(updated)
        lay.addWidget(created)
        lay.addWidget(model)
        lay.addStretch(1)

        foot = QHBoxLayout()
        tip = QLabel("点击卡片打开 →")
        tip.setObjectName("muted")
        foot.addWidget(tip)
        foot.addStretch(1)
        btn_delete = QPushButton("删除")
        btn_delete.setObjectName("linkDanger")
        btn_delete.setCursor(Qt.PointingHandCursor)
        btn_delete.clicked.connect(lambda: self.deleteRequested.emit(self.project_name))
        foot.addWidget(btn_delete)
        lay.addLayout(foot)

    def _emit_clicked(self):
        # Q2：子类把 clicked 重定义为 clicked[str]，经基类钩子带参发射
        self.clicked.emit(self.project_name)


# ============================================================
# 训练前预检后台线程（数据体检 + 参数建议；大数据集扫描不阻塞界面）
# ============================================================
class _PreflightWorker(QThread):
    done = pyqtSignal(object)   # {"issues": [HealthIssue], "findings": [Finding],
    #                             "n_images": int, "classes": [str]}
    failed = pyqtSignal(str)

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project

    def run(self):
        try:
            p = self._project
            opt = load_options()
            task = TaskType(p.task)
            d = Path(p.dataset_dir)
            if task is TaskType.CLASSIFICATION:
                samples, classes = scan_classification_dataset(d)
                scan = (samples, classes)
            else:
                samples, classes, _ = scan_labelme_dataset(d)
                scan = (samples, classes, False)
            report = run_health_check(
                task, d, opt, val_ratio=float(p.val_ratio), seed=42,
                train_ratio=getattr(p, "train_ratio", None) or None, scan=scan)
            issues = list(report.issues)
            # O4（V0.7）：缩放后实例像素尺寸校验（YOLO=letterbox、R-CNN=stretch）
            if task in (TaskType.DETECTION, TaskType.SEGMENTATION):
                from .data_health import HealthIssue, LEVEL_WARN
                from .scale_guard import scale_issue, scale_issue_text

                resize = (p.augment_ops or {}).get("resize") or {}
                w = int(resize.get("width", 0) or 0)
                h = int(resize.get("height", 0) or 0)
                if str(p.model_name or "").startswith("yolo"):
                    mode = "letterbox"   # YOLO 忽略 resize 增强只取长边（P210）
                    w = h = (max(w, h) or 640)
                else:
                    mode = "stretch"
                stat = scale_issue(samples, mode, w, h)
                txt = scale_issue_text(stat) if stat else None
                if txt:
                    issues.append(HealthIssue(
                        LEVEL_WARN, "small_instance",
                        "缩放后过小实例占比偏高", txt[0], txt[1]))
            findings = param_findings(p, dataset_facts(samples),
                                      _cuda_available())
            self.done.emit({"issues": issues, "findings": findings,
                            "n_images": report.n_images, "classes": report.classes})
        except Exception as e:  # 预检失败不阻塞训练（训练线程另有校验兜底）
            self.failed.emit(f"{type(e).__name__}: {e}")


# ============================================================
# 项目窗口：一个项目一个独立窗口
# ============================================================
class ProjectWindow(QMainWindow):
    def __init__(self, project: Project, on_renamed=None, on_closed=None):
        super().__init__()
        self.project = project
        self.on_renamed = on_renamed      # (旧名, 新名)
        self.on_closed = on_closed        # (项目名,)
        self.thread: TrainThread | None = None
        self._history: list[dict] = []        # 当前展示的历史（实时或某条历史记录）
        self._live_history: list[dict] = []   # 正在进行的训练（实时曲线数据）
        self._latest_progress: tuple | None = None  # 最近一次轮内进度 (epoch, 批序, 批总数)
        self._live_start: str = ""
        self._live_start_dt: datetime | None = None
        self._selected_run_dir: Path | None = None
        self._run_dir: Path | None = None
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(1000)
        self._live_timer.timeout.connect(self._tick_live)

        self.setWindowTitle(f"{project.name}  -  训练平台")
        self.setWindowIcon(QIcon(str(icon_path(
            _TASK_MODEL_ICON.get(project.task_type, "training.png")))))
        self.resize(1280, 900)
        _apply_theme(self)
        self._build_ui()
        self._fill_info_from_project()
        self._fill_params_from_project()
        QTimer.singleShot(0, self._refresh_stats)  # 窗口先显示，统计后台补齐（避免打开卡顿）
        self._switch_detail(0)
        QTimer.singleShot(0, self._refresh_augment_preview)  # 布局完成后渲染预览

    # ================= 界面骨架 =================
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 顶栏：标题 + 任务徽章 + 分段切换
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(64)
        head = QHBoxLayout(header)
        head.setContentsMargins(18, 8, 18, 8)
        self.det_title = QLabel(self.project.name)
        self.det_title.setObjectName("appTitle")
        head.addWidget(self.det_title)
        head.addWidget(_make_badge(self.project.task_type))
        head.addStretch(1)
        self.btn_page_info = QPushButton("项目信息")
        self.btn_page_info.setObjectName("seg")
        self.btn_page_info.setCheckable(True)
        self.btn_page_info.setChecked(True)
        self.btn_page_info.clicked.connect(lambda: self._switch_detail(0))
        self.btn_page_train = QPushButton("训练可视化")
        self.btn_page_train.setObjectName("seg")
        self.btn_page_train.setCheckable(True)
        self.btn_page_train.clicked.connect(lambda: self._switch_detail(1))
        head.addWidget(self.btn_page_info)
        head.addWidget(self.btn_page_train)
        # 齿轮入口：开发者选项（数据体检 / 预检阈值，用户批注 To2.2）。
        # ⚙ 在雅黑字体下渲染过小被裁切（用户实测），指定符号字体修复
        self.btn_dev = QPushButton("⚙")
        self.btn_dev.setFont(QFont("Segoe UI Symbol", 14))
        self.btn_dev.setFixedSize(48, 36)
        self.btn_dev.setToolTip("开发者选项：数据体检与训练前预检的阈值标准")
        self.btn_dev.clicked.connect(self._open_dev_options)
        head.addWidget(self.btn_dev)
        root.addWidget(header)

        self.detail_stack = QStackedWidget()
        root.addWidget(self.detail_stack, stretch=1)
        self.detail_stack.addWidget(self._build_info_page())   # 0
        self.detail_stack.addWidget(self._build_train_page())  # 1
        self.setCentralWidget(central)

    # ---- 页 0：项目信息（滚动）----
    def _build_info_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(12)

        # 项目信息
        g1 = QGroupBox("项目信息")
        f1 = QFormLayout(g1)
        self.info_name = QLineEdit()
        self.info_dataset = QLineEdit()
        btn_ds = QPushButton("浏览…")
        btn_ds.clicked.connect(lambda: self._browse_into(self.info_dataset, "选择数据集文件夹"))
        ds_row = QHBoxLayout()
        ds_row.addWidget(self.info_dataset, stretch=1)
        ds_row.addWidget(btn_ds)
        self.info_model_dir = QLineEdit()
        btn_md = QPushButton("浏览…")
        btn_md.clicked.connect(lambda: self._browse_into(self.info_model_dir, "选择模型储存文件夹"))
        md_row = QHBoxLayout()
        md_row.addWidget(self.info_model_dir, stretch=1)
        md_row.addWidget(btn_md)
        self.info_task = QLabel(self.project.task_type.label)
        self.info_created = QLabel(self.project.created_at)
        btn_save_info = QPushButton("保存修改")
        btn_save_info.clicked.connect(self._save_info)
        f1.addRow("项目名称", self.info_name)
        f1.addRow("数据集路径", ds_row)
        f1.addRow("模型储存路径", md_row)
        f1.addRow("任务类型", self.info_task)
        f1.addRow("创建时间", self.info_created)
        f1.addRow(btn_save_info)
        lay.addWidget(g1)

        # 数据集可视化
        g2 = QGroupBox("数据集可视化")
        v2 = QVBoxLayout(g2)
        head2 = QHBoxLayout()
        self.stats_hint = QLabel("")
        self.stats_hint.setObjectName("hint")
        btn_refresh = QPushButton("刷新统计")
        btn_refresh.clicked.connect(self._refresh_stats)
        btn_refresh.clicked.connect(lambda: self._refresh_augment_preview(new_image=True))
        # 数据体检：仅提示/警告两档，附「去标注平台处理」闭环（批次十一）
        self.btn_health = QPushButton("数据体检")
        self.btn_health.setToolTip(
            "检查类别不均衡、样本过少、验证集类别缺失、疑似重复图、尺寸离群等；\n"
            "结果仅分「提示 / 警告」两档，不会阻止训练；阈值可在右上角 ⚙ 开发者选项调整")
        self.btn_health.clicked.connect(self._run_health_report)
        head2.addWidget(self.stats_hint)
        head2.addStretch(1)
        head2.addWidget(self.btn_health)
        head2.addWidget(btn_refresh)
        v2.addLayout(head2)
        self.dist_canvas = DistributionCanvas()
        v2.addWidget(self.dist_canvas)
        # O12（V0.7）：训练/验证划分环形图（随统计刷新）
        self.split_canvas = SplitDonutCanvas()
        self.split_canvas.setMaximumHeight(240)
        v2.addWidget(self.split_canvas)
        lay.addWidget(g2)

        # O11（V0.7）：数据网格（缩略图墙；后台扫描+懒加载缩略+LRU）
        g2b = QGroupBox("数据网格（双击图片用系统看图打开）")
        v2b = QVBoxLayout(g2b)
        from .data_grid import get_data_grid_class

        self.data_grid = get_data_grid_class()()
        v2b.addWidget(self.data_grid)
        lay.addWidget(g2b)

        # 图像增强策略
        g3 = QGroupBox("图像增强策略（勾选后启用；用于数据倍增与随机变化）")
        grid = QGridLayout(g3)
        self.augment_controls: dict[str, tuple[QCheckBox, dict]] = {}
        self.augment_prob_spins: dict[str, _DoubleSpin] = {}  # 每算子启用概率
        for r, (op, label, params) in enumerate(AUGMENT_ROWS):
            check = QCheckBox(label)
            spins = {}
            for c, (pname, default, vmin, vmax, is_int) in enumerate(params, start=1):
                spin = _Spin() if is_int else _DoubleSpin()
                if is_int:
                    spin.setRange(int(vmin), int(vmax))
                    spin.setValue(int(default))
                else:
                    spin.setDecimals(2)
                    spin.setRange(vmin, vmax)
                    spin.setSingleStep(0.05)
                    spin.setValue(default)
                grid.addWidget(QLabel(pname), r, c * 2 - 1)
                grid.addWidget(spin, r, c * 2)
                spins[pname] = spin
            # 启用概率列：每个算子独立掷硬币（旧项目无 augment_probs 时回填 0.5）
            pc = len(params) * 2 + 1
            grid.addWidget(QLabel("概率"), r, pc)
            prob = _DoubleSpin()
            prob.setDecimals(2)
            prob.setRange(0.05, 1.0)
            prob.setSingleStep(0.05)
            prob.setValue(0.5)
            prob.setToolTip("该算子的启用概率：每次取样独立判定是否应用")
            grid.addWidget(prob, r, pc + 1)
            prob.valueChanged.connect(self._collect_augment_ops)
            self.augment_prob_spins[op] = prob
            check.stateChanged.connect(self._collect_augment_ops)
            grid.addWidget(check, r, 0)
            self.augment_controls[op] = (check, spins)
        resize_check, resize_spins = self.augment_controls["resize"]
        resize_check.stateChanged.connect(self._refresh_input_size_hint)
        # 注意必须用 lambda 隔离：直接连接会把 stateChanged/valueChanged 的
        # 整数实参漏进 _refresh_augment_preview(new_image=…) 形参（非 0 即真），
        # 每次调整尺寸都换成新随机图，"同图对比拉伸"预览效果失效
        resize_check.stateChanged.connect(
            lambda _s: self._refresh_augment_preview())
        for sp in resize_spins.values():
            sp.valueChanged.connect(self._refresh_input_size_hint)
            sp.valueChanged.connect(lambda _v: self._refresh_augment_preview())
        # 随机裁剪（rotate90 兜底）：勾选时自动对齐 resize 的目标宽高比
        crop_check, crop_spins = self.augment_controls["random_crop"]

        def _sync_crop_ratio():
            size = self._current_resize()
            if size and size[1]:
                crop_spins["ratio"].setValue(round(size[0] / size[1], 2))

        crop_check.stateChanged.connect(lambda _s: _sync_crop_ratio())
        btn_save_aug = QPushButton("保存增强策略")
        btn_save_aug.clicked.connect(self._save_augment)
        grid.addWidget(btn_save_aug, len(AUGMENT_ROWS), 0)

        # 效果预览：随机抽取数据集图片，按 resize 参数拉伸缩放（自适应显示）
        self.augment_preview = QLabel()
        self.augment_preview.setMinimumHeight(220)
        self.augment_preview.setAlignment(Qt.AlignCenter)
        self.augment_preview.setStyleSheet(
            "border: 1px solid #d7e2f0; border-radius: 8px; background: #f8fbff;")
        self.preview_info = QLabel("")
        self.preview_info.setObjectName("hint")
        btn_rand = QPushButton("换一张")
        btn_rand.clicked.connect(lambda: self._refresh_augment_preview(new_image=True))
        pv_head = QHBoxLayout()
        pv_head.addWidget(QLabel("效果预览"))
        pv_head.addWidget(self.preview_info, stretch=1)
        pv_head.addWidget(btn_rand)
        pv_box = QWidget()
        pv_lay = QVBoxLayout(pv_box)
        pv_lay.setContentsMargins(0, 6, 0, 0)
        pv_lay.setSpacing(6)
        pv_lay.addLayout(pv_head)
        pv_lay.addWidget(self.augment_preview, stretch=1)
        grid.addWidget(pv_box, len(AUGMENT_ROWS) + 1, 0, 1, 12)  # 跨全部列
        lay.addWidget(g3)

        # 训练参数（Quick/专家双轨，D4：默认快速档 6 核心参数；「专家模式」
        # 展开现有完整折叠表单；顶部三张预设卡一键套用并联动智能推荐；
        # 预算实时摘要明确暂缓）
        g4 = QGroupBox("训练参数")
        v4 = QVBoxLayout(g4)
        mode_row = QHBoxLayout()
        self.btn_mode_quick = QPushButton("⚡ 快速配置")
        self.btn_mode_expert = QPushButton("专家模式")
        for b in (self.btn_mode_quick, self.btn_mode_expert):
            b.setObjectName("seg")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
        self.btn_mode_quick.setToolTip(
            "只呈现 6 个核心参数，档位化配置，零经验 30 秒完成")
        self.btn_mode_expert.setToolTip("展开完整参数表单（数据与验证 / 工业与加速）")
        self.btn_mode_quick.clicked.connect(self._on_mode_quick)
        self.btn_mode_expert.clicked.connect(self._on_mode_expert)
        mode_row.addWidget(self.btn_mode_quick)
        mode_row.addWidget(self.btn_mode_expert)
        mode_row.addStretch(1)
        self.btn_recommend = QPushButton("✨ 智能推荐配置")
        self.btn_recommend.setToolTip(
            "按数据集规模与硬件给出保守建议（冻结骨干 / 工业优化 / 增强概率等）；\n"
            "应用后写回本表单并在状态栏说明理由，仍可手动调整")
        self.btn_recommend.clicked.connect(self._apply_recommendations)
        mode_row.addWidget(self.btn_recommend)
        self.btn_expand_all = QPushButton("全部展开")
        self.btn_expand_all.setToolTip("展开/折叠下方全部分区")
        self.btn_expand_all.clicked.connect(self._toggle_all_sections)
        mode_row.addWidget(self.btn_expand_all)
        v4.addLayout(mode_row)

        self.par_model = _Combo()
        self.par_model.currentTextChanged.connect(self._on_model_changed)
        self.input_size_hint = QLabel("")
        self.input_size_hint.setObjectName("accentText")
        self.input_size_hint.setToolTip(
            "训练输入尺寸与上方增强策略的 resize 宽×高一致：图像拉伸到该尺寸训练\n"
            "（比例随设定改变，预览所见即训练所见）。\n"
            "未启用 resize 时使用模型默认尺寸，由各模型自行缩放（YOLO letterbox /\n"
            "R-CNN 短边缩放 / 分类拉伸方形）——长宽比极端的图片建议保持 resize 启用")
        self.par_train_ratio = _DoubleSpin()
        self.par_train_ratio.setDecimals(2)
        self.par_train_ratio.setRange(0.1, 1.0)
        self.par_train_ratio.setSingleStep(0.05)
        self.par_train_ratio.setToolTip(
            "训练集占整体数据的比例。与验证集比例之和可以大于 100%：\n"
            "此时验证集从整体数据抽出后，训练集仍按占比从整体抽取（允许重叠），\n"
            "例如 训练集 100% + 验证集 20%（稀缺缺陷类别场景，验证集为训练集子集）")
        self.par_val_ratio = _DoubleSpin()
        self.par_val_ratio.setDecimals(2)
        self.par_val_ratio.setRange(0.0, 0.9)
        self.par_val_ratio.setSingleStep(0.05)
        self.par_val_ratio.setToolTip("验证集占整体数据的比例（为 0 时验证集复用训练集）")
        self.par_multiply = _Spin()
        self.par_multiply.setRange(1, 50)
        self.par_multiply.setToolTip(
            "每轮重复学习次数：每个 epoch 中训练集被学习的遍数；\n"
            "增强策略按随机概率（约 50%）启用，与该值无关——\n"
            "设为 1 且启用增强时，训练中也会随机增强")
        self.par_batch = _Spin()
        self.par_batch.setRange(1, 256)
        self.par_batch.setToolTip(
            "批量大小：检测/分割推荐 2~4（大分辨率下平台会按显存预算自动下调，\n"
            "必要时启用梯度检查点保证不 OOM）；分类推荐 16~32。\n"
            "建议使用默认批量大小以获得最佳性能")
        self.par_epochs = _Spin()
        self.par_epochs.setRange(1, 2000)
        self.par_epochs.setToolTip(
            "训练代数：工业小数据集建议 12~30；\n"
            "需要更多学习次数时优先调大「每轮重复学习次数」而不是代数")
        self.par_lr = _DoubleSpin()
        self.par_lr.setDecimals(6)
        self.par_lr.setRange(0.000001, 1.0)
        self.par_lr.setSingleStep(0.0001)
        self.par_lr.setToolTip(
            "选择模型时自动带入推荐值：YOLO 0.01 / R-CNN 0.005 / 分类 0.001；\n"
            "从已有权重继续训练（迁移学习）时建议调低为推荐值的 1/5~1/10")
        self.par_wd = _DoubleSpin()
        self.par_wd.setDecimals(6)
        self.par_wd.setRange(0.0, 1.0)
        self.par_eval_interval = _Spin()
        self.par_eval_interval.setRange(1, 100)
        self.par_eval_interval.setToolTip("每多少轮验证一次（YOLO 家族每轮都会验证）")
        self.par_device = _Combo()
        self.par_device.addItem("自动", "auto")
        self.par_device.addItem("CPU", "cpu")
        if _cuda_available():
            import torch

            self.par_device.addItem(f"GPU ({torch.cuda.get_device_name(0)})", "cuda")
        self.par_device.currentIndexChanged.connect(self._save_params)
        btn_save_par = QPushButton("保存参数")
        btn_save_par.clicked.connect(self._save_params)
        self.btn_start = QPushButton("▶ 开始训练")
        self.btn_start.setObjectName("accent")
        self.btn_start.clicked.connect(self._start_training)
        self.par_industrial = QCheckBox("工业场景优化（注意力 + 改进损失）")
        self.par_industrial.setToolTip(
            "针对工业缺陷检测的模型与损失改进（详见 docs/工业优化说明.md）；\n"
            "关闭即为基线模型，便于消融对照")
        # 训练加速（GPU 工控机上按需开启；CPU 机器 AMP 自动失效）
        self.par_freeze = QCheckBox("训练加速：冻结骨干网络（快速迁移模式）")
        self.par_freeze.setToolTip(
            "冻结预训练骨干，只训练分类头/检测头与工业模块（CBAM/ECA/DCN 仍参与训练）；\n"
            "每轮迭代提速约 40~60%，小数据集精度通常持平；\n"
            "领域差距极大（红外/医学等）时精度可能受限，可关闭回退；\n"
            "未使用预训练权重时该选项自动忽略")
        self.par_rpn = QCheckBox("削减 RPN 候选框（仅 Faster/Mask R-CNN 生效）")
        self.par_rpn.setToolTip(
            "每图候选 2000→256：工业单图目标少时几乎无精度损失，迭代更快、显存更省")
        self.par_mask_light = QCheckBox("轻量掩码头（仅 Mask R-CNN 生效）")
        self.par_mask_light.setToolTip(
            "掩码头 4 层 256 通道 → 2 层 128 通道：掩码分支更快、更省显存；\n"
            "工业缺陷不需要像素级细腻边缘时几乎无精度损失，可随时关闭回退")
        self.par_anchor = QCheckBox("小目标锚框自适应（仅 R-CNN 家族生效）")
        self.par_anchor.setToolTip(
            "锚框金字塔底径=入网短边/80（默认 32 针对大目标设计，\n"
            "工业小缺陷与最小锚 IoU 过低会导致 RPN 正样本稀少）；\n"
            "小目标数据集建议开启；切图训练时按 tile 边长取底径")
        self.par_slim_head = QCheckBox("slim 头 FPN64（仅 maskrcnn18 生效）")
        self.par_slim_head.setToolTip(
            "T9-5 slim 头：FPN 64 通道+两层 256 FC+4×64 掩码头（约 12.5M 参数），\n"
            "可加载 mmdet 预训练权重；\n"
            "显存占用约为标准头一半，小数据集收敛更快。仅在使用 maskrcnn18 时生效")
        self.par_p2 = QCheckBox("P2 小目标检测头（仅 YOLO 家族生效）")
        self.par_p2.setToolTip(
            "在所选 YOLO 模型（n/s，检测/分割）上增加 stride 4 的第四检测层，\n"
            "面向短边 ≤8px 的微小缺陷；骨干/颈部/原检测分支继承官方预训练，\n"
            "仅新增的 P2 分支从头学习，建议适当增加训练轮次；\n"
            "显存约为普通模型的 2 倍（1920 输入 n-seg 批量 4 约 13 GiB），\n"
            "显存不足请减小批量；训练产物自带结构，测试/部署无需再勾选；\n"
            "迁移学习/从剪枝模型继续训练时沿用起始权重的结构，本开关不生效。\n"
            "R-CNN 与分类模型忽略该选项")
        self.par_p2.toggled.connect(self._refresh_input_size_hint)
        # R-CNN 优化器三件套
        self.par_rcnn_opt = _Combo()
        self.par_rcnn_opt.addItem("AdamW（推荐）", "adamw")
        self.par_rcnn_opt.addItem("SGD+动量（旧行为）", "sgd")
        self.par_rcnn_opt.currentIndexChanged.connect(self._save_params)
        self.par_rcnn_lr = _DoubleSpin()
        self.par_rcnn_lr.setRange(0.0, 1.0)
        self.par_rcnn_lr.setDecimals(6)
        self.par_rcnn_lr.setSingleStep(1e-5)
        self.par_rcnn_lr.setValue(0.0)
        self.par_rcnn_lr.setToolTip("0=自动（AdamW→1e-4；SGD→学习率字段）")
        self.par_rcnn_lr.valueChanged.connect(self._save_params)
        self.par_rcnn_wd = _DoubleSpin()
        self.par_rcnn_wd.setRange(0.0, 0.5)
        self.par_rcnn_wd.setDecimals(4)
        self.par_rcnn_wd.setSingleStep(0.005)
        self.par_rcnn_wd.setValue(0.05)
        self.par_rcnn_wd.valueChanged.connect(self._save_params)

        self.par_amp = QCheckBox("AMP 混合精度（GPU 提速；CPU 自动关闭）")
        self.par_amp.setChecked(True)
        self.par_tf32 = QCheckBox("GPU 提速：TF32 与 cuDNN 自动调优（Ampere+ 显卡；CPU 无影响）")
        self.par_tf32.setChecked(True)
        self.par_tf32.setToolTip(
            "TF32：Ampere 及以上 NVIDIA 显卡对 FP32 计算零成本加速（约 1.5~2 倍）；\n"
            "cuDNN 自动调优：固定输入尺寸训练时自动搜索并缓存最优卷积算法（约 1.1~1.3 倍）；\n"
            "无 GPU 或较老显卡时不生效，可放心保持开启")
        self.par_epoch_w = QCheckBox("保存每轮权重（轮次权重文件夹，新训练覆盖）")
        self.par_epoch_w.setChecked(True)
        self.par_epoch_w.setToolTip(
            "每次训练共用「模型储存路径\\epoch_weights」文件夹，逐轮保存权重：\n"
            "训练开始时清空旧训练的轮次权重（覆盖式，只占最近一次训练的空间）；\n"
            ""
            "轮次权重可在测试平台直接加载对比")
        self.par_patience = _Spin()
        self.par_patience.setRange(0, 200)
        self.par_patience.setValue(10)
        self.par_patience.setToolTip("连续 N 轮验证无提升则提前停止训练；0 = 关闭早停")
        self.par_tiling = QCheckBox("切图训练（大图小目标，R-CNN 家族）")
        self.par_tiling.setToolTip(
            "把原图按滑窗切成 tile 训练：小目标占画面比例更大、更易学到；\n"
            "背景 tile 按 15% 保留、10% 原图整图混入、验证集始终整图评估；\n"
            "**推理必须配合分块**：测试平台「4K 大图分块推理」或服务 tiled=true\n"
            "（与 T9-2 两段式缩放互斥使用更清晰：切图训练时不建议再设网络输入尺寸）")
        self.par_neg = QCheckBox("负样本训练（OK 图降误检，检测/分割）")
        self.par_neg.setChecked(True)
        self.par_neg.setToolTip(
            "数据集中「OK / 良品 / 正常 / 合格 / 负样本 / 背景」等文件夹里的\n"
            "未标注图片会作为纯背景样本参与训练与验证（默认开启）：\n"
            "模型学到「这些图上什么都不该检出」，直接压低误检率；\n"
            "训练日志与历史记录会给出负样本张数与 OK 图误检率指标；\n"
            "标注图划分不受影响；负样本图数超过标注图数时会自动抽样到 1:1")
        # O5（V0.7）：颜色模式（灰度/彩色）——颜色无判别意义时用灰度防干扰
        self.par_gray = QCheckBox("灰度训练（颜色无判别意义时防颜色干扰）")
        self.par_gray.setToolTip(
            "开启后训练读图统一转灰度（复制 3 通道，骨干零改动），\n"
            "推理/部署侧按权重内记录自动同口径转灰；\n"
            "适合颜色不携带判别信息、打光颜色易漂移的工业场景；\n"
            "对检测/分割/分类家族生效（YOLO 推理侧加速导出暂不透传该标记）")
        # E3（V0.7）：确定性训练 opt-in
        self.par_det = QCheckBox("确定性训练（可复现，默认关）")
        self.par_det.setToolTip(
            "开启后固定全部随机种子并启用确定性算法（CUBLAS_WORKSPACE_CONFIG\n"
            ":4096:8 + torch.use_deterministic_algorithms）：同机同种子同依赖\n"
            "版本两次训练指标一致，run_manifest.json 标记 deterministic=true；\n"
            "代价：训练速度下降、个别算子不支持确定性实现时降级告警（不中断）")
        self.par_tile_size = _Spin()
        self.par_tile_size.setRange(256, 2048)
        self.par_tile_size.setSingleStep(32)
        self.par_tile_size.setValue(640)
        self.par_tile_size.setToolTip("切图 tile 边长（像素，32 的倍数最佳）")
        self.par_train_input = _Spin()
        self.par_train_input.setRange(0, 4096)
        self.par_train_input.setSingleStep(32)
        self.par_train_input.setValue(0)
        self.par_train_input.setToolTip(
            "两段式缩放（可选，默认 0=跟随增强画布=现状）：\n"
            "显式设 32 的倍数且小于画布短边时启用——画布增强保持画布口径，\n"
            "入网前按比例缩到该尺寸（小目标占画面比例更大、显存更省）；\n"
            "仅 R-CNN 家族生效；推理侧请配合测试平台「4K 大图分块推理」\n"
            "或服务 tiled=true（与切图训练 T9-3 配对使用效果最佳）")
        self.par_transfer = QCheckBox("迁移学习（从已有模型权重开始训练）")
        self.par_transfer.setToolTip("勾选后通过文件选择器指定起始权重（支持 .pt，"
                                     "含剪枝模型与历史训练记录的 best 权重）；"
                                     "类别需与当前数据集一致")
        self.par_transfer.toggled.connect(self._on_transfer_toggled)
        transfer_row = QHBoxLayout()
        self.par_transfer_edit = QLineEdit()
        self.par_transfer_edit.setPlaceholderText("迁移学习起始权重文件路径")
        btn_pick_model = QPushButton("浏览…")
        btn_pick_model.clicked.connect(self._pick_transfer_file)
        transfer_row.addWidget(self.par_transfer_edit, stretch=1)
        transfer_row.addWidget(btn_pick_model)

        # ---- Quick 档控件（与专家表单 par_* 双向同步；保存仍以 par_* 为准）----
        self.q_model = _Combo()
        self.q_epochs = _Spin()
        self.q_epochs.setRange(1, 2000)
        self.q_batch = _Spin()
        self.q_batch.setRange(1, 256)
        self.q_lr = _DoubleSpin()
        self.q_lr.setDecimals(6)
        self.q_lr.setRange(0.000001, 1.0)
        self.q_lr.setSingleStep(0.0001)
        self.q_device = _Combo()   # 条目在 _fill_quick_controls 与 par_device 同步
        self.q_size = QLabel("")
        self.q_size.setObjectName("accentText")
        for _qw in (self.q_model, self.q_epochs, self.q_batch, self.q_lr,
                    self.q_device):
            _qw.setMinimumHeight(34)   # Quick 档大控件
        self._rec_badges: dict = {}    # 参数键 → 「已推荐」徽章
        self._preset_values: dict = {}

        # 分区一：基础配置（默认展开）
        self.sec_basic = CollapsibleSection("基础配置", expanded=True)
        fb = QFormLayout()
        fb.setContentsMargins(0, 0, 0, 0)
        fb.addRow("训练模型", self.par_model)
        fb.addRow(self.par_transfer)
        fb.addRow("模型文件", transfer_row)
        fb.addRow("批量大小", self.par_batch)
        fb.addRow("训练轮次", self.par_epochs)
        fb.addRow("学习率", self.par_lr)
        fb.addRow("权重衰减", self.par_wd)
        fb.addRow("训练设备", self.par_device)
        self.sec_basic.add_layout(fb)
        v4.addWidget(self.sec_basic)

        # 分区二：数据与验证（默认折叠）
        self.sec_data = CollapsibleSection("数据与验证", expanded=False)
        fd = QFormLayout()
        fd.setContentsMargins(0, 0, 0, 0)
        fd.addRow("训练输入尺寸", self.input_size_hint)
        fd.addRow("网络输入尺寸（可选）", self.par_train_input)
        fd.addRow(self.par_tiling)
        fd.addRow("切图 tile 边长", self.par_tile_size)
        fd.addRow(self.par_neg)
        fd.addRow("训练集比例", self.par_train_ratio)
        fd.addRow("验证集比例", self.par_val_ratio)
        fd.addRow("数据倍增倍数", self.par_multiply)
        fd.addRow("早停耐心（轮）", self.par_patience)
        fd.addRow("评估间隔（轮）", self.par_eval_interval)
        fd.addRow(self.par_epoch_w)
        self.sec_data.add_layout(fd)
        v4.addWidget(self.sec_data)

        # 分区三：工业与加速优化（默认折叠）
        self.sec_adv = CollapsibleSection("工业与加速优化", expanded=False)
        fa = QFormLayout()
        fa.setContentsMargins(0, 0, 0, 0)
        fa.addRow(self.par_industrial)
        fa.addRow(self.par_freeze)
        fa.addRow(self.par_rpn)
        fa.addRow(self.par_mask_light)
        fa.addRow(self.par_anchor)
        fa.addRow(self.par_slim_head)
        fa.addRow(self.par_p2)
        fa.addRow(self.par_gray)  # O5（V0.7）：颜色模式
        fa.addRow(self.par_det)   # E3（V0.7）：确定性训练
        fa.addRow("R-CNN 优化器", self.par_rcnn_opt)
        fa.addRow("R-CNN 学习率", self.par_rcnn_lr)
        fa.addRow("R-CNN 权重衰减", self.par_rcnn_wd)
        fa.addRow(self.par_amp)
        fa.addRow(self.par_tf32)
        self.sec_adv.add_layout(fa)

        # ---- Quick 面板：预设卡 + 6 参数行化表单 ----
        self.quick_panel = QWidget()
        qp = QVBoxLayout(self.quick_panel)
        qp.setContentsMargins(0, 4, 0, 0)
        qp.setSpacing(12)

        presets = QHBoxLayout()
        presets.setSpacing(10)
        self._preset_cards = {}
        for key, name, desc, values, color in _PRESET_DEFS:
            card = self._make_preset_card(key, name, desc, values, color)
            presets.addWidget(card)
            self._preset_cards[key] = card
        presets.addStretch(1)
        qp.addLayout(presets)

        qgrid = QGridLayout()
        qgrid.setHorizontalSpacing(24)
        qgrid.setVerticalSpacing(10)
        qgrid.addWidget(self._quick_row("训练模型", self.q_model), 0, 0)
        qgrid.addWidget(self._quick_row("训练设备", self.q_device), 0, 1)
        qgrid.addWidget(self._quick_row("训练轮次", self.q_epochs, "轮",
                                        badge_key="epochs"), 1, 0)
        qgrid.addWidget(self._quick_row("批量大小", self.q_batch, "",
                                        badge_key="batch"), 1, 1)
        qgrid.addWidget(self._quick_row("学习率", self.q_lr, "",
                                        badge_key="lr"), 2, 0)
        self._sync_quick_size()
        qgrid.addWidget(self._quick_row("输入尺寸", self.q_size, "",  # noqa: E501
                                        badge_key="size"), 2, 1)
        qp.addLayout(qgrid)

        qtools = QHBoxLayout()
        self.btn_reset_defaults = QPushButton("↺ 恢复默认")
        self.btn_reset_defaults.setToolTip(
            "把 6 个核心参数恢复为打开项目时的默认值")
        self.btn_reset_defaults.clicked.connect(self._reset_quick_defaults)
        qtools.addWidget(self.btn_reset_defaults)
        qtools.addStretch(1)
        qp.addLayout(qtools)
        v4.addWidget(self.quick_panel)

        # ---- 专家区容器（完整折叠表单，专家模式显示）----
        self.expert_host = QWidget()
        ev = QVBoxLayout(self.expert_host)
        ev.setContentsMargins(0, 4, 0, 0)
        ev.setSpacing(4)
        ev.addWidget(self.sec_basic)
        ev.addWidget(self.sec_data)
        ev.addWidget(self.sec_adv)
        v4.addWidget(self.expert_host)

        # 初始模式（ui_prefs 持久化；默认快速档）
        self._set_param_mode(
            load_prefs().get("train_param_mode", "quick") != "expert",
            save=False)

        btns = QHBoxLayout()
        btns.addWidget(btn_save_par)
        btns.addWidget(self.btn_start)
        btns.addStretch(1)
        v4.addLayout(btns)
        lay.addWidget(g4)

        lay.addStretch(1)
        scroll.setWidget(body)
        # 纵向滚动条常显：折叠分区时滚动条出现/消失会改变视口宽度，
        # 造成整页控件回流闪动（用户回执 D1"窗口瞬间变形"）
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    # ---- 页 1：训练可视化（左：历史记录列表；右：所选记录详情）----
    def _build_train_page(self) -> QWidget:
        page = QWidget()
        lay = QHBoxLayout(page)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(12)

        # 左：训练记录列表（读取模型储存目录下的历次训练）
        panel = QFrame()
        panel.setObjectName("runPanel")
        panel.setFixedWidth(280)
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(10, 10, 10, 10)
        pv.setSpacing(8)
        cap = QLabel("训练记录")
        bind_style(cap, lambda: f"font-weight: bold; color: {token('primary_deep')};"
                                " border: none; background: transparent;")
        pv.addWidget(cap)
        self.btn_compare = QPushButton("对比所选")
        self.btn_compare.setToolTip("按住 Ctrl 点击选择 2~3 条记录后对比（新窗口：参数 / 曲线 / 混淆矩阵）")
        self.btn_compare.clicked.connect(self._open_compare)  # 常启用：点击时校验并提示
        pv.addWidget(self.btn_compare)
        self.run_hint = QLabel("")
        self.run_hint.setObjectName("hint")
        self.run_hint.setWordWrap(True)
        pv.addWidget(self.run_hint)
        self.run_list = QListWidget()
        # ExtendedSelection：单击切换详情、Ctrl+点击多选（用户回执 E1）；
        # 多选上限 3 条，超出的新选择被拒绝并提示（见 _on_run_selection_changed）
        self.run_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._prev_selected: list = []   # 上一次合法选中的路径集合（超限回退用）
        self.run_list.currentItemChanged.connect(self._on_run_selected)
        self.run_list.itemSelectionChanged.connect(self._on_run_selection_changed)
        pv.addWidget(self.run_list, stretch=1)
        # 空状态占位（与列表同槽互斥显示）
        from ..ui_widgets import EmptyState as _EmptyState

        self.run_empty = _EmptyState("🗂", "暂无训练记录",
                                     "完成一次训练后，记录会自动出现在这里")
        self.run_empty.hide()
        pv.addWidget(self.run_empty, stretch=1)
        lay.addWidget(panel)

        # 右：详情（滚动区；画布滚轮已透传，鼠标悬停画布也能直接滚动页面）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        lay2 = QVBoxLayout(body)
        lay2.setContentsMargins(4, 0, 4, 0)
        lay2.setSpacing(12)

        head = QHBoxLayout()
        self.train_status = QLabel("尚未开始训练")
        self.train_status.setStyleSheet("font-weight: bold;")
        self.gpu_mem = QLabel("")
        self.gpu_mem.setObjectName("muted")
        self.btn_stop = QPushButton("■ 停止训练")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_training)
        self.btn_open_dir = QPushButton("打开输出目录")
        self.btn_open_dir.setEnabled(False)
        self.btn_open_dir.clicked.connect(self._open_output_dir)
        head.addWidget(self.train_status)
        head.addWidget(self.gpu_mem)
        head.addStretch(1)
        head.addWidget(self.btn_stop)
        head.addWidget(self.btn_open_dir)
        lay2.addLayout(head)

        # 训练进度：开始时间 / 结束时间 / 训练时长 + 百分比进度条
        g0 = QGroupBox("训练进度")
        gd = QGridLayout(g0)
        gd.setHorizontalSpacing(10)
        gd.setVerticalSpacing(8)
        self.run_start = QLabel("—")
        self.run_end = QLabel("—")
        self.run_duration = QLabel("—")
        for col, text in enumerate(("开始时间", "结束时间", "训练时长")):
            key = QLabel(text)
            key.setObjectName("muted")
            gd.addWidget(key, 0, col * 2)
        for col, val in enumerate((self.run_start, self.run_end, self.run_duration)):
            gd.addWidget(val, 0, col * 2 + 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #d7e2f0; border-radius: 6px;"
            " background: #ffffff; text-align: center; color: #1f2937;"
            " min-height: 20px; font-weight: bold; font-size: 15px; }"
            "QProgressBar::chunk { background-color: #5ec6dd; border-radius: 5px; }")
        gd.addWidget(self.progress_bar, 1, 0, 1, 6)
        lay2.addWidget(g0)

        # 训练曲线：悬停显示最近轮次的数值（三位小数）
        g1 = QGroupBox("训练曲线")
        v1 = QVBoxLayout(g1)
        self.loss_plot = _PagePlot(axisItems={"bottom": _IntAxis(orientation="bottom")})
        _add_legend(self.loss_plot)
        self.loss_plot.setLabel("bottom", "轮次")
        self.loss_plot.showGrid(x=True, y=True, alpha=0.3)
        self.loss_plot.setMouseEnabled(x=False, y=False)
        self.loss_plot.hideButtons()
        self.metric_plot = _PagePlot(axisItems={"bottom": _IntAxis(orientation="bottom")})
        _add_legend(self.metric_plot)
        self.metric_plot.setLabel("bottom", "轮次")
        self.metric_plot.showGrid(x=True, y=True, alpha=0.3)
        self.metric_plot.setMouseEnabled(x=False, y=False)
        self.metric_plot.hideButtons()
        self.metric_plot.setYRange(0, 1, padding=0)
        _style_plot(self.loss_plot, "损失曲线")
        _style_plot(self.metric_plot, "评估指标")
        self.metric_hint = QLabel(
            "mAP/mAR：R-CNN 族为 IoU 0.10/0.25/0.50 三档平均；"
            "YOLO 族为 AP@50 与召回率；分类为 Top-1")
        self.metric_hint.setObjectName("hint")
        v1.addWidget(self.metric_hint)
        v1.addWidget(self.loss_plot)
        v1.addWidget(self.metric_plot)
        lay2.addWidget(g1)
        self._loss_hover = _PlotHover(self.loss_plot, self._loss_series)
        self._metric_hover = _PlotHover(self.metric_plot, self._metric_series)

        g2 = QGroupBox("混淆矩阵")
        v2 = QVBoxLayout(g2)
        self.cm_canvas = ConfusionMatrixCanvas()
        v2.addWidget(self.cm_canvas)
        # O1（V0.7）：逐类精确率/召回率表（VIMO"类型详情表"对照项）
        self.per_class_table = QTableWidget(0, 9)
        self.per_class_table.setHorizontalHeaderLabels(
            ["类别", "GT 实例", "检出", "TP", "FP", "FN", "精确率", "召回率", "AP@50"])
        self.per_class_table.horizontalHeader().setStretchLastSection(True)
        self.per_class_table.verticalHeader().setVisible(False)
        self.per_class_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.per_class_table.setMaximumHeight(190)
        self.per_class_table.setToolTip(
            "口径说明：\n"
            "· TP/FP/FN 来自 IoU≥0.5 同类匹配混淆矩阵（训练收尾统计）；\n"
            "· 精确率=TP/检出，召回率=TP/GT 实例；检出=TP+FP，GT 实例=TP+FN；\n"
            "· AP@50 为 COCO 口径逐类平均（与汇总 mAP@50 同口径）；\n"
            "· 曲线/摘要的 mAP・mAR 为宽松口径（IoU 0.10/0.25/0.50 三档平均），\n"
            "  两套口径并存，调优时先确认目标再对号入座；\n"
            "· \"-\" 表示该类无样本或无有效评估组合。")
        v2.addWidget(self.per_class_table)
        lay2.addWidget(g2)

        g3 = QGroupBox("本次训练参数")
        v3 = QVBoxLayout(g3)
        self.echo_form = QFormLayout()
        v3.addLayout(self.echo_form)
        self.final_metrics = QLabel("")
        bind_style(self.final_metrics,
                   lambda: f"color: {token('primary_deep')}; font-weight: bold;")
        self.final_metrics.setWordWrap(True)
        v3.addWidget(self.final_metrics)
        lay2.addWidget(g3)

        # 底部：删除本条历史记录（含输出文件夹）
        del_row = QHBoxLayout()
        self.btn_delete_run = QPushButton("删除本条训练记录（含输出文件夹）")
        self.btn_delete_run.setObjectName("danger")
        self.btn_delete_run.setEnabled(False)
        self.btn_delete_run.clicked.connect(self._delete_current_run)
        del_row.addStretch(1)
        del_row.addWidget(self.btn_delete_run)
        lay2.addLayout(del_row)

        lay2.addStretch(1)
        scroll.setWidget(body)
        lay.addWidget(scroll, stretch=1)
        return page

    # ================= 信息 / 参数 =================
    def _browse_into(self, line_edit: QLineEdit, title: str):
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            line_edit.setText(path)

    def _fill_info_from_project(self):
        p = self.project
        self.info_name.setText(p.name)
        self.info_dataset.setText(p.dataset_dir)
        self.info_model_dir.setText(p.model_dir)
        dg = getattr(self, "data_grid", None)
        if dg is not None:      # O11（V0.7）：数据网格随项目刷新
            dg.set_dataset(p.dataset_dir)
        # 快照源字典：循环内 setChecked/setValue 会同步触发 _collect_augment_ops
        # 把 project.augment_ops 替换为当前 UI 状态，若边读边写，未处理的算子
        # 会被当作"不存在"而漏勾选（rotate90 行曾因此丢失）
        saved_ops = dict(p.augment_ops or {})
        saved_probs = dict(getattr(p, "augment_probs", None) or {})
        for op, (check, spins) in self.augment_controls.items():
            params = saved_ops.get(op)
            if params is None:
                check.setChecked(False)
            else:
                check.setChecked(True)
                for pname, spin in spins.items():
                    spin.setValue(params.get(pname, spin.value()))
            prob = self.augment_prob_spins.get(op)
            if prob is not None:  # 旧项目无 augment_probs 字段 → 回填默认 0.5
                prob.setValue(float(saved_probs.get(op, 0.5)))

    def _on_transfer_toggled(self):
        self.par_transfer_edit.setEnabled(self.par_transfer.isChecked())

    def _pick_transfer_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择迁移学习起始权重", "",
            "模型权重 (*.pt);;所有文件 (*)")
        if path:
            self.par_transfer_edit.setText(path)
            self.par_transfer.setChecked(True)

    def _fill_params_from_project(self):
        p = self.project
        self.par_model.blockSignals(True)
        self.par_model.clear()
        # 仅列出算法引擎的基准模型；迁移学习改为勾选 + 指定模型文件
        from ..engine.models import list_models  # torch 延迟导入（T1-4）

        for m in list_models(p.task):
            self.par_model.addItem(f"{m['name']}  ({m['desc']})", m["name"])
        idx = self.par_model.findData(p.model_name)
        if idx < 0 and p.model_name:  # 旧项目记录的是 pruned:/run: 引用 → 回退基准名
            base = (p.model_name.split(":", 1)[-1].split("/", 1)[0])
            idx = self.par_model.findData(base)
        self.par_model.setCurrentIndex(max(idx, 0))
        self.par_model.blockSignals(False)
        self.par_transfer.setChecked(bool(p.transfer_file))
        self.par_transfer_edit.setText(p.transfer_file or "")
        self._on_transfer_toggled()
        self._refresh_input_size_hint()
        self.par_train_ratio.setValue(getattr(p, "train_ratio", 0.8) or 0.8)
        self.par_val_ratio.setValue(p.val_ratio)
        self.par_multiply.setValue(p.multiply_factor)
        self.par_batch.setValue(p.batch_size)
        self.par_epochs.setValue(p.epochs)
        self.par_lr.setValue(p.learning_rate)
        self.par_wd.setValue(p.weight_decay)
        self.par_eval_interval.setValue(p.eval_interval)
        dev_idx = self.par_device.findData(p.device)
        self.par_device.setCurrentIndex(max(dev_idx, 0))
        self.par_industrial.setChecked(getattr(p, "industrial_opt", True))
        self.par_freeze.setChecked(getattr(p, "freeze_backbone", False))
        self.par_rpn.setChecked(getattr(p, "rpn_reduced", False))
        self.par_mask_light.setChecked(getattr(p, "mask_head_light", False))
        self.par_anchor.setChecked(getattr(p, "anchor_small", False))
        self.par_slim_head.setChecked(getattr(p, "slim_head", False))
        self.par_p2.setChecked(getattr(p, "p2_head", False))
        _ri = self.par_rcnn_opt.findData(
            getattr(p, "rcnn_optimizer", "adamw"))
        self.par_rcnn_opt.setCurrentIndex(max(_ri, 0))
        self.par_rcnn_lr.setValue(getattr(p, "rcnn_learning_rate", 0.0))
        self.par_rcnn_wd.setValue(getattr(p, "rcnn_weight_decay", 0.05))
        self.par_amp.setChecked(getattr(p, "use_amp", True))
        self.par_tf32.setChecked(getattr(p, "tf32", True))
        self.par_epoch_w.setChecked(getattr(p, "save_epoch_weights", True))
        self.par_patience.setValue(getattr(p, "patience", 10))
        self.par_train_input.setValue(getattr(p, "train_input", 0))
        self.par_tiling.setChecked(getattr(p, "tiling", False))
        self.par_tile_size.setValue(getattr(p, "tile_size", 640))
        self.par_neg.setChecked(getattr(p, "use_negatives", True))
        self.par_gray.setChecked(getattr(p, "color_mode", "color") == "gray")
        self.par_det.setChecked(getattr(p, "deterministic", False))
        # Quick 档控件同步填充 + 默认值快照（「恢复默认」与推荐徽章复位用）
        self._fill_quick_controls()
        self._param_defaults = {
            "model": self.par_model.currentData(),
            "epochs": self.par_epochs.value(),
            "batch": self.par_batch.value(),
            "lr": self.par_lr.value(),
            "device": self.par_device.currentData(),
        }
        self._clear_recommended()

    def _switch_detail(self, index: int):
        self.detail_stack.setCurrentIndex(index)
        self.btn_page_info.setChecked(index == 0)
        self.btn_page_train.setChecked(index == 1)
        if index == 1:
            self._reload_history()  # 进入可视化页时重扫模型目录，历史记录保持最新

    def _collect_augment_ops(self):
        """从界面控件收集增强策略与每算子启用概率（勾选的算子才生效）。"""
        ops = {}
        probs = {}
        for op, (check, spins) in self.augment_controls.items():
            if not check.isChecked():
                continue
            ops[op] = {name: spin.value() for name, spin in spins.items()}
            probs[op] = self.augment_prob_spins[op].value()
        self.project.augment_ops = ops
        self.project.augment_probs = probs
        return ops

    def _save_info(self):
        p = self.project
        old_name = p.name
        p.name = self.info_name.text().strip()
        p.dataset_dir = self.info_dataset.text().strip()
        p.model_dir = self.info_model_dir.text().strip()
        try:
            save_project(p, old_name=old_name)
        except ValueError as e:
            QMessageBox.warning(self, "保存失败", str(e))
            return
        self.setWindowTitle(f"{p.name}  -  训练平台")
        self.det_title.setText(p.name)
        dg = getattr(self, "data_grid", None)
        if dg is not None:      # O11：数据集路径可能被修改
            dg.set_dataset(p.dataset_dir)
        if self.on_renamed is not None:
            self.on_renamed(old_name, p.name)
        toast(self, "项目信息已保存", "success")

    def _save_params(self):
        if not self.quick_panel.isHidden():
            # Quick 模式下用户可能改过快速档控件：先回写专家表单再保存
            self._experts_from_quick()
        p = self.project
        p.model_name = self.par_model.currentData() or p.model_name
        p.transfer_file = (self.par_transfer_edit.text().strip()
                           if self.par_transfer.isChecked() else "")
        p.val_ratio = self.par_val_ratio.value()
        p.train_ratio = self.par_train_ratio.value()
        p.multiply_factor = self.par_multiply.value()
        p.batch_size = self.par_batch.value()
        p.epochs = self.par_epochs.value()
        p.learning_rate = self.par_lr.value()
        p.weight_decay = self.par_wd.value()
        p.eval_interval = self.par_eval_interval.value()
        p.device = self.par_device.currentData()
        p.industrial_opt = self.par_industrial.isChecked()
        p.freeze_backbone = self.par_freeze.isChecked()
        p.rpn_reduced = self.par_rpn.isChecked()
        p.mask_head_light = self.par_mask_light.isChecked()
        p.anchor_small = self.par_anchor.isChecked()
        p.slim_head = self.par_slim_head.isChecked()
        p.p2_head = self.par_p2.isChecked()
        p.rcnn_optimizer = self.par_rcnn_opt.currentData()
        p.rcnn_learning_rate = self.par_rcnn_lr.value()
        p.rcnn_weight_decay = self.par_rcnn_wd.value()
        p.use_amp = self.par_amp.isChecked()
        p.tf32 = self.par_tf32.isChecked()
        p.save_epoch_weights = self.par_epoch_w.isChecked()
        p.patience = self.par_patience.value()
        p.train_input = self.par_train_input.value()
        p.tiling = self.par_tiling.isChecked()
        p.tile_size = self.par_tile_size.value()
        p.use_negatives = self.par_neg.isChecked()
        p.color_mode = "gray" if self.par_gray.isChecked() else "color"
        p.deterministic = self.par_det.isChecked()
        if p.train_input > 960:
            # T9-6：opt-in 大输入提示（显存显著上升）
            self.statusBar().showMessage(
                "⚠ 网络输入尺寸 >960：训练显存占用显著上升，"
                "请确认显卡预算（或降低该值 / 开启 AMP 与梯度检查点）", 15000)
        self._collect_augment_ops()
        p.save()
        toast(self, "参数已保存", "success")

    def _save_augment(self):
        self._collect_augment_ops()
        self.project.save()
        toast(self, "增强策略已保存", "success")

    def _on_model_changed(self):
        """切换模型时带入建议学习率。"""
        name = self.par_model.currentData()
        if name and name in DEFAULT_LR:
            self.par_lr.setValue(DEFAULT_LR[name])
        if hasattr(self, "augment_controls"):
            self._refresh_input_size_hint()   # P2 显存提示只对 YOLO 模型显示

    # ---- 训练输入尺寸（与增强 resize 统一） ----
    def _current_resize(self) -> tuple[int, int] | None:
        """当前增强策略里的 resize 宽高（未启用返回 None）。"""
        check, spins = self.augment_controls.get("resize", (None, {}))
        if check is None or not check.isChecked():
            return None
        return int(spins["width"].value()), int(spins["height"].value())

    def _refresh_input_size_hint(self):
        size = self._current_resize()
        if size:
            self.input_size_hint.setText(f"{size[0]} × {size[1]}　（随增强策略 resize 设置）")
        else:
            self.input_size_hint.setText("模型默认　（未启用 resize，分类将拉伸为方形）")
        name = getattr(getattr(self, "par_model", None), "currentData", lambda: None)() or ""
        p2 = getattr(self, "par_p2", None)
        if (isinstance(name, str) and name.startswith("yolo")
                and p2 is not None and p2.isChecked()):
            tip = self.input_size_hint.text()
            self.input_size_hint.setText(tip + "　· P2 小目标头：显存约 ×2，建议减小批量")
        if getattr(self, "q_size", None) is not None:
            self._sync_quick_size()   # Quick 档「输入尺寸」行同步

    @staticmethod
    def _divider():
        """浅色分隔线：用于参数表单的逻辑分区。"""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background: #eef2f8; border: none; max-height: 1px;")
        return line

    # ================= 数据集统计 =================
    def _random_dataset_image(self) -> Path | None:
        """从数据集目录随机抽一张图片。"""
        root = Path(self.project.dataset_dir)
        if not root.is_dir():
            return None
        paths = [p for p in root.rglob("*")
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                 and not p.name.startswith(".")]
        return random.choice(paths) if paths else None

    def _refresh_augment_preview(self, new_image: bool = False):
        """随机抽取数据集图片，按增强策略（等比 resize）自适应显示。"""
        if new_image or getattr(self, "_preview_source", None) is None:
            path = self._random_dataset_image()
            img = imread_unicode(path) if path else None
            if img is None:
                self._preview_source = None
                self.augment_preview.setPixmap(QPixmap())
                self.augment_preview.setText("数据集中没有可预览的图片")
                self.preview_info.setText("")
                return
            self._preview_source = img
            self._preview_name = path.name
        src = self._preview_source
        src_h, src_w = src.shape[:2]
        size = self._current_resize()
        if size:  # 与训练侧一致：拉伸到 宽×高（transforms.resize 默认口径）
            from ..engine.dataset.transforms import resize as _aug_resize

            out, _ = _aug_resize(src, [], size[0], size[1])
        else:
            out = src
        out_h, out_w = out.shape[:2]
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, out_w, out_h, 3 * out_w, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg)
        area = self.augment_preview.size()
        if area.width() > 40 and area.height() > 40:  # 自适应：等比缩放到预览区大小
            pm = pm.scaled(area, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.augment_preview.setPixmap(pm)
        tag = f"拉伸 → {out_w}×{out_h}" if size else "未启用 resize，显示原图"
        self.preview_info.setText(f"{getattr(self, '_preview_name', '')}　原始 {src_w}×{src_h}　{tag}")

    def _refresh_stats(self):
        """数据集统计（A-7①：移后台线程，主线程不再被万图目录卡住）。"""
        if getattr(self, "_stats_thread", None) is not None:
            return  # 统计中，忽略重复触发
        self.stats_hint.setText("统计中…")
        task = self.project.task_type
        dataset_dir = Path(self.project.dataset_dir)

        class _StatsThread(QThread):
            done = pyqtSignal(object)
            failed = pyqtSignal(str)

            def run(self):
                try:
                    self.done.emit(collect_stats(task, dataset_dir))
                except Exception as e:
                    self.failed.emit(f"{e}")

        # 无父对象（T2-5）：窗口可能在线程运行中被关闭——parented 运行线程
        # 会触发 Qt "Destroyed while thread is still running" 崩溃；
        # finished→auto_cleanup_thread 已接 deleteLater 兜底
        th = _StatsThread()
        th.done.connect(self._on_stats_done)
        th.failed.connect(self._on_stats_failed)
        auto_cleanup_thread(self, "_stats_thread", th)
        self._stats_thread = th
        th.start()

    def _on_stats_done(self, stats):
        n_ok = getattr(stats, "n_ok", 0)
        total = stats.n_images + n_ok
        ok_part = ""
        if n_ok and total:
            pct = n_ok / total
            ok_part = f"，OK 负样本 {n_ok} 张（占比 {pct:.0%}）"
        # O8（V0.7）：忽略区域计数可见（与数据体检提示同口径）
        n_ign = getattr(stats, "n_ignore", 0)
        if n_ign:
            ok_part += f"，__ignore__ 区域 {n_ign} 个"
        self.stats_hint.setText(
            f"共 {total} 张图片，已标注 {stats.n_labeled} 张，"
            f"类别 {len(stats.class_counts)} 个{ok_part}"
        )
        self.dist_canvas.update_stats(stats)
        # O12（V0.7）：划分环形图（n_ok 含负样本；分类任务 n_ok=0）
        self.split_canvas.update_split(
            stats.n_labeled, getattr(stats, "n_ok", 0),
            float(self.project.val_ratio or 0.0))

    def _on_stats_failed(self, msg):
        self.stats_hint.setText("")
        QMessageBox.warning(self, "统计失败", f"读取数据集失败：{msg}")

    # ================= 数据体检 / 预检 / 推荐 / 开发者选项 =================
    def _open_dev_options(self):
        DevOptionsDialog(self).exec_()

    def _run_health_report(self):
        """数据体检报告（异步扫描；仅提示/警告两档 + 去标注平台处理闭环）。"""
        worker = getattr(self, "_health_worker", None)
        if worker is not None and worker.isRunning():
            self.statusBar().showMessage("数据体检正在进行中，请稍候…", 3000)
            return
        self.btn_health.setEnabled(False)
        self.statusBar().showMessage("数据体检中…（大集合的重复图检测可能稍慢）")
        self._health_worker = _PreflightWorker(self.project, self)
        self._health_worker.done.connect(self._on_health_done)
        self._health_worker.failed.connect(self._on_health_failed)
        auto_cleanup_thread(self, "_health_worker", self._health_worker)  # A-3（T2-6）
        self._health_worker.start()

    def _on_health_done(self, result: dict):
        self.btn_health.setEnabled(True)
        self._health_worker = None
        rows = [ReportRow.from_issue(i) for i in result["issues"]]
        dlg = HealthReportDialog(
            self, "数据体检报告", rows, mode="report",
            summary=f"共 {result['n_images']} 张样本、{len(result['classes'])} 个类别。"
                    "结果仅分「提示 / 警告」两档，不会阻止训练；阈值可在 ⚙ 开发者选项调整。",
            on_annotate=self._goto_annotation)
        dlg.exec_()

    def _on_health_failed(self, message: str):
        self.btn_health.setEnabled(True)
        self._health_worker = None
        QMessageBox.warning(self, "数据体检失败",
                            f"读取数据集失败：{message}")

    def _goto_annotation(self):
        """体检闭环：拉起标注平台并定位到数据集目录（launcher 参数透传）。"""
        from .. import launcher

        dataset = self.project.dataset_dir or None
        launcher.launch_platform("annotation", dataset)
        self.statusBar().showMessage("已打开标注平台（定位到数据集目录）", 5000)

    def _apply_recommendations(self):
        """智能推荐配置：保守规则，应用后在状态栏说明理由（toast 式）。"""
        self._save_params()
        facts = dataset_facts(self._dataset_samples_for_guard())
        recs = recommend_settings(self.project, facts, _cuda_available())
        if not recs:
            self.statusBar().showMessage("当前配置已符合推荐，未做修改", 5000)
            return
        for key, value, _reason in recs:
            self._apply_recommend_key(key, value)
        self._save_params()
        self.statusBar().showMessage(
            "已应用推荐（" + "；".join(r for _, _, r in recs) + "）", 15000)

    def _apply_recommend_key(self, key, value):
        if key == "freeze_backbone":
            self.par_freeze.setChecked(True)
        elif key == "industrial_opt":
            self.par_industrial.setChecked(True)
        elif key == "rpn_reduced":
            self.par_rpn.setChecked(True)
        elif key == "use_amp":
            self.par_amp.setChecked(bool(value))
        elif key == "epochs":  # O7（V0.7）：轮次智能推荐
            self.par_epochs.setValue(int(value))
        elif key == "augment_probs":  # 已启用算子的概率提高到推荐值
            for op, prob in self.augment_prob_spins.items():
                if self.augment_controls[op][0].isChecked():
                    prob.setValue(max(prob.value(), float(value)))

    # ================= Quick/专家双轨（v0.5 UI 升级期3，D4）=================
    def _on_mode_quick(self):
        self._set_param_mode(True)

    def _on_mode_expert(self):
        self._set_param_mode(False)

    def _set_param_mode(self, quick: bool, save: bool = True):
        """快速档 ⇄ 专家模式切换；专家模式即原完整折叠表单（零迁移成本）。"""
        self.quick_panel.setVisible(quick)
        self.expert_host.setVisible(not quick)
        self.btn_expand_all.setVisible(not quick)
        self.btn_mode_quick.setChecked(quick)
        self.btn_mode_expert.setChecked(not quick)
        if quick:
            self._quick_from_experts()
        else:
            self._experts_from_quick()
        if save:
            prefs = load_prefs()
            prefs["train_param_mode"] = "quick" if quick else "expert"
            save_prefs(prefs)

    def _quick_row(self, label: str, widget, unit: str = "",
                   badge_key: str = "", tip: str = "") -> QWidget:
        """Quick 档行化参数行：label 上（可带帮助图标）、控件下 + 单位。"""
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        top = QHBoxLayout()
        top.setSpacing(6)
        lbl = QLabel(label)
        lbl.setStyleSheet(
            "font-weight: bold; border:none; background:transparent;")
        top.addWidget(lbl)
        if tip:
            info = QLabel("ⓘ")
            info.setToolTip(tip)
            bind_style(info, lambda: f"color:{_t3()}; border:none; background:transparent;")
            top.addWidget(info)
        top.addStretch(1)
        if badge_key:
            badge = Badge("已推荐", "primary")
            badge.hide()
            self._rec_badges[badge_key] = badge
            top.addWidget(badge)
        v.addLayout(top)
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(widget, stretch=1)
        if unit:
            u = QLabel(unit)
            bind_style(u, lambda: f"color:{_t2()}; border:none; background:transparent;")
            row.addWidget(u)
        v.addLayout(row)
        return host

    def _make_preset_card(self, key: str, name: str, desc: str,
                          values: dict, color: str) -> QFrame:
        """预设档位卡：点击一键套用并联动智能推荐。"""
        card = QFrame()
        card.setObjectName("card")
        card.setCursor(Qt.PointingHandCursor)
        card.setFixedSize(216, 92)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 10, 12, 8)
        lay.setSpacing(3)
        t = QLabel(name)
        t.setStyleSheet(f"font-size:16px; font-weight:bold; color:{color};"
                        "border:none; background:transparent;")
        d = QLabel(desc)
        d.setWordWrap(True)
        bind_style(d, lambda: f"font-size:15px; color:{_t2()};"
                              "border:none; background:transparent;")
        lay.addWidget(t)
        lay.addWidget(d, stretch=1)
        self._preset_values[key] = values
        card.mouseReleaseEvent = self._make_preset_click(key)
        return card

    def _make_preset_click(self, key: str):
        def _click(event):
            if event.button() == Qt.LeftButton:
                self._apply_preset(key)
        return _click

    def _fill_quick_controls(self):
        """把专家表单的模型/设备条目同步到 Quick 控件并回填当前值。"""
        self.q_model.blockSignals(True)
        self.q_model.clear()
        for i in range(self.par_model.count()):
            self.q_model.addItem(self.par_model.itemText(i),
                                 self.par_model.itemData(i))
        self.q_model.blockSignals(False)
        self.q_device.blockSignals(True)
        self.q_device.clear()
        for i in range(self.par_device.count()):
            self.q_device.addItem(self.par_device.itemText(i),
                                  self.par_device.itemData(i))
        self.q_device.blockSignals(False)
        self._quick_from_experts()

    def _quick_from_experts(self):
        """par_* → q_*（专家表单为唯一数据源）。"""
        self.q_model.blockSignals(True)
        idx = self.q_model.findData(self.par_model.currentData())
        if idx >= 0:
            self.q_model.setCurrentIndex(idx)
        self.q_model.blockSignals(False)
        self.q_epochs.setValue(self.par_epochs.value())
        self.q_batch.setValue(self.par_batch.value())
        self.q_lr.setValue(self.par_lr.value())
        dev = self.q_device.findData(self.par_device.currentData())
        if dev >= 0:
            self.q_device.setCurrentIndex(dev)
        self._sync_quick_size()

    def _experts_from_quick(self):
        """q_* → par_*（Quick 编辑后回写专家表单；lr 不被模型建议覆盖）。"""
        self.par_model.blockSignals(True)
        idx = self.par_model.findData(self.q_model.currentData())
        if idx >= 0:
            self.par_model.setCurrentIndex(idx)
        self.par_model.blockSignals(False)
        self.par_epochs.setValue(self.q_epochs.value())
        self.par_batch.setValue(self.q_batch.value())
        self.par_lr.setValue(self.q_lr.value())
        dev = self.par_device.findData(self.q_device.currentData())
        if dev >= 0:
            self.par_device.setCurrentIndex(dev)

    def _sync_quick_size(self):
        text = self.input_size_hint.text().strip()
        self.q_size.setText(text if text else "模型默认尺寸")

    def _apply_param_kv(self, key: str, value):
        """预设/推荐的参数键 → 专家表单控件（单一数据源仍为 par_*）。"""
        if key == "epochs":
            self.par_epochs.setValue(int(value))
        elif key == "batch":
            self.par_batch.setValue(int(value))
        elif key == "lr":
            self.par_lr.setValue(float(value))
        elif key == "freeze_backbone":
            self.par_freeze.setChecked(bool(value))
        elif key == "industrial_opt":
            self.par_industrial.setChecked(bool(value))
        elif key == "use_amp":
            self.par_amp.setChecked(bool(value))
        elif key == "tf32":
            self.par_tf32.setChecked(bool(value))
        elif key == "patience":
            self.par_patience.setValue(int(value))

    def _apply_preset(self, key: str):
        values = self._preset_values.get(key)
        if not values:
            return
        name = next((_n for _k, _n, _d, _v, _c in _PRESET_DEFS if _k == key),
                    key)
        for k, v in values.items():
            self._apply_param_kv(k, v)
        rec_n = 0
        try:   # 联动智能推荐（数据集事实驱动；失败不影响套用本身）
            facts = dataset_facts(self._dataset_samples_for_guard())
            recs = recommend_settings(self.project, facts,
                                      _cuda_available())
            for rkey, rvalue, _reason in recs:
                self._apply_recommend_key(rkey, rvalue)
                rec_n += 1
        except Exception:
            pass
        self._quick_from_experts()   # 预设已写 par_* → 同步到 Quick 控件
        quick_keys = {"epochs", "batch", "lr"} & set(values)
        self._mark_recommended(sorted(quick_keys))
        self._save_params()
        msg = f"已套用预设「{name}」"
        if rec_n:
            msg += f"，并联动智能推荐 {rec_n} 项"
        toast(self, msg, "success")

    def _reset_quick_defaults(self):
        d = getattr(self, "_param_defaults", None)
        if not d:
            return
        self.par_model.blockSignals(True)
        idx = self.par_model.findData(d.get("model"))
        if idx >= 0:
            self.par_model.setCurrentIndex(idx)
        self.par_model.blockSignals(False)
        self.par_epochs.setValue(d.get("epochs", self.par_epochs.value()))
        self.par_batch.setValue(d.get("batch", self.par_batch.value()))
        self.par_lr.setValue(d.get("lr", self.par_lr.value()))
        dev = self.par_device.findData(d.get("device"))
        if dev >= 0:
            self.par_device.setCurrentIndex(dev)
        self._refresh_input_size_hint()
        self._quick_from_experts()   # 默认值已写 par_* → 同步到 Quick 控件
        self._clear_recommended()
        self._save_params()
        toast(self, "已恢复默认参数", "info")

    def _mark_recommended(self, keys):
        for k in keys:
            b = self._rec_badges.get(k)
            if b is not None:
                b.show()

    def _clear_recommended(self):
        for b in self._rec_badges.values():
            b.hide()

    def _toggle_all_sections(self):
        """全部展开 ⇄ 全部折叠（用户回执 D2：需要再次点击全部折叠）。
        整批挂起重绘，避免连续三个分区逐个回流造成卡顿。"""
        sections = (self.sec_basic, self.sec_data, self.sec_adv)
        target = any(s.content.isHidden() for s in sections)
        win = self.window()
        win.setUpdatesEnabled(False)
        try:
            for sec in sections:
                sec.set_expanded(target)
        finally:
            win.setUpdatesEnabled(True)
        self.btn_expand_all.setText("全部折叠" if target else "全部展开")

    # ================= 训练 =================
    def _check_rotate_guard(self):
        """rotate90 守卫（用户规则）：数据集长宽比不一致时，旋转算子必须配合
        「随机裁剪（对齐目标宽高比）」且启用概率 100%，否则无法开始训练。

        返回 None=通过；("enable", 提示文本)=需启用裁剪兜底；
        ("prob", 提示文本)=裁剪已启用但概率不足 100%。
        """
        ops = self.project.augment_ops or {}
        if "rotate90" not in ops:
            return None
        aspects = set()
        for s in self._dataset_samples_for_guard():
            if s.width > 0 and s.height > 0:
                aspects.add(round(s.width / s.height, 3))
        return rotate_guard_decision(ops, self.project.augment_probs, aspects)

    def _dataset_samples_for_guard(self):
        """守卫用的样本元信息（宽高来自标注 JSON / 目录扫描，不解码图片）。"""
        from ..engine.dataset.base import TaskType
        from ..engine.dataset.reader import scan_classification_dataset, \
            scan_labelme_dataset

        d = Path(self.project.dataset_dir)
        if not d.is_dir():
            return []
        try:
            if TaskType(self.project.task) is TaskType.CLASSIFICATION:
                samples, _ = scan_classification_dataset(d)
                return samples
            samples, _, _ = scan_labelme_dataset(d)
            return samples
        except Exception:
            return []  # 扫描失败交由训练线程的既有校验兜底

    def _resolve_rotate_guard(self, guard: tuple) -> bool:
        """rotate90 守卫弹窗：True=已按建议写回配置可继续训练；False=用户取消。

        一键兜底：启用随机裁剪并按 resize 目标宽高比设定 ratio、启用概率
        置 100%，写回项目后同步界面控件。
        """
        kind, text = guard
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("无法开始训练")
        box.setText(text)
        btn_fix = box.addButton(
            "启用随机裁剪（100%）并开始" if kind == "enable" else "设为 100% 并开始",
            QMessageBox.AcceptRole)
        box.addButton("取消", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is not btn_fix:
            return False
        check, spins = self.augment_controls["random_crop"]
        check.setChecked(True)
        size = self._current_resize()
        ratio = round(size[0] / size[1], 2) if size and size[1] else 1.0
        spins["ratio"].setValue(ratio)
        self.augment_prob_spins["random_crop"].setValue(1.0)
        self._collect_augment_ops()
        self.project.save()
        self.statusBar().showMessage(
            f"已启用随机裁剪兜底（宽高比 {ratio:g}，概率 100%），继续训练", 5000)
        return True

    def _start_training(self):
        self._save_params()
        guard = self._check_rotate_guard()
        if guard and not self._resolve_rotate_guard(guard):
            return
        # 训练前预检（批次十一）：异步执行数据体检 + 参数检查。
        # 用户规则：尽量少弹窗——仅存在「警告」级条目时弹一个对话框；
        # 只有「提示」时状态栏一句话直接开始；预检失败不阻塞训练。
        if not load_options().preflight_enabled:
            self._begin_training()
            return
        self.btn_start.setEnabled(False)
        self.statusBar().showMessage("训练前预检中（数据体检 + 参数检查）…")
        self._preflight_worker = _PreflightWorker(self.project, self)
        self._preflight_worker.done.connect(self._on_preflight_done)
        self._preflight_worker.failed.connect(self._on_preflight_failed)
        auto_cleanup_thread(self, "_preflight_worker", self._preflight_worker)  # A-3（T2-6）
        self._preflight_worker.start()

    def _apply_preflight_value(self, key, value):
        """预检「一键应用」：把建议值写回训练表单（保存发生在开始训练时）。"""
        if key == "patience":
            self.par_patience.setValue(int(value))
        elif key == "learning_rate":
            self.par_lr.setValue(float(value))
        elif key == "batch_size":
            self.par_batch.setValue(int(value))
        elif key == "device":
            idx = self.par_device.findData(value)
            if idx >= 0:
                self.par_device.setCurrentIndex(idx)
        elif key == "resize":
            check, spins = self.augment_controls["resize"]
            check.setChecked(True)
            spins["width"].setValue(int(value[0]))
            spins["height"].setValue(int(value[1]))

    def _on_preflight_done(self, result: dict):
        self.btn_start.setEnabled(True)
        self._preflight_worker = None
        rows = [ReportRow.from_issue(i) for i in result["issues"]]
        rows += [ReportRow.from_finding(f) for f in result["findings"]]
        has_warn = any(r.level == LEVEL_WARN for r in rows)
        # 弹窗条件：存在警告级，或（用户在开发者选项开启了"提示级也弹窗"）。
        # 两者都不满足（完全干净）时状态栏一句话放行，不打扰。
        opt = load_options()
        if not has_warn and not (opt.preflight_info_popup and rows):
            self.statusBar().showMessage("预检通过，开始训练…", 5000)
            self._begin_training()
            return
        dlg = HealthReportDialog(
            self, "训练前预检", rows, mode="preflight",
            summary=f"数据集共 {result['n_images']} 张样本、{len(result['classes'])} 个类别。"
                    "以下为「提示 / 警告」两档建议，不会强制修改；折叠区可查看已应用的参数。",
            on_apply=self._apply_preflight_value)
        if dlg.exec_() != QDialog.Accepted:
            self.statusBar().showMessage("已取消训练", 3000)
            return
        self._begin_training()

    def _on_preflight_failed(self, message: str):
        self.btn_start.setEnabled(True)
        self._preflight_worker = None
        box = QMessageBox(QMessageBox.Warning, "预检失败",
                          f"训练前预检执行失败：{message}\n\n仍要直接开始训练吗？",
                          QMessageBox.Yes | QMessageBox.No, self)
        if box.exec_() == QMessageBox.Yes:
            self._begin_training()
        else:
            self.statusBar().showMessage("已取消训练", 3000)

    def _begin_training(self):
        """预检（或其跳过）之后真正启动训练线程。"""
        self._save_params()
        p = self.project
        # 数据集校验移入训练线程（大集合扫描不阻塞界面，失败弹窗提示）
        if p.model_dir:
            Path(p.model_dir).mkdir(parents=True, exist_ok=True)
        # 模型引用（pruned:/run: 前缀）含冒号等非法字符，目录名需安全化
        run_dir = Path(p.model_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}_{_safe_run_tag(p.model_name)}"

        self._live_history = []
        self._latest_progress = None
        self._history = self._live_history
        self._live_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._live_start_dt = datetime.now()
        self._selected_run_dir = run_dir
        self.loss_plot.clear()
        self.metric_plot.clear()
        self.cm_canvas.ax.clear()
        self.cm_canvas.draw()
        self.final_metrics.setText("")
        self._update_per_class_table(None)  # O1：新训练开始前清空逐类表
        self._build_echo_form(run_dir=run_dir)
        self.train_status.setText("训练中…")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(not p.model_name.startswith("yolo"))  # YOLO 训练由引擎控制，不支持中途停止
        self.btn_open_dir.setEnabled(True)

        self._run_dir = run_dir
        self.thread = TrainThread(p, run_dir)
        self.thread.epoch_changed.connect(self._on_epoch)
        self.thread.batch_progress.connect(self._on_batch_progress)
        self.thread.train_finished.connect(self._on_train_finished)
        auto_cleanup_thread(self, "thread", self.thread)  # A-3（T2-6）
        self.thread.start()
        self._switch_detail(1)  # 跳转到训练可视化页（刷新历史列表并选中本次训练）
        self._live_timer.start()
        self.statusBar().showMessage(f"训练输出：{run_dir}")

    def _build_echo_form(self, config: dict | None = None, run_dir=None):
        """把训练设置展示在训练可视化页底部；config 为历史记录里的参数快照。"""
        p = self.project
        out_dir = str(run_dir or self._run_dir or "")
        if config is None:  # 实时训练：读当前项目参数
            aug = "、".join(self._collect_augment_ops()) or "未启用"
            epoch_w = (f"每轮保存（{Path(p.model_dir) / 'epoch_weights'}）"
                       if p.save_epoch_weights else "关闭")
            rows = [
                ("项目名称", p.name), ("任务类型", p.task_type.label),
                ("训练模型", p.model_name), ("工业优化", "启用" if p.industrial_opt else "关闭"),
                ("冻结骨干", "是" if p.freeze_backbone else "否"),
                ("RPN 候选削减", "是" if p.rpn_reduced else "否"),
                ("轻量掩码头", "是" if p.mask_head_light else "否"),
        ("slim 头", "是" if getattr(p, "slim_head", False) else "否"),
        ("小目标锚框", "是" if getattr(p, "anchor_small", False) else "否"),
        ("P2 检测头", "是" if getattr(p, "p2_head", False) else "否"),
                ("AMP 混合精度", "开启" if p.use_amp else "关闭"),
                ("GPU 提速(TF32)", "开启" if p.tf32 else "关闭"),
                ("轮次权重", epoch_w),
                ("早停耐心", f"{p.patience} 轮" if p.patience else "关闭"),
                ("数据集路径", p.dataset_dir),
                ("模型储存路径", p.model_dir), ("输出目录", out_dir),
                ("训练集比例", f"{getattr(p, 'train_ratio', 0):.0%}"),
                ("验证集比例", f"{p.val_ratio:.0%}"), ("每轮重复学习", f"{p.multiply_factor} 遍"),
                ("负样本训练", "启用" if getattr(p, "use_negatives", True) else "关闭"),
                ("颜色模式", "灰度" if getattr(p, "color_mode", "color") == "gray" else "彩色"),
                ("确定性训练", "开启" if getattr(p, "deterministic", False) else "关闭"),
                ("增强策略", aug), ("批量大小", str(p.batch_size)),
                ("训练轮次", str(p.epochs)), ("学习率", f"{p.learning_rate:g}"),
                ("权重衰减", f"{p.weight_decay:g}"), ("评估间隔", f"每 {p.eval_interval} 轮"),
                ("训练设备", "自动" if p.device == "auto" else p.device.upper()),
            ]
        else:  # 历史记录：读训练时的参数快照
            aug = "、".join((config.get("augment_ops") or {}).keys()) or "未启用"
            dev = config.get("device", "auto")
            epoch_w = (f"每轮保存（{Path(p.model_dir) / 'epoch_weights'}）"
                       if config.get("save_epoch_weights", True) else "关闭")
            rows = [
                ("项目名称", p.name), ("任务类型", p.task_type.label),
                ("训练模型", config.get("model_name") or p.model_name),
                ("工业优化", "启用" if config.get("industrial_opt") else "关闭"),
                ("冻结骨干", "是" if config.get("freeze_backbone") else "否"),
                ("RPN 候选削减", "是" if config.get("rpn_reduced") else "否"),
                ("轻量掩码头", "是" if config.get("mask_head_light") else "否"),
        ("slim 头", "是" if config.get("slim_head") else "否"),
        ("小目标锚框", "是" if config.get("anchor_small") else "否"),
        ("P2 检测头", "是" if config.get("p2_head") else "否"),
                ("AMP 混合精度", "开启" if config.get("use_amp", True) else "关闭"),
                ("GPU 提速(TF32)", "开启" if config.get("tf32", True) else "关闭"),
                ("轮次权重", epoch_w),
                ("早停耐心", f"{config.get('patience', 0)} 轮" if config.get("patience") else "关闭"),
                ("数据集路径", p.dataset_dir),
                ("模型储存路径", p.model_dir), ("输出目录", out_dir),
                ("训练集比例", f"{config.get('train_ratio', 0) or round(1 - config.get('val_ratio', 0.2), 2):.0%}"),
                ("验证集比例", f"{config.get('val_ratio', 0):.0%}"),
                ("每轮重复学习", f"{config.get('multiply_factor', 1)} 遍"),
                ("负样本训练", "启用" if config.get("use_negatives", True) else "关闭"),
                ("颜色模式", "灰度" if config.get("color_mode", "color") == "gray" else "彩色"),
                ("确定性训练", "开启" if config.get("deterministic", False) else "关闭"),
                ("增强策略", aug),
                ("批量大小", str(config.get("batch_size", "—"))),
                ("训练轮次", str(config.get("epochs", "—"))),
                ("学习率", f"{config.get('learning_rate', 0):g}"),
                ("权重衰减", f"{config.get('weight_decay', 0):g}"),
                ("评估间隔", f"每 {config.get('eval_interval', 1)} 轮"),
                ("训练设备", "自动" if dev == "auto" else str(dev).upper()),
            ]
        while self.echo_form.rowCount():
            self.echo_form.removeRow(0)
        for k, v in rows:
            label = QLabel(str(v))
            label.setWordWrap(True)
            self.echo_form.addRow(k, label)

    # ================= 训练历史 =================
    def _make_run_item(self, rec: RunRecord) -> QListWidgetItem:
        """历史列表条目：状态点图标 + 时间 + 状态（按状态着色）+ 相对时间。"""
        model = rec.model_name or (self.project.model_name if rec.status == "running" else "")
        line2 = f"{rec.status_text}　{model}".strip()
        if rec.start_time:
            when = relative_time(rec.start_time)
            if when != rec.start_time:   # 解析失败时原样返回，不重复展示
                line2 = f"{rec.status_text} · {when}　{model}".strip()
        item = QListWidgetItem(QIcon(_status_dot_icon(
            STATUS_COLOR.get(rec.status, "#64748b"))),
            f"{rec.start_time or rec.run_dir.name}\n{line2}")
        item.setData(Qt.UserRole, str(rec.run_dir))
        item.setData(Qt.UserRole + 1, rec)
        item.setForeground(QBrush(QColor(STATUS_COLOR.get(rec.status, "#64748b"))))
        return item

    def _reload_history(self, select_dir: Path | None = None):
        """重扫模型储存目录刷新左侧列表；select_dir 给定时优先选中该记录。"""
        target = select_dir or self._selected_run_dir
        self.run_list.blockSignals(True)
        self.run_list.clear()
        self._prev_selected = []  # 列表已清空，缓存的多选状态随之失效
        n = 0
        if self.thread is not None and self._run_dir is not None:
            done = self._live_history[-1]["epoch"] if self._live_history else 0
            live = RunRecord(run_dir=self._run_dir, status="running",
                             start_time=self._live_start,
                             epochs_done=done, epochs_planned=self.project.epochs)
            self.run_list.addItem(self._make_run_item(live))
            n += 1
        for rec in scan_runs(self.project.model_dir):
            if self.thread is not None and self._run_dir is not None \
                    and rec.run_dir == self._run_dir:
                continue  # 仅训练进行中时跳过（进行中的目录已在顶部展示）
            self.run_list.addItem(self._make_run_item(rec))
            n += 1
        self.run_hint.setText("")
        empty = n == 0
        self.run_list.setVisible(not empty)
        self.run_empty.setVisible(empty)
        selected = None
        if target is not None:
            found = self.run_list.findItems(str(target), Qt.MatchExactly)
            if found:
                selected = found[0]
        if selected is None and self.run_list.count():
            selected = self.run_list.item(0)
        if selected is not None:
            self.run_list.setCurrentItem(selected)
        self.run_list.blockSignals(False)
        if selected is not None:
            self._on_run_selected(selected, None)
        else:
            self.train_status.setText("尚未开始训练")
            self._history = []
            self._reset_progress()
            self._update_plots()
            self.btn_delete_run.setEnabled(False)

    def _on_run_selected(self, cur, _prev=None):
        if cur is None:
            return
        rec: RunRecord = cur.data(Qt.UserRole + 1)
        self._selected_run_dir = rec.run_dir
        self.btn_open_dir.setEnabled(True)
        self._show_run(rec)

    def _show_run(self, rec: RunRecord):
        """渲染所选记录：实时训练显示进行中的数据，其余读 metrics.json。"""
        m = rec.metrics or {}
        live = rec.status == "running" and self.thread is not None
        if live:
            self._history = self._live_history
            self._build_echo_form(run_dir=rec.run_dir)
            self._update_overall_progress()
            self.run_start.setText(self._live_start or "—")
            self.run_end.setText("—")
            self._tick_duration()
        else:
            self._history = list(m.get("history") or [])
            self._build_echo_form(config=m.get("config"), run_dir=rec.run_dir)
            self._update_progress(rec.epochs_done, rec.epochs_planned)
            self.run_start.setText(rec.start_time or "—")
            self.run_end.setText(rec.end_time or "—")
            self.run_duration.setText(format_duration(rec.duration_sec))
        self._update_plots()
        if live:
            self.train_status.setText(
                f"训练中… 第 {self._live_history[-1]['epoch']} 轮" if self._live_history else "训练中…")
            self.final_metrics.setText("")
            self.cm_canvas.ax.clear()
            self.cm_canvas.draw()
            self._update_per_class_table(None)  # O1：进行中无逐类指标
            self.btn_delete_run.setEnabled(False)  # 进行中的记录不可删除
        else:
            self.train_status.setText(f"历史记录：{rec.status_text}")
            err = m.get("error")
            self.final_metrics.setText(
                f"失败原因：{err}" if err else self._summarize_metrics(m))
            matrix = m.get("confusion_matrix")
            classes = m.get("classes") or []
            self.cm_canvas.ax.clear()
            if matrix:
                self.cm_canvas.update_matrix(matrix, classes)
            else:
                self.cm_canvas.draw()
            self._update_per_class_table(m)  # O1：历史详情逐类表
            self.btn_delete_run.setEnabled(True)

    def _update_progress(self, done: int, planned: int):
        pct = int(round(100 * min(done, planned) / planned)) if planned else 0
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"%p%　第 {done} / {planned} 轮" if planned else "%p%")

    def _reset_progress(self):
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.run_start.setText("—")
        self.run_end.setText("—")
        self.run_duration.setText("—")

    def _tick_duration(self):
        if self._live_start_dt is None:
            return
        self.run_duration.setText(format_duration(
            (datetime.now() - self._live_start_dt).total_seconds()))

    def _tick_live(self):
        """训练中每秒刷新：训练时长 + 总体进度（含轮内批次比例）+ GPU 显存。"""
        if self.thread is None:
            self._live_timer.stop()
            self.gpu_mem.setText("")
            return
        if self._selected_run_dir == self._run_dir:
            self._tick_duration()
            self._update_overall_progress()
            self._update_gpu_mem()

    # ================= 曲线绘制 =================
    def _loss_series(self):
        """损失曲线数据：训练损失与验证损失（逐轮）。"""
        h = self._history
        xs = [x["epoch"] for x in h]
        out = []
        for k, label in (("train_loss", "训练损失"), ("val_loss", "验证损失")):
            if any(k in x for x in h):
                out.append((label, xs, [x.get(k) for x in h]))
        return out

    def _metric_series(self):
        """评估指标曲线：固定两条 —— mAP 与 mAR。

        R-CNN 族逐轮产出宽松口径 map_loose/mar_loose（IoU {0.10,0.25,0.50}
        三档平均），直接绘制；YOLO 族无该口径，用 map50（AP@50）与 recall
        （IoU@0.5 召回）充当同位曲线。map50_95 / mask_* 等其余键不再进曲线
        （数值仍保留在 metrics.json 与最终指标摘要中）。分类只有 val_top1。
        """
        xs = [h["epoch"] for h in self._history]
        h = self._history
        if any("val_top1" in x for x in h):
            pairs = (("val_top1", "Top-1"),)
        elif any("map_loose" in x for x in h):
            pairs = (("map_loose", "mAP"), ("mar_loose", "mAR"))
        elif any("map50" in x for x in h):
            pairs = (("map50", "mAP"), ("recall", "mAR"))
        else:
            return []
        return [(label, xs, [x.get(k) for x in h])
                for k, label in pairs if any(k in x for x in h)]

    def _on_batch_progress(self, epoch: int, batch_index: int, batch_total: int):
        """轮内进度上报：只缓存，由 1s 定时器统一刷新进度条（避免频繁重绘）。"""
        self._latest_progress = (epoch, batch_index, batch_total)

    def _update_overall_progress(self):
        """总体进度 =（已完成轮次 + 当前轮内批次比例）/ 总轮次，比纯轮次计数准确。"""
        planned = self.project.epochs
        if not planned:
            return
        done = self._live_history[-1]["epoch"] if self._live_history else 0
        frac = done / planned
        if self.thread is not None and self._latest_progress:
            e, bi, nb = self._latest_progress
            frac = max(frac, ((e - 1) + (bi / nb if nb else 1.0)) / planned)
            done = max(done, e)
        self.progress_bar.setValue(int(round(100 * min(frac, 1.0))))
        eta = ""
        if 0.02 <= frac < 1.0 and self._live_start_dt is not None:
            elapsed = (datetime.now() - self._live_start_dt).total_seconds()
            remain_s = elapsed / frac * (1.0 - frac)
            eta = f"　剩余约 {format_duration(remain_s)}"
        self.progress_bar.setFormat(f"%p%　第 {done} / {planned} 轮{eta}")

    def _update_gpu_mem(self):
        """训练页顶部实时显示 GPU 显存占用。"""
        try:
            import torch

            if not torch.cuda.is_available():
                self.gpu_mem.setText("")
                return
            free_b, total_b = torch.cuda.mem_get_info()
            used = (total_b - free_b) / 2**30
            self.gpu_mem.setText(f"GPU 显存 {used:.1f} / {total_b / 2**30:.1f} GB")
        except Exception:
            self.gpu_mem.setText("")

    def _on_epoch(self, epoch: int, logs: dict):
        self._live_history.append({"epoch": epoch, **logs})
        if self._selected_run_dir == self._run_dir:  # 正在查看本次训练 → 实时刷新
            self._history = self._live_history
            self._update_plots()
            self._update_overall_progress()
            self._tick_duration()
            self.train_status.setText(f"训练中… 第 {epoch} 轮")

    def _update_plots(self):
        xs = [h["epoch"] for h in self._history]
        width = 3  # 曲线加粗，长时观看更醒目
        # 损失曲线：训练损失 + 验证损失；每个轮次叠加空心点标注数据位置
        self.loss_plot.clear()
        for i, (key, sx, ys) in enumerate(self._loss_series()):
            color = _CURVE_COLORS[i % len(_CURVE_COLORS)]
            self.loss_plot.plot(
                sx, ys, pen=pg.mkPen(color, width=width), name=key,
                symbol="o", symbolSize=7,
                symbolPen=pg.mkPen(color, width=1.5),  # symbolBrush 默认空 → 空心点
            )
        # 评估指标固定两条：mAP 与 mAR（YOLO 用 AP@50/召回充当）
        self.metric_plot.clear()
        for i, (key, _, ys) in enumerate(self._metric_series()):
            color = _CURVE_COLORS[i % len(_CURVE_COLORS)]
            self.metric_plot.plot(
                xs, ys, pen=pg.mkPen(color, width=width), name=key,
                symbol="o", symbolSize=7, symbolPen=pg.mkPen(color, width=1.5),
            )
        # X 轴随轮次自适应（整数范围）；损失 Y 轴固定从 0 起、上限随实际损失
        # 自适应（pyqtgraph 默认自适应会把下限抬到最小值附近，损失零点参照消失）；
        # 指标图 Y 轴锁回 0~1
        x_max = max(2.5, len(self._history) + 0.5)
        loss_vals = [v for _, _, ys in self._loss_series() for v in ys
                     if isinstance(v, (int, float)) and math.isfinite(v)]
        loss_top = max(loss_vals) * 1.08 if loss_vals else 1.0
        self.loss_plot.setXRange(0.5, x_max, padding=0)
        self.loss_plot.setYRange(0, loss_top, padding=0)
        self.metric_plot.setXRange(0.5, x_max, padding=0)
        self.metric_plot.setYRange(0, 1, padding=0)
        self._loss_hover.install()   # clear() 会移除图元，重绘后重挂悬停提示
        self._metric_hover.install()

    def _on_train_finished(self, ok: bool, message: str, metrics: dict):
        self._live_timer.stop()
        self.thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if ok:
            self.train_status.setText("训练完成 ✔")
            self.final_metrics.setText(self._summarize_metrics(metrics))
            matrix = metrics.get("confusion_matrix")
            classes = metrics.get("classes") or []
            if matrix:
                self.cm_canvas.update_matrix(matrix, classes)
            self._update_per_class_table(metrics)  # O1：训练完成即出逐类表
        else:
            self.train_status.setText("训练失败 ✘")
            self.final_metrics.setText(f"失败原因：{message}")
            self._update_per_class_table(None)
            QMessageBox.critical(self, "训练失败", message)
        # 重扫历史（本次训练目录已有产出），保持选中该记录展示最终详情
        self._reload_history(select_dir=self._run_dir)

    def _update_per_class_table(self, metrics: dict | None):
        """O1（V0.7）：逐类精确率/召回率表刷新。

        metrics 为 None 或无混淆矩阵（分类任务/失败/训练中）时清空——
        表只对检测/分割的历史详情与训练完成态有意义。
        """
        from ..engine.metrics.per_class import per_class_rows

        t = self.per_class_table
        t.setRowCount(0)
        if not metrics:
            return
        cm = metrics.get("confusion_matrix")
        classes = metrics.get("classes") or []
        if not cm or not classes:
            return
        final = metrics.get("final") or {}
        try:
            rows = per_class_rows(cm, classes, final.get("per_class_ap50"))
        except ValueError:  # 尺寸不符（异常档位）不阻塞详情展示
            return

        def _s(v):
            if v is None:
                return "-"
            return f"{v:.3f}" if isinstance(v, float) else str(v)

        t.setRowCount(len(rows))
        for r, row in enumerate(rows):
            vals = [row["label"], _s(row["gt"]), _s(row["pred"]), _s(row["tp"]),
                    _s(row["fp"]), _s(row["fn"]), _s(row["precision"]),
                    _s(row["recall"]), _s(row["ap50"])]
            for c, text in enumerate(vals):
                t.setItem(r, c, QTableWidgetItem(text))

    @staticmethod
    def _summarize_metrics(metrics: dict) -> str:
        parts = []
        if metrics.get("final_val_top1") is not None:
            parts.append(f"最终验证 Top-1：{metrics['final_val_top1']:.1%}")
        if metrics.get("best_val_top1") is not None:
            parts.append(f"最优验证 Top-1：{metrics['best_val_top1']:.1%}")
        final = metrics.get("final") or {}
        for key, label in (("map_loose", "mAP"), ("map50", "mAP@50"),
                           ("map50_95", "mAP@50-95"), ("mar_loose", "mAR"),
                           ("precision", "精确率"), ("recall", "召回率"),
                           ("auroc", "AUROC"), ("ap", "AP"),
                           ("ok_fp_rate", "OK 误报率"), ("exact_match", "精确匹配"),
                           ("cer", "CER"), ("char_acc", "字符准确率")):
            if key in final and final[key] is not None:
                parts.append(f"{label}：{final[key]:.3f}")
        # O9（V0.7）：OK 负样本贡献常驻可见（能力已在，让用户看见）
        neg_train = metrics.get("neg_train_images")
        if neg_train:
            neg_val = final.get("neg_val_images")
            fp = final.get("neg_fp_per_img")
            ok_txt = f"OK 负样本：训练 {neg_train} 张"
            if neg_val is not None:
                ok_txt += f" / 验证 {neg_val} 张"
            if fp is not None:
                ok_txt += f"，误检 {fp:.2f} 框/张"
            parts.append(ok_txt)
        # O3（V0.7）：Loss 曲线自动判读（欠拟合/过拟合/NaN 一句话）
        from .loss_diagnosis import diagnose

        d = diagnose(metrics.get("history") or [])
        if d.get("text"):
            parts.append(d["text"])
        return "　|　".join(parts)

    def _stop_training(self):
        if self.thread is not None:
            self.thread.stop()
            self.train_status.setText("正在停止（当前轮结束后保存最优权重）…")
            self.btn_stop.setEnabled(False)

    def _open_output_dir(self):
        run = self._selected_run_dir or self._run_dir
        if run is not None and Path(run).exists():
            os.startfile(str(run))

    def _current_run_record(self) -> RunRecord | None:
        item = self.run_list.currentItem()
        return item.data(Qt.UserRole + 1) if item is not None else None

    # ---- 实验对比（批次十二：新窗口，2~3 条，含数据集指纹提示）----
    def _on_run_selection_changed(self):
        """多选上限 3 条：第 4 个 Ctrl+点击被拒绝（恢复上一状态）并提示。"""
        items = self.run_list.selectedItems()
        if len(items) > 3:
            keep = {it.data(Qt.UserRole) for it in self._prev_selected}
            self.run_list.blockSignals(True)
            for it in items:
                if it.data(Qt.UserRole) not in keep:
                    it.setSelected(False)
            self.run_list.blockSignals(False)
            self.statusBar().showMessage("最多选择 3 条同时对比", 4000)
            items = self.run_list.selectedItems()
        self._prev_selected = list(items)

    def _compare_candidates(self) -> list:
        """当前多选中的可对比记录（进行中的训练无 metrics，不参与）。"""
        out = []
        for item in self.run_list.selectedItems():
            rec: RunRecord = item.data(Qt.UserRole + 1)
            if rec is not None and rec.status != "running" and rec.metrics:
                out.append(rec)
        return out

    def _open_compare(self):
        recs = self._compare_candidates()
        if not 2 <= len(recs) <= 3:
            self.statusBar().showMessage("请先选择 2~3 条已完成的训练记录（按住 Ctrl 多选）", 5000)
            return
        from .compare_window import CompareWindow

        self._compare_window = CompareWindow(self.project, recs)
        self._compare_window.show()
        self._compare_window.raise_()

    def _delete_current_run(self):
        """删除当前选中的历史训练记录（连带输出文件夹，无法恢复）。"""
        rec = self._current_run_record()
        if rec is None:
            return
        if rec.status == "running" and self.thread is not None:
            QMessageBox.warning(self, "无法删除", "该记录对应的训练正在进行中，无法删除")
            return
        if QMessageBox.question(
                self, "删除训练记录",
                f"确定删除该条训练记录吗？\n{rec.run_dir}\n\n"
                "记录文件夹（含权重、指标、导出数据）将一并删除，且无法恢复。",
                ) != QMessageBox.Yes:
            return
        shutil.rmtree(rec.run_dir, ignore_errors=True)
        self._selected_run_dir = None
        self._reload_history()
        self.statusBar().showMessage("训练记录已删除", 3000)

    def closeEvent(self, event):
        for attr in ("_preflight_worker", "_health_worker"):
            worker = getattr(self, attr, None)
            if worker is not None and worker.isRunning():
                QMessageBox.warning(self, "后台任务进行中",
                                    "数据体检 / 训练前预检正在进行，请稍候再关闭窗口")
                event.ignore()
                return
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(self, "训练中", "训练正在进行，无法关闭窗口；请先等待完成或停止训练")
            event.ignore()
            return
        if self.on_closed is not None:
            self.on_closed(self.project.name)
        dg = getattr(self, "data_grid", None)
        if dg is not None:      # O11：缩略/扫描线程收尾（P144 纪律）
            dg.shutdown()
        event.accept()


# ============================================================
# 创建项目对话框
# ============================================================
class CreateProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("创建项目")
        self.setMinimumWidth(480)
        _apply_theme(self)

        form = QFormLayout(self)
        form.setContentsMargins(18, 16, 18, 14)
        form.setSpacing(10)
        self.fld_name = QLineEdit()
        self.fld_name.setPlaceholderText("例如：安全帽检测")
        self.fld_dataset = QLineEdit()
        btn_ds = QPushButton("浏览…")
        btn_ds.clicked.connect(self._browse_dataset)
        ds_row = QHBoxLayout()
        ds_row.addWidget(self.fld_dataset, stretch=1)
        ds_row.addWidget(btn_ds)
        self.fld_model_dir = QLineEdit()
        self.fld_model_dir.setPlaceholderText("留空则使用平台默认位置")
        btn_md = QPushButton("浏览…")
        btn_md.clicked.connect(self._browse_model_dir)
        md_row = QHBoxLayout()
        md_row.addWidget(self.fld_model_dir, stretch=1)
        md_row.addWidget(btn_md)
        self.cmb_task = QComboBox()
        for t in TaskType:
            self.cmb_task.addItem(t.label, t.value)
        self.cmb_task.currentIndexChanged.connect(self._fill_models)
        self.cmb_model = QComboBox()

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("创建项目")
        self.buttons.button(QDialogButtonBox.Cancel).setText("取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        form.addRow("项目名称", self.fld_name)
        form.addRow("数据集路径", ds_row)
        form.addRow("模型储存路径", md_row)
        form.addRow("任务类型", self.cmb_task)
        form.addRow("训练模型", self.cmb_model)
        form.addRow(self.buttons)
        self._fill_models()

    def _browse_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "选择数据集文件夹")
        if path:
            self.fld_dataset.setText(path)

    def _browse_model_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择模型储存文件夹")
        if path:
            self.fld_model_dir.setText(path)

    def _fill_models(self):
        task = self.cmb_task.currentData()
        self.cmb_model.clear()
        from ..engine.models import list_models  # torch 延迟导入（T1-4）

        for m in list_models(task):
            self.cmb_model.addItem(f"{m['name']}  ({m['desc']})", m["name"])

    def create(self) -> Project:
        """执行创建；失败抛 ValueError（由调用方提示）。"""
        project = create_project(
            name=self.fld_name.text(),
            dataset_dir=self.fld_dataset.text(),
            model_dir=self.fld_model_dir.text(),
            task=self.cmb_task.currentData(),
        )
        project.model_name = self.cmb_model.currentData() or ""
        project.learning_rate = DEFAULT_LR.get(project.model_name, 0.001)
        project.save()
        return project


# ============================================================
# 项目列表窗口：项目卡片墙
# ============================================================
def rotate_guard_decision(ops: dict, probs: dict, aspects: set) -> tuple | None:
    """rotate90 守卫决策（纯函数，便于测试）。

    用户规则：数据集长宽比不一致时，旋转算子必须配合「随机裁剪（对齐目标
    宽高比）」且启用概率 100%。返回 None=通过；("enable", 文本)=需启用兜底；
    ("prob", 文本)=兜底已启用但概率不足。
    """
    if "rotate90" not in ops:
        return None
    if len(aspects) <= 1:
        return None  # 长宽比一致：旋转后仍一致，训练输入形状稳定
    if "random_crop" in ops:
        crop_prob = float((probs or {}).get("random_crop", 0.5))
        if crop_prob >= 0.999:
            return None
        return ("prob",
                "「随机裁剪」作为旋转兜底时启用概率必须为 100%，\n"
                "否则部分样本旋转后长宽比不一致，训练输入形状混乱。")
    shown = "、".join(f"{a:g}" for a in sorted(aspects)[:5])
    if len(aspects) > 5:
        shown += "…"
    return ("enable",
            f"已启用「随机旋转 90°/270°」，但数据集图像长宽比不一致"
            f"（{len(aspects)} 种：{shown}）。\n"
            "旋转后宽高互换会让训练输入比例混乱。请启用「随机裁剪"
            "（对齐目标宽高比）」作为兜底，或关闭旋转算子。")


class TrainingWindow(QMainWindow):
    def __init__(self, open_project: str | None = None):
        super().__init__()
        self.setWindowTitle("XCcv 训练平台 - 项目列表")
        from PyQt5.QtGui import QIcon
        self.setWindowIcon(QIcon(str(icon_path("training.png"))))
        self.resize(1120, 720)
        _apply_theme(self)
        self._project_windows: dict[str, ProjectWindow] = {}
        self._cards: list[_ProjectCard] = []

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 顶栏
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(66)
        head = QHBoxLayout(header)
        head.setContentsMargins(22, 10, 22, 10)
        app_title = QLabel("XCcv 训练平台")
        app_title.setObjectName("appTitle")
        head.addWidget(app_title)
        head.addStretch(1)
        btn_create = QPushButton("＋ 新建项目")
        btn_create.setObjectName("accent")
        btn_create.clicked.connect(self._create_project)
        head.addWidget(btn_create)
        root.addWidget(header)

        # 内容区：项目卡片墙
        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(22, 16, 22, 12)
        body_lay.setSpacing(10)
        self.list_hint = QLabel("")
        self.list_hint.setObjectName("hint")
        body_lay.addWidget(self.list_hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.cards_host = QWidget()
        self.cards_grid = QGridLayout(self.cards_host)
        self.cards_grid.setContentsMargins(0, 0, 0, 0)
        self.cards_grid.setSpacing(14)
        self.cards_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(self.cards_host)
        body_lay.addWidget(scroll, stretch=1)
        root.addWidget(body, stretch=1)

        self.setCentralWidget(central)
        self._refresh_project_list()
        if open_project:
            # 主界面「最近项目」直达：argv 透传项目名，布局完成后自动打开
            self._pending_open = open_project
            QTimer.singleShot(0, self._open_pending_project)

    def _open_pending_project(self):
        name = getattr(self, "_pending_open", "")
        if not name:
            return
        self._pending_open = ""
        try:
            self._open_project(name)
        except Exception as e:   # 项目不存在/损坏：回退项目列表 + 人话提示
            QMessageBox.warning(self, "打开项目失败",
                                f"项目「{name}」无法打开：{e}")

    # ---- 卡片墙 ----
    def _refresh_project_list(self):
        self._cards.clear()
        while self.cards_grid.count():
            item = self.cards_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        names = list_projects()
        for i, name in enumerate(names):
            project = load_project(name)
            card = _ProjectCard(project)
            card.clicked.connect(self._open_project)
            card.deleteRequested.connect(self._confirm_delete)
            self.cards_grid.addWidget(card, i // 3, i % 3)
            self._cards.append(card)
        # 空状态引导（EmptyState 组件）：有项目时不占位
        if not names:
            from ..ui_widgets import EmptyState

            self.cards_grid.addWidget(
                EmptyState("🗂", "还没有项目",
                           "点击右上角「＋ 新建项目」，30 秒完成第一个训练项目"),
                0, 0)
        self.list_hint.setText("")

    def project_count(self) -> int:
        return len(self._cards)

    # ---- 打开 / 删除 ----
    def _open_project(self, name: str):
        from PyQt5.QtCore import Qt as _Qt

        window = self._project_windows.get(name)
        if window is not None and window.isVisible():
            window.activateWindow()
            window.raise_()
            return
        # 打开项目要加载深度学习组件（渲染卡数秒）——顶部悬浮胶囊提示
        # （U1：对齐标注平台横幅样式，单行/半透明/圆角；颜色与标注平台
        # 的琥珀底作微差取暗暖灰；V0.7）
        banner = QLabel(f"正在打开项目「{name}」…", self)
        banner.setObjectName("openProjectBanner")
        banner.setStyleSheet(
            "QLabel#openProjectBanner { background: rgba(58, 46, 34, 224);"
            " color: white; padding: 6px 18px; border-radius: 12px;"
            " font-weight: bold; font-size: 13px; }")
        banner.adjustSize()
        banner.move((self.width() - banner.width()) // 2, 12)
        banner.show()
        banner.raise_()
        QApplication.processEvents()
        self.statusBar().showMessage(f"正在打开项目「{name}」…", 8000)
        QApplication.setOverrideCursor(_Qt.WaitCursor)
        try:
            window = ProjectWindow(
                load_project(name),
                on_renamed=self._on_project_renamed,
                on_closed=self._on_project_closed,
            )
            window.show()
        finally:
            QApplication.restoreOverrideCursor()
            banner.hide()
            banner.setParent(None)
            banner.deleteLater()
        self.statusBar().showMessage(f"项目「{name}」已打开", 4000)
        self._project_windows[name] = window

    def on_platform_message(self, args):
        """单实例激活携带业务参数：args[0]=项目名 → 直接打开该项目。

        主界面「最近项目」卡片在训练平台已运行时点下：新进程把项目名经
        单实例管道转发到本钩子（第四批修复：此前只激活窗口不打开项目）。
        """
        name = args[0] if args else ""
        if not name:
            return
        QTimer.singleShot(0, lambda: self._safe_open(name))

    def _safe_open(self, name: str):
        try:
            self._open_project(name)
        except Exception as e:   # 项目不存在/损坏：人话提示
            QMessageBox.warning(self, "打开项目失败",
                                f"项目「{name}」无法打开：{e}")

    def _confirm_delete(self, name: str):
        if QMessageBox.question(self, "删除项目",
                                f"确定删除项目「{name}」吗？（不影响已保存的模型文件）"
                                ) != QMessageBox.Yes:
            return
        window = self._project_windows.pop(name, None)
        if window is not None and not window.close():
            # closeEvent 被拒绝（如训练进行中禁关窗）：取消删除，
            # 避免删掉正在使用的项目目录（P93）
            self._project_windows[name] = window
            self.statusBar().showMessage(
                f"项目「{name}」的窗口正在忙碌（训练中无法关闭），已取消删除", 5000)
            return
        delete_project(name)
        self._refresh_project_list()

    def _on_project_renamed(self, old_name: str, new_name: str):
        window = self._project_windows.pop(old_name, None)
        if window is not None:
            self._project_windows[new_name] = window
        self._refresh_project_list()

    def _on_project_closed(self, name: str):
        self._project_windows.pop(name, None)

    def open_project_windows(self) -> dict:
        """已打开的项目窗口（测试用）。"""
        return dict(self._project_windows)

    # ---- 创建 ----
    def _create_project(self):
        dialog = CreateProjectDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
        try:
            project = dialog.create()
        except ValueError as e:
            QMessageBox.warning(self, "无法创建", str(e))
            return
        self._refresh_project_list()
        self._open_project(project.name)


def main():
    from ..platform_entry import run_platform

    arg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else None
    run_platform("training", "训练平台",
                 lambda: TrainingWindow(open_project=arg), heavy=True)


if __name__ == "__main__":
    main()


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402
apply_auto_slot_module(globals())

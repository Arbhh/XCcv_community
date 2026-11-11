"""测试平台主窗口：加载训练好的模型，对图片/文件夹/数据集推理并可视化。

- 模型来源：项目（自动找最新一次训练产出的 best.pt）
  或直接选择权重文件（.pt）
- 输入来源：单张图片 / 文件夹（树形列表，子文件夹可展开）/ 数据集
- 参数：置信度阈值、IOU 阈值实时调节；标签 / 掩码 / 面积显示开关
- 结果：类别筛选（画布与列表联动）、bbox、实例掩码与掩码面积、Top-5（分类）

运行：python -m xccv.testing.testing_window
"""

if __name__ == "__main__":  # V0.6 T1-3：-m 直启先亮加载中界面，再做下方重型模块级导入
    from xccv.platform_entry import early_splash

    early_splash("testing", "测试平台", heavy=True)

import json
import logging
import sys
import time
from pathlib import Path

import cv2
from PyQt5.QtCore import QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QKeySequence, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame,
    QGraphicsScene, QGraphicsTextItem, QGraphicsView, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QShortcut, QSlider, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from ..engine.io_utils import imread_unicode
from ..gui_utils import auto_cleanup_thread, ensure_console_streams
from ..training.dataset_stats import collect_stats
from ..training.project import list_projects, load_project
from ..ui_style import APP_QSS, ACCENT, apply_app_theme, icon_path  # noqa: F401

ensure_console_streams()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_EMPTY_HINT = "请先加载模型，再打开图片或文件夹"

log = logging.getLogger(__name__)


def latest_weights(model_dir) -> Path | None:
    """在模型储存目录里找最新一次训练产出的 best.pt。"""
    candidates = list(Path(model_dir).rglob("best.pt"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


class _ModelLoadThread(QThread):
    """后台加载模型：避免大权重加载时界面卡死（网络重建耗时）。"""

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = Path(path)

    def run(self):
        # torch 延迟导入（T1-4）：引擎链在加载线程里引入，让启动闪屏先显示
        from ..engine.models import load_model

        try:
            self.done.emit(load_model(self.path))
        except Exception as e:
            log.error("模型加载失败: %s (%s)", self.path, e, exc_info=True)
            self.failed.emit(f"{type(e).__name__}: {e}")


class _BatchThread(QThread):
    """批量推理线程：逐张预测，按设置保存渲染图与检出 JSON
    （每张图片一个同名 .json，镜像相对目录防重名）。"""

    progress = pyqtSignal(int, int, str)   # 已完成数, 总数, 当前文件名
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, model, paths, conf, iou, classes, task,
                 out_dir: Path | None = None, save_json: bool = True,
                 save_images: bool = False, jpg_quality: int = 95,
                 root: Path | None = None, model_text: str = "", parent=None,
                 json_format: str = "xcv", tiled: bool = False):
        super().__init__(parent)
        self._model = model
        self._paths = list(paths)
        self._conf, self._iou = conf, iou
        self._classes = classes
        self._task = task
        self._out_dir = Path(out_dir) if out_dir else None
        self._save_json = save_json
        self._save_images = save_images
        self._jpg_quality = int(jpg_quality)
        self._root = Path(root) if root else None   # 输入文件夹（相对目录镜像用）
        self._model_text = model_text               # JSON 元信息里的权重名
        self._json_format = json_format             # P8/T8-1：xcv/labelme/coco
        self._tiled = tiled                         # P8/T8-3：4K 分块推理
        self._coco_payloads: list = []              # coco 模式的逐图缓冲
        self._costs_ms: list = []                   # P8/T8-2 统计：逐图耗时
        self.summary: dict | None = None            # 运行结束后的统计摘要

    def run(self):
        results = []
        try:
            if self._out_dir is not None:
                if self._save_images:
                    (self._out_dir / "可视化").mkdir(parents=True, exist_ok=True)
                if self._save_json:
                    (self._out_dir / "检出结果").mkdir(parents=True, exist_ok=True)
            total = len(self._paths)
            import time as _time

            for i, p in enumerate(self._paths, start=1):
                _t0 = _time.perf_counter()
                img = imread_unicode(p)
                if img is None:
                    continue
                h, w = img.shape[:2]
                if self._tiled and self._task != "classification":
                    from ..engine.tiled_infer import tiled_predict

                    pred = tiled_predict(self._model, img, conf=self._conf,
                                         iou=self._iou)
                else:
                    pred = self._model.predict(img, conf=self._conf,
                                               iou=self._iou)
                detections = []
                for j, box in enumerate(pred.boxes):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    label = (self._classes[pred.labels[j]]
                             if self._classes and pred.labels[j] < len(self._classes)
                             else str(pred.labels[j]))
                    det = {"label": label, "score": round(float(pred.scores[j]), 4),
                           "bbox": [round(x1, 1), round(y1, 1),
                                    round(x2, 1), round(y2, 1)],
                           "bbox_area": int(round((x2 - x1) * (y2 - y1)))}
                    if pred.masks is not None and j < len(pred.masks):
                        det["mask_area"] = int(pred.masks[j].sum())
                    detections.append(det)
                result = {"file": p.name, "path": str(p),
                          "width": w, "height": h,
                          "detections": detections}
                if self._save_json and self._out_dir is not None:
                    self._save_json_file(p, result, pred)
                if self._save_images and self._out_dir is not None:
                    self._save_rendered(p, img, pred)
                results.append(result)
                # 仪表化（ToE3 排查）：慢图告警 + 定期进度，若再出现"卡死"
                # 可从日志定位到具体那张图
                _cost = _time.perf_counter() - _t0
                self._costs_ms.append(round(_cost * 1000, 1))
                if _cost > 5.0:
                    log.warning("批量单图耗时 %.1f 秒: %s", _cost, p.name)
                if i % 20 == 0:
                    log.info("批量运行进度 %d/%d", i, total)
                self.progress.emit(i, total, p.name)
            if (self._json_format == "coco" and self._save_json
                    and self._out_dir is not None):
                self._save_coco_file()
            from ..engine.result_json import batch_summary, save_result_json

            self.summary = batch_summary(results, self._costs_ms)
            if self._save_json and self._out_dir is not None:
                # P8/T8-2：统计摘要随批量产物落盘（属批量任务而非 UI 职责）
                try:
                    save_result_json(
                        self._out_dir / "检出结果" / "摘要.json", self.summary)
                except Exception:  # noqa: BLE001  摘要失败不影响批量结果
                    log.warning("批量摘要落盘失败", exc_info=True)
            self.finished_ok.emit(results)
        except Exception as e:
            log.error("批量推理失败: %s", e, exc_info=True)
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _rel_target(self, path: Path, sub_dir: str, suffix: str) -> Path:
        """输出路径镜像输入的相对目录结构（不同子文件夹的同名图互不覆盖）。"""
        rel = (path.relative_to(self._root) if self._root is not None
               else Path(path.name))
        out = self._out_dir / sub_dir / rel.with_suffix(suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def _save_json_file(self, path: Path, result: dict, pred):
        """每张图片一个同名 .json——统一检出结果 schema（Q1/Q2/Q5 契约）。"""
        try:
            from ..engine.result_json import build_result_json, save_result_json

            payload = build_result_json(
                task=self._task,
                source={"file": result["file"], "path": result["path"],
                        "width": result["width"], "height": result["height"]},
                model={"file": self._model_text},
                inference={"conf": self._conf, "iou": self._iou},
                prediction=pred, classes=self._classes or [],
                flow_info=None)
            out = self._rel_target(path, "检出结果", ".json")
            if self._json_format == "labelme":
                from ..engine.result_json import to_labelme

                save_result_json(out, to_labelme(payload))
            elif self._json_format == "coco":
                # COCO 为汇总格式：逐图只缓冲，循环结束后合并落盘
                self._coco_payloads.append(payload)
                return
            else:
                save_result_json(out, payload)
        except Exception as e:
            log.error("批量检出 JSON 保存失败 %s: %s", path.name, e, exc_info=True)

    def _save_coco_file(self):
        """P8/T8-1：COCO 汇总落盘（全部分图合并为一份 results_coco.json）。"""
        try:
            from ..engine.result_json import save_result_json, to_coco

            out = self._out_dir / "检出结果" / "results_coco.json"
            save_result_json(out, to_coco(self._coco_payloads,
                                          classes=self._classes or []))
            log.info("COCO 汇总已保存: %s（%d 图）", out,
                     len(self._coco_payloads))
        except Exception as e:
            log.error("COCO 汇总保存失败: %s", e, exc_info=True)

    def _model_name_text(self) -> str:
        return self._model_text or getattr(self._model, "name", "") or "model"

    def _save_rendered(self, path: Path, img, pred):
        """渲染图与 JSON 一一对应：可视化/<相对路径>/<文件名>.jpg（单张失败不中断）。"""
        try:
            import cv2

            from .render import draw_predictions

            rendered = draw_predictions(
                img, pred, self._classes or [], self._task,
                show_labels=True,
                show_masks=(self._task == "segmentation"))
            out = self._rel_target(path, "可视化", ".jpg")
            cv2.imwrite(str(out), rendered,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self._jpg_quality])
        except Exception as e:
            log.error("批量渲染图保存失败 %s: %s", path.name, e, exc_info=True)


class _ImageView(QGraphicsView):
    """支持滚轮缩放（以鼠标为中心）与左键拖拽平移的图片显示控件。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self._hint = QGraphicsTextItem(_EMPTY_HINT)
        self._hint.setDefaultTextColor(QColor(170, 170, 170))
        self._scene.addItem(self._hint)
        self._user_zoomed = False
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QBrush(QColor(23, 24, 26)))

    def wheelEvent(self, event):
        if self._pixmap_item is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._user_zoomed = True
        self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap_item is not None and not self._user_zoomed:
            self.fit_window()

    def set_image(self, pixmap: QPixmap):
        self._scene.clear()
        self._hint = None
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._user_zoomed = False
        self.fit_window()

    def fit_window(self):
        if self._pixmap_item is None:
            return
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def one_to_one(self):
        if self._pixmap_item is None:
            return
        self._user_zoomed = True
        self.resetTransform()


class TestingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XCcv 测试平台")
        from PyQt5.QtGui import QIcon
        self.setWindowIcon(QIcon(str(icon_path("testing.png"))))
        self.resize(1520, 940)
        apply_app_theme(self)

        self.model = None            # 引擎 BaseModel 实例
        self.classes: list[str] = []
        self.task: str = ""          # classification / detection / segmentation
        self.image_paths: list[Path] = []
        self._last_image = None      # 缓存最近一次推理的原图与结果，供显示开关/筛选切换
        self._last_prediction = None
        self._last_infer_cost: float | None = None   # 最近一次单图推理耗时（秒）
        self._load_thread: _ModelLoadThread | None = None

        self._build_ui()
        from ..status_bus import attach_platform_status

        attach_platform_status(self, "测试平台", "testing")
        self._refresh_projects()
        self._add_shortcuts()

        # 滑条/筛选变化后防抖 250ms 自动重推理或重绘
        self._infer_timer = QTimer(self)
        self._infer_timer.setSingleShot(True)
        self._infer_timer.setInterval(250)
        self._infer_timer.timeout.connect(self._infer_current)

    def closeEvent(self, event):
        if self._load_thread is not None and self._load_thread.isRunning():
            self._load_thread.wait(5000)  # 等待模型加载线程收尾，避免退出崩溃
        event.accept()

    def _add_shortcuts(self):
        """XCcv：快捷键（A/D 切换图片、F 适应窗口、Ctrl+1 原始尺寸、Ctrl+O 打开）。"""
        self._shortcuts = []
        bindings = (
            ("A", lambda: self._navigate(-1)),
            ("D", lambda: self._navigate(1)),
            ("F", self.canvas.fit_window),
            ("Ctrl+1", self.canvas.one_to_one),
            ("Ctrl+O", self._open_image),
            ("Ctrl+Shift+O", self._open_folder),
        )
        for seq, slot in bindings:
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(slot)
            self._shortcuts.append(sc)

    def _navigate(self, delta: int):
        """A/D：在展平的图片顺序中切换（文件夹节点不参与导航）。"""
        if not self.image_paths:
            return
        path = self._current_image_path()
        idx = self.image_paths.index(path) if path in self.image_paths else 0
        new_idx = max(0, min(idx + delta, len(self.image_paths) - 1))
        if new_idx == idx:
            return  # 已到边界
        item = self._file_items.get(str(self.image_paths[new_idx]))
        if item is not None:
            self.image_tree.setCurrentItem(item)  # 触发选择变更 → 推理

    # ================= 界面 =================
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(64)
        head = QHBoxLayout(header)
        head.setContentsMargins(18, 8, 18, 8)
        title = QLabel("XCcv 测试平台")
        title.setObjectName("appTitle")
        head.addWidget(title)
        head.addStretch(1)
        self.header_model_info = QLabel("未加载模型")
        self.header_model_info.setObjectName("hint")
        head.addWidget(self.header_model_info)
        root.addWidget(header)

        body = QHBoxLayout()
        body.setContentsMargins(12, 12, 12, 12)
        body.setSpacing(12)
        body.addWidget(self._build_left_panel())
        body.addWidget(self._build_center(), stretch=1)
        body.addWidget(self._build_right_panel())
        root.addLayout(body)
        self.setCentralWidget(central)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(340)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        g1 = QGroupBox("模型")
        v1 = QVBoxLayout(g1)
        self.project_combo = QComboBox()
        self.project_combo.currentIndexChanged.connect(self._refresh_project_hint)
        v1.addWidget(self.project_combo)
        self.project_hint = QLabel("")
        self.project_hint.setObjectName("hint")
        self.project_hint.setWordWrap(True)
        v1.addWidget(self.project_hint)
        self.btn_load_project = QPushButton("加载该项目最新模型")
        self.btn_load_project.clicked.connect(self._load_project_model)
        v1.addWidget(self.btn_load_project)
        self.btn_pick = QPushButton("选择权重文件…")
        self.btn_pick.clicked.connect(self._pick_weights)
        v1.addWidget(self.btn_pick)
        self.model_info = QLabel("未加载模型")
        self.model_info.setObjectName("hint")
        self.model_info.setWordWrap(True)
        v1.addWidget(self.model_info)
        lay.addWidget(g1)

        g2 = QGroupBox("输入")
        v2 = QVBoxLayout(g2)
        row = QHBoxLayout()
        btn_image = QPushButton("打开图片…")
        btn_image.clicked.connect(self._open_image)
        btn_folder = QPushButton("打开文件夹…")
        btn_folder.clicked.connect(self._open_folder)
        row.addWidget(btn_image)
        row.addWidget(btn_folder)
        v2.addLayout(row)
        btn_batch = QPushButton("批量运行…")
        btn_batch.setToolTip("选择文件夹批量推理，完成后可导出检出结果 JSON")
        btn_batch.clicked.connect(self._batch_run)
        v2.addWidget(btn_batch)
        self.btn_batch = btn_batch
        # XCcv：文件树（子文件夹可展开，图片显示原名）
        self.image_tree = QTreeWidget()
        self.image_tree.setHeaderHidden(True)
        self.image_tree.itemSelectionChanged.connect(self._on_tree_selection)
        v2.addWidget(self.image_tree, stretch=1)
        nav_hint = QLabel("A / D 切换上一张、下一张图片")
        nav_hint.setObjectName("hint")
        v2.addWidget(nav_hint)
        lay.addWidget(g2)

        g3 = QGroupBox("预测参数")
        v3 = QVBoxLayout(g3)
        self.conf_slider = self._make_slider(v3, "置信度阈值", 1, 100, 25)
        self.iou_slider = self._make_slider(v3, "IOU 阈值", 10, 90, 45)
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("background: #eef2f8; border: none; max-height: 1px;")
        v3.addWidget(divider)
        self.show_labels = QCheckBox("显示类别与置信度")
        self.show_labels.setChecked(True)
        self.show_labels.stateChanged.connect(self._redraw_last)
        self.show_masks = QCheckBox("显示掩码区域")
        self.show_masks.setChecked(True)
        self.show_masks.stateChanged.connect(self._redraw_last)
        self.show_area = QCheckBox("结果显示面积")
        self.show_area.setChecked(True)
        self.show_area.stateChanged.connect(self._redraw_last)
        self.chk_tiled = QCheckBox("4K 大图分块推理（tile 640 / 重叠 15%）")
        self.chk_tiled.setToolTip(
            "整图缩到模型输入会把小目标缩没；分块推理保持模型原生口径逐块检出"
            "后合并去重（同类 NMS）。小图自动回退整图推理；分类任务不适用。\n"
            "与训练平台「切图训练」（T9-3）配对使用效果最佳。")
        v3.addWidget(self.show_labels)
        v3.addWidget(self.show_masks)
        v3.addWidget(self.show_area)
        v3.addWidget(self.chk_tiled)
        lay.addWidget(g3)

        g4 = QGroupBox("目标筛选")
        v4 = QVBoxLayout(g4)
        self.class_filter = QComboBox()
        self.class_filter.currentIndexChanged.connect(self._redraw_last)
        v4.addWidget(self.class_filter)
        filter_hint = QLabel("选定类别后，画布与列表只显示该目标")
        filter_hint.setObjectName("hint")
        filter_hint.setWordWrap(True)
        v4.addWidget(filter_hint)
        lay.addWidget(g4)
        lay.addStretch(1)
        return panel

    def _build_center(self) -> QWidget:
        center = QWidget()
        center_lay = QVBoxLayout(center)
        center_lay.setContentsMargins(0, 0, 0, 0)
        center_lay.setSpacing(6)
        self.canvas = _ImageView()
        self.canvas.setMinimumSize(560, 480)
        center_lay.addWidget(self.canvas, stretch=1)
        btn_row = QHBoxLayout()
        btn_fit = QPushButton("适应窗口")
        btn_fit.clicked.connect(self.canvas.fit_window)
        btn_11 = QPushButton("1:1")
        btn_11.clicked.connect(self.canvas.one_to_one)
        btn_row.addWidget(btn_fit)
        btn_row.addWidget(btn_11)
        btn_row.addStretch(1)
        center_lay.addLayout(btn_row)
        return center

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(320)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        g = QGroupBox("识别结果")
        v = QVBoxLayout(g)
        self.result_stats = QLabel("未加载模型")
        self.result_stats.setObjectName("hint")
        self.result_stats.setWordWrap(True)
        v.addWidget(self.result_stats)
        self.result_list = QListWidget()
        v.addWidget(self.result_list, stretch=1)
        lay.addWidget(g, stretch=1)
        return panel

    def _make_slider(self, layout, title: str, vmin: int, vmax: int, value: int) -> QSlider:
        label = QLabel(f"{title}：{value / 100:.2f}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(vmin, vmax)
        slider.setValue(value)
        layout.addWidget(label)
        layout.addWidget(slider)
        slider.valueChanged.connect(
            lambda v, lb=label, t=title: (lb.setText(f"{t}：{v / 100:.2f}"),
                                          self._infer_timer.start()))
        return slider

    # ================= 模型加载 =================
    def _refresh_projects(self):
        self.project_combo.clear()
        for name in list_projects():
            self.project_combo.addItem(name)

    def _refresh_project_hint(self):
        name = self.project_combo.currentText()
        if not name:
            self.project_hint.setText("")
            return
        try:
            project = load_project(name)
        except Exception:
            self.project_hint.setText("")
            return
        self.project_hint.setText(f"{project.task_type.label} | 输出目录：{project.model_dir}")

    def _load_project_model(self):
        name = self.project_combo.currentText()
        if not name:
            QMessageBox.information(self, "提示", "请先在下拉框选择一个项目")
            return
        weights = latest_weights(load_project(name).model_dir)
        if weights is None:
            QMessageBox.warning(self, "未找到模型",
                                f"项目「{name}」的模型储存目录下没有可加载的模型权重（best.pt），\n"
                                "请先在训练平台完成训练。")
            return
        self._load_weights(weights)

    def _pick_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择权重文件", "",
            "模型权重 (*.pt);;所有文件 (*)")
        if not path:
            return
        self._load_weights(Path(path))



    def _load_weights(self, path: Path):
        """后台线程加载模型：加载期间禁用加载入口并显示进度提示。"""
        if self._load_thread is not None and self._load_thread.isRunning():
            QMessageBox.information(self, "请稍候", "正在加载模型，请等待完成")
            return
        self.btn_load_project.setEnabled(False)
        self.btn_pick.setEnabled(False)
        self.model_info.setText(f"正在加载 {path.name} …（网络构建需要数秒）")
        self.header_model_info.setText("模型加载中…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._load_thread = _ModelLoadThread(path, self)
        self._load_thread.done.connect(
            lambda model, p=path: self._on_model_loaded(p, model))
        self._load_thread.failed.connect(self._on_model_load_failed)
        auto_cleanup_thread(self, "_load_thread", self._load_thread)  # A-3（T2-6）
        self._load_thread.start()



    def _on_model_loaded(self, path: Path, model):
        self._load_thread = None
        self.btn_load_project.setEnabled(True)
        self.btn_pick.setEnabled(True)
        QApplication.restoreOverrideCursor()
        self.model = model
        self.classes = model.classes or []
        task = getattr(model, "task", None)
        self.task = task.value if task is not None else ""
        log.info("模型已加载: %s 任务=%s 类别数=%s 设备=%s",
                 path.name, self.task, len(self.classes), model.device)
        self.model_info.setText(
            f"权重：{path.name}\n任务：{task.label if task is not None else '未知'}\n"
            f"类别数：{len(self.classes)}\n设备：{model.device}"
        )
        self.header_model_info.setText(
            f"{path.name}　|　{task.label if task is not None else '未知'}　|　{len(self.classes)} 类"
        )
        is_cls = self.task == "classification"
        is_seg = self.task == "segmentation"
        self.conf_slider.setEnabled(not is_cls)
        self.iou_slider.setEnabled(not is_cls)
        self.show_masks.setVisible(is_seg)
        self.show_area.setVisible(not is_cls)
        # 类别筛选下拉
        self.class_filter.blockSignals(True)
        self.class_filter.clear()
        self.class_filter.addItem("全部目标", None)
        for i, name in enumerate(self.classes):
            self.class_filter.addItem(name, i)
        self.class_filter.setEnabled(not is_cls)
        self.class_filter.blockSignals(False)
        self.result_stats.setText("模型已加载，请选择图片")
        self.result_list.clear()


    def _on_model_load_failed(self, message: str):
        self._load_thread = None
        self.btn_load_project.setEnabled(True)
        self.btn_pick.setEnabled(True)
        QApplication.restoreOverrideCursor()
        self.model_info.setText("未加载模型")
        self.header_model_info.setText("未加载模型")
        QMessageBox.critical(self, "加载失败", message)

    # ================= 输入 =================
    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if path:
            self._set_image_list([Path(path)], root=None)

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹（含子文件夹）")
        if not folder:
            return
        files = [p for p in sorted(Path(folder).rglob("*"))
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                 and not any(part.startswith(".") for part in p.relative_to(folder).parts[:-1])]
        if not files:
            QMessageBox.information(self, "提示", "该文件夹（含子文件夹）中没有图片")
            return
        self._set_image_list(files, root=Path(folder))

    def _set_image_list(self, paths: list[Path], root: Path | None = None):
        """填充文件树。root 给定时按子文件夹分组建树（可展开），否则平铺。"""
        self.image_paths = list(paths)
        self.image_tree.blockSignals(True)
        self.image_tree.clear()
        self._file_items = {}

        def _make_item(path: Path) -> QTreeWidgetItem:
            item = QTreeWidgetItem([path.name])
            item.setData(0, Qt.UserRole, str(path))
            self._file_items[str(path)] = item
            return item

        if root is None:
            for p in self.image_paths:
                self.image_tree.addTopLevelItem(_make_item(p))
        else:
            groups: dict = {}
            for p in self.image_paths:
                rel_dir = p.parent.relative_to(root)
                key = "" if str(rel_dir) == "." else str(rel_dir)
                groups.setdefault(key, []).append(p)
            for p in groups.get("", []):
                self.image_tree.addTopLevelItem(_make_item(p))
            for rel_dir in sorted(groups):
                if rel_dir == "":
                    continue
                folder_item = QTreeWidgetItem([Path(rel_dir).name])
                folder_item.setFlags(Qt.ItemIsEnabled)
                self.image_tree.addTopLevelItem(folder_item)
                for p in groups[rel_dir]:
                    folder_item.addChild(_make_item(p))
        self.image_tree.blockSignals(False)
        self.image_tree.setCurrentItem(self.image_tree.topLevelItem(0))  # 触发推理

    def _on_tree_selection(self):
        self._infer_current()

    # ================= 批量运行 =================
    def _batch_run(self):
        """批量推理：先弹设置（输出目录 / 是否存图等，To3.2），再后台逐张推理。"""
        if self.model is None:
            QMessageBox.information(self, "提示", "请先加载模型，再批量运行")
            return
        if getattr(self, "_batch_thread", None) is not None \
                and self._batch_thread.isRunning():
            QMessageBox.information(self, "请稍候", "批量推理正在进行中")
            return
        from .batch_dialog import BatchSettingsDialog

        dlg = BatchSettingsDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        settings = dlg.settings()
        folder = Path(settings.folder)
        files = [p for p in sorted(folder.rglob("*"))
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                 and not any(part.startswith(".") for part in p.relative_to(folder).parts[:-1])]
        if not files:
            QMessageBox.information(self, "提示", "该文件夹（含子文件夹）中没有图片")
            return
        out_dir = Path(settings.out_dir) if settings.out_dir else folder
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.warning(self, "输出目录无效", f"无法创建输出目录：{e}")
            return
        conf, iou = self._current_params()
        self.btn_batch.setEnabled(False)
        self._batch_settings = settings
        self._batch_out_dir = out_dir
        self.result_stats.setText(f"批量推理 0 / {len(files)} …")
        self._batch_thread = _BatchThread(
            self.model, files, conf, iou, self.classes or [],
            self.task, out_dir=out_dir, save_json=settings.save_json,
            save_images=settings.save_images,
            jpg_quality=settings.jpg_quality, root=folder,
            model_text=self.model_info.text().splitlines()[0].replace("权重：", ""),
            parent=self, json_format=getattr(settings, "json_format", "xcv"),
            tiled=self.chk_tiled.isChecked())
        self._batch_thread.progress.connect(
            lambda done, total, name: self.result_stats.setText(
                f"批量推理 {done} / {total}　{name}"))
        self._batch_thread.finished_ok.connect(self._on_batch_done)
        self._batch_thread.failed.connect(self._on_batch_failed)
        auto_cleanup_thread(self, "_batch_thread", self._batch_thread)  # A-3（T2-6）
        self._batch_thread.start()


    def _on_batch_done(self, results: list):
        self.btn_batch.setEnabled(True)
        summary = getattr(self._batch_thread, "summary", None) \
            if self._batch_thread is not None else None
        self._batch_thread = None
        settings = getattr(self, "_batch_settings", None)
        out_dir = getattr(self, "_batch_out_dir", None)
        if not results:
            QMessageBox.information(self, "批量运行", "没有可用的推理结果")
            return
        total_det = sum(len(r["detections"]) for r in results)
        lines = [f"批量推理完成：{len(results)} 张图，共检出 {total_det} 个目标"]
        if summary:
            pc = summary.get("per_class") or {}
            if pc:
                tops = sorted(pc.items(), key=lambda kv: -kv[1]["count"])[:5]
                lines.append("每类检出：" + "，".join(
                    f"{k}×{v['count']}（均分 {v['avg_score']:.2f}）"
                    for k, v in tops))
            lat = summary.get("latency_ms")
            if lat:
                lines.append(f"单图耗时 p50={lat['p50_ms']:.0f} ms / "
                             f"max={lat['max_ms']:.0f} ms")
            # 摘要 .json 已由批量线程落盘（检出结果/摘要.json）
        if settings is not None and settings.save_json and out_dir is not None:
            lines.append(f"检出 JSON（每图一个，同名对应）：{out_dir / '检出结果'}")
        if settings is not None and settings.save_images and out_dir is not None:
            lines.append(f"渲染图：{out_dir / '可视化'}")
        self.result_stats.setText("\n".join(lines))
        self.statusBar().showMessage(f"批量运行完成，输出目录：{out_dir}", 10000)
        from ..ui_widgets import toast

        toast(self,
              f"批量推理完成：{len(results)} 张图，共检出 {total_det} 个目标",
              "success")

    def _on_batch_failed(self, message: str):
        self.btn_batch.setEnabled(True)
        self._batch_thread = None
        self.result_stats.setText("批量推理失败")
        QMessageBox.critical(self, "批量运行失败", message)

    def _current_image_path(self) -> Path | None:
        item = self.image_tree.currentItem()
        if item is None:
            return None
        data = item.data(0, Qt.UserRole)
        return Path(data) if data else None

    # ================= 推理与可视化 =================
    def _current_params(self) -> tuple[float, float]:
        return self.conf_slider.value() / 100, self.iou_slider.value() / 100

    @staticmethod
    def _fmt_cost(seconds: float) -> str:
        """推理耗时文案：<1s 用毫秒，其余用秒（供状态摘要与测试断言）。"""
        ms = seconds * 1000.0
        return f"{ms:.0f} ms" if ms < 1000.0 else f"{seconds:.2f} s"

    def _infer_current(self):
        if self.model is None:
            self.result_stats.setText("未加载模型")
            return
        if getattr(self, "_batch_thread", None) is not None:
            # 批量运行中禁止 UI 线程并发推理（与批量线程共用同一模型对象，
            # 并发推理存在小概率卡死——用户批注 ToE3 的防护措施）
            return
        path = self._current_image_path()
        if path is None:
            return
        image = imread_unicode(path)
        if image is None:
            self.result_stats.setText(f"无法读取：{path.name}")
            return
        conf, iou = self._current_params()
        t0 = time.perf_counter()
        try:
            if self.chk_tiled.isChecked() and self.task != "classification":
                from ..engine.tiled_infer import tiled_predict

                prediction = tiled_predict(self.model, image, conf=conf,
                                           iou=iou)
            else:
                prediction = self.model.predict(image, conf=conf, iou=iou)
        except Exception as e:
            log.error("推理失败: %s: %s", type(e).__name__, e, exc_info=True)
            QMessageBox.critical(self, "推理失败", f"{type(e).__name__}: {e}")
            return
        self._last_infer_cost = time.perf_counter() - t0
        # 每次单图推理留痕（用户回传"推理变慢"排障依据：耗时/检出数/阈值可与环境对照）
        _boxes = getattr(prediction, "boxes", None)  # 判空不用真值（P150）
        log.info("单图推理 %s %dx%d 检出 %d 耗时 %.2fs（conf=%.2f iou=%.2f）",
                 path.name, image.shape[1], image.shape[0],
                 0 if _boxes is None else len(_boxes),
                 self._last_infer_cost, conf, iou)
        self._last_image = image
        self._last_prediction = prediction
        self._show_result(path.name, image, prediction)

    def _filter_prediction(self, prediction, class_idx):
        """按类别筛选预测结果；class_idx 为 None 时原样返回。"""
        if class_idx is None:
            return prediction
        keep = [i for i, l in enumerate(prediction.labels) if l == class_idx]
        masks = None
        if prediction.masks is not None:
            masks = [prediction.masks[i] for i in keep]
        from ..engine.models.base import Prediction
        return Prediction(
            labels=[prediction.labels[i] for i in keep],
            scores=[prediction.scores[i] for i in keep],
            boxes=[prediction.boxes[i] for i in keep],
            masks=masks,
            probs=prediction.probs,
        )

    def _filtered_prediction(self):
        class_idx = self.class_filter.currentData() if self.task != "classification" else None
        return self._filter_prediction(self._last_prediction, class_idx)

    def _redraw_last(self):
        """显示开关/筛选变化时直接重绘，不重新推理。"""
        if self._last_image is not None and self._last_prediction is not None:
            item = self.image_tree.currentItem()
            name = item.text(0) if item else ""
            self._show_result(name, self._last_image, self._last_prediction)


    def _show_result(self, name: str, image, raw_prediction):
        from ..engine.models.base import Prediction
        from .render import draw_predictions

        prediction = (self._filter_prediction(raw_prediction, self.class_filter.currentData())
                      if self.task != "classification" else raw_prediction)
        try:
            rendered = draw_predictions(image, prediction, self.classes, self.task,
                                        show_labels=self.show_labels.isChecked(),
                                        show_masks=self.show_masks.isChecked() and self.task == "segmentation")
            self._set_canvas_image(rendered)
            self._fill_result_list(prediction)
        except Exception as e:
            # 可视化异常只降级（显示原图+状态提示），不再整窗未捕获（T1-1 crash 面收窄）
            log.error("结果可视化失败: %s: %s", type(e).__name__, e, exc_info=True)
            try:
                self._set_canvas_image(image)
            except Exception:
                pass
            self.result_stats.setText(f"{name}\n结果可视化失败（已显示原图），详见日志")
            return
        cost_txt = (f"\n推理耗时：{self._fmt_cost(self._last_infer_cost)}"
                    if self._last_infer_cost is not None else "")
        total = len(raw_prediction.boxes) if self.task != "classification" else 0
        if self.task == "classification":
            top1 = self.classes[prediction.labels[0]] if prediction.labels else "?"
            score = prediction.scores[0] if prediction.scores else 0.0
            self.result_stats.setText(
                f"{name}\n识别为：{top1}（{score:.1%}）{cost_txt}")
        else:
            shown = len(prediction.boxes)
            extra = ""
            if self.class_filter.currentData() is not None and total:
                self.result_stats.setText(
                    f"{name}\n检出目标：{total} 个（筛选后显示 {shown} 个）{extra}{cost_txt}")
            else:
                self.result_stats.setText(
                    f"{name}\n检出目标：{shown} 个{extra}{cost_txt}")

    def _set_canvas_image(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pixmap = QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                                          rgb.shape[1] * 3, QImage.Format_RGB888))
        self.canvas.set_image(pixmap)

    def _fill_result_list(self, prediction):
        self.result_list.clear()
        if self.task == "classification":
            probs = prediction.probs or []
            ranked = sorted(enumerate(probs), key=lambda kv: -kv[1])[:5]
            for idx, prob in ranked:
                name = self.classes[idx] if idx < len(self.classes) else str(idx)
                self.result_list.addItem(f"{name}　{prob:.1%}")
            return
        show_area = self.show_area.isChecked()
        for i, (box, idx, score) in enumerate(zip(prediction.boxes, prediction.labels,
                                                  prediction.scores)):
            name = self.classes[idx] if idx < len(self.classes) else str(idx)
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            text = f"{name}　{score:.2f}　[{x1},{y1},{x2},{y2}]"
            if show_area:
                if prediction.masks is not None and i < len(prediction.masks):
                    text += f"　面积 {int(prediction.masks[i].sum())} px"
                else:
                    text += f"　面积 {int((x2 - x1) * (y2 - y1))} px²"
            self.result_list.addItem(text)


def main():
    from ..platform_entry import run_platform

    run_platform("testing", "测试平台", lambda: TestingWindow(), heavy=True)


if __name__ == "__main__":
    main()


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402
apply_auto_slot_module(globals())

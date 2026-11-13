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
    QPushButton, QScrollArea, QShortcut, QSlider, QStackedWidget, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
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

    def __init__(self, model, paths, conf, iou, flow, classes, task,
                 out_dir: Path | None = None, save_json: bool = True,
                 save_images: bool = False, jpg_quality: int = 95,
                 root: Path | None = None, model_text: str = "", parent=None,
                 json_format: str = "xcv", tiled: bool = False):
        super().__init__(parent)
        self._model = model
        self._paths = list(paths)
        self._conf, self._iou = conf, iou
        self._flow, self._classes = flow, classes
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

            # O13（V0.7）：批量预热——首图推理一次不计入统计（暖 CUDA 上下文
            # 与引擎缓存）。失败不阻塞批量本身。
            if total:
                try:
                    wimg = imread_unicode(self._paths[0])
                    if wimg is not None:
                        if self._tiled and self._task != "classification":
                            from ..engine.tiled_infer import tiled_predict

                            tiled_predict(self._model, wimg, conf=self._conf,
                                          iou=self._iou)
                        else:
                            self._model.predict(wimg, conf=self._conf,
                                                iou=self._iou)
                        log.info("批量预热完成（不计入耗时统计）: %s",
                                 self._paths[0].name)
                except Exception:  # noqa: BLE001
                    log.warning("批量预热失败（忽略）", exc_info=True)

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
                if self._flow:
                    from ..engine.flow.legacy_adapter import apply_flow_safe

                    pred, _info = apply_flow_safe(
                        pred, (w, h), self._flow, classes=self._classes or None)
                detections = []
                # O2 扩展（V0.7）：逐图灰度图（前景占比/灰度差维度的计算底图）
                gray = None
                if self._task in ("detection", "segmentation"):
                    gray = img if img.ndim == 2 else cv2.cvtColor(
                        img, cv2.COLOR_BGR2GRAY)
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
                    if gray is not None:
                        # 阈值面板多维度（_dims 内部键，不进统一 JSON）
                        from ..engine.flow.ops.det_filters import box_image_dims

                        det["_dims"] = box_image_dims(gray, [x1, y1, x2, y2])
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
            # O2/O19（V0.7）：GT 匹配统计——批量目录里同名 labelme json
            # 视为真值；无 GT 时面板与复判对话框给出说明。
            try:
                from ..engine.metrics.batch_match import collect_match_stats

                self.summary["match_stats"] = collect_match_stats(
                    results, iou_thr=self._iou)
            except Exception:  # noqa: BLE001  统计失败不影响批量结果
                log.warning("批量 GT 匹配统计失败", exc_info=True)
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
                flow_info={"removed": 0, "count_check": None}
                if self._flow else None)
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
                show_masks=(self._task in ("segmentation", "anomaly")))
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
        self._flow = None            # 检出过滤流程（项目文件夹中的 检出流程.json）
        self._flow_info = None
        self._flow_base = None       # O2：阈值应用落点
        self._loaded_weights_path = None
        self._last_batch_results = None
        self._last_match_stats = None
        self._batch_out_dir = None
        self._review_ctx = None
        self.task: str = ""          # classification / detection / segmentation / anomaly / ocr
        self.image_paths: list[Path] = []
        self._last_image = None      # 缓存最近一次推理的原图与结果，供显示开关/筛选切换
        self._last_prediction = None
        self._last_infer_cost: float | None = None   # 最近一次单图推理耗时（秒）
        self._load_thread: _ModelLoadThread | None = None

        self._build_ui()
        from ..status_bus import attach_platform_status

        attach_platform_status(self, "测试平台", "testing")
        # 勘误：底部状态栏徽章旁加 标准 / 项目复盘 页签
        self.btn_mode_std = QPushButton("标准")
        self.btn_mode_std.setObjectName("seg")
        self.btn_mode_std.setCheckable(True)
        self.btn_mode_std.setChecked(True)
        self.btn_mode_std.clicked.connect(lambda: self._switch_mode(0))
        self.btn_mode_review = QPushButton("项目复盘")
        self.btn_mode_review.setObjectName("seg")
        self.btn_mode_review.setCheckable(True)
        self.btn_mode_review.clicked.connect(lambda: self._switch_mode(1))
        self.statusBar().addWidget(self.btn_mode_std)
        self.statusBar().addWidget(self.btn_mode_review)
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
        # 勘误：复盘页临时缓存随平台关闭清理
        self.review_page.shutdown()
        event.accept()

    def _on_review_context(self, ctx: dict):
        """复盘批量完成：把上下文接到宿主共享属性——复判/AB/应用阈值/
        生成流程图四个处理器即可复用（权重/流程落点与标准批量一致）。"""
        self._last_batch_results = ctx["results"]
        self._last_match_stats = ctx["stats"]
        self._loaded_weights_path = ctx["weights"]
        self._flow_base = ctx["project_dir"]
        self._batch_out_dir = Path(ctx["out_dir"])
        self._review_ctx = ctx
        try:   # 复判/AB 复推使用与复盘批量相同的检出流程
            from ..engine.detection_flow import load_flow

            self._flow = load_flow(Path(ctx["project_dir"]) / "检出流程.json")
        except Exception:  # noqa: BLE001
            pass

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

        # 勘误（2026-11-13）：标准 / 项目复盘 双页（页签在底部状态栏）
        self._detail_stack = QStackedWidget()
        self._detail_stack.addWidget(self._build_standard_page())   # 0 标准
        from .review_page import ProjectReviewPage

        self.review_page = ProjectReviewPage()
        self.review_page.openReviewRequested.connect(self._open_batch_review)
        self.review_page.openCompareRequested.connect(self._open_model_compare)
        self.review_page.applyConfRequested.connect(self._apply_conf_to_flow)
        self.review_page.generateFlowRequested.connect(
            self._generate_threshold_flow)
        self.review_page.statusMessage.connect(
            lambda t: self.statusBar().showMessage(t, 10000))
        self.review_page.contextReady.connect(self._on_review_context)
        self._detail_stack.addWidget(self.review_page)              # 1 复盘
        # 上次会话残留的复盘缓存（关闭路径未走 closeEvent 时兜底清理）
        self.review_page.clear_stale_cache()
        root.addWidget(self._detail_stack, stretch=1)
        self.setCentralWidget(central)

    def _build_standard_page(self) -> QWidget:
        page = QWidget()
        body = QHBoxLayout(page)
        body.setContentsMargins(12, 12, 12, 12)
        body.setSpacing(12)
        body.addWidget(self._build_left_panel())
        body.addWidget(self._build_center(), stretch=1)
        body.addWidget(self._build_right_panel())
        return page

    def _switch_mode(self, index: int):
        """底部 标准 / 项目复盘 页签切换（互斥）。"""
        self._detail_stack.setCurrentIndex(index)
        self.btn_mode_std.setChecked(index == 0)
        self.btn_mode_review.setChecked(index == 1)

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
        self._loaded_weights_path = path  # O2/O20：阈值落点与回流目标解析
        task = getattr(model, "task", None)
        self.task = task.value if task is not None else ""
        self._flow = self._load_flow_for(path)
        self._warn_flow_mismatch()
        log.info("模型已加载: %s 任务=%s 类别数=%s 设备=%s 检出流程=%s",
                 path.name, self.task, len(self.classes), model.device,
                 "有" if self._flow else "无")
        # O14（V0.7）：推理形态显性化（社区版为 FP32 明文 .pt）
        form_txt = "FP32 权重（.pt）"
        self.model_info.setText(
            f"权重：{path.name}\n任务：{task.label if task is not None else '未知'}\n"
            f"类别数：{len(self.classes)}\n设备：{model.device}\n推理形态：{form_txt}"
        )
        self.header_model_info.setText(
            f"{path.name}　|　{task.label if task is not None else '未知'}　|　{len(self.classes)} 类"
        )
        is_cls = self.task == "classification"
        is_seg = self.task == "segmentation"
        is_anomaly = self.task == "anomaly"
        self.conf_slider.setEnabled(not is_cls)
        self.iou_slider.setEnabled(not is_cls)
        self.show_masks.setVisible(is_seg or is_anomaly)
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

    def _load_flow_for(self, weights_path: Path):
        """探测权重所属项目的检出过滤流程（项目文件夹中的 检出流程.json）。"""
        from ..engine import detection_flow as df

        for base in (weights_path.parent.parent, weights_path.parent):
            flow = df.load_flow(base / "检出流程.json")
            if flow:
                self._flow_base = base  # O2：阈值应用落点
                return flow
        # 无流程文件：项目目录仍作为阈值应用的默认落点
        self._flow_base = weights_path.parent.parent
        return None

    def _warn_flow_mismatch(self):
        """检出流程引用的类别与当前模型类别不一致时，状态栏提示。"""
        if not self._flow:
            return
        from ..engine import detection_flow as df

        missing = df.flow_class_mismatch(self._flow, self.classes or [])
        if missing:
            self.statusBar().showMessage(
                f"⚠ 检出流程引用的类别 {'、'.join(missing)} 不在当前模型类别中，"
                "相关类别过滤不会命中，请核对流程设置", 15000)
            log.warning("检出流程类别与模型不一致: %s（模型类别 %s）",
                        missing, self.classes)

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
            self.model, files, conf, iou, self._flow, self.classes or [],
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
                # O13（V0.7）：min/avg 补齐 + 预热口径说明
                lines.append(
                    f"单图耗时 min={lat['min_ms']:.0f} / p50={lat['p50_ms']:.0f}"
                    f" / avg={lat['avg_ms']:.0f} / max={lat['max_ms']:.0f} ms"
                    "（首张已自动预热，不计入统计）")
            # O15（V0.7）：耗时离散度告警（单图 > 中位数×3 疑似掉频/热节流）
            outliers = summary.get("slow_outliers") or {}
            if outliers.get("count"):
                lines.append(
                    f"⚠ 疑似 GPU 掉频/热节流：{outliers['count']} 张单图耗时"
                    f" > 中位数（{outliers['median_ms']:.0f} ms）×3——"
                    "查电源/驱动设置")
            # O13（V0.7）：显存水位（GPU 环境才显示）
            try:
                import torch
                if torch.cuda.is_available():
                    free_b, total_b = torch.cuda.mem_get_info()
                    lines.append(
                        f"显存占用：{(total_b - free_b) / 1048576:.0f} MB"
                        f" / 共 {total_b / 1048576:.0f} MB")
            except Exception:  # noqa: BLE001  无 CUDA/驱动异常不显示
                pass
            # 摘要 .json 已由批量线程落盘（检出结果/摘要.json）
        if settings is not None and settings.save_json and out_dir is not None:
            lines.append(f"检出 JSON（每图一个，同名对应）：{out_dir / '检出结果'}")
        if settings is not None and settings.save_images and out_dir is not None:
            lines.append(f"渲染图：{out_dir / '可视化'}")
        self.result_stats.setText("\n".join(lines))
        self.statusBar().showMessage(f"批量运行完成，输出目录：{out_dir}", 10000)
        # 标准页不再挂分数面板/复判入口——批量统计仅落盘，复盘走「项目复盘」页
        self._last_match_stats = summary.get("match_stats") if summary else None
        self._last_batch_results = results
        from ..ui_widgets import toast

        toast(self,
              f"批量推理完成：{len(results)} 张图，共检出 {total_det} 个目标",
              "success")

    # ================= O2/O19/O20（V0.7）：批量评估闭环 =================
    def _generate_threshold_flow(self, dims: dict):
        """O2 扩展：面板启用维度阈值 → 生成流程图模板。"""
        w = getattr(self, "_loaded_weights_path", None)
        if not w:
            QMessageBox.information(
                self, "生成流程图模板",
                "当前模型未关联权重文件（无法定位模型存放文件夹）；"
                "请通过「加载模型」选择权重后再生成")
            return
        weights = Path(w)
        model_dir = weights.parent.parent
        model_name = weights.stem
        from ..engine.flow.threshold_flow import save_threshold_flow

        try:
            path = save_threshold_flow(model_dir, model_name, dims,
                                       str(weights),
                                       task=str(self.task or "detection"))
        except OSError as e:
            QMessageBox.warning(self, "生成流程图模板", f"写入流程文件失败：{e}")
            return
        dims_text = "、".join(f"{k}={v:g}" for k, v in sorted(dims.items()))
        self.statusBar().showMessage(
            f"流程图模板已生成：{path}（启用维度：{dims_text}）", 15000)
        from ..ui_widgets import toast

        toast(self, "流程图模板已生成，可在流程图平台打开使用", "success")

    def _apply_conf_to_flow(self, conf: float):
        """面板阈值 → 项目「检出流程.json」置信度节点（O2 应用 + O18 留痕）。"""
        base = getattr(self, "_flow_base", None)
        if base is None:
            w = getattr(self, "_loaded_weights_path", None)
            if w is not None:
                base = Path(w).parent.parent
        if base is None:
            QMessageBox.information(
                self, "应用阈值",
                "当前模型未关联训练项目（无法定位检出流程文件）；"
                "请通过「加载该项目最新模型」或权重所在项目再应用")
            return
        from ..engine import detection_flow as df

        flow_path = Path(base) / "检出流程.json"
        flow = df.load_flow(flow_path) or df.new_flow()
        node = next((n for n in flow.get("nodes", [])
                     if n.get("op") == "confidence"), None)
        if node is None:
            node = {"op": "confidence", "params": {}}
            flow.setdefault("nodes", []).insert(0, node)
        stats = getattr(self, "_last_match_stats", None) or {}
        node.setdefault("params", {})
        node["params"]["min"] = round(float(conf), 2)
        # O18 留痕：normalize_flow 保留节点的额外参数键（apply_flow 不读）
        node["params"]["note"] = (
            f"{time.strftime('%Y-%m-%d %H:%M')} 测试平台阈值面板：基于批量 "
            f"{stats.get('n_gt_images', 0)} 张有标注图"
            f"（GT {stats.get('gt_total', 0)} 实例）")
        try:
            df.save_flow(flow_path, flow)
        except OSError as e:
            QMessageBox.warning(self, "应用阈值", f"写入检出流程失败：{e}")
            return
        self._flow = flow  # 当前会话立即生效
        self.statusBar().showMessage(
            f"默认检出阈值已应用：conf={conf:.2f} → {flow_path}", 10000)
        from ..ui_widgets import toast

        toast(self, f"已应用为项目默认检出 conf={conf:.2f}", "success")

    def _resolve_dataset_dir(self):
        """权重所属项目的数据集目录（回流默认目标；找不到返回 None）。"""
        w = getattr(self, "_loaded_weights_path", None)
        if w is None:
            return None
        import json as _json

        model_dir = Path(w).parent.parent  # run 目录的上级 = 模型储存目录
        try:
            pj = model_dir / "project.json"
            if pj.is_file():
                data = _json.loads(pj.read_text(encoding="utf-8"))
                d = data.get("dataset_dir")
                if d and Path(d).is_dir():
                    return d
            from ..training.project import list_projects, load_project

            for name in list_projects():
                try:
                    p = load_project(name)
                except Exception:
                    continue
                if Path(p.model_dir).resolve() == model_dir.resolve():
                    return (p.dataset_dir
                            if Path(p.dataset_dir).is_dir() else None)
        except Exception:
            pass
        return None

    def _open_batch_review(self):
        results = getattr(self, "_last_batch_results", None)
        if not results:
            QMessageBox.information(self, "复判", "请先完成一次批量推理")
            return
        out_dir = getattr(self, "_batch_out_dir", None)
        review_path = (Path(out_dir) / "检出结果" / "复判.json"
                       if out_dir else Path.cwd() / "复判.json")
        from .batch_review_dialog import BatchReviewDialog

        old = getattr(self, "_review_dialog", None)
        if old is not None:
            try:
                old.close()
            except Exception:  # noqa: BLE001
                pass
        self._review_dialog = BatchReviewDialog(
            results, getattr(self, "_last_match_stats", None), review_path,
            dataset_dir=self._resolve_dataset_dir(), parent=self)
        self._review_dialog.show()

    def _open_model_compare(self):
        """O21（V0.7）：模型 A/B 比对窗（非模态；B 模型复推在后台线程）。"""
        results = getattr(self, "_last_batch_results", None)
        if not results:
            QMessageBox.information(self, "A/B 比对", "请先完成一次批量推理")
            return
        ctx = getattr(self, "_review_ctx", None)
        if ctx:
            conf, iou = float(ctx["conf"]), float(ctx["iou"])
            classes = list(ctx["classes"] or [])
        else:
            conf, iou = self._current_params()
            classes = self.classes or []
        from .model_compare_dialog import ModelCompareDialog

        old = getattr(self, "_ab_dialog", None)
        if old is not None:
            try:
                old.close()
            except Exception:  # noqa: BLE001
                pass
        w = getattr(self, "_loaded_weights_path", None)
        self._ab_dialog = ModelCompareDialog(
            results, getattr(self, "_batch_out_dir", None), conf, iou,
            self._flow, classes,
            model_a_name=Path(w).name if w else "", parent=self)
        self._ab_dialog.show()

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
        if self._flow and self.task != "classification":
            from ..engine.flow.legacy_adapter import apply_flow_safe

            try:
                prediction, flow_info = apply_flow_safe(
                    prediction, (image.shape[1], image.shape[0]), self._flow,
                    classes=self.classes or None)
                self._flow_info = flow_info
                log.info("检出流程: 原始 %d → 过滤后 %d（判定 %s）",
                         len(prediction.boxes) + flow_info.get("removed", 0),
                         len(prediction.boxes), flow_info.get("count_check"))
            except Exception as e:
                # 流程应用失败只降级为不过滤，绝不打断推理
                self._flow_info = None
                log.error("检出流程应用失败，已跳过过滤: %s: %s",
                          type(e).__name__, e, exc_info=True)
                self.result_stats.setText("检出流程应用失败（已跳过过滤），详见日志")
        else:
            self._flow_info = None
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
                                        show_masks=self.show_masks.isChecked()
                                        and self.task in ("segmentation", "anomaly"))
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
            if self._flow_info:
                extra += f"（流程移除 {self._flow_info.get('removed', 0)} 个）"
                verdict = self._flow_info.get("count_check")
                if verdict:
                    extra += f"\n数量判定：{verdict}"
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
        texts = getattr(prediction, "texts", None)   # A2（V0.7）：OCR 识别文本
        for i, (box, idx, score) in enumerate(zip(prediction.boxes, prediction.labels,
                                                  prediction.scores)):
            name = self.classes[idx] if idx < len(self.classes) else str(idx)
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            text = f"{name}　{score:.2f}　[{x1},{y1},{x2},{y2}]"
            if texts is not None and i < len(texts):
                text += f"　「{texts[i]}」"
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

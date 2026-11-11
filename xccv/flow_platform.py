"""流程图平台页：主界面左侧导航的「流程图」页（与 AI 平台同等级）。

布局（参考常见流程图界面）：
- 顶栏：数据路径 + 选择文件夹；
- 左栏：图片搜索 + 数量 + 数据集文件夹树（点击图片预览）；
- 中栏：原图 / 可视化 预览；
- 右栏：模型目录、流程 JSON 选择与新建、运行流程 / 编辑流程 /
  打包 / 解包、批量运行 / 停止批量。

画布编辑仍由独立的 HTML 画布窗口（FlowGraphWebWindow）承担；流程 JSON
为磁盘事实源（编辑/运行/打包前重读）。批量运行 = 按流程对数据路径下
全部图片推理，导出检出结果 JSON 与可视化图（模型目录/批量结果_时间戳）。

运行：python -m xccv.flow_platform
"""

if __name__ == "__main__":  # V0.6 T1-3：-m 直启先亮加载中界面，再做下方重型模块级导入
    from xccv.platform_entry import early_splash

    early_splash("flowmain", "流程图平台")

import json
import logging
import time
from pathlib import Path

from PyQt5.QtCore import QFileSystemWatcher, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QSplitter, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from .engine import detection_flow as df
from .engine import flow_package as fp
from .engine.flow.ir import ImageData
from .engine.io_utils import imread_unicode
from .gui_utils import auto_cleanup_thread
from .ui_style import APP_QSS, apply_app_theme, bind_style, icon_path, token  # noqa: F401
from .modeltools.flow_pack import (
    _EMPTY_FLOW, _materialize_xvap, _write_flow_json,
)

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _list_images(root: Path):
    """递归列出数据路径下的图片（相对路径排序，文件夹分组用相对目录）。"""
    if not root.is_dir():
        return []
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."):
            out.append(p.relative_to(root))
    return sorted(out)


class _FlowBatchThread(QThread):
    """批量运行线程：逐张按流程推理，导出检出结果 JSON 与可视化图。"""

    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, flow: dict, model_file: str, images: list, root: Path,
                 out_dir: Path, parent=None):
        super().__init__(parent)
        self._flow = flow
        self._model_file = model_file
        self._images = images
        self._root = root
        self._out_dir = out_dir
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from .engine.models.registry import load_model
            from .engine.result_json import build_result_json, save_result_json
            from .testing.render import draw_predictions

            model = load_model(self._model_file)
            chain = df.graph_to_chain(self._flow)
            classes = getattr(model, "classes", None)
            task_v = getattr(getattr(model, "task", None), "value", "detection")
            vis_dir = self._out_dir / "可视化"
            json_dir = self._out_dir / "检出结果"
            vis_dir.mkdir(parents=True, exist_ok=True)
            json_dir.mkdir(parents=True, exist_ok=True)
            results = []
            for i, rel in enumerate(self._images):
                if self._stop:
                    break
                _t0 = time.perf_counter()
                img = imread_unicode(str(self._root / rel))
                if img is None:
                    results.append({"image": str(rel), "error": "无法读取"})
                    self.progress.emit(i + 1, len(self._images))
                    continue
                try:
                    pred = model.predict(img)
                    from .engine.flow.legacy_adapter import apply_flow_safe

                    filt, info = apply_flow_safe(pred, (img.shape[1], img.shape[0]),
                                                 chain, classes=classes)
                    n_det = len(filt.boxes)
                    # 统一检出结果 JSON（Q1/Q2 契约）：每图一个同名 .json，
                    # 镜像相对目录；分类任务经 classification_topk 导出
                    payload = build_result_json(
                        task=task_v,
                        source={"file": Path(rel).name, "path": str(self._root / rel),
                                "width": img.shape[1], "height": img.shape[0]},
                        model={"file": self._model_file,
                               "name": Path(self._model_file).name},
                        inference={},
                        prediction=filt, classes=classes or [], flow_info=info)
                    out_json = json_dir / rel
                    out_json.parent.mkdir(parents=True, exist_ok=True)
                    out_json = out_json.with_suffix(".json")
                    save_result_json(out_json, payload)
                    rendered = draw_predictions(img, filt, model.classes or [],
                                                task_v, show_labels=True,
                                                show_masks=(task_v == "segmentation"))
                    out_img = vis_dir / rel
                    out_img.parent.mkdir(parents=True, exist_ok=True)
                    import cv2

                    cv2.imwrite(str(out_img.with_suffix(".jpg")), rendered,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                    results.append({"image": str(rel),
                                    "count_check": info.get("count_check") or "",
                                    "detections": n_det})
                except Exception as e:  # 单张失败不中断批量
                    log.error("批量推理失败 %s: %s", rel, e, exc_info=True)
                    results.append({"image": str(rel), "error": f"{type(e).__name__}: {e}"})
                # 仪表化（E3 排查）：单图超 5 秒记录慢图；每 20 张记一次进度，
                # 若出现"卡死"，日志能定位停在即可疑的那张图
                cost = time.perf_counter() - _t0
                if cost > 5.0:
                    log.warning("批量单图耗时 %.1f 秒: %s", cost, rel)
                if (i + 1) % 20 == 0:
                    log.info("批量运行进度 %d/%d", i + 1, len(self._images))
                self.progress.emit(i + 1, len(self._images))
            payload = {"model": self._model_file, "total": len(self._images),
                       "flow": self._flow, "results": results}
            (self._out_dir / "批量摘要.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            self.finished_ok.emit(str(self._out_dir))
        except Exception as e:
            log.error("批量运行失败: %s", e, exc_info=True)
            self.failed.emit(f"{type(e).__name__}: {e}")


class _XcvBatchThread(QThread):
    """新算子（含纯传统）流程批量线程：逐图跑新引擎。

    产物与 AI 检出批量一致（Q1 统一口径）：检出结果/<相对路径>/<图名>.json
    （统一 schema，几何/测量在扩展数组、剔除无效信息）+ 可视化/<相对路径>/
    <图名>.jpg（原图 + 结果蒙版叠加）+ 批量摘要.json。
    """

    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, flow: dict, images: list, root: Path, out_dir: Path,
                 flow_name: str = "", save_json: bool = True,
                 save_images: bool = True, jpg_quality: int = 95):
        super().__init__()
        self._norm = _normalize_types(flow)
        self._images = images
        self._root = Path(root)
        self._out_dir = Path(out_dir)
        self._flow_name = flow_name
        self._save_json = save_json
        self._save_images = save_images
        self._jpg_quality = jpg_quality
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from .engine.flow.snapshot import overlay_result
            from .engine.flow.ir import ImageData as _Img
            from .engine.result_json import (build_flow_result_json,
                                             classification_topk,
                                             detection_rows, save_result_json,
                                             verdict_from_node_results)
            from .modeltools.flow_session import FlowSession

            json_dir = self._out_dir / "检出结果"
            vis_dir = self._out_dir / "可视化"
            if self._save_json:
                json_dir.mkdir(parents=True, exist_ok=True)
            if self._save_images:
                vis_dir.mkdir(parents=True, exist_ok=True)
            session = FlowSession()
            results = []
            for i, rel in enumerate(self._images):
                if self._stop:
                    break
                img_path = self._root / rel
                try:
                    norm = json.loads(json.dumps(self._norm))  # 深拷贝
                    has_loader = False
                    for n in norm.get("nodes", []):
                        if str(n.get("type", "")) == "image_loader":
                            n.setdefault("properties", {})["path"] = str(img_path)
                            has_loader = True
                    if not has_loader:
                        raise ValueError("流程缺少「读入图片」节点")
                    summary = session.run(norm, label="批量",
                                          flow_dir=self._out_dir,
                                          source_name=Path(rel).name)
                    if summary["errors"]:
                        raise RuntimeError(
                            "；".join(f"节点#{k}: {v}"
                                      for k, v in summary["errors"].items()))
                    ex_results = session.executor.results
                    img = imread_unicode(str(img_path))
                    # 按拓扑序收集全图输出（末节点可能是「保存结果」，
                    # 检出/几何/测量可能在中游节点上——与 result_output
                    # 节点的收集口径一致）
                    det = None
                    geo_list = []
                    meas_list = []
                    last_outs = {}
                    for nid in sorted(ex_results):
                        r = ex_results[nid]
                        if not r.ok:
                            continue
                        last_outs = r.outputs
                        for v in r.outputs.values():
                            if isinstance(v, _Img):
                                continue
                            elif hasattr(v, "kind"):
                                geo_list.append(v)
                            elif hasattr(v, "boxes") and v is not None:
                                det = det or v
                            elif hasattr(v, "to_json"):
                                meas_list.append(v)
                    det_list, cls_list = [], []
                    if det is not None:
                        # 类别名表：AI 节点 model_classes / 传统算子 flow_class_names
                        classes = (session.ctx.get("model_classes")
                                   or session.ctx.get("flow_class_names") or [])
                        # 与 result_output 节点共用 result_json 单一实现
                        # （P91：分类流程的 classifications 不再丢失）
                        det_list = detection_rows(det, classes)
                        cls_list = classification_topk(det, classes)
                    payload = build_flow_result_json(
                        source={"file": Path(rel).name,
                                "path": str(img_path),
                                "width": img.shape[1] if img is not None else None,
                                "height": img.shape[0] if img is not None else None},
                        detections=det_list,
                        classifications=cls_list,
                        geometries=geo_list,
                        measurements=meas_list,
                        flow_name=self._flow_name,
                        verdict=verdict_from_node_results(ex_results))
                    out_json = json_dir / rel
                    if self._save_json:
                        out_json.parent.mkdir(parents=True, exist_ok=True)
                        save_result_json(out_json.with_suffix(".json"), payload)
                    # 蒙版可视化（原图 + 末节点结果叠加）
                    if img is not None and self._save_images:
                        base = next((v.mat for v in last_outs.values()
                                     if isinstance(v, _Img)), img)
                        try:
                            rendered = overlay_result(base, last_outs)
                        except Exception:
                            rendered = base
                        out_img = vis_dir / rel
                        out_img.parent.mkdir(parents=True, exist_ok=True)
                        import cv2

                        cv2.imwrite(str(out_img.with_suffix(".jpg")), rendered,
                                    [cv2.IMWRITE_JPEG_QUALITY, self._jpg_quality])
                    results.append({"image": str(rel), "detections": len(det_list)})
                except Exception as e:  # 单张失败不中断批量
                    log.error("批量运行失败 %s: %s", rel, e, exc_info=True)
                    results.append({"image": str(rel),
                                    "error": f"{type(e).__name__}: {e}"})
                self.progress.emit(i + 1, len(self._images))
            (self._out_dir / "批量摘要.json").write_text(
                json.dumps({"flow": self._flow_name, "total": len(self._images),
                            "results": results}, ensure_ascii=False, indent=1),
                encoding="utf-8")
            self.finished_ok.emit(str(self._out_dir))
        except Exception as e:
            log.error("批量运行失败: %s", e, exc_info=True)
            self.failed.emit(f"{type(e).__name__}: {e}")


def _normalize_types(flow: dict) -> dict:
    """画布类型串 xccv2/<名> → 注册表键（flow_session 同名逻辑的本地引用）。"""
    from .modeltools.flow_session import _normalize_types as _n

    return _n(flow)


class _TreeScanWorker(QThread):
    """数据目录扫描工作线程（T2-3）：只产出相对路径列表，建树仍在主线程
    （坑 P140 纪律：线程不碰任何 QWidget/QTreeWidgetItem）。"""

    done = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, root: Path, parent=None):
        super().__init__(parent)
        self._root = Path(root)

    def run(self):
        try:
            self.done.emit([str(p) for p in _list_images(self._root)])
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class FlowPage(QWidget):
    """主界面「流程图」页：数据树 + 预览 + 流程编辑与运行面板。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_root: Path | None = None
        self.current_flow_path: Path | None = None
        self._canvas = None               # HTML 画布窗口（复用）
        self._run_model = None
        self._run_model_path = None
        self._preview_original: QImage | None = None
        self._preview_rendered: QImage | None = None
        self._preview_mode = "原图"
        self._batch_thread: _FlowBatchThread | None = None
        self._scan_thread: _TreeScanWorker | None = None   # T2-3 目录扫描线程
        self._scan_gen = 0                                  # 丢弃过期扫描结果
        self._thumb_cache: dict = {}                        # path->QImage LRU（T2-3）
        self._thumb_cache_order: list = []

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 10)
        root.setSpacing(10)

        # ---- 顶栏：数据路径 ----
        top = QHBoxLayout()
        top.addWidget(QLabel("数据路径:"))
        self.data_edit = QLineEdit()
        self.data_edit.setPlaceholderText("选择包含图片的数据集文件夹（含子文件夹）")
        self.data_edit.returnPressed.connect(self._load_data_root)
        top.addWidget(self.data_edit, stretch=1)
        btn_pick_dir = QPushButton("选择文件夹")
        btn_pick_dir.clicked.connect(self._browse_data_dir)
        top.addWidget(btn_pick_dir)
        root.addLayout(top)

        # ---- 三栏主体 ----
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(6)
        self._tree_total = 0   # 图片树未载入前搜索不报错（set_search_query 可先到）
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索文件/文件夹")
        self.search_edit.textChanged.connect(self._filter_tree)
        left_lay.addWidget(self.search_edit)
        self.count_label = QLabel("图片数量: 0/0")
        self.count_label.setObjectName("hint")
        left_lay.addWidget(self.count_label)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self._on_tree_click)
        left_lay.addWidget(self.tree, stretch=1)
        split.addWidget(left)

        center = QWidget()
        center_lay = QVBoxLayout(center)
        center_lay.setContentsMargins(0, 0, 0, 0)
        center_lay.setSpacing(6)
        mode_row = QHBoxLayout()
        mode_row.addStretch(1)
        self.btn_orig = QPushButton("原图")
        self.btn_orig.setCheckable(True)
        self.btn_orig.setChecked(True)
        self.btn_orig.clicked.connect(lambda: self._set_preview_mode("原图"))
        self.btn_vis = QPushButton("可视化")
        self.btn_vis.setCheckable(True)
        self.btn_vis.clicked.connect(lambda: self._set_preview_mode("可视化"))
        mode_row.addWidget(self.btn_orig)
        mode_row.addWidget(self.btn_vis)
        center_lay.addLayout(mode_row)
        self.preview_label = QLabel("选择左侧图片预览；「运行流程」后在「可视化」查看结果")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(420, 320)
        bind_style(self.preview_label, lambda: (
            f"border:1px solid {token('border')}; border-radius:8px;"
            f" background:{token('readonly_bg')}; color:{token('text_3')};"))
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidget(self.preview_label)
        self.preview_scroll.setWidgetResizable(True)
        center_lay.addWidget(self.preview_scroll, stretch=1)
        split.addWidget(center)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(12, 0, 0, 0)
        right_lay.setSpacing(10)
        panel = QLabel("流程编辑与运行")
        panel.setObjectName("appTitle")
        right_lay.addWidget(panel)

        model_row = QHBoxLayout()
        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setPlaceholderText("模型目录（流程 JSON / .xvap 所在位置）")
        self.model_dir_edit.returnPressed.connect(self._refresh_flow_combo)
        model_row.addWidget(self.model_dir_edit, stretch=1)
        # 📁 emoji 在雅黑下被裁切（用户实测 Q6-1），改用文字按钮；
        # 宽度不固定（用户回执 Q4：字号放大后固定 56px 会裁切"浏览/新建"）
        btn_model_dir = QPushButton("浏览")
        btn_model_dir.setToolTip("选择模型目录")
        btn_model_dir.clicked.connect(self._browse_model_dir)
        model_row.addWidget(btn_model_dir)
        right_lay.addLayout(model_row)

        flow_row = QHBoxLayout()
        self.flow_combo = QComboBox()
        self.flow_combo.currentIndexChanged.connect(self._on_flow_selected)
        flow_row.addWidget(self.flow_combo, stretch=1)
        btn_new_flow = QPushButton("新建")
        btn_new_flow.setToolTip("新建流程图")
        btn_new_flow.clicked.connect(self._create_flow)
        flow_row.addWidget(btn_new_flow)
        right_lay.addLayout(flow_row)

        grid = QGridLayout()
        grid.setSpacing(8)
        self.btn_run = QPushButton("运行流程")
        self.btn_run.clicked.connect(self._run_flow)
        self.btn_edit = QPushButton("编辑流程")
        self.btn_edit.clicked.connect(self._edit_flow)
        self.btn_pack = QPushButton("打包流程")
        self.btn_pack.clicked.connect(self._pack_flow)
        self.btn_unpack = QPushButton("解包流程")
        self.btn_unpack.clicked.connect(self._unpack_flow)
        for i, b in enumerate((self.btn_run, self.btn_edit,
                               self.btn_pack, self.btn_unpack)):
            b.setMinimumHeight(42)
            grid.addWidget(b, i // 2, i % 2)
        right_lay.addLayout(grid)

        batch_row = QHBoxLayout()
        self.btn_batch = QPushButton("批量运行")
        self.btn_batch.clicked.connect(self._start_batch)
        self.btn_batch_stop = QPushButton("停止批量")
        self.btn_batch_stop.setEnabled(False)
        self.btn_batch_stop.clicked.connect(self._stop_batch)
        batch_row.addWidget(self.btn_batch)
        batch_row.addWidget(self.btn_batch_stop)
        right_lay.addLayout(batch_row)

        self.flow_hint = QLabel("流程以 .json 保存于磁盘；「打包流程」把流程与所选模型一并封入"
                                "单个 .xvap 文件，便于分发与复用。模型节点需先在画布中选择权重文件。")
        self.flow_hint.setObjectName("hint")
        self.flow_hint.setWordWrap(True)
        right_lay.addWidget(self.flow_hint)
        right_lay.addStretch(1)
        split.addWidget(right)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([260, 560, 320])
        root.addWidget(split, stretch=1)

    # ================= 数据路径 / 树 =================
    def _browse_data_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择数据路径")
        if d:
            self.data_edit.setText(d)
            self._load_data_root()

    def _load_data_root(self):
        d = Path(self.data_edit.text().strip())
        if not d.is_dir():
            self.count_label.setText("图片数量: 路径不存在")
            return
        self.data_root = d
        self._rebuild_tree()

    def _rebuild_tree(self):
        self.tree.clear()
        self._thumb_cache.clear()
        self._thumb_cache_order.clear()
        if self.data_root is None:
            self._tree_total = 0
            self.count_label.setText("图片数量: 0/0")
            return
        # T2-3：目录扫描移后台线程（千图目录不再冻结主线程）；
        # 完成后回主线程一次性建树（QTreeWidgetItem 只能在主线程创建）
        self.count_label.setText("图片数量: 扫描中…")
        self._scan_gen += 1
        gen = self._scan_gen
        # 无父对象（T2-3）：页面可能在线程运行中被关闭（parented 运行线程
        # 会连带销毁——Qt 崩溃）；完成后经 finished→deleteLater 自清理
        worker = _TreeScanWorker(self.data_root)
        worker.done.connect(lambda rels, g=gen: self._fill_tree(rels, g))
        worker.failed.connect(lambda msg, g=gen: self._scan_failed(msg, g))
        worker.finished.connect(self._scan_thread_done)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self._scan_thread = worker

    def _scan_thread_done(self):
        w = self.sender()
        if w is self._scan_thread:
            self._scan_thread = None
        w.deleteLater()  # A-3 纪律：finished→deleteLater+置 None

    def _scan_failed(self, msg: str, gen: int):
        if gen != self._scan_gen:
            return  # 过期扫描（用户已换目录）
        self._tree_total = 0
        self.count_label.setText(f"图片数量: 扫描失败（{msg}）")

    def _fill_tree(self, rels: list, gen: int):
        if gen != self._scan_gen:
            return  # 过期扫描（用户已换目录）
        dirs: dict[str, QTreeWidgetItem] = {}

        def _dir_item(rel_dir: str) -> QTreeWidgetItem:
            if not rel_dir:
                return self.tree.invisibleRootItem()
            if rel_dir in dirs:
                return dirs[rel_dir]
            parent_rel = str(Path(rel_dir).parent)
            item = QTreeWidgetItem([Path(rel_dir).name])
            item.setData(0, Qt.UserRole, ("dir", rel_dir))
            parent = _dir_item("" if parent_rel == "." else parent_rel)
            parent.addChild(item)
            dirs[rel_dir] = item
            return item

        self.tree.setUpdatesEnabled(False)
        try:
            for rel in rels:
                rel = Path(rel)  # 工作线程给的是字符串路径
                parent = _dir_item(str(rel.parent) if str(rel.parent) != "." else "")
                it = QTreeWidgetItem([rel.name])
                it.setData(0, Qt.UserRole, ("img", str(rel)))
                parent.addChild(it)
        finally:
            self.tree.setUpdatesEnabled(True)
        self.tree.expandToDepth(0)
        total = len(rels)
        self.count_label.setText(f"图片数量: {total}/{total}")
        self._tree_total = total

    def set_search_query(self, text: str) -> None:
        """主界面全局搜索联动：过滤本页图片树（主界面搜索框与本页框同步）。"""
        if getattr(self, "search_edit", None) is None:
            return
        if self.search_edit.text() != (text or ""):
            self.search_edit.setText(text or "")   # textChanged → _filter_tree

    def _filter_tree(self, text: str):
        text = text.strip().lower()

        def match(item: QTreeWidgetItem) -> bool:
            kind, val = item.data(0, Qt.UserRole) or ("", "")
            hit = text in val.lower()
            child_hit = False
            for i in range(item.childCount()):
                if match(item.child(i)):
                    child_hit = True
            item.setHidden(not (hit or child_hit or not text))
            return hit or child_hit

        root_item = self.tree.invisibleRootItem()
        visible = 0
        for i in range(root_item.childCount()):
            if match(root_item.child(i)):
                visible += 1
        if not text:
            for i in range(root_item.childCount()):
                root_item.child(i).setHidden(False)
        self.count_label.setText(
            f"图片数量: {self._tree_total if not text else visible}/{self._tree_total}")

    def _selected_image(self) -> Path | None:
        it = self.tree.currentItem()
        if it is None:
            return None
        kind, val = it.data(0, Qt.UserRole) or ("", "")
        if kind != "img" or self.data_root is None:
            return None
        return self.data_root / val

    _THUMB_CACHE_MAX = 32  # 预览解码缓存上限（T2-3：来回点选不重复解码）

    def _on_tree_click(self, item: QTreeWidgetItem, _col: int):
        kind, val = item.data(0, Qt.UserRole) or ("", "")
        if kind != "img":
            return
        path = self.data_root / val
        key = str(path)
        cached = self._thumb_cache.get(key)
        if cached is not None:
            # LRU：命中即移到最新
            try:
                self._thumb_cache_order.remove(key)
            except ValueError:
                pass
            self._thumb_cache_order.append(key)
            self._preview_original = cached   # 缓存里已是 QImage，无需再解码
            self._preview_rendered = None
            self._apply_preview()
            return
        img = imread_unicode(key)
        if img is None:
            self.window().statusBar().showMessage(f"无法读取图片：{val}", 5000)
            return
        import cv2

        qimg = QImage(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).data,
                      img.shape[1], img.shape[0], 3 * img.shape[1],
                      QImage.Format_RGB888).copy()  # copy 脱离原缓冲再缓存
        self._thumb_cache[key] = qimg
        self._thumb_cache_order.append(key)
        while len(self._thumb_cache_order) > self._THUMB_CACHE_MAX:
            old = self._thumb_cache_order.pop(0)
            self._thumb_cache.pop(old, None)
        self._show_mat(img, rendered=False)

    def _show_mat(self, img_bgr, rendered: bool = False):
        import cv2

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        if rendered:
            self._preview_rendered = qimg
        else:
            self._preview_original = qimg
            self._preview_rendered = None
        self._apply_preview()

    def _apply_preview(self):
        qimg = (self._preview_rendered
                if self._preview_mode == "可视化" and self._preview_rendered
                else self._preview_original)
        if qimg is None:
            return
        pm = QPixmap.fromImage(qimg)
        bind_style(self.preview_label, lambda: (
            f"border:1px solid {token('border')}; border-radius:8px;"
            f" background:{token('readonly_bg')};"))
        self.preview_label.setText("")
        area = self.preview_scroll.viewport().size()
        if area.width() > 40 and area.height() > 40:
            pm = pm.scaled(area, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(pm)
        self.btn_orig.setChecked(self._preview_mode == "原图")
        self.btn_vis.setChecked(self._preview_mode == "可视化")

    def _set_preview_mode(self, mode: str):
        self._preview_mode = mode
        self._apply_preview()

    # ================= 模型目录 / 流程选择 =================
    def _browse_model_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择模型目录（流程所在位置）")
        if d:
            self.model_dir_edit.setText(d)
            self._refresh_flow_combo()

    def _watch_model_dir(self):
        """监视流程所在目录（Q7：画布保存新流程后下拉实时刷新，无需重选）。"""
        if not hasattr(self, "_dir_watcher"):
            self._dir_watcher = QFileSystemWatcher(self)
            self._dir_watcher.directoryChanged.connect(self._on_watched_dir_changed)
            # 防抖：保存/编辑会触发多次 directoryChanged
            self._combo_timer = QTimer(self)
            self._combo_timer.setSingleShot(True)
            self._combo_timer.setInterval(300)
            self._combo_timer.timeout.connect(self._refresh_flow_combo)
        if hasattr(self, "_watched_dir") and self._watched_dir:
            self._dir_watcher.removePath(self._watched_dir)
        d = self.model_dir_edit.text().strip()
        if d and Path(d).is_dir():
            self._dir_watcher.addPath(d)
            self._watched_dir = d

    def _on_watched_dir_changed(self, _path: str):
        self._combo_timer.start()

    def _refresh_flow_combo(self):
        self.flow_combo.blockSignals(True)
        self.flow_combo.clear()
        d = Path(self.model_dir_edit.text().strip())
        found = []
        if d.is_dir():
            found = sorted(p for p in d.iterdir()
                           if p.suffix.lower() in (".json", ".xvap")
                           and not p.name.startswith("~"))
        for p in found:
            self.flow_combo.addItem(p.name, str(p))
        self.flow_combo.blockSignals(False)
        if found:
            self.flow_combo.setCurrentIndex(0)
            # 首项必须手动激活（Q4）：populate 期间信号被屏蔽，默认选中
            # 不会触发 currentIndexChanged → current_flow_path 为空，
            # "运行流程"必须重选一次才生效
            self._on_flow_selected(0)
        self._watch_model_dir()   # Q7：目录内容变化（画布保存新流程）时实时刷新

    def _on_flow_selected(self, index: int):
        path = self.flow_combo.itemData(index)
        if not path:
            return
        path = Path(path)
        if path.suffix.lower() == ".xvap":
            try:
                flow_json, _ = _materialize_xvap(path)
            except Exception as e:
                log.error("解包失败: %s", e, exc_info=True)
                return
            self.current_flow_path = flow_json
        else:
            self.current_flow_path = path
        self._sync_canvas_path()

    # ================= 六个动作 =================
    def _read_flow(self) -> dict | None:
        if self.current_flow_path is None or not self.current_flow_path.is_file():
            return None
        try:
            return json.loads(self.current_flow_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.error("流程读取失败: %s", e)
            return None

    def _require_flow(self) -> Path | None:
        if self.current_flow_path is None or not self.current_flow_path.is_file():
            self.window().statusBar().showMessage(
                "请先在右侧选择或新建流程 JSON", 4000)
            return None
        return self.current_flow_path

    def _sync_canvas_path(self):
        if self._canvas is not None and self.current_flow_path is not None:
            flow = self._read_flow()
            if flow is not None:
                self._canvas.current_path = self.current_flow_path
                self._canvas.load_flow_payload(flow, self.current_flow_path.name)

    def _create_flow(self):
        d = Path(self.model_dir_edit.text().strip())
        start = str(d / "flow.json") if d.is_dir() else "flow.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "新建流程图（选择保存位置并命名）", start, "流程图 (*.json)")
        if not path:
            return
        _write_flow_json(Path(path), json.loads(json.dumps(_EMPTY_FLOW)))
        self.model_dir_edit.setText(str(Path(path).parent))
        self._refresh_flow_combo()
        self.current_flow_path = Path(path)
        self._edit_flow()

    def _edit_flow(self):
        path = self._require_flow()
        if path is None:
            return
        flow = self._read_flow()
        if flow is None:
            return
        try:
            from .modeltools.flowgraph_web_window import FlowGraphWebWindow  # 惰性：QtWebEngine
        except Exception as e:  # noqa: BLE001  常见为当前解释器缺 PyQtWebEngine
            QMessageBox.critical(
                self, "无法打开流程画布",
                "当前 Python 环境缺少 QtWebEngine 组件，流程画布无法启动。\n\n"
                f"具体错误：{type(e).__name__}: {e}\n\n"
                "请用本仓库虚拟环境启动（run_main.bat 会自动优先使用 venv\\），\n"
                "或安装依赖后重试：pip install PyQtWebEngine")
            return

        if self._canvas is None:
            self._canvas = FlowGraphWebWindow(on_closed=self._on_canvas_closed)
        self._canvas.current_path = path
        self._canvas.load_flow_payload(flow, path.name)
        self._canvas.show()
        self._canvas.raise_()
        self._canvas.activateWindow()

    def _on_canvas_closed(self):
        self._canvas = None

    def _resolve_model(self, flow: dict):
        # 旧链模型节点（新流程用 xccv2/ 模型节点走新引擎，见 _run_flow_xcv）
        model_file = next(((n.get("properties") or {}).get("model_file")
                           for n in flow.get("nodes", [])
                           if str(n.get("type", "")).startswith(("model/", "xccv/model"))
                           and (n.get("properties") or {}).get("model_file")), None)
        if not model_file:
            return None
        if self._run_model_path != model_file or self._run_model is None:
            from .engine.models.registry import load_model

            self._run_model = load_model(model_file)
            self._run_model_path = model_file
        return self._run_model

    @staticmethod
    def _is_xcv_flow(flow: dict) -> bool:
        """是否新算子流程（含任意 xccv2/ 节点即走新引擎）。"""
        return any(str(n.get("type", "")).startswith("xccv2/")
                   for n in flow.get("nodes", []))

    def _run_flow_xcv(self, flow: dict, img_path: Path,
                      flow_path: Path | None = None) -> None:
        """新算子流程运行（纯传统算子链在「流程图」页运行，Q7 修复）。"""
        from .engine.flow.snapshot import overlay_result
        from .modeltools.flow_session import FlowSession, _normalize_types

        norm = _normalize_types(flow)
        # 用当前选中的图片驱动流程（覆盖读入图片节点的路径）
        img = imread_unicode(str(img_path))
        if img is None:
            self.window().statusBar().showMessage("无法读取该图片", 4000)
            return
        has_loader = False
        for n in norm.get("nodes", []):
            if str(n.get("type", "")) == "image_loader":
                n.setdefault("properties", {})["path"] = str(img_path)
                has_loader = True
        if not has_loader:
            self.window().statusBar().showMessage(
                "新算子流程需要「读入图片」节点作为图像来源", 6000)
            return
        flow_path = flow_path or self.current_flow_path
        session = FlowSession()
        try:
            summary = session.run(norm, label="流程图平台",
                                  flow_dir=flow_path.parent if flow_path else None,
                                  source_name=img_path.name,
                                  flow_name=flow_path.name if flow_path else "")
        except Exception as e:
            self.window().statusBar().showMessage(f"运行失败：{e}", 8000)
            return
        # 结果可视化（用户批注 ToD1/ToD4 + v0.4.4 ToC2）：结果叠加收集
        # 【全部 ok 节点】的输出——末节点可能是「保存结果」（仅 saved 布尔），
        # 检出框/几何蒙版在中游节点上；此前只看末节点导致输出 JSON 后
        # 可视化"消失"。底图 = 末节点的图像输出（如转平归正），否则原图。
        results = session.executor.results
        ok_ids = [nid for nid, r in results.items() if r.ok]
        rendered = None
        if ok_ids:
            all_outs = {}
            base = None
            for nid in sorted(ok_ids):
                for v in results[nid].outputs.values():
                    if isinstance(v, ImageData):
                        base = v.mat       # 取拓扑序最后的图像输出为底图
                    else:
                        all_outs.setdefault(id(v), v)
            if base is None:
                base = img
            try:
                rendered = overlay_result(base, list(all_outs.values()))
            except Exception:
                rendered = base
        if rendered is None:
            rendered = img
        self._show_mat(rendered, rendered=True)
        self._set_preview_mode("可视化")
        ok_n, total = summary["ok_count"], summary["total"]
        has_err = bool(summary["errors"])
        msg = (f"运行完成：{ok_n}/{total} 节点成功"
               + (f"，{len(summary['errors'])} 个失败" if has_err else "")
               + "（运行日志已存 platform_data/logs/流程运行日志）")
        if has_err:
            for err in summary["errors"].values():
                msg += f" | {err}"
        self.window().statusBar().showMessage(msg, 10000)
        from .ui_widgets import toast

        toast(self.window(),
              f"流程运行完成：{ok_n}/{total} 节点成功"
              + (f"，{len(summary['errors'])} 个失败" if has_err else ""),
              "warn" if has_err else "success")

    def _run_flow(self):
        path = self._require_flow()
        if path is None:
            return
        flow = self._read_flow()
        if flow is None:
            return
        if self._is_xcv_flow(flow):
            img_path = self._selected_image()
            if img_path is None:
                self.window().statusBar().showMessage("请先在左侧选择一张图片", 4000)
                return
            self._run_flow_xcv(flow, img_path)
            return
        try:
            model = self._resolve_model(flow)
        except Exception as e:
            self.window().statusBar().showMessage(
                f"模型加载失败: {type(e).__name__}: {e}", 6000)
            return
        if model is None:
            self.window().statusBar().showMessage(
                "流程中没有已选择权重文件的模型节点（请先编辑流程）", 6000)
            return
        img_path = self._selected_image()
        if img_path is None:
            self.window().statusBar().showMessage("请先在左侧选择一张图片", 4000)
            return
        img = imread_unicode(str(img_path))
        if img is None:
            self.window().statusBar().showMessage("无法读取该图片", 4000)
            return
        pred = model.predict(img)
        base_count = len(pred.boxes)
        chain = df.graph_to_chain(flow)
        classes = getattr(model, "classes", None)
        from .engine.flow.legacy_adapter import apply_flow_safe

        filt, info = apply_flow_safe(pred, (img.shape[1], img.shape[0]), chain,
                                     classes=classes)
        from .testing.render import draw_predictions

        task_v = getattr(getattr(model, "task", None), "value", "detection")
        rendered = draw_predictions(img, filt, model.classes or [], task_v,
                                    show_labels=True,
                                    show_masks=(task_v == "segmentation"))
        self._show_mat(rendered, rendered=True)
        self._set_preview_mode("可视化")
        self.window().statusBar().showMessage(
            f"原始检出 {base_count} → 流程过滤后 {len(filt.boxes)}"
            f"（数量判定 {info.get('count_check') or '未配置'}）", 8000)

    def _pack_flow(self):
        path = self._require_flow()
        if path is None:
            return
        flow = self._read_flow()
        if flow is None:
            return
        from .modeltools.flow_pack import _show_pack_problems, \
            validate_flow_for_pack

        problems = validate_flow_for_pack(flow)
        if problems:  # 打包前校验（To3.3）：不产出坏包
            _show_pack_problems(self, problems)
            return
        try:
            models = fp.collect_model_files(flow)
        except FileNotFoundError as e:
            self.window().statusBar().showMessage(str(e), 6000)
            return
        if not models:
            has_legacy = any(str(n.get("type", "")).startswith(
                ("model/", "xccv/model", "post/", "xccv/post_"))
                for n in flow.get("nodes", []))
            if has_legacy:
                self.window().statusBar().showMessage(
                    "流程中没有已选择权重文件的模型节点", 6000)
                return
            # 纯传统算子流程（用户批注 F2/Q7）：允许打包为不含模型的 .xvap
        dst, _ = QFileDialog.getSaveFileName(
            self, "打包流程", "flow.xvap", "流程包 (*.xvap)")
        if not dst:
            return
        fp.save_xvap(dst, flow, models)
        tip = f"（含 {len(models)} 个模型）" if models else "（纯算子流程，不含模型）"
        self.window().statusBar().showMessage(
            f"已打包 {Path(dst).name}{tip}", 8000)

    def _unpack_flow(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "解包流程包", "", "流程包 (*.xvap)")
        if not path:
            return
        flow_json, _ = _materialize_xvap(Path(path))
        self.model_dir_edit.setText(str(Path(path).parent))
        self._refresh_flow_combo()
        self.current_flow_path = flow_json
        self.window().statusBar().showMessage(
            f"已解包：{flow_json.name}（模型释放到 流程模型/ 文件夹）", 8000)

    # ================= 批量运行 =================
    def _start_batch(self):
        if self._batch_thread is not None and self._batch_thread.isRunning():
            self.window().statusBar().showMessage("批量运行正在进行中", 3000)
            return
        path = self._require_flow()
        if path is None:
            return
        flow = self._read_flow()
        if flow is None:
            return
        if self.data_root is None:
            self.window().statusBar().showMessage("请先选择数据路径", 4000)
            return
        images = _list_images(self.data_root)
        if not images:
            self.window().statusBar().showMessage("数据路径下没有图片", 4000)
            return
        out_dir = self.current_flow_path.parent / f"批量结果_{time.strftime('%Y%m%d_%H%M%S')}"
        # 批量设置弹窗（用户批注 ToC5：与测试平台批量一致，可改各项信息）
        from .testing.batch_dialog import BatchSettingsDialog

        dlg = BatchSettingsDialog(self.window(), folder=str(self.data_root))
        dlg.ed_out.setPlaceholderText(
            f"留空 = {out_dir}")
        if dlg.exec_() != QDialog.Accepted:
            self.window().statusBar().showMessage("已取消批量运行", 3000)
            return
        settings = dlg.settings()
        if not settings.save_json and not settings.save_images:
            self.window().statusBar().showMessage(
                "「导出 JSON」与「保存渲染图片」至少勾选一项", 6000)
            return
        root = Path(settings.folder) if settings.folder else self.data_root
        if not root.is_dir():
            self.window().statusBar().showMessage("图片文件夹不存在", 6000)
            return
        images = _list_images(root)
        if not images:
            self.window().statusBar().showMessage("图片文件夹下没有图片", 4000)
            return
        out_dir = Path(settings.out_dir) if settings.out_dir else out_dir
        if self._is_xcv_flow(flow):
            # 纯传统/新算子流程批量（Q5：流程图要有 JSON 结果产出）——
            # 逐图跑新引擎，统一 schema JSON（与 AI 检测批量一致）+ 蒙版可视化
            self._batch_thread = _XcvBatchThread(flow, images, root, out_dir,
                                                 flow_name=path.name,
                                                 save_json=settings.save_json,
                                                 save_images=settings.save_images,
                                                 jpg_quality=settings.jpg_quality)
        else:
            model_file = next(((n.get("properties") or {}).get("model_file")
                               for n in flow.get("nodes", [])
                               if str(n.get("type", "")).startswith(("model/", "xccv/model"))
                               and (n.get("properties") or {}).get("model_file")), None)
            if not model_file:
                self.window().statusBar().showMessage(
                    "流程中没有已选择权重文件的模型节点", 6000)
                return
            self._batch_thread = _FlowBatchThread(flow, model_file, images,
                                                  self.data_root, out_dir)
        self._batch_thread.progress.connect(
            lambda done, total: self.window().statusBar().showMessage(
                f"批量推理 {done} / {total} …"))
        self._batch_thread.finished_ok.connect(self._on_batch_done)
        self._batch_thread.failed.connect(self._on_batch_failed)
        auto_cleanup_thread(self, "_batch_thread", self._batch_thread)  # A-3（T2-6）
        self.btn_batch.setEnabled(False)
        self.btn_batch_stop.setEnabled(True)
        self._batch_thread.start()

    def _stop_batch(self):
        if self._batch_thread is not None:
            self._batch_thread.stop()

    def _on_batch_done(self, out_dir: str):
        self._finish_batch(f"批量完成，结果已保存：{out_dir}")

    def _on_batch_failed(self, msg: str):
        self._finish_batch(f"批量运行失败：{msg}")

    def _finish_batch(self, msg: str):
        self.btn_batch.setEnabled(True)
        self.btn_batch_stop.setEnabled(False)
        self._batch_thread = None
        self.window().statusBar().showMessage(msg, 12000)


def main():
    from .platform_entry import run_platform

    run_platform("flowmain", "流程图平台", lambda: _wrap_window())


def _wrap_window():
    from PyQt5.QtWidgets import QMainWindow

    class _Win(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("XCcv 流程图平台")
            self.setWindowIcon(QIcon(str(icon_path("flow.png"))))   # v0.5.5追加 独立图标
            self.resize(1280, 780)
            apply_app_theme(self)
            self.setCentralWidget(FlowPage())

    return _Win()


if __name__ == "__main__":
    main()


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402

apply_auto_slot_module(globals())

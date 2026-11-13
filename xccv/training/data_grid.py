"""O11（V0.7）：训练平台数据网格（缩略图墙）。

复用 flow_platform._TreeScanWorker 的线程纪律（坑 P140：线程不碰 QWidget）：
- 扫描线程只产出 GridEntry 列表（路径/标注状态/标签/OK 目录/忽略区域数）；
- 缩略图线程按队列懒生成 QImage（cv2 解码在子线程；QPixmap 只能主线程建），
  分批投递主线程；
- 主线程 QListWidget(IconMode) 呈现 + LRU 图标缓存（上限内复用，超出淘汰）；
- 筛选：按标签 / 只看 OK / 只看未标注 / 含忽略区域；角标：[OK] 绿、⊘N
  琥珀（忽略区域数）、未标注灰字。

纯逻辑（GridEntry / scan_grid_entries / filter_grid_entries / badge_text）
可离线单测；Qt 层只有薄的展示接线。设计取舍：缩略图队列只装「当前筛选
可见」的条目（懒加载），切换筛选时重建队列，万图目录也不全量解码。
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..engine.dataset.base import IMAGE_EXTS
from ..engine.dataset.reader import is_negative_dir_name

_THUMB_SIZE = 112          # 缩略图边长（正方形 letterbox）
_LRU_CAP = 800             # QIcon 缓存上限（每项 ~112×112 ARGB ≈ 50KB）
_THUMB_BATCH = 12          # 缩略线程每批投递条数

_STATE_NAMES = {"labeled": "已标注", "clean": "已审无缺陷", "unlabeled": "未标注"}


@dataclass
class GridEntry:
    """网格单元元数据（不含像素）。state：labeled / clean / unlabeled。"""

    path: str
    has_json: bool = False
    labels: tuple = ()
    is_ok_dir: bool = False
    n_ignore: int = 0

    @property
    def state(self) -> str:
        if self.labels:
            return "labeled"
        if self.has_json:
            return "clean"      # 已审无缺陷（json 存在但无实例标注）
        return "unlabeled"


def scan_grid_entries(dataset_dir) -> list:
    """扫描数据集目录 → GridEntry 列表（后台线程调用；只读盘不解码像素）。

    与 scan_labelme_dataset 同款遍历规则：递归、跳过隐藏目录；json 最小化
    解析（只取 shapes 的 label），损坏 json 按"无标注"降级不中断扫描。
    """
    from ..engine.dataset.ignore import IGNORE_LABEL

    root = Path(dataset_dir)
    out: list = []
    for img in sorted(root.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = img.relative_to(root)
        dirs = rel.parts[:-1]
        if any(part.startswith(".") for part in dirs):
            continue
        labels: list = []
        n_ignore = 0
        jp = img.with_suffix(".json")
        has_json = jp.is_file()
        if has_json:
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
                for sh in data.get("shapes") or []:
                    lab = str(sh.get("label", "")).strip()
                    if not lab:
                        continue
                    if lab == IGNORE_LABEL:
                        n_ignore += 1
                    else:
                        labels.append(lab)
            except Exception:  # noqa: BLE001  损坏 json 按"未标注"降级
                labels, n_ignore, has_json = [], 0, False
        out.append(GridEntry(
            path=str(img), has_json=has_json, labels=tuple(labels),
            is_ok_dir=any(is_negative_dir_name(p) for p in dirs),
            n_ignore=n_ignore))
    return out


def filter_grid_entries(entries: list, label: str = "",
                        only_ok: bool = False, only_unlabeled: bool = False,
                        only_ignore: bool = False) -> list:
    """按标签 / OK / 未标注 / 含忽略区域筛选（全为叠加条件）。"""
    out = []
    for e in entries:
        if only_ok and not e.is_ok_dir:
            continue
        if only_unlabeled and e.state != "unlabeled":
            continue
        if only_ignore and e.n_ignore <= 0:
            continue
        if label and label not in e.labels:
            continue
        out.append(e)
    return out


def distinct_labels(entries: list) -> list:
    """条目集合中出现过的实例标签（排序去重）。"""
    seen: set = set()
    for e in entries:
        seen.update(e.labels)
    return sorted(seen)


def badge_text(entry: GridEntry) -> str:
    """列表项后缀角标文本（颜色由 widget 层按 state 着色）。"""
    parts = []
    if entry.is_ok_dir or entry.state == "clean":
        parts.append("[OK]")
    if entry.n_ignore:
        parts.append(f"⊘{entry.n_ignore}")
    return " ".join(parts)


def make_thumbnail(path: str, size: int = _THUMB_SIZE):
    """读图 → 正方形 letterbox 缩略（BGR ndarray，None=读取失败）。

    后台线程调用；不依赖 Qt。灰底填充保持长宽比，肉眼核对构图不变形。
    """
    from ..engine.io_utils import imread_unicode

    try:
        img = imread_unicode(str(path))
    except Exception:  # noqa: BLE001  缺文件/权限等一律按无图降级
        return None
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 235, dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = im
    return canvas


def ndarray_to_qimage(arr: np.ndarray):
    """BGR ndarray → QImage（.copy() 自持内存，跨线程投递安全）。"""
    from PyQt5.QtGui import QImage

    rgb = np.ascontiguousarray(arr[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format_RGB888).copy()


# ================= Qt 展示层（延迟构建，get_data_grid_class() 触发） =================

_grid_cls = None


def get_data_grid_class():
    """返回 DataGridWidget 类（首次调用时构建，符合窗口模块惰性导入纪律）。"""
    global _grid_cls
    if _grid_cls is None:
        _grid_cls = _build_widget_classes()
    return _grid_cls


def _build_widget_classes():
    from PyQt5.QtCore import QSize, Qt, QThread, QUrl, pyqtSignal
    from PyQt5.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPixmap
    from PyQt5.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout,
                                 QLabel, QListWidget, QListWidgetItem,
                                 QVBoxLayout, QWidget)

    def _icon_size() -> QSize:
        return QSize(_THUMB_SIZE + 8, _THUMB_SIZE + 8)

    def _placeholder_icon() -> QIcon:
        pm = QPixmap(_THUMB_SIZE, _THUMB_SIZE)
        pm.fill(QColor("#EDF1F7"))
        p = QPainter(pm)
        p.setPen(QColor("#B9C4D4"))
        p.drawRect(0, 0, _THUMB_SIZE - 1, _THUMB_SIZE - 1)
        p.end()
        return QIcon(pm)

    class _GridScanWorker(QThread):
        done = pyqtSignal(list)
        failed = pyqtSignal(str)

        def __init__(self, root, parent=None):
            super().__init__(parent)
            self._root = Path(root)

        def run(self):
            try:
                self.done.emit(scan_grid_entries(self._root))
            except Exception as e:  # noqa: BLE001
                self.failed.emit(f"{type(e).__name__}: {e}")

    class _ThumbWorker(QThread):
        ready = pyqtSignal(list)

        def __init__(self, paths, parent=None):
            super().__init__(parent)
            self._queue = list(paths)
            self._stopped = False

        def stop(self):
            self._stopped = True

        def run(self):
            batch = []
            while self._queue and not self._stopped:
                p = self._queue.pop(0)
                try:
                    arr = make_thumbnail(p)
                except Exception:  # noqa: BLE001  损坏图跳过
                    arr = None
                if arr is not None:
                    batch.append((p, ndarray_to_qimage(arr)))
                if len(batch) >= _THUMB_BATCH or (batch and not self._queue):
                    self.ready.emit(batch)
                    batch = []

    class DataGridWidget(QWidget):
        """缩略图网格：扫描/缩略全后台，主线程只做筛选与贴图。"""

        def __init__(self, parent=None):
            super().__init__(parent)
            self._entries: list = []
            self._item_by_path: dict = {}
            self._cache: OrderedDict = OrderedDict()   # path -> QIcon（LRU）
            self._scan_worker = None
            self._thumb_worker = None
            self._root = None
            self._ph_icon = _placeholder_icon()

            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            bar = QGridLayout()
            bar.setHorizontalSpacing(10)
            self.cmb_label = QComboBox()
            self.cmb_label.addItem("全部标签", "")
            self.cmb_label.currentIndexChanged.connect(self._apply_filter)
            self.cb_ok = QCheckBox("只看 OK")
            self.cb_ok.stateChanged.connect(self._apply_filter)
            self.cb_unlabeled = QCheckBox("只看未标注")
            self.cb_unlabeled.stateChanged.connect(self._apply_filter)
            self.cb_ignore = QCheckBox("含忽略区域")
            self.cb_ignore.stateChanged.connect(self._apply_filter)
            self.lbl_count = QLabel("")
            self.lbl_count.setObjectName("hint")
            bar.addWidget(QLabel("标签"), 0, 0)
            bar.addWidget(self.cmb_label, 0, 1)
            bar.addWidget(self.cb_ok, 0, 2)
            bar.addWidget(self.cb_unlabeled, 0, 3)
            bar.addWidget(self.cb_ignore, 0, 4)
            bar.addWidget(self.lbl_count, 0, 5)
            bar.setColumnStretch(5, 1)
            lay.addLayout(bar)

            self.list = QListWidget()
            self.list.setViewMode(QListWidget.IconMode)
            self.list.setResizeMode(QListWidget.Adjust)
            self.list.setMovement(QListWidget.Static)
            self.list.setSpacing(8)
            self.list.setIconSize(_icon_size())
            self.list.setUniformItemSizes(True)
            self.list.setWordWrap(True)
            self.list.itemDoubleClicked.connect(self._open_image)
            self.list.setMinimumHeight(260)
            lay.addWidget(self.list, stretch=1)

        # ---- 数据装载 ----
        def set_dataset(self, dataset_dir):
            """切换数据集：停掉旧线程 → 重新扫描（懒加载纪律）。"""
            self._stop_workers()
            self._entries = []
            self._item_by_path.clear()
            self._cache.clear()
            self.list.clear()
            self._root = Path(dataset_dir) if dataset_dir else None
            if self._root is None or not self._root.is_dir():
                self.lbl_count.setText("（未设置数据集目录）")
                return
            from ..gui_utils import auto_cleanup_thread

            w = _GridScanWorker(self._root)
            w.done.connect(self._on_scan_done)
            w.failed.connect(self._on_scan_failed)
            auto_cleanup_thread(self, "_scan_worker", w)
            self._scan_worker = w
            w.start()

        def _stop_workers(self):
            if self._thumb_worker is not None:
                self._thumb_worker.stop()
                self._thumb_worker = None
            if self._scan_worker is not None:
                self._scan_worker = None

        def shutdown(self):
            """窗口关闭前的线程收尾（training_window.closeEvent 调用）。"""
            self._stop_workers()

        def _on_scan_failed(self, msg):
            self.lbl_count.setText(f"扫描失败：{msg}")

        def _on_scan_done(self, entries: list):
            self._entries = list(entries)
            combo = self.cmb_label
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("全部标签", "")
            for lab in distinct_labels(self._entries):
                combo.addItem(lab, lab)
            combo.blockSignals(False)
            self._apply_filter()

        # ---- 筛选与填充 ----
        def _apply_filter(self, *_):
            label = self.cmb_label.currentData() or ""
            visible = filter_grid_entries(
                self._entries, label=label,
                only_ok=self.cb_ok.isChecked(),
                only_unlabeled=self.cb_unlabeled.isChecked(),
                only_ignore=self.cb_ignore.isChecked())
            self.list.clear()
            self._item_by_path.clear()
            for e in visible:
                text = Path(e.path).name
                badge = badge_text(e)
                if badge:
                    text += "  " + badge
                item = QListWidgetItem(self._ph_icon, text)
                item.setData(Qt.UserRole, e.path)
                tip = f"{e.path}\n状态：{_STATE_NAMES[e.state]}"
                if e.labels:
                    tip += f"\n标签：{'、'.join(e.labels)}"
                if e.n_ignore:
                    tip += f"\n忽略区域：{e.n_ignore}"
                item.setToolTip(tip)
                color = {"labeled": QColor("#25406B"), "clean": QColor("#2E9E5B"),
                         "unlabeled": QColor("#8A8F98")}[e.state]
                item.setForeground(color)
                self.list.addItem(item)
                self._item_by_path[e.path] = item
            self.lbl_count.setText(f"显示 {len(visible)} / 共 {len(self._entries)} 张")
            self._start_thumbs([e.path for e in visible])   # 懒加载：只解码可见项

        def _start_thumbs(self, paths):
            if self._thumb_worker is not None:
                self._thumb_worker.stop()
                self._thumb_worker = None
            todo = [p for p in paths if p not in self._cache]
            if not todo:
                return
            from ..gui_utils import auto_cleanup_thread

            w = _ThumbWorker(todo)
            w.ready.connect(self._on_thumbs)
            auto_cleanup_thread(self, "_thumb_worker", w)
            self._thumb_worker = w
            w.start()

        def _on_thumbs(self, batch):
            for p, qimg in batch:
                icon = QIcon(QPixmap.fromImage(qimg))
                self._cache[p] = icon
                self._cache.move_to_end(p)
                while len(self._cache) > _LRU_CAP:
                    self._cache.popitem(last=False)
                item = self._item_by_path.get(p)
                if item is not None:
                    item.setIcon(icon)

        def _open_image(self, item):
            QDesktopServices.openUrl(QUrl.fromLocalFile(item.data(Qt.UserRole)))

    return DataGridWidget

# -*- coding: utf-8 -*-
"""标注平台数量统计面板（批注第二批 F8）。

当前图每类计数（实时） + 数据集目录每类计数（后台线程扫描 labelme json）。
"""

import json
import os
import threading
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QHeaderView,
                             QLabel, QProgressDialog, QTableWidget,
                             QTableWidgetItem, QVBoxLayout)


class StatsDialog(QDialog):
    """数量统计：文件夹/当前图每类计数。"""

    _folder_done = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数量统计")
        self.resize(420, 360)
        lay = QVBoxLayout(self)
        self._lb_img = QLabel("当前图：—")
        self._lb_img.setAlignment(Qt.AlignLeft)
        self._lb_folder = QLabel("文件夹：扫描中…")
        lay.addWidget(self._lb_img)
        lay.addWidget(self._lb_folder)
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(
            ["类别", "当前图", "文件夹"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self._table)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        bb.clicked.connect(self.close)
        lay.addWidget(bb)
        self._folder_done.connect(self._apply_folder)

        self.refresh()

    def refresh(self):
        """当前图实时计数 + 后台扫描目录（不卡 UI）。"""
        win = self.parent()
        canvas = getattr(win, "canvas", None)
        img_counts = {}
        folder = ""
        if canvas is not None:
            for sh in canvas.shapes:
                if sh.label:
                    img_counts[sh.label] = img_counts.get(sh.label, 0) + 1
            folder = getattr(win, "lastOpenDir", "") or ""
            if win.filename and not folder:
                folder = os.path.dirname(win.filename)
        self._render(img_counts, None)
        if folder and os.path.isdir(folder):
            self._scan_folder_async(folder, img_counts)

    def _scan_folder_async(self, folder, img_counts):
        prog = QProgressDialog("扫描数据集标注…", "取消", 0, 0, self)
        prog.setWindowTitle("数量统计")
        prog.setWindowModality(Qt.WindowModal)

        def _work():
            counts = {}
            n = 0
            try:
                for f in Path(folder).glob("*.json"):
                    n += 1
                    try:
                        data = json.loads(
                            f.read_text(encoding="utf-8"))
                        for sh in data.get("shapes") or []:
                            lb = str(sh.get("label") or "")
                            if lb:
                                counts[lb] = counts.get(lb, 0) + 1
                    except (OSError, ValueError):
                        continue
                    if prog.wasCanceled():
                        break
            except Exception:
                pass
            prog.close()
            self._folder_done.emit(counts)

        threading.Thread(target=_work, daemon=True).start()

    def _apply_folder(self, counts: dict):
        self._folder_counts = counts
        self._render(getattr(self, "_img_counts", {}), counts)
        total = sum(counts.values())
        self._lb_folder.setText(
            f"文件夹：共 {len(counts)} 类 / {total} 个目标")

    def _render(self, img_counts, folder_counts):
        self._img_counts = img_counts
        labels = sorted(set(img_counts) | set(folder_counts or {}))
        self._table.setRowCount(len(labels))
        for i, lb in enumerate(labels):
            self._table.setItem(i, 0, QTableWidgetItem(lb))
            self._table.setItem(
                i, 1, QTableWidgetItem(str(img_counts.get(lb, 0))))
            fc = "…" if folder_counts is None else str(
                (folder_counts or {}).get(lb, 0))
            self._table.setItem(i, 2, QTableWidgetItem(fc))
        total = sum(img_counts.values())
        self._lb_img.setText(
            f"当前图：{len(img_counts)} 类 / {total} 个目标")

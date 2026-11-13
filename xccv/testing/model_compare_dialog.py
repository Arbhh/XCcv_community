"""O21（V0.7 评审项补全）：模型 A/B 批量比对对话框（Qt 展示层）。

用法：测试平台完成一次批量（模型 A）后打开本对话框 → 选择模型 B 权重
（.pt）→ 后台逐图复推（与 A 同 conf/iou、同检出流程）→ 逐图
对比表 + 判定统计；产物落 `检出结果/比对.json`。

线程纪律（坑 P144/P140）：CompareThread 无父对象 + auto_cleanup_thread
收尾；线程内不碰任何 QWidget，产物经信号回主线程。
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QDialog, QFileDialog, QFormLayout, QHBoxLayout,
                             QHeaderView, QLabel, QMessageBox, QProgressBar,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

log = logging.getLogger(__name__)

_VERDICT_COLORS = {"A优": "#2E6FBB", "B优": "#2E9E5B", "一致": "#25406B",
                   "未评估": "#8A8F98"}


class CompareThread(QThread):
    """模型 B 复推线程：加载 → 逐图 predict（同 conf/iou/流程）→ 回传检出。"""

    progress = pyqtSignal(int, int, str)
    done = pyqtSignal(dict)          # {path: [det...]}
    failed = pyqtSignal(str)

    def __init__(self, model_path, paths, conf, iou, flow, classes,
                 parent=None):
        super().__init__(parent)
        self._model_path = Path(model_path)
        self._paths = list(paths)
        self._conf, self._iou = conf, iou
        self._flow, self._classes = flow, classes

    def run(self):
        try:
            from ..engine.models import load_model

            model = load_model(self._model_path)
            out: dict = {}
            total = len(self._paths)
            for i, p in enumerate(self._paths, start=1):
                from ..engine.io_utils import imread_unicode

                img = imread_unicode(p)
                if img is None:
                    continue
                h, w = img.shape[:2]
                pred = model.predict(img, conf=self._conf, iou=self._iou)
                if self._flow:
                    from ..engine.flow.legacy_adapter import apply_flow_safe

                    pred, _info = apply_flow_safe(
                        pred, (w, h), self._flow, classes=self._classes or None)
                dets = []
                for j, box in enumerate(pred.boxes):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    label = (self._classes[pred.labels[j]]
                             if self._classes and pred.labels[j] < len(self._classes)
                             else str(pred.labels[j]))
                    dets.append({"label": label,
                                 "score": round(float(pred.scores[j]), 4),
                                 "bbox": [x1, y1, x2, y2]})
                out[str(p)] = dets
                self.progress.emit(i, total, p.name)
            self.done.emit(out)
        except Exception as e:  # noqa: BLE001
            log.error("A/B 比对复推失败: %s", e, exc_info=True)
            self.failed.emit(f"{type(e).__name__}: {e}")


class ModelCompareDialog(QDialog):
    """A/B 比对窗：选 B 模型 → 复推 → 逐图判定表 + 统计 + 比对.json 落盘。"""

    def __init__(self, results_a: list, out_dir, conf: float, iou: float,
                 flow, classes, model_a_name: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("模型 A/B 批量比对")
        self.resize(860, 560)
        self._results_a = list(results_a)
        self._paths = [Path(str(r.get("path"))) for r in self._results_a
                       if r.get("path")]
        self._conf, self._iou = float(conf), float(iou)
        self._flow, self._classes = flow, classes
        self._model_a_name = model_a_name
        self._out_dir = Path(out_dir) if out_dir else None
        self._thread = None
        self._payload: dict | None = None

        lay = QVBoxLayout(self)
        head = QFormLayout()
        self.lbl_a = QLabel(model_a_name or "（模型 A：本次批量所用权重）")
        self.lbl_b = QLabel("—")
        head.addRow("模型 A", self.lbl_a)
        head.addRow("模型 B", self.lbl_b)
        lay.addLayout(head)

        bar = QHBoxLayout()
        self.btn_pick = QPushButton("选择 B 模型…")
        self.btn_pick.clicked.connect(self._pick_model)
        self.btn_run = QPushButton("开始比对")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._start)
        self.progress = QProgressBar()
        self.progress.setMaximum(1)
        self.progress.setValue(0)
        bar.addWidget(self.btn_pick)
        bar.addWidget(self.btn_run)
        bar.addWidget(self.progress, stretch=1)
        lay.addLayout(bar)

        self.lbl_summary = QLabel(
            "判定口径：优势分 = 命中 GT 数 − 误检数，A/B 取差值；"
            "OK 图比误检；无 GT 记「未评估」")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setObjectName("hint")
        lay.addWidget(self.lbl_summary)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["文件", "GT", "A 检出", "A 命中", "B 检出", "B 命中", "判定", "备注"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table, stretch=1)

    # ---- B 模型选择与复推 ----
    def _pick_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型 B 权重", "",
            "模型权重 (*.pt);;全部文件 (*.*)")
        if not path:
            return
        self._model_b_path = Path(path)
        self.lbl_b.setText(self._model_b_path.name)
        self.btn_run.setEnabled(True)

    def _start(self):
        if getattr(self, "_model_b_path", None) is None:
            return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "请稍候", "比对正在进行中")
            return
        self.btn_run.setEnabled(False)
        self.btn_pick.setEnabled(False)
        self.progress.setMaximum(len(self._paths))
        self.progress.setValue(0)
        from ..gui_utils import auto_cleanup_thread

        th = CompareThread(self._model_b_path, self._paths, self._conf,
                           self._iou, self._flow, self._classes)
        th.progress.connect(self._on_progress)
        th.done.connect(self._on_done)
        th.failed.connect(self._on_failed)
        auto_cleanup_thread(self, "_thread", th)
        self._thread = th
        th.start()

    def _on_progress(self, done: int, total: int, name: str):
        self.progress.setValue(done)
        self.setWindowTitle(f"模型 A/B 批量比对（{done}/{total} {name}）")

    def _on_failed(self, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_pick.setEnabled(True)
        QMessageBox.warning(self, "比对失败",
                            f"模型 B 复推失败：{msg}")

    def _on_done(self, predictions_b: dict):
        self.btn_run.setEnabled(True)
        self.btn_pick.setEnabled(True)
        from .model_compare import (build_compare_payload, compare_batch,
                                    save_compare_json)

        records = compare_batch(self._results_a, predictions_b, self._iou)
        self._fill_table(records)
        payload = build_compare_payload(
            self._model_a_name, getattr(self, "_model_b_path", Path()).name,
            self._conf, self._iou, records)
        self._payload = payload
        s = payload["summary"]["counts"]
        self.lbl_summary.setText(
            f"判定统计：一致 {s.get('一致', 0)}，A 优 {s.get('A优', 0)}，"
            f"B 优 {s.get('B优', 0)}，未评估 {s.get('未评估', 0)}"
            f"　（共 {len(records)} 图，conf={self._conf:g} iou={self._iou:g}）")
        if self._out_dir is not None:
            try:
                out = self._out_dir / "检出结果" / "比对.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                save_compare_json(out, payload)
                self.lbl_summary.setText(self.lbl_summary.text()
                                         + f"　已保存：{out}")
            except OSError as e:
                QMessageBox.warning(self, "保存比对结果", f"写入失败：{e}")

    def _fill_table(self, records: list):
        self.table.setRowCount(len(records))
        for i, rec in enumerate(records):
            a, b = rec["a"], rec["b"]
            cells = [rec.get("file") or Path(rec.get("path", "")).name,
                     str(rec["gt"]) if rec["gt_available"] else "—",
                     str(a["n"]), str(a["hit"]), str(b["n"]), str(b["hit"]),
                     rec["verdict"],
                     "OK 图" if (rec["gt_available"] and rec["gt"] == 0)
                     else ("" if rec["gt_available"] else "无 GT")]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 6:
                    color = QColor(_VERDICT_COLORS.get(rec["verdict"], "#25406B"))
                    item.setForeground(color)
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(i, c, item)

    def payload(self):
        return self._payload

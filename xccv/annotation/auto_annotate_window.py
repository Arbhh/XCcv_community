# -*- coding: utf-8 -*-
"""AI 批量预标注窗口（V0.6 T5-4）：选模型/文件夹 → 后台线程逐图预标注。

入口：标注平台菜单「工具 → AI 批量预标注」（annotator_window 接线）。
线程纪律：QThread 无父对象 + auto_cleanup_thread 收尾（P142/P144）；
窗口关闭即取消（cancel flag），不驻留后台。
"""

from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton,
    QVBoxLayout,
)

from ..gui_utils import auto_cleanup_thread


class _AnnotateThread(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, kwargs: dict, parent=None):
        super().__init__(parent)
        self._kw = kwargs
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            summary = self._kw.pop("runner")(
                progress_cb=lambda d, t, n: self.progress.emit(d, t, n),
                cancel=lambda: self._cancelled,
                **self._kw)
            self.finished_ok.emit(summary)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class AutoAnnotateDialog(QDialog):
    """批量预标注对话框（非模态；窗口随标注平台）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI 批量预标注")
        self.setMinimumWidth(560)
        self._thread = None

        v = QVBoxLayout(self)
        form = QFormLayout()
        row = QHBoxLayout()
        self.model_edit = QLineEdit()
        btn_model = QPushButton("浏览…")
        btn_model.clicked.connect(self._pick_model)
        row.addWidget(self.model_edit, stretch=1)
        row.addWidget(btn_model)
        form.addRow("模型权重（.pt）", row)

        row2 = QHBoxLayout()
        self.folder_edit = QLineEdit()
        btn_dir = QPushButton("浏览…")
        btn_dir.clicked.connect(self._pick_folder)
        row2.addWidget(self.folder_edit, stretch=1)
        row2.addWidget(btn_dir)
        form.addRow("图片文件夹", row2)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setMinimumWidth(120)
        form.addRow("置信度", self.conf_spin)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.1, 0.9)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setMinimumWidth(120)
        form.addRow("IOU", self.iou_spin)
        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0.0, 0.02)
        self.eps_spin.setSingleStep(0.001)
        self.eps_spin.setDecimals(4)
        self.eps_spin.setValue(0.002)
        self.eps_spin.setToolTip("掩码转多边形的简化比例（0=不简化，顶点最密）")
        self.eps_spin.setMinimumWidth(120)
        form.addRow("多边形简化比例", self.eps_spin)
        self.skip_check = QCheckBox("跳过已有同名 JSON 的图片")
        self.skip_check.setChecked(True)
        form.addRow("", self.skip_check)
        try:
            from ..ui_widgets import load_prefs

            _p = load_prefs()
            self.model_edit.setText(_p.get("aa_model") or "")
            self.folder_edit.setText(_p.get("aa_folder") or "")
        except Exception:  # noqa: BLE001  偏好缺失不影响对话框
            pass
        v.addLayout(form)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        v.addWidget(self.bar)
        self.status = QLabel("选择模型与文件夹后点「开始」")
        self.status.setWordWrap(True)
        v.addWidget(self.status)

        btns = QDialogButtonBox()
        self.btn_start = btns.addButton("开始", QDialogButtonBox.AcceptRole)
        self.btn_cancel_run = btns.addButton("取消运行", QDialogButtonBox.DestructiveRole)
        self.btn_close = btns.addButton(QDialogButtonBox.Close)
        self.btn_cancel_run.setEnabled(False)
        self.btn_start.clicked.connect(self._start)
        self.btn_cancel_run.clicked.connect(self._cancel_run)
        self.btn_close.clicked.connect(self.close)
        v.addWidget(btns)

    # ---------- 选择 ----------
    def _pick_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重", "", "模型 (*.pt)")
        if path:
            self.model_edit.setText(path)

    def _pick_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self.folder_edit.setText(path)

    # ---------- 运行 ----------
    def _start(self):
        from . import auto_annotate

        model_path = self.model_edit.text().strip()
        folder = self.folder_edit.text().strip()
        if not model_path or not Path(model_path).is_file():
            self.status.setText("请选择有效的模型权重文件")
            return
        if not folder:
            self.status.setText("请选择图片文件夹")
            return
        self.btn_start.setEnabled(False)
        self.btn_cancel_run.setEnabled(True)
        self.bar.setValue(0)
        try:
            from ..ui_widgets import save_prefs

            save_prefs({"aa_model": model_path, "aa_folder": folder})
        except Exception:  # noqa: BLE001  记忆失败不影响运行
            pass
        self._thread = _AnnotateThread(dict(
            runner=auto_annotate.annotate_folder,
            model_path=model_path, folder=folder,
            conf=float(self.conf_spin.value()), iou=float(self.iou_spin.value()),
            skip_annotated=self.skip_check.isChecked(),
            eps_ratio=float(self.eps_spin.value())))
        self._thread.progress.connect(self._on_progress)
        self._thread.finished_ok.connect(self._on_done)
        self._thread.failed.connect(self._on_failed)
        auto_cleanup_thread(self, "_thread", self._thread)
        self._thread.start()

    def _cancel_run(self):
        if self._thread is not None:
            self._thread.cancel()
            self.status.setText("取消中…（当前图完成后停止）")

    def _on_progress(self, done: int, total: int, name: str):
        self.bar.setValue(int(done * 100 / max(total, 1)))
        self.status.setText(f"{done}/{total}：{name}")

    def _on_done(self, summary: dict):
        self.btn_start.setEnabled(True)
        self.btn_cancel_run.setEnabled(False)
        self.bar.setValue(100)
        text = (f"完成：{summary['annotated']} 张已标注，{summary['skipped']} 张跳过"
                f"，{summary['failed']} 张失败"
                + ("（已取消，未跑完全部）" if summary.get("cancelled") else ""))
        self.status.setText(text)

    def _on_failed(self, msg: str):
        self.btn_start.setEnabled(True)
        self.btn_cancel_run.setEnabled(False)
        self.status.setText(f"失败：{msg}")

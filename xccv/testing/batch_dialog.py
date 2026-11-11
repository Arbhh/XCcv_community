"""测试平台「批量运行」设置弹窗（用户批注 To3.2：点击批量运行先弹设置）。

可配置：输入文件夹、输出目录、是否导出检出结果 JSON、是否保存
渲染图片（画框/掩码可视化，与 JSON 一一对应）及图片质量。
"""

from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
)


@dataclass
class BatchSettings:
    """一次批量运行的设置（out_dir 为空时由调用方回退为输入文件夹）。

    json_format：检出结果导出格式（P8/T8-1）——
    xcv=统一检出 JSON（每图一份，平台契约）；labelme=每图一份（标注平台
    可直接打开复核）；coco=合并为一份 results_coco.json。
    """

    folder: str = ""
    out_dir: str = ""
    save_json: bool = True
    save_images: bool = True
    jpg_quality: int = 95
    json_format: str = "xcv"


class BatchSettingsDialog(QDialog):
    """批量运行设置。accept 后用 settings() 取值。"""

    def __init__(self, parent=None, folder: str = ""):
        super().__init__(parent)
        self.setWindowTitle("批量运行设置")
        self.setMinimumWidth(560)

        form = QFormLayout(self)
        form.setContentsMargins(18, 16, 18, 14)
        form.setSpacing(10)

        self.ed_folder = QLineEdit(folder)
        btn_folder = QPushButton("浏览…")
        btn_folder.clicked.connect(self._browse_folder)
        row_folder = QHBoxLayout()
        row_folder.addWidget(self.ed_folder, stretch=1)
        row_folder.addWidget(btn_folder)
        form.addRow("图片文件夹", row_folder)

        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText("留空 = 保存到图片文件夹")
        btn_out = QPushButton("浏览…")
        btn_out.clicked.connect(self._browse_out)
        row_out = QHBoxLayout()
        row_out.addWidget(self.ed_out, stretch=1)
        row_out.addWidget(btn_out)
        form.addRow("输出目录", row_out)

        self.ck_json = QCheckBox("导出检出结果（格式见下；xcv/labelme 为每图一份同名文件）")
        self.ck_json.setChecked(True)
        form.addRow(self.ck_json)
        from PyQt5.QtWidgets import QComboBox

        self.fmt_combo = QComboBox()
        self.fmt_combo.addItem("统一检出 JSON（每图一份，平台契约，C# 直读）", "xcv")
        self.fmt_combo.addItem("labelme（每图一份，标注平台可直接打开复核）", "labelme")
        self.fmt_combo.addItem("COCO（全部图合并为一份 results_coco.json）", "coco")
        form.addRow("检出结果格式", self.fmt_combo)
        self.ck_images = QCheckBox("保存渲染图片（画框/掩码可视化，存于输出目录「可视化」文件夹）")
        self.ck_images.setChecked(True)
        self.ck_images.toggled.connect(self._toggle_quality)
        form.addRow(self.ck_images)
        self.sp_quality = QSpinBox()
        self.sp_quality.setRange(50, 100)
        self.sp_quality.setValue(95)
        self.sp_quality.setToolTip("渲染图 JPEG 质量（越高越清晰、占用越大）")
        form.addRow("图片质量", self.sp_quality)

        note = QLabel("批量运行会对文件夹（含子文件夹）逐张应用当前模型与检出流程。")
        note.setObjectName("hint")
        form.addRow(note)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.button(QDialogButtonBox.Ok).setText("开始批量运行")
        box.button(QDialogButtonBox.Cancel).setText("取消")
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

    # ---- 浏览 / 校验 ----
    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择批量推理的图片文件夹（含子文件夹）")
        if d:
            self.ed_folder.setText(d)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            self.ed_out.setText(d)

    def _toggle_quality(self, on: bool):
        self.sp_quality.setEnabled(on)

    def _accept(self):
        folder = self.ed_folder.text().strip()
        if not folder or not Path(folder).is_dir():
            show_warning(self, "请选择有效的图片文件夹")
            return
        if not (self.ck_json.isChecked() or self.ck_images.isChecked()):
            show_warning(self, "「导出 JSON」与「保存渲染图片」至少勾选一项，"
                               "否则批量运行没有产出")
            return
        self.accept()

    def settings(self) -> BatchSettings:
        return BatchSettings(
            folder=self.ed_folder.text().strip(),
            out_dir=self.ed_out.text().strip(),
            save_json=self.ck_json.isChecked(),
            save_images=self.ck_images.isChecked(),
            jpg_quality=self.sp_quality.value(),
            json_format=self.fmt_combo.currentData() or "xcv",
        )


def show_warning(dialog, text: str):
    """设置校验提示（独立函数便于测试打桩）。"""
    from PyQt5.QtWidgets import QMessageBox

    QMessageBox.warning(dialog, "批量运行设置", text)


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402

apply_auto_slot_module(globals())

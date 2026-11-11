import PIL.Image
import PIL.ImageEnhance
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage


class BrightnessContrastDialog(QtWidgets.QDialog):
    _base_value = 50

    def __init__(self, img, callback, parent=None):
        super(BrightnessContrastDialog, self).__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Brightness/Contrast")

        assert isinstance(img, PIL.Image.Image)
        # PNG 可能是 RGBA/L 模式：直接按 RGB888 重解释会内存错位（卡死/闪退），
        # 统一转 RGB；缓冲持有在 self._buf 上（QImage 不拷贝字节）
        self._src = img.convert("RGB")
        self._buf = b""
        self.callback = callback
        # 防抖：拖动滑条时每 tick 全图 PIL 运算会卡死 UI（大图尤甚），
        # 停止 150ms 后才真正计算
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self.onNewValue)

        sliders = {}
        layouts = {}
        for title in ["Brightness:", "Contrast:"]:
            layout = QtWidgets.QHBoxLayout()
            title_label = QtWidgets.QLabel(self.tr(title))
            title_label.setFixedWidth(75)
            layout.addWidget(title_label)
            #
            slider = QtWidgets.QSlider(Qt.Horizontal)
            slider.setRange(0, 3 * self._base_value)
            slider.setValue(self._base_value)
            layout.addWidget(slider)
            #
            value_label = QtWidgets.QLabel(f"{slider.value() / self._base_value:.2f}")
            value_label.setAlignment(Qt.AlignRight)
            layout.addWidget(value_label)
            #
            # lambda 必须接收 *args：Cython 编译态下编译函数无 code object，
            # PyQt 无法按签名裁剪 valueChanged(int) 的实参，无参 lambda 会
            # 被多传参 TypeError → 进程 abort（坑清单 5.1.1 同族）
            slider.valueChanged.connect(
                lambda *_, s=slider, lb=value_label:
                    lb.setText(f"{s.value() / self._base_value:.2f}"))
            slider.valueChanged.connect(lambda *_: self._debounce.start())
            layouts[title] = layout
            sliders[title] = slider

        self.slider_brightness = sliders["Brightness:"]
        self.slider_contrast = sliders["Contrast:"]
        del sliders

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layouts["Brightness:"])
        layout.addLayout(layouts["Contrast:"])
        del layouts
        self.setLayout(layout)

    def onNewValue(self, *_):
        brightness = self.slider_brightness.value() / self._base_value
        contrast = self.slider_contrast.value() / self._base_value

        img = self._src
        if brightness != 1:
            img = PIL.ImageEnhance.Brightness(img).enhance(brightness)
        if contrast != 1:
            img = PIL.ImageEnhance.Contrast(img).enhance(contrast)

        self._buf = img.tobytes()
        qimage = QImage(
            self._buf, img.width, img.height, img.width * 3, QImage.Format_RGB888
        )
        self.callback(qimage)


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402
apply_auto_slot_module(globals())

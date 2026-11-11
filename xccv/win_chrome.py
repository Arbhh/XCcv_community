"""XCcv 无边框窗口基座（v0.5.5 终版，依用户反馈第三轮定案）。

- 四角圆角回归：WA_TranslucentBackground + chromeShell 12px 圆角
  （顶栏圆上角、内容圆下角，最大化自动拍平）——上一版"未生效"的根因
  （第三方标题栏脱节）已随自研基座移除，本轮四角全部生效；
- 顶栏：右侧 最小化/最大化/关闭 三键改为 **QPainter 矢量绘制**
  （统一 12px 视觉尺寸，替代字符字形——此前"—"明显偏大）；顶栏
  **去除底部分割线**；左侧图标/标题可隐藏（主界面由侧栏品牌上移填充，
  避免重复字样）；
- 拖拽/双击最大化/边缘缩放：沿用 WM_NCHITTEST + startSystemMove。
"""

import ctypes
import ctypes.wintypes

from PyQt5.QtCore import QPoint, Qt, QEvent  # noqa: F401  QPoint: NCHITTEST 坐标换算
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget,
)

from .ui_style import icon_path, token

_EDGE = 6
_HEADER_H = 38
_RADIUS = 12
_HT = {"L": 10, "R": 11, "T": 12, "TL": 13, "TR": 14, "B": 15,
       "BL": 16, "BR": 17}
_HT_CAPTION = 2


class ChromeButton(QPushButton):
    """顶栏窗口按钮：矢量绘制（min=横线 / max=方框 / close=斜叉），统一
    12px 视觉尺寸；悬停底色高亮，关闭悬停红底白叉。"""

    def __init__(self, kind: str, tip: str, handler, parent=None):
        super().__init__(parent)
        self._kind = kind
        self.setObjectName("chrome" + kind.capitalize())
        self.setToolTip(tip)
        self.setFixedSize(44, _HEADER_H)
        self.setCursor(Qt.PointingHandCursor)
        self.setFlat(True)
        if handler is not None:
            self.clicked.connect(handler)
        self._load_colors()
        self.setMouseTracking(True)

    def _load_colors(self):
        self._c_line = QColor(token("text_2", "#5f7086"))
        self._c_line_hover = QColor(token("text_1", "#1f2937"))
        self._c_hover_bg = QColor(token("btn_hover", "#f0f7fb"))
        self._c_close_bg = QColor(token("danger", "#e05252"))

    def refresh_theme(self):
        """主题实时切换（V0.6 F2-5）：矢量键颜色按当前 token 重取。"""
        self._load_colors()
        self.update()

    def paintEvent(self, event):
        from PyQt5.QtGui import QColor as _C

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        hover = self.underMouse() and self.isEnabled()
        if hover:
            if self._kind == "close":
                p.fillRect(self.rect(), self._c_close_bg)
            else:
                p.fillRect(self.rect(), self._c_hover_bg)
        ink = QColor("#ffffff") if (self._kind == "close" and hover) else (
            self._c_line_hover if hover else self._c_line)
        cx, cy = self.width() / 2.0, self.height() / 2.0
        pen = QPen(ink, 1.6)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        if self._kind == "min":
            p.drawLine(int(cx - 6), int(cy), int(cx + 6), int(cy))
        elif self._kind == "max":
            p.drawRoundedRect(int(cx - 6), int(cy - 6), 12, 12, 2, 2)
        else:   # close
            p.drawLine(int(cx - 5), int(cy - 5), int(cx + 5), int(cy + 5))
            p.drawLine(int(cx - 5), int(cy + 5), int(cx + 5), int(cy - 5))

    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)


def _header_qss() -> str:
    return f"""
    #chromeHeader {{
        background: {token("panel")};
        border-bottom: 1px solid {token("divider")};
        border-top-left-radius: {_RADIUS}px;
        border-top-right-radius: {_RADIUS}px;
    }}
    #chromeHeader[maximized="true"] {{
        border-top-left-radius: 0; border-top-right-radius: 0;
    }}
    #chromeHeader QLabel {{
        color: {token("text_2")}; font-size: 15px;
        background: transparent; border: none;
    }}
    """


def _shell_qss() -> str:
    return f"""
    #chromeShell {{
        background: {token("surface")};
        border: 1px solid {token("border")};
        border-radius: {_RADIUS}px;
    }}
    #chromeShell[maximized="true"] {{
        border-radius: 0; border: none;
    }}
    #chromeBody {{
        background: transparent;
        border-bottom-left-radius: {_RADIUS}px;
        border-bottom-right-radius: {_RADIUS}px;
    }}
    #chromeBody[maximized="true"] {{
        border-bottom-left-radius: 0; border-bottom-right-radius: 0;
    }}
    #chromePage {{ background: transparent; }}
    """


class ChromeMainWindow(QMainWindow):
    """XCcv 无边框窗口基座：圆角壳 + 顶栏（品牌位 + 矢量三键）。"""

    def __init__(self, icon_file: str = "app.png", show_title: bool = True):
        super().__init__()
        self._icon_file = icon_file
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        from PyQt5.QtGui import QIcon

        self.setWindowIcon(QIcon(str(icon_path(icon_file))))
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        shell = QWidget()
        shell.setObjectName("chromeShell")
        shell.setStyleSheet(_shell_qss())
        outer = QVBoxLayout(shell)
        outer.setContentsMargins(1, 1, 1, 1)   # 1px 让边框描边可见
        outer.setSpacing(0)

        self._header = self._build_header(show_title)
        outer.addWidget(self._header)
        self._body = QWidget()
        self._body.setObjectName("chromeBody")
        self._body.setMouseTracking(True)
        body_lay = QVBoxLayout(self._body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)
        outer.addWidget(self._body, stretch=1)

        self.setCentralWidget(shell)
        self._apply_maximized_flag()

    # ---- 顶栏 ----
    def _build_header(self, show_title: bool) -> QWidget:
        header = QWidget()
        header.setObjectName("chromeHeader")
        self._header_h = _HEADER_H   # 可经 set_header_height 增高（品牌位）
        header.setFixedHeight(self._header_h)
        header.setMouseTracking(True)
        lay = QHBoxLayout(header)
        lay.setContentsMargins(12, 0, 0, 0)
        lay.setSpacing(8)

        self._icon_lbl = QLabel()
        self._icon_lbl.setPixmap(QPixmap(
            str(icon_path(self._icon_file))).scaled(
            18, 18, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lay.addWidget(self._icon_lbl)
        self._title_lbl = QLabel(self.windowTitle() or "XCcv")
        lay.addWidget(self._title_lbl)
        if not show_title:
            self._icon_lbl.hide()
            self._title_lbl.hide()
        self._brand_host = lay   # 品牌位（主界面注入侧栏品牌）
        lay.addStretch(1)

        self._btn_min = ChromeButton("min", "最小化", self.showMinimized)
        self._btn_max = ChromeButton("max", "最大化 / 还原", self._toggle_maximize)
        self._btn_close = ChromeButton("close", "关闭", self.close)
        for b in (self._btn_min, self._btn_max, self._btn_close):
            lay.addWidget(b)
        header.setStyleSheet(_header_qss())
        return header

    def getTitleBar(self):
        """兼容旧测试名：返回顶栏。"""
        return self._header

    def refresh_theme(self):
        """主题实时切换（V0.6 F2-5）：圆角壳/顶栏的 token 样式重算。"""
        shell = self.centralWidget()
        if shell is not None and shell.objectName() == "chromeShell":
            shell.setStyleSheet(_shell_qss())
        self._header.setStyleSheet(_header_qss())

    def add_header_widget(self, widget: QWidget, index: int = 1) -> None:
        """顶栏中部插入业务控件（默认紧跟品牌/图标位、弹簧之前）。"""
        self._header.layout().insertWidget(index, widget)

    def add_header_stretch(self, index: int) -> None:
        """顶栏弹簧位（业务控件组与窗口按钮之间）。"""
        self._header.layout().insertStretch(index)

    def take_header_buttons(self) -> tuple:
        """取出三枚窗口键（不再置于顶栏布局，交由调用方自定义排布）。

        键的信号已接线（最小化/最大化/关闭），换父后依旧生效。
        """
        lay = self._header.layout()
        btns = (self._btn_min, self._btn_max, self._btn_close)
        for b in btns:
            lay.removeWidget(b)
            b.setParent(None)
        return btns

    def set_header_height(self, h: int) -> None:
        """顶栏增高（品牌较高的窗口用）；拖拽/命中判定区域随动。"""
        self._header_h = max(_HEADER_H, int(h))
        self._header.setFixedHeight(self._header_h)

    def set_header_brand(self, widget: QWidget) -> None:
        """把品牌部件注入顶栏左侧（并隐藏默认图标/标题，避免重复字样）。"""
        lay = self._header.layout()
        lay.insertWidget(0, widget)
        self._icon_lbl.hide()
        self._title_lbl.hide()

    def set_chrome_central(self, widget: QWidget) -> None:
        """挂载业务页面（顶栏之下；页面背景透明，由壳体提供圆角底色）。"""
        if not widget.objectName():
            widget.setObjectName("chromePage")
        extra = "#chromePage { background: transparent; }"
        cur = self.styleSheet()
        if "chromePage" not in cur:
            self.setStyleSheet(cur + extra)
        self._body.layout().addWidget(widget)

    def setWindowTitle(self, title: str) -> None:  # noqa: N802（Qt 命名）
        super().setWindowTitle(title)
        self._title_lbl.setText(title)

    def _toggle_maximize(self):
        if self.windowState() & Qt.WindowMaximized:
            self.showNormal()
        else:
            self.showMaximized()

    # ---- 命中判定：交互控件不参与拖拽 ----
    def _is_draggable_zone(self, pos) -> bool:
        child = self.childAt(pos)
        # 祖先链上带 _chrome_drag_exempt 的业务控件不参与拖拽
        # （QLabel 子件会被下方规则误判为可拖区，如服务胶囊/搜索框）
        w = child
        while isinstance(w, QWidget):
            if getattr(w, "_chrome_drag_exempt", False):
                return False
            w = w.parentWidget()
        if child is self._header or isinstance(child, QLabel):
            return True
        return False

    def mouseDoubleClickEvent(self, event):
        if (event.pos().y() <= self._header_h
                and self._is_draggable_zone(event.pos())):
            self._toggle_maximize()
        super().mouseDoubleClickEvent(event)

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            self._apply_maximized_flag()
        return super().changeEvent(event)

    def _apply_maximized_flag(self):
        maximized = bool(self.windowState() & Qt.WindowMaximized)
        for w in (self._header, self.centralWidget()):
            if w is None:
                continue
            if w.property("maximized") != maximized:
                w.setProperty("maximized", maximized)
                w.style().unpolish(w)
                w.style().polish(w)

    # ---- 边缘缩放：WM_NCHITTEST 原生命中 ----
    def nativeEvent(self, eventType, message):
        if eventType == b"windows_generic_MSG":
            msg = ctypes.wintypes.MSG.from_address(int(message))
            if msg.message == 0x0084:   # WM_NCHITTEST
                x = ctypes.c_short(msg.lParam & 0xFFFF).value
                y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
                pos = self.mapFromGlobal(QPoint(x, y))
                w, h = self.width(), self.height()
                m = _EDGE
                left = pos.x() <= m
                right = pos.x() >= w - m
                top = pos.y() <= m
                bottom = pos.y() >= h - m
                if top and left:
                    return True, _HT["TL"]
                if top and right:
                    return True, _HT["TR"]
                if bottom and left:
                    return True, _HT["BL"]
                if bottom and right:
                    return True, _HT["BR"]
                if left:
                    return True, _HT["L"]
                if right:
                    return True, _HT["R"]
                if top:
                    return True, _HT["T"]
                if bottom:
                    return True, _HT["B"]
                if pos.y() <= self._header_h and self._is_draggable_zone(pos):
                    return True, _HT_CAPTION
        return super().nativeEvent(eventType, message)


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402

apply_auto_slot_module(globals())

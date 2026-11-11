"""UI 升级公共组件（v0.5 UI 批次）：徽章 / 状态点 / Toast / 空状态 / 图标绘制。

设计 Token 见 ui_style.C/R/SP；本模块只依赖 ui_style 与轻量路径工具，
禁 torch / QtWebEngine 依赖（P66：可复用逻辑不进重模块）。

Cython 约束：信号一律连 self 具名方法（禁无参 lambda——P06/P48）；
本模块末尾 apply_auto_slot_module(globals())。
"""

from datetime import datetime
import math

from PyQt5.QtCore import QEasingCurve, QPointF, QRectF, Qt, QTimer, QVariantAnimation, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QFrame, QGraphicsDropShadowEffect, QGraphicsOpacityEffect, QLabel,
    QVBoxLayout, QWidget,
)

from . import app_paths
from .ui_style import C

__all__ = [
    "Badge", "StatusDot", "EmptyState", "HoverCard", "PlatformCard", "toast",
    "nav_icon", "gradient_tile", "relative_time", "load_prefs", "save_prefs",
]


# ============================================================
# 徽章 Badge
# ============================================================

_KIND_OBJECT = {
    "primary": "badge",
    "info": "badge",
    "orange": "badgeDet",
    "success": "badgeSeg",
    "warn": "badgeWarn",
    "danger": "badgeDanger",
    "neutral": "badgeNeutral",
}


class Badge(QLabel):
    """圆角胶囊徽章：Badge("检测", "orange") / Badge("运行中", "success")。"""

    def __init__(self, text: str = "", kind: str = "primary", parent=None):
        super().__init__(text, parent)
        self.set_kind(kind)

    def set_kind(self, kind: str) -> None:
        self._kind = kind if kind in _KIND_OBJECT else "primary"
        self.setObjectName(_KIND_OBJECT[self._kind])

    def kind(self) -> str:
        return self._kind


# ============================================================
# 状态点 StatusDot（可选呼吸动画）
# ============================================================

class StatusDot(QWidget):
    """彩色圆点状态指示。start_breath() 后透明度 0.35↔1.0 呼吸（1.6s 循环）。"""

    def __init__(self, diameter: int = 12, color: str = C.text_3, parent=None):
        super().__init__(parent)
        self._d = diameter
        self._color = QColor(color)
        self._alpha = 1.0
        self.setFixedSize(diameter + 2, diameter + 2)
        self._anim: QVariantAnimation | None = None

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def start_breath(self) -> None:
        if self._anim is not None:
            return
        anim = QVariantAnimation(self)
        anim.setDuration(1600)
        anim.setLoopCount(-1)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.InOutSine)
        anim.valueChanged.connect(self._on_breath)
        anim.finished.connect(self._on_breath_done)
        self._anim = anim
        anim.start()

    def stop_breath(self) -> None:
        if self._anim is None:
            return
        anim = self._anim
        self._anim = None
        anim.stop()
        anim.deleteLater()
        self._alpha = 1.0
        self.update()

    def breathing(self) -> bool:
        return self._anim is not None

    def _on_breath(self, v: float):
        # 0→1→0 三角波映射到透明度 0.35↔1.0
        wave = 1.0 - abs(2.0 * float(v) - 1.0)
        self._alpha = 0.35 + 0.65 * wave
        self.update()

    def _on_breath_done(self):
        self._anim = None
        self._alpha = 1.0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = QColor(self._color)
        color.setAlphaF(max(0.0, min(1.0, self._alpha)))
        p.setBrush(color)
        p.setPen(Qt.NoPen)
        d = self._d
        p.drawEllipse(QRectF((self.width() - d) / 2.0, (self.height() - d) / 2.0,
                             d, d))


# ============================================================
# Toast 轻提示（右下角浮出，2.5s 自动消失）
# ============================================================

_TOAST_KINDS = {"info": "toastInfo", "success": "toastSuccess",
                "warn": "toastWarn", "error": "toastError"}


class _ToastFrame(QLabel):
    """单条 Toast（父级为顶层窗口；淡入 → 停留 → 淡出自毁）。"""

    def __init__(self, parent: QWidget, text: str, kind: str, duration: int):
        super().__init__(parent)
        self.setObjectName(_TOAST_KINDS.get(kind, "toastInfo"))
        self.setText(text)
        self._duration = duration
        self._closing = False
        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(0.0)
        self.setGraphicsEffect(self._effect)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on_alpha)
        self._anim.finished.connect(self._on_fade_done)

    # ---- 动画 ----
    def play_in(self):
        self.adjustSize()
        win = self.parentWidget()
        if win is not None:
            x = win.width() - self.width() - 26
            y = win.height() - self.height() - 34
            self.move(max(12, x), max(12, y))
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()

    def _play_out(self):
        import sip

        if sip.isdeleted(self):   # 窗口已销毁本条（竞态：停留定时器晚于 close_now）
            return
        self._closing = True
        self._anim.setStartValue(self._effect.opacity())
        self._anim.setEndValue(0.0)
        self._anim.start()

    def close_now(self):
        self._closing = True
        try:
            self._release_from_window()
            self._anim.stop()
            self.close()
            self.deleteLater()
        except RuntimeError:
            pass   # C++ 对象已随 deleteLater 回收，Python 壳尚存——无需处理

    def _on_alpha(self, v: float):
        self._effect.setOpacity(float(v))

    def _release_from_window(self):
        # 淡出销毁后窗口仍持有死引用会导致下一条 toast 触发
        # RuntimeError（P129：训练平台按钮失效根因），销毁路径必须先摘除
        try:
            win = self.parentWidget()
        except RuntimeError:
            return
        if win is not None and getattr(win, "_active_toast", None) is self:
            win._active_toast = None

    def _on_fade_done(self):
        if self._closing:
            self._release_from_window()
            try:
                self.close()
                self.deleteLater()
            except RuntimeError:
                pass
        else:
            QTimer.singleShot(self._duration, self._play_out)


def toast(parent: QWidget, text: str, kind: str = "info",
          duration: int = 2500) -> _ToastFrame | None:
    """在 parent 所属窗口右下角弹一条轻提示（同窗旧 Toast 即刻让位）。

    parent 非真实 QWidget（如测试桩）时安全跳过。
    """
    if not isinstance(parent, QWidget):
        return None
    win = parent.window()
    if not isinstance(win, QWidget):
        return None
    old = getattr(win, "_active_toast", None)
    if old is not None:
        old.close_now()
    frame = _ToastFrame(win, text, kind, duration)
    win._active_toast = frame
    frame.show()
    frame.raise_()
    frame.play_in()
    return frame


# ============================================================
# 空状态 EmptyState
# ============================================================

class EmptyState(QWidget):
    """空状态占位：大图标 + 标题 + 一句提示（居中）。"""

    def __init__(self, glyph: str = "🗂", title: str = "暂无内容",
                 hint: str = "", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 24, 16, 24)
        lay.setSpacing(8)
        lay.setAlignment(Qt.AlignCenter)
        icon = QLabel(glyph)
        icon.setObjectName("emptyIcon")
        icon.setAlignment(Qt.AlignCenter)
        t = QLabel(title)
        t.setObjectName("emptyTitle")
        t.setAlignment(Qt.AlignCenter)
        lay.addWidget(icon)
        lay.addWidget(t)
        if hint:
            h = QLabel(hint)
            h.setObjectName("emptyHint")
            h.setAlignment(Qt.AlignCenter)
            h.setWordWrap(True)
            lay.addWidget(h)


# ============================================================
# 可点击卡片 HoverCard（悬停阴影抬升 + QSS hover 底色）
# ============================================================

class HoverCard(QFrame):
    """平台卡片基件：objectName=card（QSS 圆角/hover），静态阴影 + 点击信号。

    Q1 教训（2026-09-21）：**不要在运行中修改 QGraphicsDropShadowEffect 属性**
    （逐帧动画 blur/offset）——QSS 样式化的卡片会被渲染成空白（widget 消失
    但仍可点击，用户实测）。阴影只在构造时静态设置一次，悬停反馈交给
    QSS `#card:hover`（边框主色 + 底色变化）。

    Q2 教训：子类可能重定义 clicked 为带参信号（如 clicked[str]）——
    mouseReleaseEvent 统一经 `_emit_clicked()` 钩子发射，子类覆写该钩子。
    """

    clicked = pyqtSignal()

    def __init__(self, parent=None, shadow_color: str = "#178fb4"):
        super().__init__(parent)
        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)
        # 静态阴影：只在此处设置一次，之后绝不修改任何属性
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(16)
        self._shadow.setOffset(0, 3)
        col = QColor(shadow_color)
        col.setAlpha(40)
        self._shadow.setColor(col)
        self.setGraphicsEffect(self._shadow)

    def _emit_clicked(self):
        """点击发射钩子：带参信号的子类覆写此方法（勿覆写 mouseReleaseEvent）。"""
        self.clicked.emit()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._emit_clicked()
        super().mouseReleaseEvent(event)


# ============================================================
# 平台卡片 PlatformCard（AI 平台页 / 部署页共用）
# ============================================================

class PlatformCard(HoverCard):
    """渐变图标底 + 标题 + 描述 + 打开提示的平台卡片。"""

    def __init__(self, key: str, title: str, desc: str, icon: str,
                 grad: tuple, parent=None):
        super().__init__(parent)
        self.card_key = key
        self.card_title = title
        self.card_desc = desc
        self.setFixedSize(310, 240)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 20, 16)
        lay.setSpacing(10)

        icon_label = QLabel()
        icon_label.setPixmap(gradient_tile(str(icon), grad[0], grad[1], 76, 20))
        icon_label.setFixedSize(76, 76)
        lay.addWidget(icon_label)

        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        lay.addWidget(title_label)

        desc_label = QLabel(desc)
        desc_label.setObjectName("muted")
        desc_label.setWordWrap(True)
        lay.addWidget(desc_label, stretch=1)

        tip = QLabel("打开 →")
        tip.setStyleSheet(
            f"color: {grad[1]}; font-size: 15px; font-weight: bold;"
            "background: transparent; border: none;")
        tip.setAlignment(Qt.AlignRight)
        lay.addWidget(tip)


# ============================================================
# 图标绘制（导航 / 平台卡渐变底，避免新增二进制资源）
# ============================================================

def nav_icon(kind: str, color: str, size: int = 40) -> QIcon:
    """程序化绘制线性风格导航图标（ai/trad/flow/deploy/settings/doc/about）。"""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color), max(2.0, size * 0.055))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    s = size
    lo, hi = s * 0.18, s * 0.82
    mid = s / 2.0

    def star4(cx: float, cy: float, r_out: float, r_in: float) -> QPainterPath:
        path = QPainterPath()
        for i in range(8):
            ang = -90 + i * 45
            rad = ang * 3.141592653589793 / 180.0
            r = r_out if i % 2 == 0 else r_in
            pt = QPointF(cx + _cos(rad) * r, cy + _sin(rad) * r)
            if i == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        path.closeSubpath()
        return path

    if kind == "ai":  # 四角星 + 小星（AI 火花）
        p.fillPath(star4(mid, mid, s * 0.30, s * 0.12), QColor(color))
        p.fillPath(star4(s * 0.78, s * 0.24, s * 0.12, s * 0.05), QColor(color))
    elif kind == "trad":  # 调节滑杆
        for i, y in enumerate((0.3, 0.5, 0.7)):
            yv = s * y
            p.drawLine(QPointF(lo, yv), QPointF(hi, yv))
            knob_x = s * (0.62 if i != 1 else 0.38)
            p.setBrush(QColor("#ffffff"))
            p.drawEllipse(QPointF(knob_x, yv), s * 0.085, s * 0.085)
            p.setBrush(Qt.NoBrush)
    elif kind == "flow":  # 流程节点
        r = s * 0.13
        p.drawRoundedRect(QRectF(s * 0.32, lo, s * 0.36, s * 0.22), r, r)
        p.drawLine(QPointF(mid, s * 0.40), QPointF(mid, s * 0.52))
        p.drawLine(QPointF(s * 0.32, s * 0.52), QPointF(s * 0.68, s * 0.52))
        p.drawLine(QPointF(s * 0.32, s * 0.52), QPointF(s * 0.32, s * 0.62))
        p.drawLine(QPointF(s * 0.68, s * 0.52), QPointF(s * 0.68, s * 0.62))
        p.drawRoundedRect(QRectF(s * 0.16, s * 0.62, s * 0.32, s * 0.22), r, r)
        p.drawRoundedRect(QRectF(s * 0.52, s * 0.62, s * 0.32, s * 0.22), r, r)
    elif kind == "deploy":  # 上行箭头 + 底座（部署/发布）
        p.drawLine(QPointF(mid, s * 0.20), QPointF(mid, s * 0.62))
        p.drawLine(QPointF(mid, s * 0.20), QPointF(s * 0.36, s * 0.40))
        p.drawLine(QPointF(mid, s * 0.20), QPointF(s * 0.64, s * 0.40))
        p.drawLine(QPointF(lo, s * 0.80), QPointF(hi, s * 0.80))
    elif kind == "settings":  # 齿轮（圆 + 齿）
        p.drawEllipse(QPointF(mid, mid), s * 0.20, s * 0.20)
        for i in range(8):
            ang = i * 45 * 3.141592653589793 / 180.0
            r1, r2 = s * 0.26, s * 0.34
            p.drawLine(QPointF(mid + _cos(ang) * r1, mid + _sin(ang) * r1),
                       QPointF(mid + _cos(ang) * r2, mid + _sin(ang) * r2))
    elif kind == "doc":  # 文档页 + 行
        p.drawRoundedRect(QRectF(s * 0.26, lo, s * 0.48, s * 0.64), s * 0.06, s * 0.06)
        for y in (0.38, 0.5, 0.62):
            p.drawLine(QPointF(s * 0.36, s * y), QPointF(s * 0.64, s * y))
    elif kind == "about":  # 信息圈
        p.drawEllipse(QPointF(mid, mid), s * 0.30, s * 0.30)
        p.setBrush(QColor(color))
        p.drawEllipse(QPointF(mid, s * 0.42), s * 0.035, s * 0.035)
        p.setBrush(Qt.NoBrush)
        p.drawLine(QPointF(mid, s * 0.52), QPointF(mid, s * 0.66))
    else:  # 兜底：实心圆
        p.setBrush(QColor(color))
        p.drawEllipse(QPointF(mid, mid), s * 0.24, s * 0.24)
    p.end()
    return QIcon(pm)


def _cos(x: float) -> float:
    return math.cos(x)


def _sin(x: float) -> float:
    return math.sin(x)


def gradient_tile(icon_file: str, color_from: str = "", color_to: str = "",
                  size: int = 96, radius: int = 22) -> QPixmap:
    """平台图标瓦片：v0.5.2 起图标自带渐变圆角底（透明四角），此处仅高质量缩放。

    color_from/color_to 参数保留（旧调用兼容），不再参与绘制。
    """
    from pathlib import Path

    from .ui_style import icon_path

    src_path = Path(icon_file)
    if not src_path.is_absolute():
        src_path = icon_path(icon_file)
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    src = QPixmap(str(src_path))
    p.drawPixmap(0, 0, size, size, src)
    p.end()
    return pm


# ============================================================
# 相对时间 / UI 偏好持久化
# ============================================================

def relative_time(value) -> str:
    """把时间戳 / ISO 字符串转「N 分钟前」样式的相对时间。"""
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return value
    elif isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromtimestamp(float(value))
    delta = datetime.now() - dt
    secs = delta.total_seconds()
    if secs < 60:
        return "刚刚"
    if secs < 3600:
        return f"{int(secs // 60)} 分钟前"
    if secs < 86400:
        return f"{int(secs // 3600)} 小时前"
    if secs < 86400 * 30:
        return f"{int(secs // 86400)} 天前"
    return dt.strftime("%Y-%m-%d")


def _prefs_path():
    return app_paths.app_root() / "platform_data" / "ui_prefs.json"


def load_prefs() -> dict:
    """界面偏好（平台数据目录 ui_prefs.json；读不到给空表）。"""
    import json

    try:
        return json.loads(_prefs_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_prefs(prefs: dict) -> None:
    import json

    try:
        path = _prefs_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = load_prefs()
        merged.update(prefs)
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    except Exception:
        pass  # 偏好写失败不致影响功能


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402

apply_auto_slot_module(globals())

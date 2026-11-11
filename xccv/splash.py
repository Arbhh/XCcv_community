"""全局"加载中"启动界面（V0.6 T1-3）。

任意平台入口点击后立即（<500ms）显示：平台图标 + 名称 + "加载中…" +
不确定进度动画；超过 20s 提示"仍在加载，可继续等待或查看日志"。
平台窗口 show 之后由 finish() 关闭——先 setParent(None) 再 deleteLater，
避免残影（坑 P134 教训）。

只依赖 PyQt5 与 ui_style，不引入任何重型库，可在毫秒级显示。
绘制为圆角白色卡片 + 品牌色顶条，风格与主界面一致。
"""

from PyQt5.QtCore import QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap, QPainterPath
from PyQt5.QtWidgets import QApplication, QWidget

from .ui_style import ACCENT, FG, MUTED, PANEL

_PATIENCE_SECONDS = 20  # 超过该时长提示可继续等待或查看日志

_EARLY = None  # 入口顶部 show_early() 预显示的加载卡；show_splash() 接管（T1-3）


def _alive(widget) -> bool:
    try:
        from PyQt5 import sip

        return widget is not None and not sip.isdeleted(widget) and widget.isVisible()
    except Exception:
        return False


class LoadingSplash(QWidget):
    """无框置顶加载卡：轻量帧动画，窗口就绪后 finish() 关闭并销毁。"""

    def __init__(self, name: str, message: str = "", icon=None):
        super().__init__(None)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint
                            | Qt.WindowStaysOnTopHint)
        self.setFixedSize(460, 176)
        self._name = name
        self._message = message or "正在加载组件…"
        self._icon = icon  # QPixmap | None
        self._phase = 0.0  # 动画相位（0~1 循环）

        self._anim = QTimer(self)
        self._anim.setInterval(33)  # ~30fps，轻量
        self._anim.timeout.connect(self._tick)
        self._patience = QTimer(self)
        self._patience.setSingleShot(True)
        self._patience.timeout.connect(self._on_patience)

    # ---- 生命周期 ----
    @classmethod
    def show_splash(cls, name: str, message: str = "", icon=None) -> "LoadingSplash":
        """立即创建并显示加载卡（processEvents 保证先于窗口构建绘制）。

        入口顶部已 show_early() 预显示时直接接管同一张卡（更新名称/文案/
        图标；不闪烁、不重复创建；耐心计时从首次显示算起）。内容未变时
        **不重绘**：实测接管时的 processEvents 重绘又付一次 ~270ms 的中文
        字体加载（drawText），轻量平台主窗因此晚出 ~300ms。
        """
        global _EARLY
        early, _EARLY = _EARLY, None
        if _alive(early):
            changed = early._name != name or bool(message and early._message != message)
            early._name = name
            if message:
                early._message = message
            if icon is not None and early._icon is None:
                early._icon = icon
                changed = True
            if changed:
                early.update()
                QApplication.processEvents()
            return early
        splash = cls(name, message, icon)
        splash.show()
        splash._anim.start()
        splash._patience.start(_PATIENCE_SECONDS * 1000)
        QApplication.processEvents()
        return splash

    @classmethod
    def show_early(cls, name: str, message: str = "", icon=None) -> "LoadingSplash":
        """入口模块顶部预显示（模块级重型导入之前），随后由 show_splash 接管。"""
        global _EARLY
        splash = cls.show_splash(name, message, icon)
        _EARLY = splash
        return splash

    def title_text(self) -> str:
        """卡片标题：平台名已含品牌前缀时不重复（部署族标题为"XCcv 推理服务"）。"""
        return self._name if self._name.startswith("XCcv") else f"XCcv {self._name}"

    def finish(self, main_window=None) -> None:
        """平台窗口 show 之后调用：关闭 + setParent(None)+deleteLater（防残影）。"""
        global _EARLY
        if _EARLY is self:
            _EARLY = None
        self._anim.stop()
        self._patience.stop()
        self.close()
        if main_window is not None:
            try:  # 贴到主窗口中心再关闭，视觉上"加载卡融入平台窗口"
                geo = main_window.geometry()
                self.move(geo.center().x() - self.width() // 2,
                          geo.center().y() - self.height() // 2)
            except Exception:
                pass
        self.setParent(None)
        self.deleteLater()

    def _tick(self):
        self._phase = (self._phase + 0.02) % 1.0
        self.update()

    def _on_patience(self):
        self._message = "仍在加载，可继续等待或查看日志（logs/）"
        self.update()

    # ---- 绘制 ----
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        card = QRectF(self.rect().adjusted(6, 6, -6, -6))
        path = QPainterPath()
        path.addRoundedRect(card, 16, 16)
        p.fillPath(path, QColor(PANEL))
        # 顶部品牌色条
        bar = QPainterPath()
        bar.addRoundedRect(card.left(), card.top(), card.width(), 6, 3, 3)
        p.fillPath(bar, QColor(ACCENT))

        icon = self._icon
        text_left = int(card.left()) + 36
        if icon is not None and not icon.isNull():
            size = 56
            p.drawPixmap(int(card.left()) + 32, int(card.top()) + 40,
                         icon.scaled(size, size, Qt.KeepAspectRatio,
                                     Qt.SmoothTransformation))
            text_left = int(card.left()) + 108

        p.setPen(QColor(FG))
        p.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        p.drawText(text_left, int(card.top()) + 62, self.title_text())
        p.setPen(QColor(MUTED))
        font = p.font()
        font.setPointSize(9)
        font.setBold(False)
        p.setFont(font)
        y = int(card.top()) + 92
        for line in self._message.split("\n")[:2]:  # 逐行绘制（drawText(x,y,s) 不解析 \n）
            p.drawText(text_left, y, line)
            y += 22

        # 不确定进度条：一段高亮滑块在浅色槽内循环
        groove_y = int(card.top()) + 118
        groove_l, groove_r = int(card.left()) + 36, int(card.right()) - 36
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(ACCENT).darker(220))
        p.drawRoundedRect(groove_l, groove_y, groove_r - groove_l, 6, 3, 3)
        span = (groove_r - groove_l) * 0.28
        pos = groove_l + (groove_r - groove_l - span) * self._phase
        p.setBrush(QColor(ACCENT))
        p.drawRoundedRect(int(pos), groove_y, int(span), 6, 3, 3)
        p.end()


def make_splash(name: str, message: str = "") -> LoadingSplash:
    """兼容旧入口的工厂（v0.5.x 语义）：显示一张居中的加载提示卡。"""
    return LoadingSplash.show_splash(name, message)

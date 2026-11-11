"""XCcv Community 主界面：统一入口（平台卡片墙 + 最近项目 + 全局搜索）。

主界面本身保持轻量（不含 torch 等重型依赖）。点击平台卡片时以独立子进程
拉起标注 / 训练 / 测试 / 缺陷样本合成四个平台，平台崩溃互不影响。

运行：python -m xccv.app
"""

if __name__ == "__main__":  # V0.6 T1-3：-m 直启先亮加载中界面，再做下方重型模块级导入
    from xccv.platform_entry import early_splash

    early_splash("main", "主界面")

# QtWebEngine 早期初始化：主界面进程内要创建流程画布（FlowGraphWebWindow），
# 必须在 QApplication 建立前导入并开启共享 OpenGL 上下文（否则画布崩溃/白屏）。
from PyQt5.QtCore import Qt as _Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication as _QApp  # noqa: E402

if _QApp.instance() is None:
    _QApp.setAttribute(_Qt.AA_ShareOpenGLContexts, True)
try:
    import PyQt5.QtWebEngineWidgets  # noqa: F401,E402
except Exception:  # noqa: BLE001  环境缺件时流程图编辑会给出中文指引
    pass

import json
import os
import sys
from pathlib import Path

from .gui_utils import ensure_console_streams, install_excepthook

ensure_console_streams()  # pythonw 下 stdout/stderr 为 None，先兜底
os.environ.setdefault("QT_API", "pyqt5")

from PyQt5.QtCore import QEvent, QSize, Qt, QTimer, QVariantAnimation, pyqtSignal  # noqa: E402
from PyQt5.QtGui import QIcon, QPixmap  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication, QDialog, QFrame, QGraphicsOpacityEffect, QGridLayout,
    QComboBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QStackedWidget, QVBoxLayout, QWidget,
)

from . import __version__, launcher  # noqa: E402  版本号统一在包 __init__ 维护
from .app_paths import app_root  # noqa: E402
from .ui_style import C, apply_app_theme, bind_style, token  # noqa: E402
from .ui_widgets import (  # noqa: E402
    HoverCard, PlatformCard, StatusDot, gradient_tile, nav_icon,
    relative_time,
)
from .win_chrome import ChromeMainWindow  # noqa: E402  自定义标题栏基座
_ASSETS = Path(__file__).resolve().parent / "assets"

# 平台卡片：key / 标题 / 描述 / 图标 / 渐变底（起, 止）
_PLATFORMS = [
    ("annotation", "标注平台", "绘制矩形 / 多边形，自动保存 labelme json；AI 智能标注",
     "annotation.png", ("#5b9df5", "#3a7ce0")),
    ("training", "训练平台", "项目管理、数据集可视化、增强策略、参数配置与实时训练曲线",
     "training.png", ("#f6a13c", "#ef7f1a")),
    ("testing", "测试平台", "加载模型推理，置信度 / IOU 实时调节，bbox 与掩码可视化",
     "testing.png", ("#57c793", "#2fae66")),
    ("defectgen", "缺陷样本合成", "缺陷样本合成工具：把缺陷贴图自然融合到良品图，扩充训练样本",
     "defectgen.png", ("#9d6bf0", "#7c4fe0")),
]

# 左侧导航：图标 + 文字；底部系统区（设置/文档/关于）
_NAV_ITEMS = [("ai", "AI平台", "ai"), ("flow", "流程图", "flow")]
_PAGE_TITLES = {"ai": "AI 平台", "flow": "流程图"}


def icon_path(name: str) -> Path:
    return _ASSETS / name


def recent_projects(limit: int = 4, root: Path | None = None) -> list[dict]:
    """最近项目（按 project.json 修改时间倒序；主界面轻量直读 JSON）。"""
    base = root or (app_root() / "platform_data" / "projects")
    rows = []
    if base.is_dir():
        for d in base.iterdir():
            pf = d / "project.json"
            if not (d.is_dir() and pf.is_file()):
                continue
            try:
                data = json.loads(pf.read_text(encoding="utf-8"))
                mtime = pf.stat().st_mtime
            except (OSError, ValueError):
                continue
            rows.append({"name": d.name, "task": str(data.get("task") or ""),
                         "mtime": mtime})
    rows.sort(key=lambda r: -r["mtime"])
    return rows[:limit]


class _RecentChip(QFrame):
    """最近项目小卡：项目名 + 相对时间，点击直达训练平台并打开该项目。"""

    clicked = pyqtSignal(str)

    def __init__(self, name: str, when: str, parent=None):
        super().__init__(parent)
        self.setObjectName("recentChip")
        self.project_name = name
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"打开项目「{name}」")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 9)
        lay.setSpacing(2)
        title = QLabel(name)
        bind_style(title, lambda: f"font-size:16px; font-weight:bold; color:{token('text_1')};")
        time_label = QLabel(when)
        bind_style(time_label, lambda: f"font-size:15px; color:{token('text_3')};")
        lay.addWidget(title)
        lay.addWidget(time_label)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.project_name)
        super().mouseReleaseEvent(event)


_PlatformCard = PlatformCard   # 平台卡片统一由 ui_widgets 提供（部署页共用）


class HomeWindow(ChromeMainWindow):
    def __init__(self):
        # 品牌由侧栏上移至顶栏（去除顶栏重复字样，见 v0.5.5 批注）
        super().__init__(show_title=False)
        self.setWindowTitle("XCcv 视觉模型训练平台（社区版）")
        self.setWindowIcon(QIcon(str(icon_path("app.png"))))
        self.resize(1280, 780)
        self._platform_cards = []
        self._recent_chips = []
        self._build_ui()

    # ---- UI ----
    def _build_ui(self):
        central = QWidget()
        apply_app_theme(central)  # 仅作用于主界面子树；主题切换实时重设（F2-5）
        central.setObjectName("homeCentral")
        central.setStyleSheet(
            central.styleSheet()
            + "\nQWidget#homeCentral { background: transparent; }")
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- 左侧导航（图标 + 文字；底部系统区）----
        side = QFrame()
        side.setFixedWidth(200)
        bind_style(side, lambda: (
            f"background:{token('nav_side_bg', '#f7fafd')};"
            f" border-right:1px solid {token('border', '#e2ecf7')};"
            " border-bottom-left-radius: 12px;"))
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(14, 12, 14, 16)
        side_lay.setSpacing(10)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("navList")
        for _key, text, icon_kind in _NAV_ITEMS:
            self.nav_list.addItem(QListWidgetItem(
                nav_icon(icon_kind, token("text_2", "#5f7086"), 45), text))
        self.nav_list.setIconSize(QSize(45, 45))
        self.nav_list.setCurrentRow(0)
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)
        side_lay.addWidget(self.nav_list, stretch=1)

        # 品牌区改为固定尺寸的左上角块（随 build_header_content 重构，
        # 不再随窗口边框/标题栏伸缩），页标题/搜索/胶囊一并并入窗口顶栏
        self._build_header_content()

        # 底部系统区（占位动作，设置中心后续版本接入）
        self._nav_foot = []
        for text, kind, handler in (
                ("设置", "settings", self._show_settings_placeholder),
                ("使用文档", "doc", self._open_docs),
                ("关于", "about", self._show_about)):
            btn = QPushButton(text)
            btn.setObjectName("navFoot")
            btn.setIcon(nav_icon(kind, token("text_2", "#5f7086"), 36))
            btn.setIconSize(QSize(36, 36))
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(handler)
            side_lay.addWidget(btn)
            self._nav_foot.append((btn, kind))
        outer.addWidget(side)

        # ---- 右侧：页面栈（页标题/搜索/服务胶囊已并入窗口顶栏）----
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_ai_page())    # 平台页
        from .flow_platform import FlowPage  # 流程图页（图片树 / 运行 / 画布入口）

        self.stack.addWidget(FlowPage())               # 流程图
        right_lay.addWidget(self.stack, stretch=1)
        outer.addWidget(right, stretch=1)
        # 无边框基座：内容追加到标题栏之下（setCentralWidget 会覆盖标题栏）
        self.set_chrome_central(central)

    # ---- 窗口顶栏（品牌角块 + 原页面顶栏内容合并）----
    def _build_header_content(self):
        """品牌区做成固定尺寸的左上角块，不再随窗口边框（标题栏）伸缩。

        蓝框规格（用户批注）：宽 = 侧栏宽（200），底边与页标题行分割线
        齐平（顶栏总高 120 = 原标题栏 58 + 原页面顶栏 62）；logo 48px。
        原页面顶栏的标题/搜索/服务胶囊移入顶栏弹簧两侧，搜索框与胶囊
        标记 _chrome_drag_exempt（可点击，不被当作标题栏拖拽区）。
        """
        brand = QWidget()
        brand.setFixedWidth(200)
        brand.setStyleSheet("background: transparent; border: none;")
        b_lay = QHBoxLayout(brand)
        b_lay.setContentsMargins(20, 0, 4, 0)
        b_lay.setSpacing(14)
        b_logo = QLabel()
        b_logo.setPixmap(QPixmap(str(icon_path("app.png"))).scaled(
            48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        b_logo.setFixedSize(48, 48)
        b_text = QVBoxLayout()
        b_text.setContentsMargins(0, 0, 0, 0)
        b_text.setSpacing(2)
        b_big = QLabel("XCcv")
        bind_style(b_big, lambda: f"font-size:20px; font-weight:bold; color:{token('text_1')};"
                                  "background:transparent; border:none;")
        b_small = QLabel("视觉模型训练平台")
        bind_style(b_small, lambda: f"font-size:14px; color:{token('text_2')};"
                                    "background:transparent; border:none;")
        b_text.addWidget(b_big)
        b_text.addWidget(b_small)
        b_lay.addWidget(b_logo, 0, Qt.AlignVCenter)
        b_lay.addLayout(b_text)
        b_lay.setAlignment(b_text, Qt.AlignVCenter)
        self.set_header_brand(brand)
        self.set_header_height(120)   # 58（原标题栏）+ 62（原页面顶栏）

        # 页标题紧随品牌角块；搜索框与服务胶囊移到顶栏右侧红框位
        # （窗口键上移贴顶、其下右对齐），不再占据标题行中部
        self.page_title = QLabel(_PAGE_TITLES["ai"])
        self.page_title.setObjectName("appTitle")
        bind_style(self.page_title,
                   lambda: f"font-size:22px; font-weight:bold; color:{token('text_1')};"
                           "background:transparent; border:none;")
        self.add_header_widget(self.page_title, 1)
        self.add_header_stretch(2)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索平台 / 项目…")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setFixedWidth(220)
        # T2-7：连续输入 300ms 防抖后才重过滤（直接调 _apply_search 的派发点
        # ——导航切换/窗口激活等——仍即时生效）
        self._search_debounce = QTimer(self)
        self._search_debounce.setSingleShot(True)
        self._search_debounce.setInterval(300)
        self._search_debounce.timeout.connect(self._on_search_debounced)
        self.search_edit.textChanged.connect(self._on_search_typed)
        self.search_edit._chrome_drag_exempt = True   # 搜索框可点可拖选，不当标题栏
        # 胶囊/搜索框已移出 homeCentral 子树，APP_QSS 触达不到——内联自带样式
        bind_style(self.search_edit, lambda: (
            f"QLineEdit {{ background:{token('panel')};"
            f" border:1px solid {token('border_strong')};"
            f" border-radius:{token('r_input', '10px')}; padding:7px 12px;"
            f" color:{token('text_1')}; font-size:15px; }}"
            f"QLineEdit:focus {{ border:1px solid {token('primary')};"
            f" background:{token('focus_bg', '#ffffff')}; }}"))
        # 右侧容器：上行 = 窗口键（上移贴顶），下行 = 搜索框 + 胶囊（红框位）
        right_side = QWidget()
        right_side.setStyleSheet("background: transparent; border: none;")
        rs_lay = QVBoxLayout(right_side)
        rs_lay.setContentsMargins(0, 8, 24, 14)
        rs_lay.setSpacing(6)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(0)
        btn_row.addStretch(1)
        for b in self.take_header_buttons():
            btn_row.addWidget(b, 0, Qt.AlignTop)
        rs_lay.addLayout(btn_row)
        rs_lay.addStretch(1)
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(12)
        ctrl_row.addStretch(1)
        ctrl_row.addWidget(self.search_edit)
        rs_lay.addLayout(ctrl_row)
        right_side._chrome_drag_exempt = True   # 容器内全是可点控件，不当标题栏
        self.add_header_widget(right_side, 3)
        # 清掉顶栏尾部遗留（旧弹簧/隐藏的默认图标标题），否则右组被夹在
        # 两个弹簧之间变成水平居中（visual-judge 实测 x700+ 全空）
        lay = self._header.layout()
        while lay.count() > 4:   # 期望终态：品牌 | 页标题 | 弹簧 | 右侧容器
            item = lay.takeAt(lay.count() - 1)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    # ---- 最近项目实时刷新 ----
    def _refresh_recent_projects(self):
        """重建最近项目小卡（用户需求 v0.5.5追加-①：不再只在启动时读一次）。

        触发时机：主界面显示（showEvent）、切回 AI 平台页（导航）、
        窗口重新激活（从训练平台开/关项目后回到主界面）。训练平台是
        独立进程，打开/退出项目对 project.json 的 mtime 影响在回到
        主界面的这三个时机全部体现。
        """
        lay = getattr(self, "_recent_chips_lay", None)
        if lay is None:
            return
        self._recent_chips.clear()
        while lay.count():   # 旧小卡与尾部弹簧一并清空
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                # setParent(None) 立即脱离视图：仅 deleteLater 会残留旧卡
                # 几帧（布局已让位、旧卡仍画在原位），与新卡叠字（P134）
                w.setParent(None)
                w.deleteLater()
        for rec in recent_projects():
            chip = _RecentChip(rec["name"], relative_time(rec["mtime"]))
            chip.clicked.connect(self._open_recent_project)
            lay.addWidget(chip)
            self._recent_chips.append(chip)
        lay.addStretch(1)
        if getattr(self, "search_edit", None) is not None:
            self._apply_search(self.search_edit.text())   # 按当前关键词定可见性
        else:
            self.recent_row.setVisible(bool(self._recent_chips))

    def showEvent(self, event):
        self._refresh_recent_projects()
        super().showEvent(event)

    def changeEvent(self, event):
        # 用户在训练平台打开/退出项目后切回主界面即刷新；Qt 激活变化投递
        # 的是 ActivationChange（WindowActivate 不进 changeEvent）。
        # 激活与去激活都刷新：成本低（读几个 project.json），回窗时必然新鲜
        if event.type() == QEvent.ActivationChange:
            self._refresh_recent_projects()
        super().changeEvent(event)

    def _build_ai_page(self) -> QWidget:
        page = QWidget()
        page_lay = QVBoxLayout(page)
        page_lay.setContentsMargins(44, 26, 44, 14)
        page_lay.setSpacing(16)

        # Hero 区（搜索激活时隐藏，呈现纯结果视图）
        self.hero_row = QWidget()
        hero = QHBoxLayout(self.hero_row)
        hero.setSpacing(16)
        logo = QLabel()
        logo.setPixmap(QPixmap(str(icon_path("app.png"))).scaled(
            56, 56, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo.setFixedSize(56, 56)
        hero.addWidget(logo)
        hero_text = QVBoxLayout()
        hero_text.setSpacing(4)
        big_title = QLabel("欢迎使用 XCcv 视觉模型训练平台")
        big_title.setObjectName("pageTitle")
        subtitle = QLabel("标注 → 训练 → 测试，三步完成模型落地（另附缺陷样本合成工具）")
        subtitle.setObjectName("pageSub")
        hero_text.addWidget(big_title)
        hero_text.addWidget(subtitle)
        hero.addLayout(hero_text)
        hero.setAlignment(Qt.AlignVCenter)
        page_lay.addWidget(self.hero_row)

        # 最近项目（横排小卡，读 platform_data/projects 的 mtime；
        # 内容随主界面显示/AI 页切换/窗口激活实时重建——见 _refresh_recent_projects）
        self.recent_row = QWidget()
        recent_lay = QVBoxLayout(self.recent_row)
        recent_lay.setContentsMargins(0, 0, 0, 0)
        recent_lay.setSpacing(8)
        self.recent_hint = QLabel("最近项目")
        self.recent_hint.setObjectName("muted")
        recent_lay.addWidget(self.recent_hint)
        self._recent_chips_lay = QHBoxLayout()
        self._recent_chips_lay.setSpacing(10)
        recent_lay.addLayout(self._recent_chips_lay)
        self._refresh_recent_projects()
        page_lay.addWidget(self.recent_row)

        # 平台卡片（2×2 网格，四个平台）
        grid = QGridLayout()
        grid.setSpacing(18)
        grid.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        for i, (key, name, desc, icon, grad) in enumerate(_PLATFORMS):
            card = _PlatformCard(key, name, desc, icon_path(icon), grad)
            card.clicked.connect(lambda k=key: self._open_platform(k))
            grid.addWidget(card, i // 2, i % 2)
            self._platform_cards.append(card)
        page_lay.addLayout(grid, stretch=1)

        version = QLabel(f"XCcv v{__version__}")
        version.setObjectName("muted")
        version.setAlignment(Qt.AlignRight)
        page_lay.addWidget(version)
        return page

    # ---- 顶栏交互 ----
    def _on_search_typed(self, _text: str) -> None:
        """键盘输入路径：300ms 防抖（T2-7），落定后才执行一次全量过滤。"""
        self._search_debounce.start()

    def _on_search_debounced(self) -> None:
        if getattr(self, "search_edit", None) is not None:
            self._apply_search(self.search_edit.text())

    def _apply_search(self, text: str) -> None:
        """全局搜索：按当前平台页过滤内容。

        AI 页：过滤平台卡（最近项目整节与 Hero 在搜索态隐藏，仅呈现
        匹配的平台卡）；部署页/流程页派发给页面的 set_search_query。
        """
        query = (text or "").strip().lower()
        for card in self._platform_cards:
            card.setVisible(not query or query in card.card_title.lower()
                            or query in card.card_desc.lower())
        # 最近项目保留搜索（用户更正：上轮只该修残影 bug，不该砍功能）——
        # 命中的项目卡作为结果显示；"最近项目"提示语仅在有内容时出现
        hit = 0
        for chip in self._recent_chips:
            show = not query or query in chip.project_name.lower()
            chip.setVisible(show)
            hit += show
        if getattr(self, "hero_row", None) is not None:
            self.hero_row.setVisible(not query)
        self.recent_row.setVisible(bool(query and hit)
                                   or (not query and bool(self._recent_chips)))
        if getattr(self, "recent_hint", None) is not None:
            self.recent_hint.setVisible(self.recent_row.isVisible())
        page = self.stack.currentWidget() if hasattr(self, "stack") else None
        if page is not None and hasattr(page, "set_search_query"):
            page.set_search_query(text)   # 流程图页过滤图片树

    # ---- 导航切换 ----
    def _on_nav_changed(self, row: int):
        key = _NAV_ITEMS[row][0] if 0 <= row < len(_NAV_ITEMS) else ""
        self.page_title.setText(_PAGE_TITLES.get(key, ""))
        self.stack.setCurrentIndex(max(0, row))
        if key == "ai":
            self._refresh_recent_projects()   # 回到 AI 平台页即刷新最近项目
        if getattr(self, "search_edit", None) is not None:
            self._apply_search(self.search_edit.text())   # 新页应用当前关键词
        self._animate_page_in()

    # ---- 页面切换淡入（150ms；WebEngine 页不参与透明度动效）----
    def _animate_page_in(self):
        page = self.stack.currentWidget()
        if page is None:
            return
        effect = page.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(page)
            page.setGraphicsEffect(effect)
        anim = QVariantAnimation(self)
        anim.setDuration(150)
        anim.setStartValue(0.3)
        anim.setEndValue(1.0)
        anim.valueChanged.connect(self._on_fade_value)
        anim.finished.connect(self._on_fade_done)
        self._fade_effect = effect
        effect.setOpacity(0.3)
        self._fade_anim = anim
        anim.start()

    def _on_fade_value(self, v):
        effect = getattr(self, "_fade_effect", None)
        if effect is not None:
            effect.setOpacity(float(v))

    def _on_fade_done(self):
        # 淡入结束即摘除透明度效果：遗留的 QGraphicsEffect 会与卡片静态
        # 阴影形成嵌套效果（Qt 已知渲染陷阱），恢复页面普通渲染
        page = self.stack.currentWidget()
        if page is not None:
            page.setGraphicsEffect(None)
        self._fade_effect = None
        self._fade_anim = None

    # ---- 平台启动（独立子进程，单实例由平台侧守卫） ----
    def _open_platform(self, key: str):
        try:
            launcher.launch_platform(key)
        except Exception as e:
            QMessageBox.critical(
                self, "无法启动平台",
                f"「{dict((p[0], p[1]) for p in _PLATFORMS).get(key, key)}」启动失败：\n"
                f"{type(e).__name__}: {e}")

    def _open_recent_project(self, name: str):
        """最近项目小卡：直达训练平台并打开该项目（argv 透传）。"""
        try:
            launcher.launch_platform("training", name)
        except Exception as e:
            QMessageBox.critical(self, "无法打开项目",
                                 f"项目「{name}」打开失败：\n"
                                 f"{type(e).__name__}: {e}")

    # ---- 系统区 ----
    def refresh_theme(self):
        """主题实时切换（V0.6 F2-5）：圆角壳/顶栏 + 导航矢量图标按当前 token 重绘。"""
        super().refresh_theme()
        ink = token("text_2", "#5f7086")
        for i, (_key, _text, icon_kind) in enumerate(_NAV_ITEMS):
            item = self.nav_list.item(i)
            if item is not None:
                item.setIcon(nav_icon(icon_kind, ink, 45))
        for btn, kind in getattr(self, "_nav_foot", []):
            btn.setIcon(nav_icon(kind, ink, 36))

    def _show_settings_placeholder(self):
        """设置对话框：界面主题切换**即时生效**（V0.6 F2-5；此前需重启）+ 版本信息。

        本进程 set_theme 立即重绘；写入 ui_prefs.json 后，已打开的各平台进程
        经文件监听（install_theme_watcher）同步切换。
        """
        from .ui_style import ACTIVE_THEME
        from .ui_widgets import load_prefs

        dialog = QDialog(self)
        dialog.setWindowTitle("设置 - XCcv")
        lay = QVBoxLayout(dialog)
        lay.setSpacing(12)

        row = QHBoxLayout()
        row.addWidget(QLabel("界面主题"))
        combo = QComboBox()
        combo.addItem("浅色", "light")
        combo.addItem("暗色", "dark")
        current = load_prefs().get("theme", ACTIVE_THEME)
        idx = combo.findData(current if current in ("light", "dark") else "light")
        combo.setCurrentIndex(max(idx, 0))
        row.addWidget(combo, stretch=1)
        lay.addLayout(row)

        note = QLabel("保存后立即生效：主界面与已打开的各平台窗口同步切换。")
        note.setObjectName("hint")
        lay.addWidget(note)

        btns = QHBoxLayout()
        btns.addStretch(1)
        ok = QPushButton("保存")
        ok.setObjectName("accent")
        ok.clicked.connect(dialog.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(dialog.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        lay.addLayout(btns)

        if dialog.exec_() != QDialog.Accepted:
            return
        self.apply_theme_choice(combo.currentData())

    def apply_theme_choice(self, theme: str) -> None:
        """保存主题偏好并即时切换（设置对话框与测试共用）。"""
        from .ui_style import build_qss, set_theme
        from .ui_widgets import load_prefs, save_prefs

        build_qss(theme)   # 校验可生成（失败即抛，不写坏偏好）
        prefs = load_prefs()
        if prefs.get("theme") != theme:
            prefs["theme"] = theme
            save_prefs(prefs)   # 其他平台进程经文件监听同步
        set_theme(theme)        # 本进程即时生效

    def _open_docs(self):
        doc = app_root() / "docs" / "quickstart.md"
        if doc.is_file():
            try:
                os.startfile(str(doc))  # noqa: S606  Windows 关联程序打开
                return
            except OSError:
                pass
        QMessageBox.information(self, "使用文档",
                                "未找到本地文档（docs/quickstart.md）。\n"
                                "请查阅仓库 README 与 docs 目录。")

    def _show_about(self):
        QMessageBox.about(self, "关于 XCcv Community",
                          f"XCcv 视觉模型训练平台（社区版）v{__version__}\n\n"
                          "零基础工业视觉模型训练平台：\n"
                          "标注 → 训练 → 测试（另附缺陷样本合成工具）\n\n"
                          "本版本为社区版，基于 AGPL-3.0 开源。")


def main():
    from .gui_utils import apply_app_font, install_freeze_watchdog
    from .logsetup import setup_logging
    from .taskbar import set_process_app_user_model_id

    set_process_app_user_model_id("XCcv.Main")   # 任务栏独立条目（v0.5.5）
    from .win_qos import opt_out_power_throttling

    opt_out_power_throttling()  # 退出 EcoQoS 节流（V0.6 用户回传：后台推理被降频）
    setup_logging("main")
    install_excepthook("XCcv 主界面")
    if "--selftest" in sys.argv:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QApplication.instance() or QApplication(sys.argv)
    apply_app_font(app)  # 全局字体放大
    install_freeze_watchdog()  # T2-1：主线程卡顿留痕
    from .ui_style import install_theme_watcher

    install_theme_watcher()  # F2-5：主题偏好文件监听（OS 通知，非轮询）
    app.setWindowIcon(QIcon(str(icon_path("app.png"))))
    # 全局"加载中"界面（V0.6 T1-3）：主界面与其他九入口行为一致
    from .splash import LoadingSplash

    splash = LoadingSplash.show_splash("主界面", "正在加载…",
                                       icon=QPixmap(str(icon_path("app.png"))))
    try:
        window = HomeWindow()
    except Exception:
        splash.finish()
        raise
    window.show()
    splash.finish(window)
    if "--selftest" in sys.argv:
        app.processEvents()
        print("SELFTEST OK 主界面")
        window.close()
        return
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402
apply_auto_slot_module(globals())

"""XCcv 统一主题（设计 Token 表 + QSS 模板生成器，各平台共用）。

v0.5 UI 升级（2026-09）：主题从「一坨 QSS 字符串」升级为 Token 表 + 模板：
换主色 / 未来明暗双主题只改 Token 重新生成，一处换全局换。

三层结构：
1. C / R / SP 三个 Token 容器（颜色 / 圆角 / 间距栅格）；
2. build_qss(theme)：由 _QSS_TEMPLATE 按 Token 生成全局 QSS
   （APP_QSS = build_qss()，既有 import 不变）；
3. 兼容别名：BG/PANEL/CARD/BORDER/FG/MUTED/ACCENT/ACCENT2/OK/DANGER/GRID
   保留（图表、splash 等仍在用）。

约束不放松：QSS 字号下限 15px（test_batch11_ui.FontSizeAuditTest 正则锁）；
QSS 规则顺序即优先级——列表选中态四态显式声明，hover 必须写在 selected 之后
的顺序保持不变（用户回执 E1 前科）。
"""

import json
import os
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def icon_path(name: str) -> Path:
    """平台图标路径（assets 目录下）。"""
    return ASSETS_DIR / name


# ============================================================
# 1. Token 表（浅色主题；未来暗色主题在此并列追加）
# ============================================================

class C:
    """颜色 Token。primary 系主色 = 青蓝 #5ec6dd（用户选定 D2）。"""

    # ---- 背景三层 ----
    surface = "#f4f7fb"        # 窗口底层
    panel = "#ffffff"          # 面板 / 输入框 / 列表
    card = "#ffffff"           # 卡片
    card_hover = "#f7fbfd"     # 卡片悬停底色
    # ---- 边框 ----
    border = "#e3ecf4"         # 常规极细边框
    border_strong = "#cfe0ec"  # 输入框 / 按钮（需要可辨识的边）
    divider = "#e9eff7"        # 分割线 / 表头底线
    # ---- 主色（青蓝）----
    primary = "#5ec6dd"        # 品牌主色（填充 / 渐变起点 / 焦点）
    primary_hover = "#74d0e4"
    primary_pressed = "#41b0c9"
    primary_grad_to = "#33a9c4"   # 强调按钮渐变终点
    primary_deep = "#11809c"   # 可读深青：链接文字 / 选中文字 / 图表曲线
    primary_tint = "#e4f6fa"   # 浅青底（导航悬停 / 徽章底）
    primary_tint_deep = "#cfeef6"  # 选中态底色
    # ---- 功能色 ----
    success = "#2fbf71"
    success_deep = "#1d8a53"
    success_tint = "#e3f7ed"
    warn = "#f59e0b"
    warn_deep = "#b45309"
    warn_tint = "#fdf3e0"
    danger = "#e05252"
    danger_deep = "#b02a2a"
    danger_tint = "#fdecec"
    # ---- 文字三级 ----
    text_1 = "#1f2937"
    text_2 = "#5f7086"
    text_3 = "#93a3b5"
    # ---- 其它 ----
    accent2 = "#e8853d"        # 次强调橙（图表第二曲线 / 检测徽章）
    grid = "#dfe7f1"           # 图表网格线
    scroll_handle = "#c7d6e6"
    scroll_handle_hover = "#a9bcd6"
    toast_info = "#0f7f9b"     # Toast 深底（白字）
    on_primary = "#ffffff"


class R:
    """圆角 Token（px）。"""

    card = 14
    panel = 12
    btn = 8
    input = 8
    list = 12
    badge = 999
    tooltip = 6


class SP:
    """间距栅格（px）。"""

    gap_s = 8
    gap_m = 14
    gap_l = 24


def _theme_tokens() -> dict:
    """浅色主题 Token（@名 → 值）。"""
    c, r = C, R
    return {
        # 颜色
        "surface": c.surface, "panel": c.panel, "card": c.card,
        "card_hover": c.card_hover,
        "border": c.border, "border_strong": c.border_strong,
        "divider": c.divider,
        "primary": c.primary, "primary_hover": c.primary_hover,
        "primary_pressed": c.primary_pressed,
        "primary_grad_to": c.primary_grad_to,
        "primary_deep": c.primary_deep,
        "primary_tint": c.primary_tint,
        "primary_tint_deep": c.primary_tint_deep,
        "success": c.success, "success_deep": c.success_deep,
        "success_tint": c.success_tint,
        "warn": c.warn, "warn_deep": c.warn_deep, "warn_tint": c.warn_tint,
        "danger": c.danger, "danger_deep": c.danger_deep,
        "danger_tint": c.danger_tint,
        "text_1": c.text_1, "text_2": c.text_2, "text_3": c.text_3,
        "accent2": c.accent2, "grid": c.grid,
        "scroll_handle": c.scroll_handle,
        "scroll_handle_hover": c.scroll_handle_hover,
        "toast_info": c.toast_info,
        "on_primary": c.on_primary,
        # 控件交互底色（主题间差异最大的字面量全部 Token 化）
        "btn_hover": "#f0f7fb", "btn_pressed": "#e2eff4",
        "btn_disabled_bg": "#f2f5fa",
        "focus_bg": "#fbfdff", "readonly_bg": "#f6f9fc",
        "list_hover": "#edf6fa", "seg_border": "#bfe5ef",
        "nav_side_bg": "#f7fafd",
        # 圆角
        "r_card": f"{r.card}px", "r_panel": f"{r.panel}px",
        "r_btn": f"{r.btn}px", "r_input": f"{r.input}px",
        "r_list": f"{r.list}px", "r_badge": f"{r.badge}px",
        "r_tooltip": f"{r.tooltip}px",
    }


def _dark_overlay() -> dict:
    """暗色主题覆盖层（D7 二期：与浅色并列，只覆盖有差异的 Token）。"""
    return {
        "surface": "#1c2634", "panel": "#232f3e", "card": "#232f3e",
        "card_hover": "#2a3a4d",
        "border": "#33445c", "border_strong": "#3f5674", "divider": "#2b3a4d",
        "primary_deep": "#7cd7ea",           # 暗底上的可读强调色用亮青
        "primary_tint": "#1d3d4a", "primary_tint_deep": "#25505f",
        "success_deep": "#5fd99b", "success_tint": "#1d4030",
        "warn_deep": "#e8a23c", "warn_tint": "#43331a",
        "danger_deep": "#f08c8c", "danger_tint": "#48242a",
        "text_1": "#e7eef7", "text_2": "#a6b6ca", "text_3": "#708299",
        "grid": "#33445c",
        "scroll_handle": "#3b4f68", "scroll_handle_hover": "#4c6480",
        "btn_hover": "#2c3c50", "btn_pressed": "#35485f",
        "btn_disabled_bg": "#263244",
        "focus_bg": "#28374a", "readonly_bg": "#1f2a38",
        "list_hover": "#2c3c50", "seg_border": "#2f6d80",
        "nav_side_bg": "#18212d",
    }


def active_theme() -> str:
    """激活主题：环境变量 XCCV_THEME 优先，其次 ui_prefs.json 的 theme 键。"""
    env = os.environ.get("XCCV_THEME")
    if env in ("light", "dark"):
        return env
    try:
        from .app_paths import app_root

        data = json.loads((app_root() / "platform_data" / "ui_prefs.json")
                          .read_text(encoding="utf-8"))
        theme = str(data.get("theme", "light"))
        return theme if theme in ("light", "dark") else "light"
    except Exception:
        return "light"


ACTIVE_THEME = active_theme()


# 兼容别名（图表 / splash / dialogs 等既有 import 不变）。
# 跟随激活主题取值（暗色下 ACCENT/文字/网格等切到可读的亮色变体）。
_OVERLAY = _dark_overlay() if ACTIVE_THEME == "dark" else {}


def _t(name: str, light_value: str) -> str:
    return _OVERLAY.get(name, light_value)


BG = _t("surface", C.surface)
PANEL = _t("panel", C.panel)
CARD = _t("card", C.card)
BORDER = _t("border", C.border)
FG = _t("text_1", C.text_1)
MUTED = _t("text_2", C.text_2)
ACCENT = _t("primary_deep", C.primary_deep)
ACCENT2 = C.accent2
OK = _t("success_deep", C.success_deep)
DANGER = C.danger
GRID = _t("grid", C.grid)


# ============================================================
# 2. QSS 模板（@token 占位，build_qss() 替换生成）
# ============================================================

_QSS_TEMPLATE = """
QMainWindow, QDialog, QWidget {
    background: @surface;
    color: @text_1;
    font-size: 16px;
}
QFrame#header {
    background: @panel;
    border-bottom: 1px solid @divider;
}
QFrame#card {
    background: @card;
    border: 1px solid @border;
    border-radius: @r_card;
}
QFrame#card:hover {
    border: 1px solid @primary;
    background: @card_hover;
}
QFrame#runPanel {
    background: @panel;
    border: 1px solid @border;
    border-radius: @r_card;
}
QLabel#appTitle {
    font-size: 22px; font-weight: bold; color: @text_1;
    background: transparent; border: none;
}
QLabel#cardTitle {
    font-size: 19px; font-weight: bold; color: @text_1;
    background: transparent; border: none;
}
QLabel#muted {
    color: @text_2; font-size: 15px;
    background: transparent; border: none;
}
QLabel#hint { color: @text_2; font-size: 15px; }
QLabel#title { font-size: 24px; font-weight: bold; color: @text_1; }
QLabel#accentText { color: @primary_deep; font-weight: bold; }
QLabel#pageTitle {
    font-size: 26px; font-weight: bold; color: @text_1;
    background: transparent; border: none;
}
QLabel#pageSub {
    color: @text_2; font-size: 15px;
    background: transparent; border: none;
}
/* ---- 徽章体系（ui_widgets.Badge / 既有 badgeDet/badgeSeg 不变）---- */
QLabel#badge {
    border: none; border-radius: @r_badge;
    padding: 4px 14px; font-size: 15px; font-weight: bold;
    background: @primary_tint; color: @primary_deep;
}
QLabel#badgeDet {
    border: none; border-radius: @r_badge;
    padding: 4px 14px; font-size: 15px; font-weight: bold;
    background: rgba(232, 133, 61, 0.15); color: #b25e1a;
}
QLabel#badgeSeg {
    border: none; border-radius: @r_badge;
    padding: 4px 14px; font-size: 15px; font-weight: bold;
    background: @success_tint; color: @success_deep;
}
QLabel#badgeWarn {
    border: none; border-radius: @r_badge;
    padding: 4px 14px; font-size: 15px; font-weight: bold;
    background: @warn_tint; color: @warn_deep;
}
QLabel#badgeDanger {
    border: none; border-radius: @r_badge;
    padding: 4px 14px; font-size: 15px; font-weight: bold;
    background: @danger_tint; color: @danger_deep;
}
QLabel#badgeNeutral {
    border: none; border-radius: @r_badge;
    padding: 4px 14px; font-size: 15px; font-weight: bold;
    background: #edf2f8; color: @text_2;
}
QGroupBox {
    border: 1px solid @border;
    border-radius: @r_panel;
    margin-top: 16px;
    padding: 14px 14px 12px 14px;
    background: @panel;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 6px;
    color: @primary_deep;
    font-size: 17px;
    font-weight: bold;
}
QPushButton {
    background: @panel;
    border: 1px solid @border_strong;
    border-radius: @r_btn;
    padding: 8px 18px;
    color: @text_1;
}
QPushButton:hover { background: @btn_hover; border-color: @primary; }
QPushButton:pressed { background: @btn_pressed; }
QPushButton:disabled { color: @text_3; background: @btn_disabled_bg; border-color: @border; }
QPushButton#accent, QPushButton#btnAccent {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 @primary, stop:1 @primary_grad_to);
    border: none;
    color: @on_primary;
    font-weight: bold;
    padding: 9px 22px;
}
QPushButton#accent:hover, QPushButton#btnAccent:hover { background: @primary_hover; }
QPushButton#accent:pressed, QPushButton#btnAccent:pressed { background: @primary_pressed; }
QPushButton#seg {
    background: transparent;
    border: none;
    color: @text_2;
    padding: 8px 18px;
    font-weight: bold;
    font-size: 17px;
}
QPushButton#seg:hover { color: @text_1; }
QPushButton#seg:checked {
    background: @primary_tint;
    color: @primary_deep;
    border-radius: @r_btn;
    border: 1px solid @seg_border;
}
QPushButton#linkDanger {
    background: transparent;
    border: none;
    color: @danger;
    padding: 2px 6px;
    font-size: 15px;
}
QPushButton#linkDanger:hover { color: @danger_deep; }
QPushButton#danger {
    background: @panel;
    border: 1px solid #eec3c3;
    border-radius: @r_btn;
    padding: 8px 18px;
    color: @danger;
}
QPushButton#danger:hover { background: @danger_tint; border-color: @danger; }
QPushButton#danger:pressed { background: #fbe3e3; }
QPushButton#danger:disabled { color: @text_3; background: @btn_disabled_bg; border-color: @border; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: @panel;
    border: 1px solid @border_strong;
    border-radius: @r_input;
    padding: 7px 12px;
    color: @text_1;
    selection-background-color: @primary_deep;
    selection-color: @on_primary;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid @primary;
    background: @focus_bg;
}
QLineEdit[readOnly="true"] { background: @readonly_bg; color: @text_2; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView {
    background: @panel;
    color: @text_1;
    border: 1px solid @border_strong;
    border-radius: @r_input;
    selection-background-color: @primary_deep;
    selection-color: @on_primary;
    outline: none;
    font-size: 16px;
}
QListWidget, QTreeWidget {
    background: @panel;
    border: 1px solid @border;
    border-radius: @r_list;
    outline: none;
    color: @text_1;
    font-size: 16px;
}
QListWidget::item { padding: 9px 11px; border-radius: 6px; }
/* 选中态四态显式声明：hover 规则写在 selected 之后会覆盖其背景，导致
   "白字 + 浅底"不可见（用户回执 E1/ToE3）；窗口失焦（如对比窗口打开后
   切回训练页）用 :!active 给出深字浅底的可读组合 */
QListWidget::item:selected { background: @primary_tint_deep; color: @primary_deep; }
QListWidget::item:selected:!active { background: @primary_tint; color: @text_1; }
QListWidget::item:hover { background: @list_hover; color: @text_1; }
QListWidget::item:selected:hover { background: #c2e9f3; color: @primary_deep; }
QTreeWidget::item { padding: 7px 9px; }
/* 多行文本（日志/说明框）：缺此规则时暗色下沿用系统白底 + 全局浅色字 =
   白底浅字不可读（F2-5 暗色截图实测，工作台日志/算子说明框） */
QPlainTextEdit, QTextEdit {
    background: @panel;
    color: @text_1;
    border: 1px solid @border;
    border-radius: @r_input;
    selection-background-color: @primary_deep;
    selection-color: @on_primary;
}
QPlainTextEdit[readOnly="true"], QTextEdit[readOnly="true"] { background: @readonly_bg; }
QGraphicsView {
    background: @panel;
    border: 1px solid @border;
    border-radius: 10px;
}
QSlider::groove:horizontal {
    height: 6px; background: #dde6f1; border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 14px; margin: -5px 0;
    background: @primary; border-radius: 7px;
}
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
QScrollBar::handle:vertical {
    background: @scroll_handle; border-radius: 4px; min-height: 32px;
}
QScrollBar::handle:vertical:hover { background: @scroll_handle_hover; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QScrollBar:horizontal { background: transparent; height: 8px; margin: 2px; }
QScrollBar::handle:horizontal {
    background: @scroll_handle; border-radius: 4px; min-width: 32px;
}
QStatusBar { color: @text_2; font-size: 15px; }
QToolTip {
    background: @panel; color: @text_1;
    border: 1px solid @border_strong; border-radius: @r_tooltip; padding: 6px;
    font-size: 15px;
}
/* ---- 主界面左侧图标导航（app.py navList）---- */
QListWidget#navList {
    background: transparent; border: none; outline: 0;
    font-size: 17px;
}
QListWidget#navList::item {
    color: @text_2; padding: 10px 12px; border-radius: 10px; margin: 3px 0;
}
QListWidget#navList::item:hover { background: @primary_tint; color: @text_1; }
QListWidget#navList::item:selected {
    background: @primary_tint_deep; color: @primary_deep; font-weight: bold;
}
QListWidget#navList::item:selected:!active {
    background: @primary_tint; color: @text_1;
}
/* 顶栏服务状态胶囊（点击跳部署平台） */
QFrame#statusChip {
    background: @panel; border: 1px solid @border; border-radius: @r_badge;
}
QFrame#statusChip:hover {
    border: 1px solid @primary; background: @card_hover;
}
QFrame#statusChip QLabel {
    background: transparent; border: none; color: @text_2; font-size: 15px;
}
/* 最近项目横排小卡（AI 平台页） */
QFrame#recentChip {
    background: @panel; border: 1px solid @border; border-radius: 10px;
}
QFrame#recentChip:hover {
    border: 1px solid @primary; background: @card_hover;
}
QFrame#recentChip QLabel {
    background: transparent; border: none;
}
/* 左导航底部系统区按钮（设置/使用文档/关于） */
QPushButton#navFoot {
    background: transparent; border: none; border-radius: 10px;
    text-align: left; padding: 8px 12px; color: @text_2; font-size: 16px;
}
QPushButton#navFoot:hover { background: @primary_tint; color: @text_1; }
/* ---- Toast 轻提示（ui_widgets.toast，深底白字圆角）---- */
QLabel#toastInfo {
    background: @toast_info; color: @on_primary;
    border: none; border-radius: 10px; padding: 10px 18px; font-size: 15px;
}
QLabel#toastSuccess {
    background: @success_deep; color: @on_primary;
    border: none; border-radius: 10px; padding: 10px 18px; font-size: 15px;
}
QLabel#toastWarn {
    background: @warn_deep; color: @on_primary;
    border: none; border-radius: 10px; padding: 10px 18px; font-size: 15px;
}
QLabel#toastError {
    background: @danger_deep; color: @on_primary;
    border: none; border-radius: 10px; padding: 10px 18px; font-size: 15px;
}
/* ---- 空状态（ui_widgets.EmptyState）---- */
QLabel#emptyIcon {
    font-size: 40px; color: @text_3;
    background: transparent; border: none;
}
QLabel#emptyTitle {
    font-size: 18px; font-weight: bold; color: @text_2;
    background: transparent; border: none;
}
QLabel#emptyHint {
    font-size: 15px; color: @text_3;
    background: transparent; border: none;
}
"""


# 暗色专用补充规则（V0.6 F2-5 暗色截图实测）：这些控件浅色下用系统原生外观
# （保持浅色像素不变），暗色下不显式着色就会白底浅字/白框——如算子编辑器页签。
_DARK_EXTRA = """
QTabWidget::pane {
    border: 1px solid @border; background: @panel;
    border-radius: @r_input; top: -1px;
}
QTabBar::tab {
    background: @surface; color: @text_2;
    border: 1px solid @border; border-bottom: none;
    padding: 7px 16px; margin-right: 2px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
}
QTabBar::tab:selected { background: @panel; color: @primary_deep; }
QTabBar::tab:hover { color: @text_1; }
QHeaderView::section {
    background: @surface; color: @text_2; border: none;
    border-bottom: 1px solid @divider; border-right: 1px solid @divider;
    padding: 6px 8px;
}
QTableView, QTableWidget {
    background: @panel; alternate-background-color: @readonly_bg;
    color: @text_1; gridline-color: @divider; border: 1px solid @border;
    selection-background-color: @primary_tint_deep; selection-color: @text_1;
}
QTableCornerButton::section { background: @surface; border: none; }
QMenu { background: @panel; color: @text_1; border: 1px solid @border_strong; }
QMenu::item:selected { background: @primary_tint_deep; color: @primary_deep; }
QProgressBar {
    background: @readonly_bg; border: 1px solid @border; border-radius: 6px;
    color: @text_1; text-align: center;
}
QProgressBar::chunk { background: @primary; border-radius: 6px; }
QToolButton { color: @text_1; }
QSplitter::handle { background: @divider; }
"""


def build_qss(theme: str | None = None) -> str:
    """由 Token 表生成全局 QSS。

    theme=None 取当前激活主题（ACTIVE_THEME，set_theme 后随之变化）；
    "dark" 为暗色覆盖层（另追加 _DARK_EXTRA：浅色下沿用系统原生外观的控件
    在暗色下必须显式着色，否则白底浅字）。
    """
    if theme is None:
        theme = ACTIVE_THEME
    if theme not in ("light", "dark"):
        raise ValueError(f"暂未支持的主题：{theme}")
    tokens = _theme_tokens()
    qss = _QSS_TEMPLATE
    if theme == "dark":
        tokens.update(_dark_overlay())
        qss = qss + _DARK_EXTRA
    # 键长降序替换：避免 @warn 抢先匹配 @warn_deep 的前缀（QSS 解析报脏色）
    for key, val in sorted(tokens.items(), key=lambda kv: -len(kv[0])):
        qss = qss.replace("@" + key, val)
    return qss


def token(name: str, default: str = "") -> str:
    """给非 QSS 场景（代码内 setStyleSheet / 自绘）按激活主题取 Token 值。

    每次调用读当前 ACTIVE_THEME：绘制时/refresh_theme() 里调用即随主题切换。
    """
    base = _theme_tokens()
    return (_dark_overlay() if ACTIVE_THEME == "dark" else {}).get(
        name, base.get(name, default))


# 全局 QSS（模块级生成，既有 `from .ui_style import APP_QSS` 不变；
# 注意：导入方绑定的是导入时的字符串——需要随主题切换的窗口改用 apply_app_theme）
APP_QSS = build_qss()


# ============================================================
# 3. 主题实时切换（V0.6 F2-5：用户要求浅/暗切换不再需要重启）
# ============================================================
# 机制：
# - set_theme(name)：重算别名常量与 APP_QSS → ThemeBus.theme_changed 广播；
# - apply_app_theme(widget)：窗口/子树根套用当前 QSS 并登记，切换时自动
#   重设（保留登记后追加的局部规则，如 homeCentral/chromePage 透明底）；
# - 自绘/内联样式部件实现 refresh_theme()：切换时对 QApplication.allWidgets()
#   中实现了该方法的部件逐一调用（绘制时取 token() 的部件只需 update()）；
# - install_theme_watcher()：每个平台进程入口挂 QFileSystemWatcher 监听
#   ui_prefs.json（操作系统文件通知，非轮询——不违反"无常驻负载"红线），
#   主界面设置里切换 → 已打开的各平台进程同步变色。
import logging as _logging
import weakref as _weakref

from PyQt5.QtCore import QObject as _QObject, pyqtSignal as _pyqtSignal

_LOG = _logging.getLogger(__name__)
_THEMED = _weakref.WeakSet()
_BUS = None
_WATCHER = None


class ThemeBus(_QObject):
    """主题广播总线（进程内单例）：set_theme 后发出 theme_changed(主题名)。"""

    theme_changed = _pyqtSignal(str)


def theme_bus() -> ThemeBus:
    global _BUS
    if _BUS is None:
        _BUS = ThemeBus()
        _BUS.theme_changed.connect(_on_theme_changed)
    return _BUS


def set_theme(name: str) -> bool:
    """切换激活主题并实时生效（进程内）；返回是否真的发生了切换。"""
    global ACTIVE_THEME, _OVERLAY, APP_QSS
    global BG, PANEL, CARD, BORDER, FG, MUTED, ACCENT, OK, GRID
    if name not in ("light", "dark"):
        raise ValueError(f"暂未支持的主题：{name}")
    if name == ACTIVE_THEME:
        return False
    ACTIVE_THEME = name
    _OVERLAY = _dark_overlay() if name == "dark" else {}
    BG = _t("surface", C.surface)
    PANEL = _t("panel", C.panel)
    CARD = _t("card", C.card)
    BORDER = _t("border", C.border)
    FG = _t("text_1", C.text_1)
    MUTED = _t("text_2", C.text_2)
    ACCENT = _t("primary_deep", C.primary_deep)
    OK = _t("success_deep", C.success_deep)
    GRID = _t("grid", C.grid)
    APP_QSS = build_qss(name)
    _LOG.info("界面主题切换为 %s（实时生效）", name)
    theme_bus().theme_changed.emit(name)
    return True


def apply_app_theme(widget) -> None:
    """窗口/子树根套用当前全局 QSS 并登记：之后 set_theme 自动重设。

    取代各窗口的 `setStyleSheet(APP_QSS)`（导入期绑定的字符串不会随主题变）。
    """
    qss = build_qss()
    widget.setStyleSheet(qss)
    widget._xccv_theme_qss = qss
    _THEMED.add(widget)
    theme_bus()   # 确保中继已接线


_STYLED = _weakref.WeakSet()


def bind_style(widget, make_qss) -> None:
    """内联样式随主题刷新：make_qss() 内用 token() 取色，切换主题时重算重设。

    取代 `w.setStyleSheet(f"color:{ACCENT}")` 这类构造期定死颜色的写法。
    """
    widget.setStyleSheet(make_qss())
    widget._xccv_style_fn = make_qss
    _STYLED.add(widget)
    theme_bus()


def _set_sheet(w, qss: str) -> None:
    """重设样式表。未显示过的部件上二次 setStyleSheet 会被 Qt 忽略（仍按首次
    样式渲染；Fusion 样式下"先清空再设"也不够——P173 实测），对未显示部件
    显式 unpolish/polish 整个子树；已显示部件 Qt 自行重抛光，免重复开销。"""
    if w.isVisible():
        w.setStyleSheet(qss)
        return
    w.setStyleSheet("")
    w.setStyleSheet(qss)
    from PyQt5.QtWidgets import QWidget

    for x in [w] + w.findChildren(QWidget):
        st = x.style()
        st.unpolish(x)
        st.polish(x)


def _restyle_bound() -> None:
    for w in list(_STYLED):
        try:
            _set_sheet(w, w._xccv_style_fn())
        except RuntimeError:
            _STYLED.discard(w)


def _restyle_registered() -> None:
    new = build_qss()
    for w in list(_THEMED):
        try:
            cur = w.styleSheet()
            old = getattr(w, "_xccv_theme_qss", "")
            tail = cur[len(old):] if old and cur.startswith(old) else ""
            _set_sheet(w, new + tail)   # 登记后追加的局部规则原样保留
            w._xccv_theme_qss = new
        except RuntimeError:   # C++ 对象已销毁
            _THEMED.discard(w)


def _on_theme_changed(_name: str) -> None:
    _restyle_registered()
    _restyle_bound()
    try:
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance()
        widgets = app.allWidgets() if app is not None else []
    except Exception:
        widgets = []
    for w in widgets:
        fn = getattr(w, "refresh_theme", None)
        if callable(fn):
            try:
                fn()
            except Exception:   # 单个部件刷新失败不影响其余（留痕便于排查）
                _LOG.warning("部件 %s 主题刷新失败", type(w).__name__, exc_info=True)
        else:
            try:
                w.update()   # 绘制时取 token() 的自绘部件重画即生效
            except RuntimeError:
                pass


def _prefs_path() -> Path:
    from .app_paths import app_root

    return app_root() / "platform_data" / "ui_prefs.json"


def _prefs_theme():
    try:
        data = json.loads(_prefs_path().read_text(encoding="utf-8"))
        theme = str(data.get("theme", "light"))
        return theme if theme in ("light", "dark") else None
    except Exception:
        return None


def _norm(p) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def _watch_paths() -> None:
    if _WATCHER is None:
        return
    p = _prefs_path()
    files = {_norm(f) for f in _WATCHER.files()}
    dirs = {_norm(d) for d in _WATCHER.directories()}
    if p.is_file() and _norm(p) not in files:
        _WATCHER.addPath(str(p))
    if p.parent.is_dir() and _norm(p.parent) not in dirs:
        _WATCHER.addPath(str(p.parent))   # 偏好文件被替换/首次创建时由目录事件补挂


def _on_prefs_changed(_path=None) -> None:
    _watch_paths()
    if os.environ.get("XCCV_THEME") in ("light", "dark"):
        return   # 环境变量钉死主题（截图/调试）：不跟随偏好文件
    theme = _prefs_theme()
    if theme and theme != ACTIVE_THEME:
        set_theme(theme)


def install_theme_watcher() -> bool:
    """平台进程入口调用（QApplication 建好之后）：跨进程主题同步。返回是否已安装。"""
    global _WATCHER
    if _WATCHER is not None:
        return True
    try:
        from PyQt5.QtCore import QFileSystemWatcher
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            return False
        theme_bus()
        _WATCHER = QFileSystemWatcher(app)
        _WATCHER.fileChanged.connect(_on_prefs_changed)
        _WATCHER.directoryChanged.connect(_on_prefs_changed)
        _watch_paths()
        return True
    except Exception:
        _LOG.warning("主题文件监听安装失败（不影响使用，仅跨进程同步失效）", exc_info=True)
        return False

"""XCcv 标注平台入口。

直接使用本项目内置的开源 labelme 修改版主窗口（仓库根目录 labelme/），
保留 labelme 全部原生功能（撤销/缩放/快捷键/文件导航等），叠加平台扩展：
- 自动保存（config auto_save=true）
- 保存时同步 classes.txt（app.py: _sync_classes_txt）
- 分类归档：File 菜单"归类到类别"（app.py: archiveToClass）

运行：python -m xccv.annotation.annotator_window [数据集目录]
"""

if __name__ == "__main__":  # V0.6 T1-3：-m 直启先亮加载中界面，再做下方重型模块级导入
    from xccv.platform_entry import early_splash

    early_splash("annotation", "标注平台")

import os
import sys

from ..gui_utils import ensure_console_streams

ensure_console_streams()  # pythonw 下 stdout/stderr 为 None，先兜底

os.environ.setdefault("QT_API", "pyqt5")


def build_window(dataset=None):
    """构建标注主窗口（主界面整合时复用）。"""
    from labelme.app import MainWindow
    from labelme.config import get_default_config

    window = MainWindow(config=get_default_config())
    try:   # 第三批补充：恢复用户上次「切换图片时保持缩放」选择（默认关）
        from ..ui_widgets import load_prefs

        window.enableKeepPrevScale(
            bool(load_prefs().get("keep_prev_scale", False)))
    except Exception:  # noqa: BLE001  偏好读取失败走默认
        pass
    from PyQt5.QtGui import QIcon
    from ..ui_style import icon_path
    window.setWindowIcon(QIcon(str(icon_path("annotation.png"))))
    # v0.5.4 状态栏分层：标注平台徽章（labelme 为冻结区，从外部挂）
    from ..status_bus import attach_platform_status

    attach_platform_status(window, "标注平台", "annotation")
    # V0.6 T5-4：AI 批量预标注入口（工具菜单；labelme 冻结区从外部挂菜单）
    try:
        from PyQt5.QtWidgets import QAction

        menu = window.menuBar().addMenu("工具")
        act_auto = QAction("AI 批量预标注…", window)
        act_auto.triggered.connect(lambda: _open_auto_annotate(window))
        menu.addAction(act_auto)
        menu.addSeparator()
        act_stats = QAction("数量统计…", window)
        act_stats.triggered.connect(
            lambda: _open_stats_dialog(window))
        menu.addAction(act_stats)
    except Exception:  # noqa: BLE001  菜单挂载失败不影响标注主功能
        pass
    if dataset:
        window.importDirImages(dataset, load=True)
    return window


def _open_auto_annotate(parent):
    from .auto_annotate_window import AutoAnnotateDialog

    dlg = AutoAnnotateDialog(parent)
    dlg.setModal(False)
    dlg.show()


def _open_stats_dialog(parent):
    # 批注第二批 F8：数量统计面板（当前图/文件夹每类计数）
    from .stats_dialog import StatsDialog

    dlg = StatsDialog(parent)
    dlg.setModal(False)
    dlg.show()


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else None
    from ..platform_entry import run_platform

    run_platform("annotation", "标注平台", lambda: build_window(dataset),
                 enable_logging=False)  # 标注平台暂不启用日志


if __name__ == "__main__":
    main()

"""平台独立进程统一入口。

打包后的标注 / 训练 / 测试平台 exe 与开发态的 python -m 启动都走这里，
统一提供：崩溃日志与错误弹窗、平台日志、单实例守卫、重型平台的加载闪屏，
以及供打包验证用的 --selftest 离屏自检（构建窗口后立即退出）。
"""

import os
import sys

# QtWebEngine 硬性约束：必须在 QApplication 创建前导入/初始化
# （缺陷样本合成工具的内嵌 Web 界面依赖它）。
# 必须发生在本模块导入时——各平台的 QApplication 都在 run_platform 内才创建。
from PyQt5.QtCore import Qt as _Qt
from PyQt5.QtWidgets import QApplication as _QApp

if _QApp.instance() is None:
    _QApp.setAttribute(_Qt.AA_ShareOpenGLContexts, True)
try:
    import PyQt5.QtWebEngineWidgets  # noqa: F401,E402
except Exception:
    pass  # 环境缺失时保持旧有行为（流程包加载会给出明确报错）

from .gui_utils import ensure_console_streams, install_excepthook

ensure_console_streams()

# 平台 key -> 加载中界面图标（assets/ 下；缺失时回退 app.png）
_SPLASH_ICONS = {
    "annotation": "annotation.png",
    "training": "training.png",
    "testing": "testing.png",
    "defectgen": "defectgen.png",  # 缺陷样本合成
}


def _splash_icon(key: str):
    """加载中界面平台图标（QPixmap；离屏/缺失环境安全回退 None）。"""
    from PyQt5.QtGui import QPixmap

    from .ui_style import icon_path

    name = _SPLASH_ICONS.get(key, "app.png")
    try:
        pm = QPixmap(str(icon_path(name)))
        return pm if not pm.isNull() else None
    except Exception:
        return None


def _splash_message(heavy: bool) -> str:
    """加载中文案（early_splash 与 run_platform 共用：一致则接管时不重绘）。"""
    return ("正在加载深度学习组件…\n首次启动约需 1~2 分钟，请耐心等待"
            if heavy else "正在加载…")


_EARLY_APP = None  # early_splash 创建的 QApplication（模块级保活，见下）


def early_splash(key: str, display_name: str, heavy: bool = False) -> None:
    """入口模块最顶部调用：先于模块级重型导入显示加载中界面（V0.6 T1-3）。

    `pythonw -m <平台模块>` 直启时，模块级导入（pyqtgraph/matplotlib/cv2…）
    全部发生在 main()→run_platform 之前——实测训练平台首窗 765ms，超出
    <500ms 验收线。入口模块在 docstring 之后第一句写
    `if __name__ == "__main__": early_splash(...)`：建 QApplication 并立即
    显示加载卡；随后 run_platform / app.main 的 LoadingSplash.show_splash
    接管同一张卡。已有同平台实例时直接激活旧窗口并退出（不白跑整套重型
    导入、也不闪加载卡）。--selftest 不显示。编译态（.pyd）模块不以
    __main__ 运行，不经此路径，行为同 run_platform 内显示。
    """
    if "--selftest" in sys.argv:
        return
    from .taskbar import aumid_for, set_process_app_user_model_id
    from .win_qos import opt_out_power_throttling

    # AUMID 须先于任何窗口（任务栏分组，v0.5.5）；QoS 越早越好
    set_process_app_user_model_id("XCcv.Main" if key == "main" else aumid_for(key))
    opt_out_power_throttling()
    from PyQt5.QtWidgets import QApplication

    global _EARLY_APP
    # 必须保活：局部变量持有的 QApplication 在函数返回时被 GC 销毁（连带加载卡），
    # run_platform 会再建一个新 app+新卡、重付字体加载 ~300ms（坑 P157 实测）
    app = _EARLY_APP = QApplication.instance() or QApplication(sys.argv)
    if key != "main":
        from .single_instance import already_running

        if already_running(key):
            sys.exit(0)  # 已发激活消息给旧实例，本进程退出
    from .gui_utils import apply_app_font
    from .splash import LoadingSplash

    apply_app_font(app)  # 幂等（setFont 固定字号），与 run_platform 一致
    LoadingSplash.show_early(display_name, _splash_message(heavy),
                             icon=_splash_icon(key))


def run_platform(key: str, display_name: str, build, heavy: bool = False,
                 enable_logging: bool = True) -> None:
    """运行一个平台直到退出。

    任务栏分组（v0.5.5）：进入即设进程 AUMID——各平台在 Windows 任务栏
    独立成条目；同进程子窗口（如项目窗口）归在所属平台条目之下。

    build: 无参工厂函数，返回主窗口；重型导入必须放在工厂内部，
    这样加载中界面才能在导入 torch 之前显示出来。
    heavy: 平台是否包含 torch 等重型依赖——**静态声明**（V0.6 T1-3 修 P94：
    原先在运行时探测 torch 是否已导入，但调用方模块的引擎链导入使探测恒
    为 False，闪屏永不显示）。全部入口统一显示"加载中"界面，
    heavy 仅影响提示文案。
    enable_logging: 是否初始化平台日志文件（平台排障用）。
    """
    from .taskbar import aumid_for, set_process_app_user_model_id
    from .win_qos import opt_out_power_throttling

    set_process_app_user_model_id(aumid_for(key))
    high_qos = opt_out_power_throttling()  # 退出 EcoQoS：后台/未响应时推理不被降频（3.4s→8.8s 实测）
    log_path = None
    if enable_logging:
        from .logsetup import setup_logging

        log_path = setup_logging(key)
        import logging

        logging.getLogger("xccv").info(
            "进程 QoS：%s", "HighQoS（已退出系统 EcoQoS 节流）" if high_qos
            else "系统默认（未能设置，后台时可能被节流）")

    if "--selftest" in sys.argv:
        _selftest(display_name, build)
        return

    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    from .gui_utils import apply_app_font, install_freeze_watchdog

    apply_app_font(app)  # 全局字体放大（含标注平台），减少长时间使用疲劳
    # 第三批：双击判定统一 250ms（Windows 默认 500ms 偏慢；标注/训练/测试
    # 等各平台共用此入口，一次设置全覆盖）
    app.setDoubleClickInterval(250)
    install_excepthook(display_name)
    install_freeze_watchdog()  # T2-1：主线程卡顿留痕（50ms 级心跳，无常驻负载）
    from .ui_style import install_theme_watcher

    install_theme_watcher()  # F2-5：主界面切主题 → 本平台进程同步（文件通知，非轮询）

    from .single_instance import already_running, register

    if already_running(key):
        return  # 已有实例：激活旧窗口后本进程退出

    heavy = bool(heavy)  # 静态声明，不做运行时探测（修 P94，见 docstring）
    from .splash import LoadingSplash

    splash = LoadingSplash.show_splash(
        display_name, _splash_message(heavy), icon=_splash_icon(key))
    try:
        window = build()
    except Exception:
        splash.finish()
        raise
    window.show()
    splash.finish(window)
    register(key, window)
    if log_path is not None:
        import logging

        logging.getLogger("xccv").info("平台窗口就绪，进入消息循环")
    sys.exit(app.exec_())


def _selftest(display_name: str, build) -> None:
    """打包自检：离屏构建各平台窗口，能构建即认为运行环境完整。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    window = build()
    window.show()
    app.processEvents()
    print(f"SELFTEST OK {display_name}")
    window.close()

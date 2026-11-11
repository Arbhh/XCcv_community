"""GUI 运行辅助。

用 pythonw（无控制台）启动时，sys.stdout / sys.stderr 是 None，
任何库往里写内容（如 torchvision 下载预训练权重的进度条、ultralytics
的训练日志）都会抛 AttributeError: 'NoneType' object has no attribute 'write'。
GUI 入口统一先调用 ensure_console_streams() 兜底重定向。
"""

import os
import sys
from datetime import datetime
from pathlib import Path


def ensure_console_streams() -> None:
    """兜底重定向 stdout/stderr：None（pythonw 正常态）或**继承自无控制台
    父进程的无效句柄**（GUI 主界面 Popen 不带重定向时，子进程写即异常
    退出 exit 1——v0.5 推理服务启动失败实锤）都统一指向 devnull。"""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            setattr(sys, name,
                    open(os.devnull, "w", encoding="utf-8", errors="replace"))
            continue
        try:
            stream.flush()
        except Exception:
            try:
                setattr(sys, name,
                        open(os.devnull, "w", encoding="utf-8",
                             errors="replace"))
            except OSError:
                pass


def crash_log_path() -> Path:
    """崩溃日志位置：运行根（app_paths.app_root）旁的 platform_data/logs，
    不可写则回退临时目录。"""
    from .app_paths import app_root

    candidates = [app_root() / "platform_data" / "logs",
                  Path(os.environ.get("TEMP", str(app_root()))) / "xccv"]
    for d in candidates:
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_probe"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            return d / "crash.log"
        except Exception:
            continue
    return candidates[-1] / "crash.log"


def apply_app_font(app) -> None:
    """全局应用字体：微软雅黑 11pt（比系统默认大一号），减轻长时间使用的视觉疲劳。

    走 APP_QSS 的平台在此基础上由样式表进一步放大；标注平台（labelme）
    无样式表，直接受益于此全局字体。
    """
    from PyQt5.QtGui import QFont

    font = QFont("Microsoft YaHei", 11)
    font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(font)


def install_excepthook(context: str) -> None:
    """未捕获异常写入 crash.log 与平台日志并弹窗提示，避免 GUI 进程静默失败。"""

    def _hook(tp, val, tb):
        import logging
        import traceback

        text = "".join(traceback.format_exception(tp, val, tb))
        try:
            logging.getLogger("xccv.crash").critical(
                f"未捕获异常（{context}）\n{text}")
        except Exception:
            pass
        path = crash_log_path()
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] {context}\n{text}")
        except Exception:
            pass
        try:
            from PyQt5.QtWidgets import QApplication, QMessageBox

            if QApplication.instance() is not None:
                QMessageBox.critical(
                    None, "XCcv 发生错误",
                    f"{context}\n\n{tp.__name__}: {val}\n\n详细堆栈已保存到：\n{path}")
        except Exception:
            pass

    sys.excepthook = _hook


def install_freeze_watchdog(threshold_ms: int = 200, heartbeat_ms: int = 50):
    """主线程卡顿看门狗（V0.6 T2-1）。

    心跳：主线程 QTimer 每 heartbeat_ms 触发一次喂 threading.Event；
    监视线程：超过 threshold_ms 没等到心跳即判定卡顿——开始时记录一次
    "卡顿开始"+主线程栈顶三帧，心跳恢复时补记"卡顿结束+时长"
    （写入 logs/freeze_watchdog.log，异常必须留痕，P75 纪律）。

    负载纪律（用户 R3）：一个 50ms 级 daemon 线程 + 主线程低频 QTimer，
    无轮询数据库/文件，不构成常驻性能负载；主要入口统一装配。
    返回 stop() 卸载函数（测试用）。

    生命周期（坑 P153）：产品入口丢弃返回值——心跳 QTimer 若只被 stop 闭包
    引用会随之被 GC，心跳永停：每次启动只记一条"卡顿开始"、此后永久失明
    （V0.6 首版真机日志 48 条开始/0 条结束）。故 QTimer 挂 QApplication 为父
    并登记模块级保活表。布防：连续 3 次准时心跳（事件循环已稳定运转）后才
    开始判定——启动期窗口构建本就无事件循环（有加载中界面兜底），不计卡顿。
    """
    import threading

    from PyQt5.QtCore import QCoreApplication, QTimer

    log_path = crash_log_path().parent / "freeze_watchdog.log"
    beat = threading.Event()
    state = {"frozen": False, "start": 0.0, "stop": False, "armed": False}
    main_tid = threading.get_ident()

    timer = QTimer(QCoreApplication.instance())  # 挂 app 为父：随进程存活（P153）
    timer.setInterval(heartbeat_ms)

    def _beat():
        beat.set()

    timer.timeout.connect(_beat)

    def _watch():
        import time as _time

        good = 0
        while not state["stop"] and good < 3:  # 布防：等事件循环稳定（连续 3 次准时心跳）
            if beat.wait(threshold_ms / 1000.0):
                beat.clear()
                good += 1
            else:
                good = 0
        state["armed"] = True
        while not state["stop"]:
            ok = beat.wait(threshold_ms / 1000.0)
            if ok:
                beat.clear()  # Event.wait 不清标志（坑 P141），必须显式清
            now = _time.perf_counter()
            if ok:
                if state["frozen"]:
                    state["frozen"] = False
                    _write(f"卡顿结束：持续 {now - state['start']:.2f}s")
                continue
            if not state["frozen"]:
                state["frozen"] = True
                state["start"] = now
                _write(f"卡顿开始（>{threshold_ms}ms）主线程栈顶: {_stack_top()}")

    def _stack_top() -> str:
        import sys as _sys

        frame = _sys._current_frames().get(main_tid)
        if frame is None:
            return "(无栈信息)"
        frames = []
        f = frame
        for _ in range(3):
            if f is None:
                break
            frames.append(f"{Path(f.f_code.co_filename).name}:{f.f_lineno}"
                          f"({f.f_code.co_name})")
            f = f.f_back
        return " <- ".join(frames)

    def _write(text: str):
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{stamp}] {text}\n")
        except Exception:
            pass

    timer.start()
    thread = threading.Thread(target=_watch, daemon=True,
                              name="xccv-freeze-watchdog")
    thread.start()

    def stop():
        state["stop"] = True
        beat.set()
        try:
            timer.stop()
            timer.deleteLater()
        except Exception:
            pass
        try:
            _WATCHDOG_KEEPALIVE.remove(entry)
        except ValueError:
            pass

    entry = (timer, thread, state)
    _WATCHDOG_KEEPALIVE.append(entry)  # 模块级保活（返回值被丢弃时心跳仍在，P153）
    return stop


_WATCHDOG_KEEPALIVE: list = []  # [(QTimer, Thread, state)]；state["armed"] 供测试观测


def auto_cleanup_thread(owner, attr: str, thread):
    """QThread 卫生收尾（V0.6 T2-6，A-3 模式）。

    finished→安全置 None（owner 线程队列投递；owner 先亡自动断开）+
    finished→deleteLater 释放 Qt 底层对象。thread 参数须显式传入
    （连接建立于 start 之前；普通函数槽会直连在工作线程执行，
    跨线程碰已亡 C++ 对象即段错误——坑 P146）。
    """
    from PyQt5.QtCore import QObject

    class _CleanupProxy(QObject):
        """以 owner 子对象身份接收 finished：队列投递回 owner 线程，
        owner 销毁时随亡、连接自动断开。"""

        def _on_finished(self):
            worker = self.sender()
            if worker is None:
                return
            if getattr(owner, attr, None) is worker:
                setattr(owner, attr, None)

    proxy = _CleanupProxy(owner)
    thread.finished.connect(proxy._on_finished)
    thread.finished.connect(thread.deleteLater)
    return proxy


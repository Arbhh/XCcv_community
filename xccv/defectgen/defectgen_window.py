# -*- coding: utf-8 -*-
"""DefectAugmentStudio 缺陷生成窗口（V0.6 P6/T6-2）：creat_seg 移植的 Qt 壳。

- 后端：`xccv/defectgen/backend/app.py`（FastAPI，create_app）经 **uvicorn 子线程**
  托管，端口=动态分配（port=0 由 OS 分派），窗口关闭即 `should_exit` 停服——
  不驻留后台、不写死端口（与流程图编辑器同时打开无冲突，T6-4 验收项）；
- 前端：QtWebEngine 内嵌 `http://127.0.0.1:<port>/`（本机回环；**不用
  webbrowser.open**）；
- QtWebEngine 必须在 QApplication 建立前导入（AA_ShareOpenGLContexts），
  与流程图 web 窗口同款纪律；关窗先销毁页面再退出（同款段错误防线）。
"""

import os
import threading

# QtWebEngine 初始化纪律（同 flowgraph_web_window）
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox --disable-gpu")
from PyQt5.QtCore import Qt as _Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication as _QApp  # noqa: E402

if _QApp.instance() is None:
    _QApp.setAttribute(_Qt.AA_ShareOpenGLContexts, True)
from PyQt5.QtCore import QTimer, QUrl  # noqa: E402
from PyQt5.QtWebEngineWidgets import QWebEnginePage, QWebEngineView  # noqa: E402,E501
from PyQt5.QtWidgets import QLabel, QMainWindow  # noqa: E402


class _Page(QWebEnginePage):
    """页面：JS 报错进日志（内嵌浏览器无开发者工具，否则前端异常无从排查）。"""

    def javaScriptConsoleMessage(self, level, message, line, source):
        if level >= QWebEnginePage.WarningMessageLevel:
            from loguru import logger

            logger.warning("defectgen JS: {} ({}:{})", message, source, line)


# 固定优先端口：资产/历史存于浏览器 IndexedDB，按 origin（host:port）
# 分区——端口每次启动漂移会让 origin 变化，资产表现成"每次重进都为空"
#（第三批 2026-11-07）。优先绑定固定端口保持 origin 稳定；被占用时向后
# 顺延，全部失败才回退 OS 动态分配（该会话资产暂不可见，不影响功能）。
_DEFECTGEN_PORT_PREFERRED = 39771
# 本进程已分出、服务尚未退出的端口：探测绑定后立即释放，服务线程稍后才
# 真正监听——同进程连开两个窗口时第二个会探到同一端口（绑定失败）
_CLAIMED: set = set()
_CLAIM_LOCK = threading.Lock()


def _pick_port() -> int:
    import socket

    with _CLAIM_LOCK:
        for p in range(_DEFECTGEN_PORT_PREFERRED,
                       _DEFECTGEN_PORT_PREFERRED + 48):
            if p in _CLAIMED:
                continue
            s = socket.socket()
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                continue
            finally:
                s.close()
            _CLAIMED.add(p)
            return p
    return 0


def _release_port(port: int) -> None:
    with _CLAIM_LOCK:
        _CLAIMED.discard(port)


class _UvicornThread(threading.Thread):
    """uvicorn 服务线程：优先固定端口（资产持久化），失败回退动态端口；
    daemon=True 不阻退出。"""

    def __init__(self):
        super().__init__(daemon=True, name="defectgen-uvicorn")
        self._port = _pick_port()
        self.server = self._make_server(self._port)

    @staticmethod
    def _make_server(port: int):
        import uvicorn

        from .backend.app import create_app

        return uvicorn.Server(uvicorn.Config(
            create_app(), host="127.0.0.1", port=port, log_level="warning"))

    def run(self):
        try:
            try:
                self.server.run()
            except SystemExit:
                # 探测后、监听前被其它进程抢占（uvicorn 绑定失败即 sys.exit）：
                # 回退 OS 动态分配 port=0，服务照常可用
                if self._port and not self.server.should_exit:
                    _release_port(self._port)
                    self._port = 0
                    self.server = self._make_server(0)
                    self.server.run()
        finally:
            _release_port(self._port)

    def actual_port(self) -> int | None:
        """服务启动后取 OS 实际分派的端口（未就绪/已停返回 None）。"""
        try:
            servers = getattr(self.server, "servers", None)
            if servers:
                return int(servers[0].sockets[0].getsockname()[1])
        except Exception:  # noqa: BLE001
            return None
        return None


class DefectGenWindow(QMainWindow):
    """缺陷生成主窗口（QtWebEngine 内嵌本地服务）。"""

    tool_key = "defectgen"
    tool_title = "缺陷生成"

    def __init__(self, on_closed=None):
        super().__init__()
        self.on_closed = on_closed
        from PyQt5.QtGui import QIcon

        from ..ui_style import icon_path

        self.setWindowTitle(f"{self.tool_title}  -  XCcv DefectAugmentStudio")
        self.setWindowIcon(QIcon(str(icon_path("defectgen.png"))))
        self.resize(1360, 860)

        self._server = _UvicornThread()
        self._server.start()

        self.page = _Page(self)
        # 内嵌浏览器无下载栏：不接此信号 <a download> 导出会静默丢失
        self.page.profile().downloadRequested.connect(self._on_download)
        view = QWebEngineView(self)
        view.setPage(self.page)
        view.setContextMenuPolicy(_Qt.NoContextMenu)
        self.setCentralWidget(view)

        self.status = QLabel("正在启动缺陷生成服务…")
        self.status.setStyleSheet("padding:4px 10px; color:#64748b;")
        self.statusBar().addWidget(self.status, 1)

        # 轮询动态端口（uvicorn 启动完成回填 URL；200ms × 最多 150 次=30s）
        self._poll = QTimer(self)
        self._poll.timeout.connect(self._poll_port)
        self._poll.start(200)

    # ---------- 端口回填 ----------
    def _poll_port(self):
        port = self._server.actual_port()
        if port is None:
            return
        self._poll.stop()
        self.port = port
        self.status.setText(f"服务已就绪：127.0.0.1:{port}（关闭窗口即停止）")
        self.page.setUrl(QUrl(f"http://127.0.0.1:{port}/"))

    # ---------- 导出保存 ----------
    def _on_download(self, item):
        if item.page() is not self.page:
            return
        from PyQt5.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(self, "保存文件", item.path())
        if not path:
            item.cancel()
            return
        item.setPath(path)
        item.accept()
        self.status.setText(f"已保存：{path}")

    # ---------- 生命周期 ----------
    def closeEvent(self, ev):
        if self.on_closed:
            self.on_closed(self.tool_key)
        try:
            self.page.profile().downloadRequested.disconnect(self._on_download)
        except (TypeError, RuntimeError):
            pass
        # 停服（uvicorn 子线程随窗口结束，不驻留）
        try:
            self._server.server.should_exit = True
        except Exception:  # noqa: BLE001
            pass
        # QtWebEngine 页面先于应用退出销毁（段错误防线，同 flowgraph_web_window）；
        # closeEvent 可能二次触发（窗口重复 close），已删对象静默跳过
        try:
            self.page.deleteLater()
        except RuntimeError:  # noqa: BLE001  页面已随视图销毁
            pass
        central = self.centralWidget()
        if central is not None:
            central.deleteLater()
        super().closeEvent(ev)


def build_window(on_closed=None):
    """主界面/launcher 工厂。"""
    return DefectGenWindow(on_closed=on_closed)


def main():
    import sys

    from ..platform_entry import run_platform

    run_platform("defectgen", "缺陷生成", build_window, heavy=False,
                 enable_logging=False)


if __name__ == "__main__":
    from ..platform_entry import early_splash

    early_splash("defectgen", "缺陷生成")
    main()

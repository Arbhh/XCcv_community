"""平台单实例守卫。

每个平台是独立进程，重复启动（主界面再点一次 / 用户直接双击 exe）时
应激活已有窗口而不是开第二个进程。实现：QLocalServer 具名管道——
先探测服务是否存在，存在则发 activate 消息并让新进程退出；
否则本进程注册服务，收到 activate 时激活自己的窗口。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtNetwork import QLocalServer, QLocalSocket


def already_running(key: str) -> bool:
    """是否已有同平台实例在运行（轻量探测，重型导入前调用）。"""
    sock = QLocalSocket()
    sock.connectToServer(f"xccv-{key}")
    connected = sock.waitForConnected(300)
    if connected:
        sock.write(b"activate")
        sock.waitForBytesWritten(300)
        sock.disconnectFromServer()
    return connected


def register(key: str, window) -> None:
    """注册单实例服务：后续重复启动会激活 window。"""
    name = f"xccv-{key}"
    QLocalServer.removeServer(name)  # 清理异常退出残留（Windows 上为无害空操作）
    server = QLocalServer()
    if not server.listen(name):
        return  # 注册失败不阻断正常使用

    def _activate():
        conn = server.nextPendingConnection()
        if conn is None:
            return
        conn.waitForReadyRead(300)
        conn.disconnectFromServer()
        window.setWindowState((window.windowState() & ~Qt.WindowMinimized) | Qt.WindowActive)
        window.show()
        window.raise_()
        window.activateWindow()

    server.newConnection.connect(_activate)
    window._xccv_single_instance_server = server  # 保活


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402
apply_auto_slot_module(globals())

"""全局日志：各平台进程独立日志文件，便于工控机等远程环境排障。

- 位置：platform_data/logs/<平台>.log（与 crash.log 同目录，不可写时回退临时目录）；
- 每平台一个文件（训练/测试/模型工具/主界面各自独立进程，互不锁文件），
  RotatingFileHandler 单文件 5MB、保留 4 份历史，防止无限膨胀；
- 每次启动写分隔头（平台、版本、时间），方便把一段完整会话回传分析；
- 模块内用法：logger = logging.getLogger(__name__)，在关键路径埋点；
  未捕获异常由 excepthook 同步写入日志（install_excepthook）。
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .gui_utils import crash_log_path

_FMT_FILE = "%(asctime)s  %(levelname)-7s [%(name)s]  %(message)s"
_FMT_CONSOLE = "%(levelname)s [%(name)s] %(message)s"

_configured = False
_fault_file = None  # faulthandler 的文件句柄需常驻引用，防止被 GC 后失效


def logs_dir() -> Path:
    """日志目录（与崩溃日志同目录）。"""
    return crash_log_path().parent


def setup_logging(platform: str, console_level: int = logging.WARNING) -> Path:
    """初始化根日志器并写会话分隔头；返回日志文件路径。重复调用安全。"""
    global _configured, _fault_file
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    log_file = logs_dir() / f"{platform}.log"
    if not _configured:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=4, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FMT_FILE))
        root.addHandler(file_handler)
        console = logging.StreamHandler(sys.stdout)  # pythonw 下为 devnull，无副作用
        console.setLevel(console_level)
        console.setFormatter(logging.Formatter(_FMT_CONSOLE))
        root.addHandler(console)
        try:
            import faulthandler

            _fault_file = open(crash_log_path(), "a", encoding="utf-8",
                               errors="replace")
            faulthandler.enable(file=_fault_file)  # native 级崩溃也留 Python 调用栈
        except Exception:
            pass  # faulthandler 不可用不影响正常日志
        _configured = True
    from . import __version__

    logging.getLogger("xccv").info(
        f"===== XCcv {platform} 平台启动  v{__version__} =====")
    return log_file


def get_logger(name: str) -> logging.Logger:
    """模块 logger 快捷入口（名字自动带 xccv 前缀层级）。"""
    return logging.getLogger(name)

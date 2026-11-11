"""状态栏分层总线（v0.5.4 任务三）。

分层规则（用户定案）：**大功能平台拥有专属底部状态栏**（训练平台、测试
平台、标注平台、模型工具、推理服务、模型加载、C# 测试）；其下的小项目、
子功能窗口不再各自显示状态栏，消息统一路由到所属大平台的状态栏。

用法：
- 大平台主窗口：``attach_platform_status(win, "训练平台", "training")``
  —— 底部状态栏左侧加平台名常驻徽章，并注册为该平台的消息宿主；
- 子窗口 / 子功能：``show_status("training", "参数已保存")`` —— 消息显示
  在训练平台状态栏；子窗口自身应 ``self.setStatusBar(None)``。
"""

import weakref

from PyQt5.QtWidgets import QLabel

# 平台 key 常量（各窗口注册与子窗口路由共用，避免散落字符串）
TRAINING = "training"
TESTING = "testing"
ANNOTATION = "annotation"
DEPLOY = "deploy"          # 推理服务
MODEL_LOAD = "model_load"
CSHARP = "csharp"

_hosts = {}   # key -> weakref.ref(QMainWindow)


def attach_platform_status(win, label: str, key: str) -> None:
    """大平台主窗口：状态栏左侧加平台名徽章并注册为消息宿主。"""
    badge = QLabel(f"  {label}  ")
    badge.setObjectName("badge")
    win.statusBar().addWidget(badge)
    ref = weakref.ref(win, _make_cleaner(key))
    _hosts[key] = ref


def _make_cleaner(key):
    def _clean(_ref):
        if _hosts.get(key) is not None:
            _hosts.pop(key, None)
    return _clean


def show_status(key: str, text: str, timeout: int = 5000) -> None:
    """向指定大平台的状态栏发消息（子窗口/子功能统一入口）。"""
    ref = _hosts.get(key)
    win = ref() if ref is not None else None
    if win is not None:
        win.statusBar().showMessage(text, timeout)


report = show_status   # 语义别名


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402

apply_auto_slot_module(globals())

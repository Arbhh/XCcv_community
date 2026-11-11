"""PyQt5 槽函数自动包装：修复 Cython 编译态信号槽参数不适配问题。

背景：PyQt5 发射内置信号（如 clicked(bool)、textChanged(str)）时，靠检查
槽函数的 __code__ 裁剪信号附带的多余参数。Cython 编译后的方法不是标准
Python 函数（无 code object），PyQt5 无法裁剪 → 所有无参槽方法被多传参数
→ TypeError → PyQt5.15 把槽内异常转为进程 abort（表现为"所有按钮点击
都报错后闪退"）。给编译函数加 @pyqtSlot 装饰也会硬崩，均不可用。

机制：apply_auto_slot(module_globals) 扫描模块内定义的 QObject 子类，把
类中本类定义的普通方法包一层纯 Python 包装函数（PyQt5 对标准 Python
函数能正确裁剪；包装为参数透传，不改变任何行为）。开发态（源码 .py）
包装后行为同样一致，因此 dev/编译态统一逻辑、无分支。

用法：模块末尾一行 ——
    from xccv.qt_slots import apply_auto_slot
    apply_auto_slot(globals())
"""

import inspect
import os
from functools import wraps

from PyQt5.QtCore import QObject


_PASS_THROUGH = -1


def _positional_args(fn) -> int:
    """fn 除 self 外可接受的位置参数个数（取默认值参数计入）。

    实测要点：槽被调用时若实参数超过 fn 形参（如 clicked 的 checked 传给
    无参槽），PyQt5.15 会把 TypeError 直接转为进程 abort（退出码 127），
    而非打印异常——所以包装层必须按原函数签名"显式裁剪"。
    *args 型返回 _PASS_THROUGH：签名本就"来者不拒"，必须原样透传
    （曾返回 0 被当成无参槽，removeRows(row, 1) 实参全丢→删除多边形报错）。
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return 0  # 签名不可知时按无参处理（绝大多数 UI 槽即如此）
    n = 0
    for p in list(sig.parameters.values())[1:]:  # 跳过 self
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
        elif p.kind is p.VAR_POSITIONAL:
            return _PASS_THROUGH
    return n


def _wrap(fn):
    """纯 Python 包装：PyQt5 可对它正确裁剪参数；按原签名数调用原实现。

    实测两条铁律（PyQt5.15 + offscreen 下违反即进程 abort，退出码 127）：
    1. 不能透传 *args——clicked 的 checked 等多余实参传进无参槽会
       TypeError，PyQt5 把它转为 abort 而非打印；必须按签名显式裁剪；
    2. 不能用 @pyqtSlot 装饰（编译函数配 pyqtSlot 派发即硬崩）。
    """
    n = _positional_args(fn)

    if n == _PASS_THROUGH:
        def wrapper(self, *args, **kwargs):
            return fn(self, *args, **kwargs)
    elif n == 0:
        def wrapper(self, *args, **kwargs):
            return fn(self)
    else:
        def wrapper(self, *args, **kwargs):
            # kwargs 必须原样转发：_set_image_list(paths, root=…) /
            # _build_echo_form(config=…) 等关键字调用曾因丢 kwargs 退化为 None
            return fn(self, *args[:n], **kwargs)

    wrapper.__name__ = getattr(fn, "__name__", "slot_wrapper")
    wrapper.__wrapped__ = fn       # 检查/测试可追溯到原实现
    wrapper._autoslot_wrapped = True  # 幂等标记（编译函数自带 __wrapped__，不可用作标记）
    return wrapper


def _is_plain_method(obj) -> bool:
    """本类定义的、需要包装的普通方法（函数对象）。"""
    if not callable(obj) or isinstance(obj, (staticmethod, classmethod, property)):
        return False
    if hasattr(obj, "__pyqtSignature__"):
        return False  # 已显式 pyqtSlot 声明
    return inspect.isfunction(obj) or inspect.isbuiltin(obj) or \
        type(obj).__name__ == "cython_function_or_method"


def apply_auto_slot(cls) -> None:
    """对单个 QObject 子类应用槽包装；重复调用无害（幂等）。"""
    if os.environ.get("XCCV_DISABLE_AUTOSLOT"):
        return
    if not (isinstance(cls, type) and issubclass(cls, QObject)):
        return
    for name, obj in list(vars(cls).items()):
        if name.startswith("__") or not _is_plain_method(obj):
            continue
        if getattr(obj, "_autoslot_wrapped", False):
            continue  # 已是包装函数（幂等保护；编译函数自带 __wrapped__，不能用它判断）
        setattr(cls, name, _wrap(obj))


def apply_auto_slot_module(globals_: dict) -> None:
    """扫描模块命名空间，对其内定义的全部 QObject 子类应用槽包装。

    只处理 __module__ 与当前模块一致的类，避免重复包装从其他模块导入的类。
    """
    mod = globals_.get("__name__")
    for obj in list(globals_.values()):
        if isinstance(obj, type) and obj.__module__ == mod:
            apply_auto_slot(obj)

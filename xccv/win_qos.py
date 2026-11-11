"""进程 QoS：显式退出 Windows EcoQoS（电源节流），保证推理/训练速度稳定。

背景（V0.6 用户回传，2026-11-02 实测于 Intel Core 7 250H 混合架构笔记本）：
Windows 11 会对"后台"进程自动施加 EcoQoS——降频并优先调度到能效核（E 核）。
测试平台单图推理在主线程执行（maskrcnn 2821×2144 约 3.4s），窗口一旦失去
前台（用户切走窗口，或主线程阻塞超过约 5s 被 DWM 判"未响应"、换成幽灵
窗口），进程即被节流：同一推理 3.4s → 8.8s（SetProcessInformation 强制
EcoQoS 复现，与用户"3s 飙到 8s 以上"吻合）。

XCcv 是计算密集型桌面工具：各进程入口显式声明 HighQoS（ControlMask=
EXECUTION_SPEED、StateMask=0 → 系统不再自动节流）。仅 Windows 生效；
任何失败静默返回 False（不影响功能，只是速度可能受系统节流）。
"""

import sys

_PROCESS_POWER_THROTTLING = 4            # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
_THROTTLING_CURRENT_VERSION = 1          # PROCESS_POWER_THROTTLING_CURRENT_VERSION
_THROTTLING_EXECUTION_SPEED = 0x1        # PROCESS_POWER_THROTTLING_EXECUTION_SPEED


def _kernel32():
    import ctypes
    import ctypes.wintypes as wt

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wt.HANDLE
    for fn in (k32.SetProcessInformation, k32.GetProcessInformation):
        fn.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
        fn.restype = wt.BOOL
    return k32


def _state_struct():
    import ctypes
    import ctypes.wintypes as wt

    class _State(ctypes.Structure):  # PROCESS_POWER_THROTTLING_STATE
        _fields_ = [("Version", wt.ULONG), ("ControlMask", wt.ULONG),
                    ("StateMask", wt.ULONG)]

    return _State


def opt_out_power_throttling() -> bool:
    """当前进程退出 EcoQoS（HighQoS）。成功返回 True；非 Windows/失败返回 False。

    幂等，可在多个入口重复调用。须尽早调用（进程入口，重型计算之前）。
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        k32 = _kernel32()
        state = _state_struct()(_THROTTLING_CURRENT_VERSION,
                                _THROTTLING_EXECUTION_SPEED, 0)
        return bool(k32.SetProcessInformation(
            k32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING,
            ctypes.byref(state), ctypes.sizeof(state)))
    except Exception:
        return False


def power_throttling_state() -> tuple | None:
    """查询当前进程 (ControlMask, StateMask)；不支持查询的系统返回 None（测试/排障用）。"""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        k32 = _kernel32()
        state = _state_struct()(_THROTTLING_CURRENT_VERSION, 0, 0)
        if not k32.GetProcessInformation(
                k32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING,
                ctypes.byref(state), ctypes.sizeof(state)):
            return None
        return int(state.ControlMask), int(state.StateMask)
    except Exception:
        return None

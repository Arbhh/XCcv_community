"""任务栏分组：为当前进程设置显式 AppUserModelID（AUMID）。

背景（v0.5.5 用户批注）：开发态所有平台都由同一个 pythonw.exe 启动，
Windows 默认按 exe 路径做任务栏分组 → 主界面/训练/测试/部署…全部被
并进同一个任务栏按钮。为每个平台进程设置独立 AUMID 后，任务栏按
「大平台」划分条目（同进程的子项目/子功能窗口自动归在所属平台之下，
如项目窗口归训练平台），与打包态（各平台独立 exe）行为一致。

仅在 Windows 生效；调用须在首个窗口创建之前（platform_entry / app.main
入口处）。失败静默（不影响功能，仅分组退化）。
"""

import ctypes
import sys


def set_process_app_user_model_id(aumid: str) -> bool:
    """设置进程级 AUMID（如 "XCcv.Training"）。Windows 以外返回 False。"""
    if sys.platform != "win32":
        return False
    try:
        prop = ctypes.c_wchar_p(aumid)
        hr = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            prop)
        return hr == 0   # S_OK == 0（此前 bool(hr) 把成功误判为失败）
    except Exception:
        return False


def aumid_for(key: str) -> str:
    """平台 key → AUMID（platform_entry 的 key 直接复用）。"""
    mapping = {"deploy": "XCcv.InferenceService",
               "model_load": "XCcv.ModelLoading",
               "csharp_test": "XCcv.CSharpTesting"}
    return mapping.get(key, f"XCcv.{key}")

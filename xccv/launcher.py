"""平台独立进程启动器。

打包后每个平台是独立 exe（与主界面同在安装根），主界面点击卡片时优先
以子进程拉起同目录 exe；exe 缺失时回退解释器直接调用平台 main()（安装
态模块已编译为 .pyd，不能用 `python -m`，必须走 importlib）。开发态
（源码 .py）用 pythonw -m。任何平台出错都不会影响主界面与其他平台。
单实例由平台侧 single_instance 守卫负责。
"""

import subprocess
import sys
from pathlib import Path

from .app_paths import app_root, is_installed

# 平台 key -> (独立 exe 名, 模块名)
PLATFORMS = {
    "annotation": ("XCcv标注平台.exe", "xccv.annotation.annotator_window"),
    "training": ("XCcv训练平台.exe", "xccv.training.training_window"),
    "testing": ("XCcv测试平台.exe", "xccv.testing.testing_window"),
    # 缺陷样本合成（DefectAugmentStudio，独立进程）
    "defectgen": ("XCcv缺陷生成.exe", "xccv.defectgen.defectgen_window"),
}


def _pythonw() -> str:
    """子进程解释器：优先 venv 自身的 pythonw。

    venv 里的 python.exe 可能是基础解释器的符号链接/副本，sys.executable 会
    解析到基环境（丢失 venv 的 site-packages，QtWebEngine 等组件随之缺失）。
    sys.prefix 才是当前 venv 根——只要它的 Scripts 下有解释器就用它。
    """
    scripts = Path(sys.prefix) / "Scripts"
    for name in ("pythonw.exe", "python.exe"):
        cand = scripts / name
        if cand.is_file():
            return str(cand)
    python = Path(sys.executable)
    w = python.with_name("pythonw.exe")
    return str(w if w.is_file() else python)


def launch_platform(key: str, arg: str | None = None) -> subprocess.Popen:
    """以独立进程启动平台，返回进程句柄。

    arg 为可选的命令行参数（如数据体检的「去标注平台处理」把数据集目录
    传给标注平台；annotator_window.main 读取 sys.argv[1]）。
    """
    exe_name, module = PLATFORMS[key]
    if getattr(sys, "frozen", False) or is_installed():
        # 冻结/安装态：优先同目录独立 exe，缺失回退 importlib 调 main()
        exe = Path(sys.executable).resolve().parent / exe_name
        if exe.is_file():
            cmd = [str(exe)]
            if arg:
                cmd.append(arg)
            return subprocess.Popen(cmd, cwd=str(exe.parent), close_fds=True)
        code = (f"import importlib; importlib.import_module('{module}').main()")
        cmd = [_pythonw(), "-c", code]
    else:  # 开发态：源码模块可直接 -m
        cmd = [_pythonw(), "-m", module]
    if arg:
        cmd.append(arg)
    return subprocess.Popen(cmd, cwd=str(app_root()), close_fds=True)

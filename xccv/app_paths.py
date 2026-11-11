"""安装根定位：日志/权重/项目数据等运行期根目录的统一解析。

三种形态：
- 安装态（Cython 编译为 .pyd 随完整解释器分发，xccv 位于 site-packages）：
  根 = sys.prefix，即安装器铺出的环境目录本身
  （weights/、bin/、platform_data/ 都在环境根下）；
- 冻结态（Nuitka/PyInstaller 兼容）：根 = exe 所在目录；
- 开发态（源码工作区）：根 = xccv 包的上级目录（仓库根）。
"""

import sys
from pathlib import Path


def is_installed() -> bool:
    """xccv 是否以安装形态位于 site-packages（编译产物随解释器环境分发）。"""
    if getattr(sys, "frozen", False):
        return False
    return "site-packages" in Path(__file__).resolve().parts


def app_root() -> Path:
    """运行期根目录（platform_data/、weights/、bin/ 的落点）。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    here = Path(__file__).resolve()
    if "site-packages" in here.parts:
        return Path(sys.prefix).resolve()
    return here.parents[1]  # 开发态：xccv 包上级 = 仓库根

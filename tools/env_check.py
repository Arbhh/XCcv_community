# -*- coding: utf-8 -*-
"""启动环境诊断（中文）：按 run_*.bat 的优先级逐个检查候选解释器。

优先级：XCCV_PY 环境变量 → python_path.txt → PATH 的 python → py -3
        → C:\XCcv\venv（产品环境，可选）→ 本目录 venv/.venv（可选兜底）
注意：本目录 venv 只是最后兜底，不要求存在。
"""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cfg_file = ROOT / "python_path.txt"
cfg_value = ""
if cfg_file.is_file():
    try:
        cfg_value = (cfg_file.read_text(encoding="utf-8", errors="ignore").strip()
                     .splitlines() or [""])[0].strip()
    except OSError:
        cfg_value = ""

CANDIDATES = [
    ("XCCV_PY 环境变量", os.environ.get("XCCV_PY") or "", []),
    ("python_path.txt", cfg_value, []),
    ("PATH 中的 python", shutil.which("python") or "", []),
    ("py 启动器（-3）", shutil.which("py") or "", ["-3"]),
    ("产品环境 C:\\XCcv\\venv", r"C:\XCcv\venv\Scripts\python.exe", []),
    ("本目录 venv（可选）", str(ROOT / "venv" / "Scripts" / "python.exe"), []),
    ("本目录 .venv（可选）", str(ROOT / ".venv" / "Scripts" / "python.exe"), []),
]

print("=" * 62)
print("XCcv 社区版 · 启动环境诊断")
print(f"程序目录: {ROOT}")
print("=" * 62)
ok = None
for label, exe, extra in CANDIDATES:
    if not exe or not Path(exe).is_file():
        print(f"[不可用] {label}: {exe or '（未设置 / 未找到）'}")
        continue
    try:
        p = subprocess.run([exe, *extra, "-c",
                            "import sys, PyQt5.QtWebEngineWidgets; print(sys.version.split()[0])"],
                           capture_output=True, text=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        print(f"[不可用] {label}: {exe} -> {type(e).__name__}: {e}")
        continue
    if p.returncode == 0:
        print(f"[可用]   {label}: {exe}  (Python {p.stdout.strip()})")
        ok = ok or (label, exe)
    else:
        tail = ((p.stderr or p.stdout).strip().splitlines() or [""])[-1]
        print(f"[不可用] {label}: {exe}")
        print(f"         原因: {tail[:150]}")

print("-" * 62)
if ok:
    print(f"结论：将使用「{ok[0]}」-> {ok[1]}")
    print("      run_*.bat 会自动选中它，可直接双击启动。")
else:
    print("结论：没有找到可用的解释器（需要 PyQtWebEngine）。")
    print("      任选一种方式指定：")
    print("        1. 在本目录新建 python_path.txt，写入一行完整路径，例如：")
    print("             C:\\XCcv\\venv\\Scripts\\python.exe")
    print("        2. 启动前设置环境变量：")
    print("             set XCCV_PY=C:\\XCcv\\venv\\Scripts\\python.exe")
    print("        3. 或给某个 Python 3.11 安装依赖后使用：")
    print("             <python.exe> -m pip install -r requirements.txt")
print("=" * 62)
raise SystemExit(0 if ok else 1)

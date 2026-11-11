@echo off
rem XCcv Community launcher - resolves interpreter, verifies QtWebEngine, then runs.
setlocal
cd /d "%~dp0"

set "PY=%XCCV_PY%"
if not defined PY if exist "%~dp0venv\Scripts\python.exe" set "PY=%~dp0venv\Scripts\python.exe"
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY set "PY=python"

"%PY%" -c "import PyQt5.QtWebEngineWidgets" >nul 2>nul
if errorlevel 1 goto nogui
"%PY%" -m xccv.testing.testing_window

exit /b %errorlevel%

:nogui
echo.
echo [XCcv] 启动失败：当前 Python 环境缺少 PyQtWebEngine，或解释器指向了错误环境。
echo         当前解释器: %PY%
echo         解决办法（任选其一）：
echo           1. 在本目录创建虚拟环境并安装依赖：
echo              python -m venv venv
echo              venv\Scripts\pip install -r requirements.txt
echo           2. 设置环境变量 XCCV_PY 指向已装好依赖的 python.exe
echo.
pause
exit /b 1
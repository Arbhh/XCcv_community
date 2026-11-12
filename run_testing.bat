@echo off
rem XCcv Community launcher (ASCII only).
rem Interpreter precedence:
rem   1) XCCV_PY environment variable
rem   2) python_path.txt in this folder (one line: full path to python.exe)
rem   3) python on PATH
rem   4) py launcher (-3)
rem   5) C:\XCcv\venv (the product environment, if present)
rem   6) .\venv and .\.venv in this folder (optional, last resort)
rem Each candidate is accepted only if it can import PyQt5.QtWebEngineWidgets.
setlocal EnableExtensions
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PYEXE="
set "PYARGS="
set "PYCFG="
if exist "%ROOT%python_path.txt" set /p PYCFG=<"%ROOT%python_path.txt"

call :try "%XCCV_PY%" ""
call :try "%PYCFG%" ""
call :try "python" ""
call :try "py" "-3"
call :try "C:\XCcv\venv\Scripts\python.exe" ""
call :try "%ROOT%venv\Scripts\python.exe" ""
call :try "%ROOT%.venv\Scripts\python.exe" ""
if not defined PYEXE goto nogui

"%PYEXE%" %PYARGS% -m xccv.testing.testing_window

exit /b %errorlevel%

:try
if defined PYEXE exit /b 0
if "%~1"=="" exit /b 0
"%~1" %~2 -c "import PyQt5.QtWebEngineWidgets" >nul 2>nul
if errorlevel 1 exit /b 0
set "PYEXE=%~1"
set "PYARGS=%~2"
exit /b 0

:nogui
echo.
echo [XCcv] No usable Python found (PyQtWebEngine is required).
echo         Candidates tried, in order:
echo           XCCV_PY, python_path.txt, python on PATH, py -3,
echo           C:\XCcv\venv, .\venv, .\.venv
echo.
echo         How to fix (either one):
echo           1) Tell XCcv which interpreter to use - create python_path.txt
echo              in this folder with one line, e.g.:
echo                C:\XCcv\venv\Scripts\python.exe
echo              or set an environment variable before launching:
echo                set XCCV_PY=C:\XCcv\venv\Scripts\python.exe
echo           2) Install the dependencies into your own Python 3.11:
echo                pip install -r requirements.txt
echo.
if exist "%ROOT%tools\env_check.py" call :diag
pause
exit /b 1

:diag
python "%ROOT%tools\env_check.py" 2>nul
if errorlevel 1 echo (env_check.py needs a working python on PATH; skipped)
exit /b 0
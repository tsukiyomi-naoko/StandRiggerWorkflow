@echo off
rem StandRig Setup launcher: finds a Python with tkinter (ComfyUI Desktop's own first) and opens the GUI.
setlocal EnableDelayedExpansion
set "SCRIPT=%~dp0installer\standrig_setup.py"
set "PY="

rem 1) ComfyUI Desktop: basePath from %APPDATA%\ComfyUI\config.json -> <basePath>\.venv\Scripts\python.exe
for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$f = Join-Path $env:APPDATA 'ComfyUI\config.json'; if (Test-Path $f) { $b = (Get-Content $f -Raw | ConvertFrom-Json).basePath; if ($b) { $p = Join-Path $b '.venv\Scripts\python.exe'; if (Test-Path $p) { $p } } }"`) do set "PY=%%P"
if defined PY call :try "!PY!"

rem 2) Python launcher / python on PATH
if not defined OK for /f "delims=" %%P in ('where py 2^>nul') do if not defined OK call :try "%%P" -3
if not defined OK for /f "delims=" %%P in ('where python 2^>nul') do if not defined OK call :try "%%P"

if not defined OK (
  echo.
  echo Could not find a Python with tkinter.
  echo  - Start ComfyUI Desktop once so it finishes installing, then run this again, or
  echo  - install Python 3.10+ from https://www.python.org/downloads/ ^(tick "tcl/tk"^).
  pause
  exit /b 1
)

rem Start without a console window when pythonw is next to python
set "PYW=%OKPY:python.exe=pythonw.exe%"
if exist "%PYW%" (
  start "" "%PYW%" %OKARGS% "%SCRIPT%" %*
) else (
  start "" "%OKPY%" %OKARGS% "%SCRIPT%" %*
)
exit /b 0

:try
"%~1" %~2 -c "import tkinter, urllib.request, zipfile" >nul 2>nul
if %errorlevel%==0 (
  set "OK=1"
  set "OKPY=%~1"
  set "OKARGS=%~2"
)
exit /b 0

@echo off
setlocal
cd /d %~dp0

if "%PORT%"=="" set PORT=8000

where py >nul 2>nul
if %errorlevel%==0 (
  set PYTHON_CMD=py -3
) else (
  set PYTHON_CMD=python
)

echo Starting Agentic AI-BI MVP on http://127.0.0.1:%PORT%
start "" http://127.0.0.1:%PORT%
%PYTHON_CMD% server.py

echo.
echo If the port is busy, try:
echo   set PORT=8001
echo   start_windows.bat
pause

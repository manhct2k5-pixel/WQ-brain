@echo off
setlocal

title brain-learn WSL alpha feed

where wsl >nul 2>nul
if errorlevel 1 (
  echo WSL was not found on this machine.
  echo Install or enable WSL first, then try again.
  pause
  exit /b 1
)

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_DIR=%%~fI"
for /f "usebackq delims=" %%I in (`wsl wslpath -a "%REPO_DIR%"`) do set "WSL_DIR=%%I"

if not defined WSL_DIR (
  echo Failed to convert the repo path into a WSL path.
  pause
  exit /b 1
)

echo Running brain-learn alpha-feed workflow in WSL...
echo Repo: %REPO_DIR%
echo.
echo Steps:
echo   1. doctor
echo   2. feed
echo.

wsl bash -lc "cd '%WSL_DIR%' && bash run_wsl.sh doctor && bash run_wsl.sh feed"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
  echo Workflow completed.
  echo Check these files if needed:
  echo   %REPO_DIR%artifacts\bang_tin_alpha.md
  echo   %REPO_DIR%artifacts\bao_cao_moi_nhat.md
  echo   %REPO_DIR%artifacts\duyet_tay.md
  echo   %REPO_DIR%artifacts\lo_tiep_theo.json
) else (
  echo Workflow finished with errors. Exit code: %EXIT_CODE%
  echo Check the terminal output above and the logs folder.
)

echo.
pause
exit /b %EXIT_CODE%

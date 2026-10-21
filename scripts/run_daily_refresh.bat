@echo off
REM Daily hub refresh for Task Scheduler. Runs from project root.
setlocal
cd /d "%~dp0.."
if not exist "data\logs" mkdir "data\logs"
set "LOG=data\logs\daily_refresh.log"
echo ===== %date% %time% =====>> "%LOG%"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\run_refresh_hub.py >> "%LOG%" 2>&1
) else (
  python scripts\run_refresh_hub.py >> "%LOG%" 2>&1
)
set "RC=%ERRORLEVEL%"
echo exit_code=%RC%>> "%LOG%"
echo.>> "%LOG%"
exit /b %RC%

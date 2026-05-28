@echo off
REM ==================================================================
REM  Arc 10 sidecar — NSSM service install
REM ==================================================================
REM  Prereqs:
REM    1. NSSM installed and on PATH (https://nssm.cc)
REM    2. Python 3.11+ with deps from requirements-dev.txt installed
REM    3. MetaTrader 5 terminal installed + 5ers login configured
REM    4. ``configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml``
REM       checked in alongside this repo
REM    5. Run as administrator (NSSM service install requires it)
REM ==================================================================

setlocal

set SERVICE_NAME=Arc10Sidecar
set PYTHON_EXE=C:\Users\panap\AppData\Local\Python\bin\python.exe
set REPO_ROOT=C:\Users\panap\Documents\Forex-Backtester
set WINNING_CFG=%REPO_ROOT%\configs\l_arc_10_v3.0.2_utc_rerun\winning_config.yaml
REM SIDECAR_ROOT must live under Terminal\Common\Files because the EA
REM uses FILE_COMMON for all file IO (required for Strategy Tester
REM compatibility; see deployment/README.md §1).
set SIDECAR_ROOT=%APPDATA%\MetaQuotes\Terminal\Common\Files\Arc10
set LOG_DIR=%SIDECAR_ROOT%\logs

if not exist "%PYTHON_EXE%" (
  echo ERROR: PYTHON_EXE not found at %PYTHON_EXE%
  echo Edit this script with the correct path.
  exit /b 1
)
if not exist "%WINNING_CFG%" (
  echo ERROR: winning config not found at %WINNING_CFG%
  exit /b 1
)
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo Installing service %SERVICE_NAME% ...
nssm install %SERVICE_NAME% "%PYTHON_EXE%" ^
  -m deployment.sidecar ^
  --winning-config "%WINNING_CFG%" ^
  --sidecar-root "%SIDECAR_ROOT%" ^
  --log-level INFO
if errorlevel 1 (
  echo ERROR: nssm install failed
  exit /b 1
)
nssm set %SERVICE_NAME% AppDirectory "%REPO_ROOT%"
nssm set %SERVICE_NAME% AppStdout "%LOG_DIR%\sidecar.stdout.log"
nssm set %SERVICE_NAME% AppStderr "%LOG_DIR%\sidecar.stderr.log"
nssm set %SERVICE_NAME% AppRotateFiles 1
nssm set %SERVICE_NAME% AppRotateBytes 10485760
nssm set %SERVICE_NAME% AppRestartDelay 5000
nssm set %SERVICE_NAME% AppExit Default Restart
nssm set %SERVICE_NAME% Description "Arc 10 DLR Phase 1 sidecar (UTC native)"
nssm set %SERVICE_NAME% Start SERVICE_AUTO_START

echo.
echo Service %SERVICE_NAME% installed.
echo   Start:  net start %SERVICE_NAME%
echo   Stop:   net stop  %SERVICE_NAME%
echo   Logs:   %LOG_DIR%\sidecar.*.log
echo.
endlocal

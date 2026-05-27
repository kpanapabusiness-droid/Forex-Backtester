@echo off
set SERVICE_NAME=Arc10Sidecar

echo Stopping %SERVICE_NAME% ...
net stop %SERVICE_NAME% 2>nul

echo Removing %SERVICE_NAME% ...
nssm remove %SERVICE_NAME% confirm
if errorlevel 1 (
  echo ERROR: nssm remove failed.
  exit /b 1
)
echo %SERVICE_NAME% uninstalled.

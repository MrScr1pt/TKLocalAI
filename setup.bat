@echo off
REM TKLocalAI Setup - Launches PowerShell setup script
REM ===================================================

echo.
echo Starting TKLocalAI Setup...
echo.

REM Run the PowerShell setup script with execution policy bypass
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0setup.ps1"

if errorlevel 1 (
    echo.
    echo Setup encountered errors. See above for details.
)

pause

@echo off
REM TKLocalAI Setup - Launches PowerShell setup script
REM ===================================================

echo.
echo Starting TKLocalAI Setup...
echo.

REM Run the PowerShell setup script with execution policy bypass
powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1"

pause

@echo off
REM TKLocalAI Launcher
REM ==================

if "%1"=="" (
    echo Usage: start.bat [desktop^|server]
    echo.
    echo   start.bat desktop  - Launch desktop app with UI
    echo   start.bat server   - Launch API server only
    echo.
    pause
    exit /b 1
)

powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" %1

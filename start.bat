@echo off
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo Starting ANPR with virtual environment...
    .venv\Scripts\python.exe Run.py
) else (
    echo No .venv found. Trying system python...
    python Run.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Could not start. See message above.
    pause
)

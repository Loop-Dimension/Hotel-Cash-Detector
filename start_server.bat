@echo off
echo ========================================
echo Starting Hotel Cash Detector Server
echo ========================================
echo.

REM Check if waitress is installed
python -c "import waitress" 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Installing waitress for production server...
    pip install waitress
    echo.
)

echo [INFO] Starting production server...
echo [INFO] Access at: http://localhost:5000
echo [INFO] Press Ctrl+C to stop
echo.

python run_production.py


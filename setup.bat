@echo off
REM TEDR System Installation and Setup Script for Windows

echo =========================================
echo TEDR - Object Detection System Setup
echo =========================================

REM Check Python version
echo.
echo Checking Python version...
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
if not exist "venv\" (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Verify installation
echo.
echo Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"

echo.
echo =========================================
echo Installation complete!
echo =========================================
echo.
echo To start the API server:
echo   venv\Scripts\activate
echo   python backend\main.py
echo.
echo Then open your browser to:
echo   http://localhost:8000/static/index.html
echo.
echo For API documentation:
echo   http://localhost:8000/docs
echo.

pause

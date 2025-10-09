@echo off
echo ========================================
echo  Job Market Analysis Agentic AI
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found: 
python --version

REM Check if virtual environment exists
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Check if requirements are installed
echo.
echo Checking if dependencies are installed...
python -c "import fastapi, streamlit, pandas, numpy, plotly, requests" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Some dependencies may not be installed.
    echo Please run: pip install -r requirements.txt
    echo.
    echo Do you want to continue anyway? (Press any key to continue or Ctrl+C to exit)
    pause
)

REM Create data directories
echo.
echo Creating data directories...
if not exist "data" mkdir data
if not exist "data\raw" mkdir data\raw
if not exist "data\preprocessed" mkdir data\preprocessed
if not exist "logs" mkdir logs
if not exist "charts" mkdir charts

echo.
echo ========================================
echo  Setup Complete! Starting Application
echo ========================================
echo.

REM Start backend in new window
echo Starting FastAPI backend...
start "Job Market AI - Backend" cmd /k "call venv\Scripts\activate.bat && python main.py"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend in new window
echo Starting Streamlit frontend...
start "Job Market AI - Frontend" cmd /k "call venv\Scripts\activate.bat && streamlit run streamlit_app.py"

echo.
echo ========================================
echo  Application Started Successfully!
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Frontend: http://localhost:8501
echo.
echo Press any key to exit this window...
pause >nul
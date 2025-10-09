@echo off
echo Starting Job Vacancies Agent...
echo.

echo Starting Backend Server...
start "Backend Server" cmd /k "python main.py"

echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak >nul

echo Starting Frontend Server...
start "Frontend Server" cmd /k "streamlit run streamlit_app.py"

echo.
echo Both servers are starting...
echo - Backend: http://localhost:8000
echo - Frontend: http://localhost:8501
echo.
echo Press any key to exit...
pause >nul



@echo off
chcp 65001 > nul
title Thai Financial Document OCR Server

echo ============================================================
echo   Starting Thai Financial Document OCR & Extraction System
echo ============================================================
echo.

cd /d "%~dp0"

:: 1. Check & Start PostgreSQL Docker Container
echo [1/3] Checking PostgreSQL Docker container...
docker ps -q -f name=expense-reimbursement-postgres > nul 2>&1
for /f "tokens=*" %%i in ('docker ps -q -f name=expense-reimbursement-postgres 2^>nul') do set DOCKER_RUNNING=%%i

if "%DOCKER_RUNNING%"=="" (
    echo       Container is not running. Attempting to start...
    docker start expense-reimbursement-postgres > nul 2>&1
    if errorlevel 1 (
        echo       [!] Warning: Could not start Docker container 'expense-reimbursement-postgres'.
        echo           Please make sure Docker Desktop is running.
    ) else (
        echo       [OK] Docker container started successfully.
    )
) else (
    echo       [OK] PostgreSQL Docker container is already running.
)

:: 2. Check & Start Ollama (Port 11434)
echo.
echo [2/3] Checking Ollama Service (Port 11434)...
netstat -ano | findstr :11434 > nul
if errorlevel 1 (
    echo       Ollama is not running. Starting Ollama in background...
    start "" ollama serve
    timeout /t 3 /nobreak > nul
    echo       [OK] Ollama started.
) else (
    echo       [OK] Ollama is already running.
)

:: 3. Check Virtual Environment & Start Server
echo.
echo [3/3] Starting OCR Web Server...
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe!
    echo Please make sure .venv is installed in this directory.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Server is running at: http://localhost:8000
echo   Swagger API Docs at:  http://localhost:8000/docs
echo   Press Ctrl + C in this window to stop the server.
echo ============================================================
echo.

:: Automatically open browser to Swagger API Docs
start "" cmd /c "timeout /t 2 /nobreak > nul && start http://localhost:8000/docs"

:: Launch FastAPI Web Server
.\.venv\Scripts\python.exe -m src.app

pause

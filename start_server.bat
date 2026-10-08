@echo off
chcp 65001 > nul
title Thai Financial Document OCR & Extraction Server

echo ============================================================
echo   Starting Thai Financial Document OCR & Extraction System
echo ============================================================
echo.

cd /d "%~dp0"

:: 0. Hardware Acceleration & GPU Auto-Detection
echo [Hardware Setup] Detecting GPU Acceleration...
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2^>nul') do (
        echo       [OK] NVIDIA GPU Detected: %%g
    )
    :: Configure Ollama and Inference Engines to utilize GPU
    set OLLAMA_NUM_GPU=-1
    set OLLAMA_FLASH_ATTENTION=1
    set OCR_DEVICE=gpu
    echo       [OK] GPU Acceleration Active (OLLAMA_NUM_GPU=-1, FlashAttention Enabled)
) else (
    echo       [INFO] No dedicated NVIDIA GPU tool (nvidia-smi) found in PATH.
    echo       [INFO] Running in Adaptive Mode (OLLAMA_NUM_GPU=-1 auto-detects Metal/ROCm/Vulkan or falls back to CPU).
    set OLLAMA_NUM_GPU=-1
    set OCR_DEVICE=cpu
)
echo.

:: 1. Check & Start PostgreSQL Docker Container
echo [1/3] Checking PostgreSQL Docker container...
docker ps -q -f name=expense-reimbursement-postgres > nul 2>&1
for /f "tokens=*" %%i in ('docker ps -q -f name=expense-reimbursement-postgres 2^>nul') do set DOCKER_RUNNING=%%i

if "%DOCKER_RUNNING%"=="" (
    echo       Container is not running. Attempting to start...
    docker start expense-reimbursement-postgres > nul 2>&1
    if errorlevel 1 (
        echo       [!] Warning: Could not start Docker container 'expense-reimbursement-postgres'.
        echo           Please make sure Docker Desktop is running if you need database features.
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
    echo       Ollama is not running. Starting Ollama with GPU acceleration...
    where ollama >nul 2>&1
    if errorlevel 1 (
        if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
            set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
        )
    )
    start "" ollama serve
    timeout /t 4 /nobreak > nul
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
echo   Interactive UI Studio: http://localhost:8000/
echo   Swagger API Docs:      http://localhost:8000/docs
echo   ReDoc Documentation:   http://localhost:8000/redoc
echo   Ollama LLM Host at:    http://localhost:11434 (GPU: %OLLAMA_NUM_GPU%)
echo   Press Ctrl + C in this window to stop the server.
echo ============================================================
echo.

:: Automatically open browser to Interactive UI Studio
start "" cmd /c "timeout /t 2 /nobreak > nul && start http://localhost:8000/"

:: Launch FastAPI Web Server
.\.venv\Scripts\python.exe -m src.app

pause

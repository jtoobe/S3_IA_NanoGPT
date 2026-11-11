@echo off
title NanoGPT ESP32 Build Pipeline

echo ==========================================
echo NanoGPT ESP32 - Build Pipeline
echo ==========================================
echo.

REM ==========================================
REM Verificar Python
REM ==========================================

python --version >nul 2>&1

if errorlevel 1 (
    echo ERROR: Python no esta instalado.
    pause
    exit /b 1
)

echo Python OK
echo.

REM ==========================================
REM Crear entorno virtual
REM ==========================================

if not exist venv (

    echo Creando entorno virtual...

    python -m venv venv

    if errorlevel 1 (
        echo ERROR creando entorno virtual
        pause
        exit /b 1
    )

)

echo Entorno virtual OK

REM ==========================================
REM Activar entorno virtual
REM ==========================================

call venv\Scripts\activate.bat

if errorlevel 1 (
    echo ERROR activando entorno virtual
    pause
    exit /b 1
)

echo VENV OK
echo.

REM ==========================================
REM Actualizar pip
REM ==========================================

echo Actualizando pip...

python -m pip install --upgrade pip

echo.

REM ==========================================
REM Verificar requirements.txt
REM ==========================================

if not exist requirements.txt (

    echo ERROR: requirements.txt no encontrado
    pause
    exit /b 1

)

echo Instalando dependencias...
echo.

pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR instalando dependencias
    pause
    exit /b 1
)

echo.
echo Dependencias instaladas
echo.

REM ==========================================
REM Verificacion librerias
REM ==========================================

python -c "import torch, numpy, tqdm"

if errorlevel 1 (
    echo ERROR verificando dependencias
    pause
    exit /b 1
)

echo Torch OK
echo NumPy OK
echo tqdm OK
echo.

REM ==========================================
REM Verificar scripts
REM ==========================================

if not exist tokenizer.py (
    echo ERROR: tokenizer.py no encontrado
    pause
    exit /b 1
)

if not exist build_tokens.py (
    echo ERROR: build_tokens.py no encontrado
    pause
    exit /b 1
)

if not exist train.py (
    echo ERROR: train.py no encontrado
    pause
    exit /b 1
)

if not exist quantize.py (
    echo ERROR: quantize.py no encontrado
    pause
    exit /b 1
)

if not exist export_model.py (
    echo ERROR: export_model.py no encontrado
    pause
    exit /b 1
)

if not exist export_vocab.py (
    echo ERROR: export_vocab.py no encontrado
    pause
    exit /b 1
)

echo Scripts OK
echo.

REM ==========================================
REM DATASET
REM ==========================================

if not exist dataset.txt (
    echo ERROR: dataset no encontrado
    pause
    exit /b 1
)

echo Dataset OK
echo.

REM ==========================================
REM PIPELINE
REM ==========================================

echo [1/6] tokenizer.py
python tokenizer.py
if errorlevel 1 goto error

echo.
echo [2/6] build_tokens.py
python build_tokens.py
if errorlevel 1 goto error

echo.
echo [3/6] train.py
python train.py
if errorlevel 1 goto error

echo.
echo [4/6] quantize.py
python quantize.py
if errorlevel 1 goto error

echo.
echo [5/6] export_model.py
python export_model.py
if errorlevel 1 goto error

echo.
echo [6/6] export_vocab.py
python export_vocab.py
if errorlevel 1 goto error

echo.
echo ==========================================
echo BUILD FINALIZADO CORRECTAMENTE
echo ==========================================
echo.

echo Archivos generados:
echo.
echo   vocab.json
echo   train_data.pt
echo   nanogpt_words.pth
echo   nanogpt_words_int8.pth
echo   model.h
echo   vocab.h
echo.

pause
exit /b 0

:error

echo.
echo ==========================================
echo ERROR DURANTE EL BUILD
echo ==========================================
echo.

pause
exit /b 1

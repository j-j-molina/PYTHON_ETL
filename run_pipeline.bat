@echo off
REM ===================================================
REM run_pipeline.bat
REM Ejecuta el pipeline MLOps completo en orden.
REM Prerequisito: haber corrido set_up.bat al menos una vez.
REM ===================================================

setlocal EnableDelayedExpansion

REM ── Detectar venv ──────────────────────────────────
set "VENV_PYTHON="
if exist "cdp_credito-venv\Scripts\python.exe" (
    set "VENV_PYTHON=cdp_credito-venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set "VENV_PYTHON=python"
    ) else (
        echo [ERROR] No se encontro Python. Ejecuta set_up.bat primero.
        exit /b 1
    )
)

echo.
echo =====================================================
echo  CDP Credito — Pipeline MLOps
echo  Python: %VENV_PYTHON%
echo =====================================================
echo.

REM ── Directorio de scripts ──────────────────────────
set "SRC=mlops_pipeline\src"

REM ── Helper: ejecutar paso y abortar si falla ───────
REM   Uso: call :run_step "Nombre" script.py
goto :main

:run_step
    set "STEP_NAME=%~1"
    set "STEP_SCRIPT=%~2"
    echo [>>] %STEP_NAME%
    "%VENV_PYTHON%" "%SRC%\%STEP_SCRIPT%"
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo [ERROR] Fallo en: %STEP_NAME%
        echo         Revisa los logs antes de continuar.
        exit /b 1
    )
    echo [OK] %STEP_NAME%
    echo.
    exit /b 0

:main

call :run_step "1/6  Feature Engineering"  ft_engineering.py
if %ERRORLEVEL% NEQ 0 exit /b 1

call :run_step "2/6  Modelo Heuristico"     heuristic_model.py
if %ERRORLEVEL% NEQ 0 exit /b 1

call :run_step "3/6  Entrenamiento"         model_training.py
if %ERRORLEVEL% NEQ 0 exit /b 1

call :run_step "4/6  Evaluacion"            model_evaluation.py
if %ERRORLEVEL% NEQ 0 exit /b 1

call :run_step "5/6  Monitoreo"             model_monitoring.py
if %ERRORLEVEL% NEQ 0 exit /b 1

call :run_step "6/6  Deploy (artefactos)"   model_deploy.py
if %ERRORLEVEL% NEQ 0 exit /b 1

echo =====================================================
echo  Pipeline completado exitosamente.
echo  Artefactos: mlops_pipeline\artifacts\
echo  Reportes:   mlops_pipeline\reports\
echo =====================================================
echo.
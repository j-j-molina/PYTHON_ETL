@echo off
REM ===================================
REM set_up.bat
REM Purpose: Crear venv e instalar requirements para el proyecto CDP III
REM ===================================

echo.
echo === Python Virtual Environment Setup ===
echo.

setlocal EnableDelayedExpansion

REM 1) Encontrar config.json (tu caso: dentro de "source" = src)
set "CONFIG_PATH="

if exist "mlops_pipeline\src\config.json" (
    set "CONFIG_PATH=mlops_pipeline\src\config.json"
) else if exist "src\config.json" (
    set "CONFIG_PATH=src\config.json"
) else (
    echo Error: No se encontró config.json en "mlops_pipeline\src\" ni en "src\".
    goto :eof
)

echo Usando config: %CONFIG_PATH%

REM 2) Leer project_code desde config.json
for /f "usebackq tokens=2 delims=:" %%A in (`findstr "project_code" "%CONFIG_PATH%"`) do (
    set "line=%%A"
    set "line=!line:,=!"
    set "line=!line:"=!"
    set "project_code=!line:~1!"
)

echo Project code encontrado: [%project_code%]
echo.

REM 3) Crear venv
echo Creando nuevo ambiente virtual: %project_code%-venv
py -m venv %project_code%-venv

echo Activando virtual environment...
call %project_code%-venv\Scripts\activate

if %ERRORLEVEL% NEQ 0 (
    echo Error activando el ambiente virtual.
    goto :eof
)

echo.
echo Ambiente virtual creado con exito!
where python
echo Directorio actual: %cd%
echo.

REM 4) Instalar requirements
echo === Instalando requisitos ===
if exist requirements.txt (
    echo requirements.txt encontrado, instalando librerias...
    "%project_code%-venv\Scripts\python.exe" -m pip install --no-cache-dir -r requirements.txt
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo Todas las librerías instaladas correctamente.
        echo.
        echo === Registrando ambiente virtual con Jupyter ===
        python -m ipykernel install --user --name=%project_code%-venv --display-name="%project_code%-venv Python"
    ) else (
        echo Error instalando las librerías desde requirements.txt.
    )
) else (
    echo Advertencia: requirements.txt no fue encontrado en el directorio actual.
)

echo.

#!/usr/bin/env bash
# ===================================================
# set_up.sh
# Purpose: Crear venv e instalar requirements para el proyecto CDP III
# Uso: bash set_up.sh   (desde la raíz del repo mlops_pipeline/)
# ===================================================

set -euo pipefail

echo ""
echo "=== Python Virtual Environment Setup ==="
echo ""

# ── 1. Encontrar config.json ────────────────────────
CONFIG_PATH=""
if   [[ -f "mlops_pipeline/src/config.json" ]]; then
    CONFIG_PATH="mlops_pipeline/src/config.json"
elif [[ -f "src/config.json" ]]; then
    CONFIG_PATH="src/config.json"
else
    echo "Error: No se encontró config.json en 'mlops_pipeline/src/' ni en 'src/'."
    exit 1
fi

echo "Usando config: $CONFIG_PATH"

# ── 2. Leer project_code desde config.json ──────────
PROJECT_CODE=$(python3 -c "
import json, sys
with open('$CONFIG_PATH') as f:
    print(json.load(f)['project_code'])
")
echo "Project code encontrado: [$PROJECT_CODE]"
echo ""

# ── 3. Crear venv ───────────────────────────────────
VENV_NAME="${PROJECT_CODE}-venv"
echo "Creando nuevo ambiente virtual: $VENV_NAME"
python3 -m venv "$VENV_NAME"

echo "Activando virtual environment..."
source "$VENV_NAME/bin/activate"

echo ""
echo "Ambiente virtual creado con éxito!"
which python
echo "Directorio actual: $(pwd)"
echo ""

# ── 4. Instalar requirements ────────────────────────
echo "=== Instalando requisitos ==="
if [[ -f "requirements.txt" ]]; then
    echo "requirements.txt encontrado, instalando librerías..."
    pip install --no-cache-dir -r requirements.txt

    echo ""
    echo "Todas las librerías instaladas correctamente."
    echo ""
    echo "=== Registrando ambiente virtual con Jupyter ==="
    python -m ipykernel install --user \
        --name="$VENV_NAME" \
        --display-name="$VENV_NAME Python"
else
    echo "Advertencia: requirements.txt no fue encontrado en el directorio actual."
fi

echo ""
echo "Setup completo. Para activar el entorno en una nueva terminal:"
echo "  source $VENV_NAME/bin/activate"
echo ""
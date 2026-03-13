#!/usr/bin/env bash
# ===================================================
# run_pipeline.sh
# Ejecuta el pipeline MLOps completo en orden.
# Prerequisito: haber corrido set_up.sh al menos una vez.
# Uso: bash run_pipeline.sh   (desde la raíz del repo mlops_pipeline/)
# ===================================================

set -euo pipefail

SRC="src"

# ── Detectar Python ─────────────────────────────────
if [[ -f "cdp_credito-venv/bin/python" ]]; then
    PYTHON="cdp_credito-venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "[ERROR] No se encontró Python. Ejecuta set_up.sh primero."
    exit 1
fi

echo ""
echo "====================================================="
echo " CDP Credito — Pipeline MLOps"
echo " Python: $PYTHON"
echo "====================================================="
echo ""

# ── Helper ──────────────────────────────────────────
run_step() {
    local label="$1"
    local script="$2"
    echo "[>>] $label"
    "$PYTHON" "$SRC/$script"
    echo "[OK] $label"
    echo ""
}

run_step "1/6  Feature Engineering"  ft_engineering.py
run_step "2/6  Modelo Heurístico"    heuristic_model.py
run_step "3/6  Entrenamiento"        model_training.py
run_step "4/6  Evaluación"           model_evaluation.py
run_step "5/6  Monitoreo"            model_monitoring.py
run_step "6/6  Deploy (artefactos)"  model_deploy.py

echo "====================================================="
echo " Pipeline completado exitosamente."
echo " Artefactos : mlops_pipeline/artifacts/"
echo " Reportes   : mlops_pipeline/reports/"
echo "====================================================="
echo ""
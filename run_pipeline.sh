#!/usr/bin/env bash
set -euo pipefail

SRC="src"
USE_CASE="scoring_mora"

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
echo " Python: $PYTHON  |  use_case: $USE_CASE"
echo "====================================================="
echo ""

# ── Helper ──────────────────────────────────────────
run_step() {
    local label="$1"
    local script="$2"
    shift 2
    echo "[>>] $label"
    "$PYTHON" "$SRC/$script" "$@"
    echo "[OK] $label"
    echo ""
}

run_step "1/6  Feature Engineering"  ft_engineering.py  --use-case "$USE_CASE"
run_step "2/6  Modelo Heurístico"    heuristic_model.py --use-case "$USE_CASE"
run_step "3/6  Entrenamiento"        model_training.py  --use-case "$USE_CASE"
run_step "4/6  Evaluación"           model_evaluation.py --use-case "$USE_CASE"
run_step "5/6  Monitoreo"            model_monitoring.py --use-case "$USE_CASE"
run_step "6/6  Deploy (artefactos)"  model_deploy.py    --use-case "$USE_CASE"

echo "====================================================="
echo " Pipeline completado exitosamente."
echo " Artefactos : artifacts/$USE_CASE/"
echo " Reportes   : reports/$USE_CASE/"
echo "====================================================="
echo ""
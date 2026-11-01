#!/usr/bin/env bash
# SessionStart: asegura dependencias + Graphify en contenedores efímeros (Claude Code web).
# Idempotente y silencioso. Reconstruye el grafo AST si falta (gratis, sin LLM).
set -u
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0
if ! command -v graphify >/dev/null 2>&1 || ! python3 -c "import yfinance, pyarrow" >/dev/null 2>&1; then
  pip install -q -r requirements.txt >/dev/null 2>&1 || true
fi
if command -v graphify >/dev/null 2>&1 && [ ! -f graphify-out/graph.json ]; then
  graphify update . >/dev/null 2>&1 || true
fi
# Contexto mínimo para el agente (stdout se inyecta en la sesión)
echo "Quant-Predictor: usa 'graphify query \"<pregunta>\"' antes de leer código. Estado del proyecto: docs/PROJECT_STATE.md"
exit 0

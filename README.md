# Quant-Predictor

Sistema cuantitativo para **predecir rentabilidades de ETFs y construir una estrategia de inversión**
para una cartera ficticia, al estilo de los algoritmos de los fondos quant.

## Fases
| # | Fase | Estado |
|---|---|---|
| 1 | Ingesta: universo amplio de ETFs (todas las clases de activo) + macro + datos alternativos | ✅ |
| 2 | Investigación: papers, foros y estudios sobre las mejores técnicas | ✅ |
| 3 | Estudio propio: split temporal train/test y comparación de estrategias | ✅ |
| 4 | Análisis de fallos e iteración hasta que no haya mejora | ⏳ |
| 5 | Exportar modelo descargable + informe matemático | ⏳ |

## Objetivo actual
Cartera simulada de ETFs que intenta **batir al S&P 500**, entrenada con datos diarios y valorada
**minuto a minuto** frente al SPY (`src/quantpred/live`). Resumen: [`reports/RELATIVE_SUMMARY.md`](reports/RELATIVE_SUMMARY.md).

## Dataset (Fase 1)
- **~5.800 ETFs/ETNs cotizados en EEUU** (acciones EEUU/internacionales, sectores, factores, bonos,
  materias primas, divisas, REITs, cripto, volatilidad, apalancados, alternativos), histórico diario completo.
- **~3.600 ETFs/ETCs/ETNs europeos UCITS** (lista oficial Xetra, deduplicados por ISIN).
- **~75 series de mercado de historia larga** (índices mundiales, futuros, divisas, tipos, volatilidad).
- **~74 series macro FRED** y **~260 series alternativas** (factores Fama-French y AQR, predictores
  Goyal-Welch, CAPE, estructura temporal de volatilidad Cboe, posicionamiento CFTC, sentimiento,
  riesgo geopolítico, prima por plazo…).

Detalle, fuentes y justificación académica: [`docs/DATA.md`](docs/DATA.md).
Estado y decisiones: [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md).

## Uso
```bash
pip install -r requirements.txt
scripts/run_all.sh                            # EEUU + Europa UCITS + macro Yahoo (reanudable)
python scripts/run_ingestion.py --region eu   # solo una región
PYTHONPATH=src python -m quantpred.ingest.alt_data   # datos alternativos
python -m pytest -q tests
```
```python
import sys; sys.path.insert(0, "src")
from quantpred.ingest.store import load_prices, load_panel, load_metadata
px = load_panel("adj_close", asset_classes=["fixed_income", "commodity"])  # fecha x ticker
```

## Ahorro de tokens para agentes (Claude Code)
- **Graphify** (`.claude/skills/graphify`, hooks en `.claude/settings.json`): grafo de conocimiento del
  código en `graphify-out/`; el agente consulta `graphify query` en vez de releer ficheros.
- **Caveman** (`.claude/skills/caveman`): respuestas cortas y densas por defecto.
- Hook `SessionStart` reinstala dependencias y reconstruye el grafo en contenedores efímeros.

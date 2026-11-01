# Quant-Predictor — contexto para agentes

Programa quant: predecir precios de ETFs y construir estrategia para cartera ficticia.
Idioma del proyecto: español. Estado actual y decisiones: `docs/PROJECT_STATE.md` (leer primero, es corto).

## Ahorro de tokens (obligatorio)
1. **Caveman activo por defecto** (`.claude/skills/caveman/SKILL.md`): chat corto y denso. Excepción: informes/docs.
2. **Graphify primero**: nunca releer el repo entero. Ver sección graphify abajo.
3. Datos: nunca volcar parquet/CSV en el chat. Usar `head`, `describe`, conteos.
4. Al cerrar una tarea: actualizar `docs/PROJECT_STATE.md` (estado + decisiones, máx. ~80 líneas) y `graphify update .`.

## Mapa rápido
- `src/quantpred/config.py` — rutas, listas de series (Yahoo macro, FRED).
- `src/quantpred/ingest/` — Fase 1: `universe` (NASDAQ Trader), `yahoo` (cliente HTTP), `prices`,
  `metadata` (clasificación por clase de activo), `quality`, `store` (parquet + loaders), `macro`, `alt_data`.
- `scripts/run_ingestion.py` — pipeline Fase 1 (reanudable).
- `data/processed/` — dataset versionado. `data/raw/` — no versionado.
- `docs/DATA.md` — diccionario de datos y fuentes con referencias académicas.

## Reglas de datos (anti look-ahead / sesgos)
- `date` = fecha de referencia del dato. Desplazar por `pub_lag_days` (catálogos en `data/meta/`) antes de usar como feature.
- Universo con sesgo de supervivencia (solo ETFs vivos). En backtests usar solo ETFs con datos en t.
- Split train/test siempre temporal (walk-forward, purging + embargo). Nunca aleatorio.

## Yahoo
- Cliente educado (`ingest/yahoo.py`): pausa, 2 hilos, respeta 429/Retry-After. No intentar evadir límites.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.

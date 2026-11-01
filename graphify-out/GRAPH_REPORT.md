# Graph Report - Quant-Predictor  (2026-11-01)

## Corpus Check
- 32 files · ~17,563 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 2 file(s) not represented in the graph (top: (none) 2)

## Summary
- 316 nodes · 735 edges · 15 communities (13 shown, 2 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 17 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c9110b0c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- pandas
- prices.py
- metadata.py
- pandas
- store.py
- numpy
- Fase 1 — Dataset: fuentes, diccionario y justificación
- run_study.py
- Quant-Predictor — contexto para agentes
- run_all.sh
- backtest.py
- Fase 2 — Investigación: qué funciona para predecir y asignar ETFs
- Fase 3 — Resultados del estudio comparativo
- Fases 3-4 — Resumen de resultados y lecciones

## God Nodes (most connected - your core abstractions)
1. `_get()` - 16 edges
2. `_long()` - 15 edges
3. `main()` - 13 edges
4. `main()` - 10 edges
5. `main()` - 10 edges
6. `chart()` - 10 edges
7. `build_panel()` - 10 edges
8. `risk_parity()` - 10 edges
9. `cftc()` - 9 edges
10. `load_prices()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `Fase 1 — Dataset: fuentes, diccionario y justificación` --references--> `fetch_fred()`  [INFERRED]
  docs/DATA.md → src/quantpred/ingest/macro.py
- `Decisiones clave` --references--> `classify_by_name()`  [INFERRED]
  docs/PROJECT_STATE.md → src/quantpred/ingest/metadata.py
- `main()` --calls--> `build_panel()`  [EXTRACTED]
  scripts/run_phase4.py → src/quantpred/research/features.py
- `main()` --calls--> `build_panel()`  [EXTRACTED]
  scripts/run_phase4b.py → src/quantpred/research/features.py
- `main()` --calls--> `build_panel()`  [EXTRACTED]
  scripts/run_study.py → src/quantpred/research/features.py

## Import Cycles
- None detected.

## Communities (15 total, 2 thin omitted)

### Community 0 - "pandas"
Cohesion: 0.19
Nodes (19): aaii(), acm_term_premium(), ads_index(), aqr(), baker_wurgler(), cboe(), cftc(), _cftc_files() (+11 more)

### Community 1 - "prices.py"
Cohesion: 0.14
Nodes (10): fetch_fred(), fetch_yahoo_macro(), _fred_one(), download_all(), download_batch(), download_symbols(), load_chart_meta(), load_raw() (+2 more)

### Community 2 - "metadata.py"
Cohesion: 0.07
Nodes (23): Decisiones clave, build_metadata(), classify(), classify_by_name(), fetch_summaries(), one(), _name_flags(), _parse_summary() (+15 more)

### Community 3 - "pandas"
Cohesion: 0.15
Nodes (16): build_panel(), etf_features(), load_daily(), macro_features(), _month_ends(), risk_free_monthly(), _series_asof(), _alt_series() (+8 more)

### Community 4 - "store.py"
Cohesion: 0.22
Nodes (6): load_macro(), load_metadata(), load_panel(), load_prices(), _write(), write_prices()

### Community 5 - "numpy"
Cohesion: 0.11
Nodes (16): clean_prices(), add_targets(), rank_ensemble(), sel_two_level(), f(), smooth(), allocate(), _cluster_var() (+8 more)

### Community 6 - "Fase 1 — Dataset: fuentes, diccionario y justificación"
Cohesion: 0.09
Nodes (19): 1. Universo de ETFs (activos invertibles), 1b. Universo europeo UCITS — `data/processed/etf_prices_eu/`, 2. Series de mercado de historia larga (Yahoo) — `data/processed/macro_yahoo.parquet`, 3. Macro FRED — `data/processed/macro_fred.parquet` (~74 series), 4. Datos alternativos / features — `data/processed/alt/*.parquet`, 5. Reglas para usar el dataset sin trampas, Fase 1 — Dataset: fuentes, diccionario y justificación, Estado del proyecto (leer primero; mantener < 80 líneas) (+11 more)

### Community 7 - "run_study.py"
Cohesion: 0.08
Nodes (17): main(), build_pred(), evaluate(), get_pred(), main(), calmonth(), main(), run_overlay() (+9 more)

### Community 8 - "Quant-Predictor — contexto para agentes"
Cohesion: 0.29
Nodes (6): Ahorro de tokens (obligatorio), graphify, Mapa rápido, Quant-Predictor — contexto para agentes, Reglas de datos (anti look-ahead / sesgos), Yahoo

### Community 11 - "backtest.py"
Cohesion: 0.11
Nodes (14): _cap_classes(), feature_cols(), ic_series(), Result, run(), sel_core_tsmom(), sel_dual_momentum(), sel_fixed() (+6 more)

### Community 12 - "Fase 2 — Investigación: qué funciona para predecir y asignar ETFs"
Cohesion: 0.33
Nodes (5): 1. Lecciones generales (lo que la evidencia dice de verdad), 2. Técnicas candidatas, 3. Diseño experimental para la Fase 3, Fase 2 — Investigación: qué funciona para predecir y asignar ETFs, Fuentes

### Community 13 - "Fase 3 — Resultados del estudio comparativo"
Cohesion: 0.50
Nodes (3): Desarrollo, Fase 3 — Resultados del estudio comparativo, TEST (fuera de muestra)

### Community 14 - "Fases 3-4 — Resumen de resultados y lecciones"
Cohesion: 0.40
Nodes (4): Decisión para la Fase 5, Fases 3-4 — Resumen de resultados y lecciones, Qué NO funcionó (y por qué), Qué SÍ es robusto

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `fetch_fred()` connect `prices.py` to `pandas`, `Fase 1 — Dataset: fuentes, diccionario y justificación`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Should `prices.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13768115942028986 - nodes in this community are weakly interconnected._
- **Why does `Fase 1 — Dataset: fuentes, diccionario y justificación` connect `Fase 1 — Dataset: fuentes, diccionario y justificación` to `prices.py`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Should `metadata.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07123034227567067 - nodes in this community are weakly interconnected._
- **Why does `classify_by_name()` connect `metadata.py` to `pandas`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Should `pandas` be split into smaller, more focused modules?**
  _Cohesion score 0.14616755793226383 - nodes in this community are weakly interconnected._
- **Should `numpy` be split into smaller, more focused modules?**
  _Cohesion score 0.11428571428571428 - nodes in this community are weakly interconnected._
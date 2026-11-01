# Graph Report - Quant-Predictor  (2026-11-01)

## Corpus Check
- 29 files · ~15,280 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 2 file(s) not represented in the graph (top: (none) 2)

## Summary
- 282 nodes · 641 edges · 14 communities (12 shown, 2 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 16 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b9e87aa2`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- pandas
- prices.py
- metadata.py
- yahoo.py
- pandas
- numpy
- Fase 1 — Dataset: fuentes, diccionario y justificación
- run_study.py
- Quant-Predictor — contexto para agentes
- run_all.sh
- backtest.py
- Fase 2 — Investigación: qué funciona para predecir y asignar ETFs
- Fase 3 — Resultados del estudio comparativo

## God Nodes (most connected - your core abstractions)
1. `_get()` - 16 edges
2. `_long()` - 15 edges
3. `main()` - 10 edges
4. `main()` - 10 edges
5. `chart()` - 10 edges
6. `risk_parity()` - 10 edges
7. `cftc()` - 9 edges
8. `load_prices()` - 9 edges
9. `_get_json()` - 9 edges
10. `run()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `Decisiones clave` --references--> `classify_by_name()`  [INFERRED]
  docs/PROJECT_STATE.md → src/quantpred/ingest/metadata.py
- `Fase 1 — Dataset: fuentes, diccionario y justificación` --references--> `fetch_fred()`  [INFERRED]
  docs/DATA.md → src/quantpred/ingest/macro.py
- `main()` --calls--> `build_panel()`  [EXTRACTED]
  scripts/run_phase4.py → src/quantpred/research/features.py
- `main()` --calls--> `build_panel()`  [EXTRACTED]
  scripts/run_study.py → src/quantpred/research/features.py
- `test_classify_by_category()` --calls--> `classify()`  [EXTRACTED]
  tests/test_ingest.py → src/quantpred/ingest/metadata.py

## Import Cycles
- None detected.

## Communities (14 total, 2 thin omitted)

### Community 0 - "pandas"
Cohesion: 0.19
Nodes (19): aaii(), acm_term_premium(), ads_index(), aqr(), baker_wurgler(), cboe(), cftc(), _cftc_files() (+11 more)

### Community 1 - "prices.py"
Cohesion: 0.15
Nodes (9): fetch_fred(), fetch_yahoo_macro(), _fred_one(), download_all(), download_batch(), download_symbols(), load_chart_meta(), load_raw() (+1 more)

### Community 2 - "metadata.py"
Cohesion: 0.12
Nodes (12): build_metadata(), classify(), classify_by_name(), _name_flags(), _parse_summary(), _raw(), _xetra_override(), fetch_universe() (+4 more)

### Community 3 - "yahoo.py"
Cohesion: 0.13
Nodes (11): fetch_summaries(), one(), fetch_universe_eu(), resolve_isin(), resolve_missing(), chart(), _get_crumb(), _get_json() (+3 more)

### Community 4 - "pandas"
Cohesion: 0.15
Nodes (13): load_macro(), load_metadata(), load_panel(), load_prices(), _write(), write_prices(), build_panel(), etf_features() (+5 more)

### Community 5 - "numpy"
Cohesion: 0.11
Nodes (16): clean_prices(), add_targets(), rank_ensemble(), sel_two_level(), f(), smooth(), allocate(), _cluster_var() (+8 more)

### Community 6 - "Fase 1 — Dataset: fuentes, diccionario y justificación"
Cohesion: 0.09
Nodes (19): 1. Universo de ETFs (activos invertibles), 1b. Universo europeo UCITS — `data/processed/etf_prices_eu/`, 2. Series de mercado de historia larga (Yahoo) — `data/processed/macro_yahoo.parquet`, 3. Macro FRED — `data/processed/macro_fred.parquet` (~74 series), 4. Datos alternativos / features — `data/processed/alt/*.parquet`, 5. Reglas para usar el dataset sin trampas, Fase 1 — Dataset: fuentes, diccionario y justificación, Decisiones clave (+11 more)

### Community 7 - "run_study.py"
Cohesion: 0.10
Nodes (13): main(), build_pred(), evaluate(), get_pred(), main(), main(), models(), deflated_sharpe() (+5 more)

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

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `fetch_fred()` connect `prices.py` to `pandas`, `Fase 1 — Dataset: fuentes, diccionario y justificación`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Should `metadata.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12333333333333334 - nodes in this community are weakly interconnected._
- **Why does `Fase 1 — Dataset: fuentes, diccionario y justificación` connect `Fase 1 — Dataset: fuentes, diccionario y justificación` to `prices.py`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Should `yahoo.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12615384615384614 - nodes in this community are weakly interconnected._
- **Why does `classify_by_name()` connect `metadata.py` to `Fase 1 — Dataset: fuentes, diccionario y justificación`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Should `numpy` be split into smaller, more focused modules?**
  _Cohesion score 0.11428571428571428 - nodes in this community are weakly interconnected._
- **Should `Fase 1 — Dataset: fuentes, diccionario y justificación` be split into smaller, more focused modules?**
  _Cohesion score 0.09090909090909091 - nodes in this community are weakly interconnected._
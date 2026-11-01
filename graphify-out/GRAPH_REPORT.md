# Graph Report - Quant-Predictor  (2026-11-01)

## Corpus Check
- 43 files · ~22,906 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 2 file(s) not represented in the graph (top: (none) 2)

## Summary
- 425 nodes · 993 edges · 19 communities (16 shown, 3 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 43 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `89f0b676`
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
- numpy
- SimulatedPortfolio
- feeds.py

## God Nodes (most connected - your core abstractions)
1. `Policy` - 19 edges
2. `_get()` - 16 edges
3. `main()` - 15 edges
4. `_long()` - 15 edges
5. `main()` - 13 edges
6. `_get_json()` - 11 edges
7. `main()` - 10 edges
8. `main()` - 10 edges
9. `chart()` - 10 edges
10. `SimulatedPortfolio` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Fase 1 — Dataset: fuentes, diccionario y justificación` --references--> `fetch_fred()`  [INFERRED]
  docs/DATA.md → src/quantpred/ingest/macro.py
- `Decisiones clave` --references--> `classify_by_name()`  [INFERRED]
  docs/PROJECT_STATE.md → src/quantpred/ingest/metadata.py
- `live()` --uses--> `YahooMinuteFeed`  [INFERRED]
  scripts/live_demo.py → src/quantpred/live/feeds.py
- `main()` --calls--> `load_metadata()`  [INFERRED]
  scripts/run_relative.py → src/quantpred/ingest/store.py
- `crypto_tickers()` --calls--> `load_metadata()`  [EXTRACTED]
  scripts/live_demo.py → src/quantpred/ingest/store.py

## Import Cycles
- None detected.

## Communities (19 total, 3 thin omitted)

### Community 0 - "pandas"
Cohesion: 0.10
Nodes (28): aaii(), acm_term_premium(), ads_index(), aqr(), baker_wurgler(), cboe(), cftc(), _cftc_files() (+20 more)

### Community 1 - "prices.py"
Cohesion: 0.20
Nodes (7): download_all(), download_batch(), download_symbols(), load_chart_meta(), load_raw(), _one(), chart()

### Community 2 - "metadata.py"
Cohesion: 0.11
Nodes (17): Decisiones clave, build_metadata(), classify(), classify_by_name(), fetch_summaries(), one(), _name_flags(), _parse_summary() (+9 more)

### Community 3 - "pandas"
Cohesion: 0.21
Nodes (9): _alt_series(), build_regime_frame(), expanding_quantile(), har_features(), month_ends(), spy_daily(), state_features(), targets() (+1 more)

### Community 4 - "store.py"
Cohesion: 0.13
Nodes (13): load_macro(), load_metadata(), load_panel(), load_prices(), _write(), write_prices(), build_panel(), etf_features() (+5 more)

### Community 5 - "numpy"
Cohesion: 0.14
Nodes (14): add_targets(), rank_ensemble(), sel_two_level(), f(), smooth(), allocate(), _cluster_var(), equal_weight() (+6 more)

### Community 6 - "Fase 1 — Dataset: fuentes, diccionario y justificación"
Cohesion: 0.06
Nodes (28): 1. Universo de ETFs (activos invertibles), 1b. Universo europeo UCITS — `data/processed/etf_prices_eu/`, 2. Series de mercado de historia larga (Yahoo) — `data/processed/macro_yahoo.parquet`, 3. Macro FRED — `data/processed/macro_fred.parquet` (~74 series), 4. Datos alternativos / features — `data/processed/alt/*.parquet`, 5. Reglas para usar el dataset sin trampas, Fase 1 — Dataset: fuentes, diccionario y justificación, Estado del proyecto (leer primero; mantener < 80 líneas) (+20 more)

### Community 7 - "run_study.py"
Cohesion: 0.07
Nodes (18): plot(), main(), build_pred(), evaluate(), get_pred(), main(), calmonth(), main() (+10 more)

### Community 8 - "Quant-Predictor — contexto para agentes"
Cohesion: 0.29
Nodes (6): Ahorro de tokens (obligatorio), graphify, Mapa rápido, Quant-Predictor — contexto para agentes, Reglas de datos (anti look-ahead / sesgos), Yahoo

### Community 11 - "backtest.py"
Cohesion: 0.12
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

### Community 15 - "numpy"
Cohesion: 0.07
Nodes (23): baseline_weights(), hold_every(), idx_of(), main(), metrics(), clean_prices(), build_daily(), cross_section() (+15 more)

### Community 16 - "SimulatedPortfolio"
Cohesion: 0.15
Nodes (6): crypto_tickers(), live(), replay(), LiveEngine, SimulatedPortfolio, test_simulated_portfolio_vs_benchmark()

### Community 17 - "feeds.py"
Cohesion: 0.15
Nodes (3): PriceFeed, ReplayFeed, YahooMinuteFeed

## Knowledge Gaps
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `fetch_fred()` connect `pandas` to `Fase 1 — Dataset: fuentes, diccionario y justificación`?**
  _High betweenness centrality (0.101) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `Policy` (e.g. with `main()` and `LiveEngine`) actually correct?**
  _`Policy` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Should `pandas` be split into smaller, more focused modules?**
  _Cohesion score 0.10064935064935066 - nodes in this community are weakly interconnected._
- **Why does `Fase 1 — Dataset: fuentes, diccionario y justificación` connect `Fase 1 — Dataset: fuentes, diccionario y justificación` to `pandas`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Should `metadata.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11494252873563218 - nodes in this community are weakly interconnected._
- **Why does `classify_by_name()` connect `metadata.py` to `pandas`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Should `store.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1330049261083744 - nodes in this community are weakly interconnected._
# Graph Report - Quant-Predictor  (2026-11-01)

## Corpus Check
- 19 files · ~8,787 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 2 file(s) not represented in the graph (top: (none) 2)

## Summary
- 170 nodes · 378 edges · 11 communities (9 shown, 2 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5bb782d9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- pandas
- prices.py
- metadata.py
- yahoo.py
- store.py
- test_ingest.py
- Fase 1 — Dataset: fuentes, diccionario y justificación
- run_ingestion.py
- Quant-Predictor — contexto para agentes
- run_all.sh

## God Nodes (most connected - your core abstractions)
1. `_get()` - 16 edges
2. `_long()` - 15 edges
3. `chart()` - 10 edges
4. `cftc()` - 9 edges
5. `_get_json()` - 9 edges
6. `ken_french()` - 8 edges
7. `aqr()` - 8 edges
8. `acm_term_premium()` - 8 edges
9. `goyal_welch()` - 7 edges
10. `shiller()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Decisiones clave` --references--> `classify_by_name()`  [INFERRED]
  docs/PROJECT_STATE.md → src/quantpred/ingest/metadata.py
- `Fase 1 — Dataset: fuentes, diccionario y justificación` --references--> `fetch_fred()`  [INFERRED]
  docs/DATA.md → src/quantpred/ingest/macro.py
- `test_classify_by_category()` --calls--> `classify()`  [EXTRACTED]
  tests/test_ingest.py → src/quantpred/ingest/metadata.py
- `test_classify_by_name()` --calls--> `classify()`  [EXTRACTED]
  tests/test_ingest.py → src/quantpred/ingest/metadata.py
- `fetch_summaries()` --uses--> `RateLimited`  [INFERRED]
  src/quantpred/ingest/metadata.py → src/quantpred/ingest/yahoo.py

## Import Cycles
- None detected.

## Communities (11 total, 2 thin omitted)

### Community 0 - "pandas"
Cohesion: 0.18
Nodes (19): aaii(), acm_term_premium(), ads_index(), aqr(), baker_wurgler(), cboe(), cftc(), _cftc_files() (+11 more)

### Community 1 - "prices.py"
Cohesion: 0.20
Nodes (7): download_all(), download_batch(), download_symbols(), load_chart_meta(), load_raw(), _one(), chart()

### Community 2 - "metadata.py"
Cohesion: 0.18
Nodes (11): build_metadata(), classify(), classify_by_name(), fetch_summaries(), one(), _name_flags(), _parse_summary(), _raw() (+3 more)

### Community 3 - "yahoo.py"
Cohesion: 0.18
Nodes (8): fetch_universe_eu(), resolve_isin(), resolve_missing(), _get_crumb(), _get_json(), quote_summary(), RateLimited, _session()

### Community 4 - "store.py"
Cohesion: 0.21
Nodes (6): load_macro(), load_metadata(), load_panel(), load_prices(), _write(), write_prices()

### Community 5 - "test_ingest.py"
Cohesion: 0.24
Nodes (5): clean_prices(), fetch_universe(), yahoo_symbol(), test_clean_prices_flags_and_bounds(), test_yahoo_symbol()

### Community 6 - "Fase 1 — Dataset: fuentes, diccionario y justificación"
Cohesion: 0.09
Nodes (19): 1. Universo de ETFs (activos invertibles), 1b. Universo europeo UCITS — `data/processed/etf_prices_eu/`, 2. Series de mercado de historia larga (Yahoo) — `data/processed/macro_yahoo.parquet`, 3. Macro FRED — `data/processed/macro_fred.parquet` (~74 series), 4. Datos alternativos / features — `data/processed/alt/*.parquet`, 5. Reglas para usar el dataset sin trampas, Fase 1 — Dataset: fuentes, diccionario y justificación, Decisiones clave (+11 more)

### Community 7 - "run_ingestion.py"
Cohesion: 0.14
Nodes (4): main(), fetch_fred(), fetch_yahoo_macro(), _fred_one()

### Community 8 - "Quant-Predictor — contexto para agentes"
Cohesion: 0.29
Nodes (6): Ahorro de tokens (obligatorio), graphify, Mapa rápido, Quant-Predictor — contexto para agentes, Reglas de datos (anti look-ahead / sesgos), Yahoo

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `fetch_fred()` connect `run_ingestion.py` to `test_ingest.py`, `Fase 1 — Dataset: fuentes, diccionario y justificación`?**
  _High betweenness centrality (0.152) - this node is a cross-community bridge._
- **Should `Fase 1 — Dataset: fuentes, diccionario y justificación` be split into smaller, more focused modules?**
  _Cohesion score 0.09090909090909091 - nodes in this community are weakly interconnected._
- **Why does `Fase 1 — Dataset: fuentes, diccionario y justificación` connect `Fase 1 — Dataset: fuentes, diccionario y justificación` to `run_ingestion.py`?**
  _High betweenness centrality (0.132) - this node is a cross-community bridge._
- **Should `run_ingestion.py` be split into smaller, more focused modules?**
  _Cohesion score 0.14210526315789473 - nodes in this community are weakly interconnected._
- **Why does `classify_by_name()` connect `metadata.py` to `Fase 1 — Dataset: fuentes, diccionario y justificación`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
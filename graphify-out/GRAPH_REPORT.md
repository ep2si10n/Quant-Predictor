# Graph Report - Quant-Predictor  (2026-11-01)

## Corpus Check
- 16 files · ~7,545 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 2 file(s) not represented in the graph (top: (none) 2)

## Summary
- 150 nodes · 342 edges · 10 communities (8 shown, 2 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

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

## God Nodes (most connected - your core abstractions)
1. `_get()` - 16 edges
2. `_long()` - 15 edges
3. `cftc()` - 9 edges
4. `chart()` - 9 edges
5. `ken_french()` - 8 edges
6. `aqr()` - 8 edges
7. `acm_term_premium()` - 8 edges
8. `goyal_welch()` - 7 edges
9. `shiller()` - 7 edges
10. `gpr()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `test_classify_by_category()` --calls--> `classify()`  [EXTRACTED]
  tests/test_ingest.py → src/quantpred/ingest/metadata.py
- `test_classify_by_name()` --calls--> `classify()`  [EXTRACTED]
  tests/test_ingest.py → src/quantpred/ingest/metadata.py
- `fetch_summaries()` --uses--> `RateLimited`  [INFERRED]
  src/quantpred/ingest/metadata.py → src/quantpred/ingest/yahoo.py
- `test_clean_prices_flags_and_bounds()` --calls--> `clean_prices()`  [EXTRACTED]
  tests/test_ingest.py → src/quantpred/ingest/quality.py
- `test_yahoo_symbol()` --calls--> `yahoo_symbol()`  [EXTRACTED]
  tests/test_ingest.py → src/quantpred/ingest/universe.py

## Import Cycles
- None detected.

## Communities (10 total, 2 thin omitted)

### Community 0 - "pandas"
Cohesion: 0.20
Nodes (19): aaii(), acm_term_premium(), ads_index(), aqr(), baker_wurgler(), cboe(), cftc(), _cftc_files() (+11 more)

### Community 1 - "prices.py"
Cohesion: 0.18
Nodes (9): fetch_fred(), fetch_yahoo_macro(), _fred_one(), download_all(), download_batch(), download_symbols(), load_chart_meta(), load_raw() (+1 more)

### Community 2 - "metadata.py"
Cohesion: 0.18
Nodes (10): build_metadata(), classify(), classify_by_name(), fetch_summaries(), one(), _name_flags(), _parse_summary(), _raw() (+2 more)

### Community 3 - "yahoo.py"
Cohesion: 0.20
Nodes (6): chart(), _get_crumb(), _get_json(), quote_summary(), RateLimited, _session()

### Community 4 - "store.py"
Cohesion: 0.21
Nodes (6): load_macro(), load_metadata(), load_panel(), load_prices(), _write(), write_prices()

### Community 5 - "test_ingest.py"
Cohesion: 0.21
Nodes (5): clean_prices(), fetch_universe(), yahoo_symbol(), test_clean_prices_flags_and_bounds(), test_yahoo_symbol()

### Community 6 - "Fase 1 — Dataset: fuentes, diccionario y justificación"
Cohesion: 0.15
Nodes (11): 1. Universo de ETFs (activos invertibles), 2. Series de mercado de historia larga (Yahoo) — `data/processed/macro_yahoo.parquet`, 3. Macro FRED — `data/processed/macro_fred.parquet` (~74 series), 4. Datos alternativos / features — `data/processed/alt/*.parquet`, 5. Reglas para usar el dataset sin trampas, Fase 1 — Dataset: fuentes, diccionario y justificación, Ahorro de tokens para agentes (Claude Code), Dataset (Fase 1) (+3 more)

### Community 8 - "Quant-Predictor — contexto para agentes"
Cohesion: 0.29
Nodes (6): Ahorro de tokens (obligatorio), graphify, Mapa rápido, Quant-Predictor — contexto para agentes, Reglas de datos (anti look-ahead / sesgos), Yahoo

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `chart()` connect `yahoo.py` to `pandas`, `prices.py`, `test_ingest.py`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Why does `fetch_summaries()` connect `metadata.py` to `pandas`, `yahoo.py`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Why does `load_prices()` connect `store.py` to `pandas`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
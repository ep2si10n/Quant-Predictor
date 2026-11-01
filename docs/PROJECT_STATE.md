# Estado del proyecto (leer primero; mantener < 80 líneas)

## Objetivo
Predictor quant de ETFs + estrategia para cartera ficticia. 5 fases (ver README).

## Fase actual: 1 — Ingesta ✅ (2026-11-01)
- Universo EEUU: NASDAQ Trader, todos los ETFs/ETNs (~5.8k). Universo EU: Xetra UCITS (~3.6k, por ISIN).
- Precios Yahoo `v8/chart` con period1 explícito (range=max degradaba a semanal/mensual → guardia anti no-diario).
- Macro: ~75 series Yahoo (índices, futuros, FX, vol) + ~74 FRED.
- Alternativos (`data/processed/alt/`): Ken French, AQR, Goyal-Welch-Zafirov, Shiller, Cboe, CFTC COT
  (legacy/disagg/TFF), AAII, Baker-Wurgler, GPR, ACM term premium, GSCPI, ADS, SF Fed news sentiment.
- Catálogos: `data/meta/feature_catalog_alt.csv`, `macro_fred.parquet` (col `pub_lag_days`).
- Números exactos: `data/meta/dataset_stats.json`.

## Decisiones clave
- D1: Dos regiones separadas (`region=us|eu|all`): EEUU en USD, UCITS Xetra mayoritariamente EUR.
  Fallback ISIN→otra plaza si el mnemónico Xetra no está en Yahoo. Ojo duplicados de índice entre regiones.
- D2: Cliente Yahoo propio (sin crumb) para precios; yfinance sufría 429 en cookie/crumb.
  Cortesía: 2 hilos, pausa 0.35 s, respeta Retry-After. No evadir límites.
- D3: Clasificación `asset_class`: Morningstar (quoteSummary) si disponible, si no reglas por nombre
  (`metadata.classify_by_name`, tests en `tests/test_ingest.py`).
- D4: QA conservador: sin interpolar; flags `ret_outlier` (>40%/día) y `stale`.
- D5: Parquet zstd particionado por clase, <90 MB/fichero → versionable en GitHub sin LFS.
- D6: Features con `pub_lag_days` para evitar look-ahead; validación walk-forward con purging/embargo.

## Limitaciones conocidas
- Sesgo de supervivencia (sin ETFs deslistados).
- OAS ICE BofA en FRED solo 3 años → usar Baa-Aaa / `gw_dfy` para historia larga.
- Macro revisada (no vintages ALFRED).

## Fases 2-4 ✅ (ver reports/PHASE4_SUMMARY.md)
- F2: docs/RESEARCH.md. F3: run_study.py (20 estrategias). F4: run_phase4.py (selección dos niveles).
  F4b: run_phase4b.py (vol HAR + régimen + MLP). 125 intentos registrados.
- Robusto: predicción de vol HAR; 60/40 vol-managed baja drawdown. No robusto: selección ML,
  clasificadores de régimen (AUC~0,5), combos elegidos en dev (PBO 0,60).
- Decisión F5: modelo = 60/40 + gestión de volatilidad HAR, L=1.

## Próximo: Fase 5 — exportar modelo (paquete + señal mensual) e informe matemático.

## Herramientas agente
Graphify (`graphify query`), Caveman (respuestas cortas). Hook SessionStart reinstala deps.

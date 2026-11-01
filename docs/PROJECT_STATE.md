# Estado del proyecto (leer primero; mantener < 80 líneas)

## Objetivo
Predictor quant de ETFs + estrategia para cartera ficticia. 5 fases (ver README).

## Fase actual: 1 — Ingesta ✅ (2026-11-01)
- Universo: NASDAQ Trader, todos los ETFs/ETNs EEUU (~5.8k). Precios Yahoo `v8/chart` range=max.
- Macro: ~75 series Yahoo (índices, futuros, FX, vol) + ~74 FRED.
- Alternativos (`data/processed/alt/`): Ken French, AQR, Goyal-Welch-Zafirov, Shiller, Cboe, CFTC COT
  (legacy/disagg/TFF), AAII, Baker-Wurgler, GPR, ACM term premium, GSCPI, ADS, SF Fed news sentiment.
- Catálogos: `data/meta/feature_catalog_alt.csv`, `macro_fred.parquet` (col `pub_lag_days`).
- Números exactos: `data/meta/dataset_stats.json`.

## Decisiones clave
- D1: Universo EEUU = cobertura global de clases de activo con un solo calendario/divisa (USD).
  UCITS europeos: extensión opcional futura.
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

## Próximo: Fase 2 — Investigación de técnicas
Candidatas a revisar: TSMOM/cross-sectional momentum, risk parity/HRP, Black-Litterman, modelos de
regímenes (HMM), GBDT (LightGBM) con features cross-asset, redes (LSTM/Transformer) para series,
ensembles, Kelly fraccional, meta-labeling y triple barrier (López de Prado), shrinkage de covarianzas.

## Herramientas agente
Graphify (`graphify query`), Caveman (respuestas cortas). Hook SessionStart reinstala deps.

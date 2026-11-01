# Fase 1 — Dataset: fuentes, diccionario y justificación

Regenerar todo: `scripts/run_all.sh` (EEUU + Europa UCITS + macro Yahoo) y
`PYTHONPATH=src python -m quantpred.ingest.alt_data` (alternativos + FRED vía `macro.fetch_fred`).
Cargar: `from quantpred.ingest.store import load_prices, load_panel, load_metadata, load_macro`.

## 1. Universo de ETFs (activos invertibles)

| Elemento | Detalle |
|---|---|
| Fuente del universo | NASDAQ Trader `nasdaqtraded.txt`: todos los valores negociados en EEUU con flag `ETF=Y` (NASDAQ, NYSE, NYSE Arca, Cboe BZX, IEX…) |
| Tamaño | ~5.800 ETFs/ETNs/ETPs (acciones EEUU e internacionales, sectores, factores, bonos de todo tipo, materias primas, divisas, inmobiliario, cripto, volatilidad, apalancados/inversos, alternativos, multi-activo) |
| Precios | Yahoo Finance `v8/chart`, diario, `period1=1900` (desde 1993 para SPY); guardia que rechaza series no diarias |
| Campos | `open high low close adj_close volume dividends splits capital_gains` |
| Clasificación | `asset_class` ∈ {equity, fixed_income, commodity, real_estate, currency, volatility, crypto, multi_asset, alternative, leveraged_inverse}; categoría Morningstar cuando Yahoo la sirve (`class_source`), si no reglas por nombre. Flags `is_leveraged`, `is_inverse`, `is_etn` |
| Almacenamiento | `data/processed/etf_prices/asset_class=<clase>/part-XX.parquet` (zstd, <90 MB/fichero) |
| QA | `data/meta/quality_report.csv` por ticker: fechas, filas, outliers (>40%/día), % días sin negociación, volumen $ mediano, dividendos, splits |

### 1b. Universo europeo UCITS — `data/processed/etf_prices_eu/`
| Elemento | Detalle |
|---|---|
| Fuente | Lista oficial de instrumentos negociables de **Xetra** (Deutsche Börse), mayor mercado de ETPs UCITS de Europa |
| Tamaño | ~3.600 (≈3.070 ETFs, 205 ETCs, 360 ETNs), deduplicados por ISIN |
| Precios | Yahoo `<mnemónico>.DE`; si no existe, mismo ISIN en otra plaza (Frankfurt, Ámsterdam, París, Milán, Londres, SIX…) vía búsqueda por ISIN (`yahoo_resolved_by_isin`) |
| Divisa | Mayoritariamente EUR (columna `currency` en `etf_metadata_eu`); convertir con `EURUSD=X` antes de mezclar con EEUU |
| Clasificación | Reglas por nombre completo (Yahoo) + grupo de producto Xetra (RENTEN→renta fija, ETC→materias primas) |
| Carga | `load_prices(region="eu")`, `load_metadata("eu")`, o `region="all"` |

Ventajas: ETFs acumulativos, exposiciones europeas (bonos euro, sectores Stoxx), ETCs físicos de metales.
Muchos replican los mismos índices que los de EEUU → útil para validación cruzada, pero cuidado con
duplicar exposiciones al combinar regiones (deduplicar por índice/benchmark en la fase de selección).

**Por qué un universo tan amplio**: el modelo necesita ver todas las clases de activo y regímenes para
aprender relaciones cross-asset (bonos vs. acciones, dólar vs. commodities, etc.) y después seleccionar
el sub-universo invertible (filtros de liquidez, historia mínima, sin apalancados) en la fase de estrategia.

**Limitaciones conocidas**
- *Sesgo de supervivencia*: solo ETFs vivos hoy (Yahoo no sirve deslistados). Mitigación: en cada fecha
  del backtest el universo = ETFs con datos en esa fecha; resultados de selección cross-sectional pueden
  estar ligeramente sobreestimados. Se documentará en los backtests.
- Historia corta para muchos ETFs (mediana pocos años). Por eso se añaden **índices/futuros de historia larga**
  (abajo) como proxies de clase de activo para entrenar en más regímenes (1970s, 2000, 2008…).
- `adj_close` de Yahoo ajusta por dividendos y splits (retorno total aproximado).

## 2. Series de mercado de historia larga (Yahoo) — `data/processed/macro_yahoo.parquet`
~75 series: índices bursátiles EEUU e internacionales (S&P 500 desde 1927, IBEX, DAX, Nikkei…),
volatilidad (VIX, VIX3M, VVIX, SKEW, MOVE), tipos (13s, 5Y, 10Y, 30Y), futuros de materias primas
(energía, metales, agrícolas, ganado), futuros financieros (ES, NQ, ZN, ZB…), divisas y cripto.

## 3. Macro FRED — `data/processed/macro_fred.parquet` (~74 series)
Tipos (curva completa, pendientes 10Y-2Y/10Y-3M, real 10Y, breakevens), crédito (Baa-Aaa, OAS IG/HY/BBB/CCC),
liquidez (reservas, reverse repo, TGA, balance Fed), condiciones financieras (NFCI, ANFCI, STLFSI),
actividad (paro, nóminas, producción, CFNAI, JOLTS, Sahm, prob. recesión), inflación (IPC, PCE, PPI,
expectativas Cleveland/Michigan), crédito bancario (SLOOS), vivienda, divisas, vol. implícita oro/crudo, EPU.
Columna `pub_lag_days`: D=1, W=7, M=35, Q=90.
Nota: FRED solo sirve 3 años de los OAS ICE BofA (licencia) → para historia larga usar Baa-Aaa y `gw_dfy`.

## 4. Datos alternativos / features — `data/processed/alt/*.parquet`
Formato largo `date, series, value`. Catálogo con frecuencia, lag, motivo y referencia:
`data/meta/feature_catalog_alt.csv`.

| Fuente | Contenido | Por qué (evidencia) |
|---|---|---|
| Ken French Data Library | FF5 + momentum + reversión corto/largo (EEUU y desarrollados), 12 industrias, diario | Factores de riesgo sistemático; exposiciones de ETFs y rotación de factores (Fama-French 1993/2015, Carhart 1997, Jegadeesh-Titman 1993) |
| AQR Data Library | TSMOM multi-activo (58 futuros), BAB diario, QMJ, Value & Momentum Everywhere, Century of Factor Premia | Primas cross-asset robustas (Moskowitz-Ooi-Pedersen 2012; Frazzini-Pedersen 2014; Asness-Moskowitz-Pedersen 2013; Ilmanen et al. 2021) |
| Goyal-Welch-Zafirov | 44 predictores mensuales de la prima de riesgo desde 1871 (d/p, e/p, b/m, tms, dfy, infl, ntis, svar, cay, vrp, skew, ogap…) | Benchmark académico de predicción; la mayoría falla OOS → útil para no engañarse (Welch-Goyal 2008, RFS 2024) |
| Shiller | CAPE, dividendos, beneficios, IPC desde 1871 | Valoración predice rentabilidad a largo plazo (Campbell-Shiller 1988) |
| Cboe | VIX 9D/3M/6M/1Y (estructura temporal), VVIX, SKEW, vol. Nasdaq/Russell/oro/crudo/bonos/EM, put/call | VIX y su pendiente predicen a corto plazo; prima de riesgo de varianza (Bollerslev-Tauchen-Zhou 2009; Bekaert-Hoerova 2014) |
| CFTC COT | Posiciones semanales por mercado: legacy (1986+), disaggregated (2006+, managed money), TFF (2006+, leveraged funds/asset managers) | Presión de cobertura y crowding en futuros (De Roon-Nijman-Veld 2000; Bessembinder 1992) |
| AAII | Sentimiento minorista semanal desde 1987 | Contrarian en extremos (Brown-Cliff 2004/2005) |
| Baker-Wurgler | Índice de sentimiento compuesto (1965-2018) | Sentimiento y sección cruzada (Baker-Wurgler 2006/2007) |
| GPR | Riesgo geopolítico diario y mensual (desde 1900) | Caldara-Iacoviello (AER 2022) |
| NY Fed ACM | Prima por plazo y tipo neutral esperado 2/5/10Y, diario | Predicción de excesos de bonos (Adrian-Crump-Moench 2013) |
| NY Fed GSCPI | Presión cadenas de suministro | Inflación/commodities (Benigno et al. 2022) |
| Philly Fed ADS | Condiciones de negocio diarias | Nowcasting de ciclo (Aruoba-Diebold-Scotti 2009) |
| SF Fed News Sentiment | Sentimiento económico diario de prensa | Shapiro-Sudhof-Wilson (2022) |

Señales que **se derivan de precios** en la fase de features (no requieren descarga): momentum/TSMOM
multi-horizonte, reversión, volatilidad realizada y su ratio, carry (rendimiento por dividendos, pendientes
de curva), correlaciones y betas rolling, drawdown, liquidez (Amihud), estacionalidad.

## 5. Reglas para usar el dataset sin trampas
1. Desplazar cada feature por su `pub_lag_days` (ALFRED vintages no incluidos: las series macro revisadas
   tienen algo de look-ahead residual; usar preferentemente series de mercado o sin revisión).
2. Validación temporal (walk-forward con purging + embargo, López de Prado 2018), nunca k-fold aleatorio.
3. Costes de transacción y filtros de liquidez antes de declarar una estrategia ganadora.

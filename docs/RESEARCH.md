# Fase 2 — Investigación: qué funciona para predecir y asignar ETFs

Síntesis de literatura académica, estudios empíricos y práctica quant. Cada técnica lleva su evidencia,
sus riesgos y cómo se prueba en la Fase 3.

## 1. Lecciones generales (lo que la evidencia dice de verdad)
1. **La predictibilidad es pequeña.** Gu-Kelly-Xiu (RFS 2020): R² OOS mensual ~0.3–0.4% en acciones
   individuales con los mejores modelos; aun así genera valor económico grande (Sharpe > 2 en long-short
   de acciones; mucho menos en ETFs/índices). → Objetivo realista: ranking relativo, no precio exacto.
2. **Árboles (GBDT) y redes ganan a lo lineal** por capturar interacciones no lineales; las señales
   dominantes son siempre variantes de **momentum, volatilidad y liquidez** (GKX 2020).
3. **La mayoría de predictores macro fallan fuera de muestra** (Goyal-Welch-Zafirov, RFS 2024): la media
   histórica es difícil de batir para timing del mercado. → Macro sirve más como *condicionante de régimen*
   que como predictor directo.
4. **Primas cross-asset robustas** (décadas, muchos mercados): time-series momentum (Moskowitz-Ooi-Pedersen
   2012), value y momentum everywhere (Asness-Moskowitz-Pedersen 2013), carry (Koijen et al. 2018),
   low-beta/BAB (Frazzini-Pedersen 2014). → Base de features y estrategias benchmark.
5. **La asignación importa tanto como la predicción.** Mean-variance con medias estimadas es inestable
   (DeMiguel-Garlappi-Uppal 2009: 1/N difícil de batir). Mejoran OOS: shrinkage de covarianza
   (Ledoit-Wolf 2004), risk parity/inverse-vol, HRP (López de Prado 2016); la evidencia empírica HRP vs.
   LW-mean-variance es mixta → hay que probarlo en nuestros datos.
6. **El mayor riesgo es el sobreajuste del backtest.** Bailey-López de Prado: Deflated Sharpe Ratio (DSR)
   y Probability of Backtest Overfitting (PBO/CSCV); validación con purging + embargo (López de Prado 2018).

## 2. Técnicas candidatas
| Bloque | Técnica | Evidencia | Riesgo |
|---|---|---|---|
| Señal | Momentum cross-sectional 12-1, 6-1, 3-1 | Jegadeesh-Titman; robusto en ETFs sectoriales/país | Crashes de momentum (Daniel-Moskowitz 2016) |
| Señal | TSMOM con escalado por volatilidad | MOP 2012, AQR datos 1985+ | Mercados laterales (whipsaw) |
| Señal | Dual momentum (absoluto + relativo) | Antonacci 2014 | Pocos activos → alta varianza |
| Señal | Reversión de corto plazo (1 semana/1 mes) | Lehmann 1990; ST reversal FF | Costes de transacción |
| Señal | Carry/yield (dividendos, pendiente curva) | Koijen et al. 2018 | Disponibilidad de datos en ETFs |
| Modelo | Ridge / Elastic Net sobre features | Baseline lineal penalizado GKX | No capta no linealidades |
| Modelo | GBDT (LightGBM) regresión/ranking | GKX 2020; Kaggle/práctica quant | Sobreajuste → regularizar, early stopping temporal |
| Modelo | Redes (MLP/LSTM/TCN/Transformer) | GKX (NN3 mejor); ICLR 2026 TCN cross-sectional | Datos limitados en ETFs, inestabilidad |
| Modelo | Regímenes (HMM 2-3 estados, clustering macro) | Ang-Bekaert 2002; Hamilton 1989 | Detección tardía |
| Modelo | Ensembles (media de modelos/semillas) | Reduce varianza del pronóstico | — |
| Cartera | 1/N, inverse-vol, risk parity (ERC) | DeMiguel 2009; Maillard-Roncalli-Teiletche 2010 | Concentración en bonos |
| Cartera | Mínima varianza con Ledoit-Wolf | Ledoit-Wolf 2004 | Ignora retornos |
| Cartera | HRP | López de Prado 2016 | Evidencia mixta |
| Cartera | Mean-variance con μ del modelo + shrinkage (Black-Litterman) | He-Litterman 1999 | Sensibilidad a μ |
| Cartera | Volatility targeting | Moreira-Muir 2017 | Lag en shocks |
| Validación | Walk-forward expandido, purging + embargo | López de Prado 2018 | — |
| Validación | DSR, PBO (CSCV), costes y turnover | Bailey-LdP 2014; Bailey et al. 2017 | Contar todos los intentos |

## 3. Diseño experimental para la Fase 3
- **Universo invertible**: ETFs EEUU líquidos (volumen $ mediano alto), sin apalancados/inversos/ETNs,
  ≥ 3 años de historia en cada fecha; universo dinámico (solo ETFs existentes en t).
- **Horizonte**: predicción de exceso de retorno a 21 días; rebalanceo mensual (también semanal como test).
- **Split temporal**: entrenamiento/validación hasta 2018-12 (walk-forward anual expandido);
  **test final intocable 2019-01 → hoy** (COVID, inflación 2022, 2023-26). El test se mira una sola vez
  por versión de estrategia y se registran todos los intentos (para DSR).
- **Costes**: 5 pb por operación + slippage proporcional a iliquidez.
- **Métricas**: CAGR, volatilidad, Sharpe, Sortino, máx. drawdown, Calmar, turnover, hit-rate, IC de rango
  (Spearman) de las predicciones, DSR y PBO.
- **Benchmarks**: SPY, 60/40 (SPY/AGG), 1/N del universo.

## Fuentes
- Gu, Kelly & Xiu (2020) [NBER w25398](https://www.nber.org/papers/w25398), [SSRN](https://papers.ssrn.com/abstract=3159577)
- Goyal, Welch & Zafirov (2024) [SFI](https://www.sfi.ch/en/publications/a-comprehensive-2022-look-at-the-empirical-performance-of-equity-premium-prediction)
- AQR data library: [TSMOM](https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Factors-Monthly)
- Bailey, Borwein, López de Prado & Zhu — [Probability of Backtest Overfitting](https://papers.ssrn.com/abstract=2326253); [Purged CV](https://en.wikipedia.org/wiki/Purged_cross-validation)
- HRP/shrinkage: [Molyboga, modified HRP](https://papers.ssrn.com/author=3588908), [Comp. Economics 2025](https://link.springer.com/10.1007/s10614-025-10848-w), [CRISP preprint 2026](https://www.emergentmind.com/papers/2604.23833)
- ETF/ML aplicados: [Karatas & Hirsa 2021](https://arxiv.org/abs/2108.02838), [UOC LightGBM walk-forward](https://openaccess.uoc.edu/bitstream/10609/149633/5/jginerabFMDP0224report.pdf), [Chalmers stocks vs bonds](https://odr.chalmers.se/items/7545236f-f5c7-48ca-bded-9a7d1beca55a), [Network momentum (CXO)](https://cxoadvisory.com/momentum-investing/multi-class-network-momentum), [ICLR 2026 cross-sectional](https://iclr.cc/virtual/2026/10016547)

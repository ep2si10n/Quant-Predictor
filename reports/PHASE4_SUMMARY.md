# Fases 3-4 — Resumen de resultados y lecciones

Protocolo: desarrollo 2008-2018 (selección), test 2019-2026 (una sola evaluación), costes 5 pb,
125 configuraciones registradas en total (Fases 3, 4 y 4b) → Deflated Sharpe con N real.

## Qué NO funcionó (y por qué)
| Enfoque | Dev Sharpe | Test Sharpe | Diagnóstico |
|---|---|---|---|
| ML selección de ETFs (LightGBM/Ridge, top-20) | -0,29 … 0,12 | 0,14 … 0,61 | IC>0 procedía de evitar activos de beta baja (aprendía la prima de mercado), rotación 15x/año |
| Dos niveles: ERC entre clases + selección (F4) | 0,53 | 0,29 | Commodities 2008-15 y filtro de tendencia lastran en 2019-26 |
| Clasificador "mes malo" (logit, LightGBM, MLP) | AUC 0,45-0,53 | AUC 0,42-0,59 | Sin capacidad: el riesgo de cola no es predecible con estas variables |
| Combo vol + régimen elegido en dev | **1,16** | 0,35 | Sobreajuste (PBO 0,60) |

## Qué SÍ es robusto
- **Predicción de volatilidad HAR** (Corsi 2009): R² frente a naive +17% (dev) y +19% (test).
- **60/40 gestionada por volatilidad** (exposición = σ_ref/σ̂, sin apalancamiento):

| Test 2019-2026 | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|
| 60/40 base | 10,8% | 0,74 | -20,0% | 0,54 |
| 60/40 vol-managed (HAR, L=1) | 8,4% | 0,68 | -14,3% | 0,59 |
| SPY | 16,7% | 0,87 | -23,9% | 0,70 |

Correlación entre Sharpe de desarrollo y de test de las variantes: 0,12 → la ordenación en
desarrollo apenas predice el test; preferir reglas simples y con base teórica.

## Decisión para la Fase 5
Modelo final = **60/40 con gestión de volatilidad HAR, sin apalancamiento** (pocos parámetros,
mecanismo documentado, menor drawdown). La selección ML de ETFs se mantiene como módulo de
investigación, no como estrategia recomendada.

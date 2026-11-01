# Estrategia relativa al S&P 500 — cartera simulada de ETFs que intenta batir al SPY

## Objetivo
Elegir cada día una cartera long-only entre los ~80 ETFs más líquidos de EEUU (renta variable de todo
el mundo, sectores, bonos, oro y materias primas, REITs…; sin apalancados, inversos ni cripto) para **batir al S&P 500**.
Entrenada con datos **diarios**; en producción la cartera se **valora minuto a minuto** frente al SPY
(`src/quantpred/live/`), lista para superponer ambas curvas en tiempo real.

## Recompensa (la pedida, calibrada)
Δ = rendimiento de la cartera − rendimiento del SPY, acumulado en ventanas de 21 días (neto de costes).

    f(Δ) = Δ + κ·max(Δ,0) − λ·min(Δ,0)²        κ = 0,5   λ = 50

Batir al SPY puntúa (1+κ)Δ; quedarse cerca del SPY cuesta poco; quedarse muy por debajo se penaliza
de forma cuadrática.

**Lección importante:** con κ>0 la recompensa es convexa y **premia la volatilidad en sí misma**
(efecto opción): para Δ ~ N(0,σ) se cumple E[f] = κσ/√(2π) − λσ²/2. En la primera versión (Δ diario,
κ=1, λ=25) la política aprendió a apostar por ETFs volátiles: recompensa media positiva con retorno
activo de −3,3% al año. Corrección: medir Δ sobre 21 días y exigir λ > 2κ/(σ√2π), de modo que el ruido
puro tenga recompensa esperada negativa (verificado en `tests/test_relative.py`).

## Modelo
- Puntuación lineal por ETF s = θ·x (23 features: momentum y retorno relativo a 1-252 días,
  volatilidad, beta/correlación con el SPY, medias móviles, drawdown, RSI e interacciones con el VIX),
  z-score cross-sectional.
- Pesos w = (1−γ)·SPY + γ·softmax(s): núcleo SPY más satélite.
- θ se entrena **maximizando directamente f** con gradiente analítico exacto (verificado contra
  diferencias finitas, error relativo de 1e-9) y L-BFGS. Se probó también una red MLP: rindió peor en validación.

## Resultados (TRAIN 2005-15 · VALIDACIÓN 2016-18 · TEST 2019-26, coste 2 pb por operación)
Configuración elegida en validación (γ=1, la más agresiva). Test:

| estrategia | CAGR | activo/año | TE | IR | MaxDD | % ventanas 21d batiendo al SPY | rotación/año |
|---|---|---|---|---|---|---|---|
| Política diaria (elegida) | 23,4% | +10,0% | 27,8% | 0,36 | −53% | 60% | 224× |
| Política semanal | 11,3% | −1,3% | 26,1% | −0,05 | −62% | 47% | 91× |
| SPY | 16,9% | 0 | 0 | — | −34% | — | 0 |
| Equiponderada 80 | 11,8% | −4,9% | 6,4% | −0,77 | −31% | 40% | 3× |
| Momentum top-10 | 7,9% | −7,8% | 18,7% | −0,42 | −41% | 39% | 60× |

Variante **núcleo-satélite** (γ=0,6, coste de entrenamiento 10 pb). Era la más estable en validación;
su test **se consultó después** del de la elegida, así que se informa con esa salvedad:

| | CAGR | activo/año | TE | IR | MaxDD | caída relativa máx. | % ventanas 21d | rotación |
|---|---|---|---|---|---|---|---|---|
| Validación | 13,8% vs 9,7% | +4,0% | 5,3% | 0,75 | −18,5% | −4,2% | 67% | 27× |
| **Test** | **22,4% vs 16,9%** | **+5,7%** | 9,9% | **0,57** | −27,5% | −13,9% | 64% | 30× |

## Lectura honesta
- La política agresiva gana en riqueza final pero viola el espíritu de la recompensa: rachas de
  −50% frente al SPY y rotación diaria muy alta. Su ventaja desaparece con rebalanceo semanal: es una
  señal de muy corto plazo (reversión a 5 días) y frágil ante costes reales.
- La variante núcleo-satélite cumple mejor lo pedido: sigue al S&P 500 de cerca (TE ~10%) y lo bate
  en ~2/3 de los meses. Es el **modelo recomendado para producción**
  (`models/relative_policy_core_satellite.json`, reentrenado con todo el histórico).
- Riesgos: sesgo de supervivencia del universo, costes reales mayores que 2 pb en ETFs menos
  líquidos, ejecución al cierre (MOC) y un único periodo de test.

## Uso en tiempo real
```python
from quantpred.live.engine import LiveEngine, SimulatedPortfolio
eng = LiveEngine.from_files("models/relative_policy_core_satellite.json").prepare(closes, dollar_vol)
pf = SimulatedPortfolio(100_000); pf.rebalance(eng.target_weights(), precios, ts)
punto = pf.on_tick(ts, precios_minuto)   # {'ts','modelo','sp500','diff_pct',...} -> gráfico
```
Demo: `python scripts/live_demo.py --replay` (histórico día a día) o `--yahoo 30` (30 minutos en vivo).

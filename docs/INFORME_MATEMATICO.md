# Informe matemático — Quant-Predictor

Cartera simulada de ETFs entrenada para batir al S&P 500. Este documento recoge la formulación
completa: datos, features, función de recompensa, modelo, optimización, protocolo de validación y
métricas, junto con los métodos de las fases de investigación (3-4b). Los resultados numéricos están
en `reports/` (resumen en la sección 9).

---

## 1. Datos y notación

- Universo: ETFs de EEUU (NASDAQ Trader, ~5.800) y UCITS europeos (Xetra, ~3.600), precios diarios
  ajustados por dividendos y splits, $P_{i,t}$. Los UCITS se convierten a USD: $P^{USD}_{i,t} = P_{i,t}\,X_t$,
  donde $X_t$ es el tipo de cambio (EURUSD, GBPUSD, 1/USDCHF, 1/USDJPY; peniques/100).
- Retorno diario $r_{i,t} = P_{i,t}/P_{i,t-1} - 1$. Benchmark: SPY, $r_{m,t}$.
- Universo invertible en el mes $\tau$ (fijado el primer día hábil, sin mirar el futuro): los $N_{US}=80$
  ETFs de EEUU y $N_{EU}=40$ UCITS de mayor volumen negociado mediano en 63 días, con al menos 252
  días de historia y $\sigma_{252}\ge 3\%$ (excluye equivalentes de liquidez). Se excluyen productos
  apalancados, inversos, de volatilidad y cripto. El SPY ocupa siempre el hueco $i=0$.
- **Ejecución**: la decisión se toma al cierre de EEUU en $t$. Un ETF de EEUU gana $r_{i,t+1}$; un UCITS
  (Xetra cierra antes que Wall Street) se ejecuta en el cierre europeo de $t+1$ y gana $r_{i,t+2}$.
  Así no hay look-ahead por los husos horarios.

## 2. Features (vector $x_{i,t}\in\mathbb{R}^{F}$, $F=35$)

Para cada activo y fecha, solo con información hasta $t$:

| Grupo | Definición |
|---|---|
| Momentum | $R^{(h)}_{i,t} = P_{i,t}/P_{i,t-h}-1$, $h\in\{1,5,21,63,126,252\}$ |
| Relativo al SPY | $A^{(h)}_{i,t} = R^{(h)}_{i,t} - R^{(h)}_{m,t}$, $h\in\{5,21,63,252\}$ |
| Riesgo | $\sigma^{(21)},\sigma^{(63)}$ (anualizadas); $\beta_{63}=\mathrm{Cov}(r_i,r_m)/\mathrm{Var}(r_m)$; $\rho_{63}$ |
| Tendencia | $P/\overline{P}_{50}-1$, $P/\overline{P}_{200}-1$, $P/\max_{252}P - 1$ |
| Oscilador | RSI(14) centrado: $\frac{\bar u}{\bar u+\bar d}-\frac12$ |
| Régimen | $z^{VIX}_t$ (z-score del VIX a 252 días), $z^{SPY21}_t$; interacciones $\beta\cdot z^{VIX}$, $\beta\cdot z^{SPY21}$, $A^{(21)} z^{VIX}$, $\sigma^{(63)} z^{VIX}$ |
| Estáticas (v2) | indicador SPY, indicador UCITS, one-hot de clase (renta variable, renta fija, materias primas, inmobiliario, alternativos) |
| Clase × régimen (v2) | $\mathbb{1}[\text{clase}=c]\cdot z^{VIX}_t$, $\mathbb{1}[\text{clase}=c]\cdot z^{SPY21}_t$ para renta fija, materias primas e inmobiliario |

Normalización cross-sectional en cada fecha (salvo las estáticas):
$\tilde x_{i,t,f} = \mathrm{clip}\big((x_{i,t,f}-\mu_{t,f})/\sigma_{t,f},\,-5,\,5\big)$. Los valores ausentes pasan a 0.

## 3. Función de recompensa

Sea $w_t$ la cartera y $c$ el coste por unidad de rotación. El rendimiento activo neto diario es
$$\Delta_t = \sum_i w_{i,t} r_{i,t+1} - c\sum_i |w_{i,t}-w_{i,t-1}| - r_{m,t+1}.$$
La recompensa se evalúa sobre el rendimiento relativo acumulado en ventanas de $H=21$ días,
$D_t = \sum_{\tau=t-H+1}^{t}\Delta_\tau$:
$$f(D) = D + \kappa\max(D,0) - \lambda\min(D,0)^2,\qquad \kappa=0{,}5,\ \lambda=50.$$
- Batir al SPY se premia con $(1+\kappa)D$; quedar por debajo penaliza de forma lineal más cuadrática.
- **Convexidad y calibración.** Con $D\sim\mathcal N(0,\sigma^2)$ (ruido sin habilidad):
  $\mathbb E[f(D)] = \kappa\,\sigma/\sqrt{2\pi} - \lambda\sigma^2/2$. Si $\kappa>0$ el primer término premia
  la varianza: es el efecto opción de cualquier bonus convexo. Para que el ruido puntúe negativo hace
  falta $\lambda > 2\kappa/(\sigma\sqrt{2\pi})$; con $\sigma\approx 2\%$ a 21 días sale $\lambda>40\kappa$.
  En la primera versión (Δ diario, κ=1, λ=25) la condición no se cumplía y la política aprendió a
  maximizar dispersión: recompensa positiva con retorno activo negativo.
- Versión suave para el gradiente (nitidez $k=200$): $\max(D,0)\approx \mathrm{sp}_k(D)=\tfrac1k\log(1+e^{kD})$,
  $\min(D,0)^2\approx \mathrm{sp}_k(-D)^2$, y
  $f'(D) = 1 + \kappa\,\mathrm{sig}(kD) + 2\lambda\,\mathrm{sp}_k(-D)\,\mathrm{sig}(-kD)$.

## 4. Modelo (política de cartera)

**Puntuación** por activo: lineal $s_{i,t}=\theta^\top\tilde x_{i,t}$, o MLP
$s_{i,t}=v^\top\tanh(W\tilde x_{i,t}+b)$.

**Pesos** (long-only, 100% invertidos), con núcleo SPY:
$$\omega_{i,t} = \frac{e^{s_{i,t}}\,\mathbb 1[i\in\mathcal U_t]}{\sum_{j\in\mathcal U_t}e^{s_{j,t}}},\qquad
w_{i,t} = \gamma_t\,\omega_{i,t} + (1-\gamma_t)\,\mathbb 1[i=0].$$
- $\gamma$ fijo: fracción satélite (por ejemplo 0,6 → al menos 40% en SPY).
- $\gamma_t$ dinámico (v3): $\gamma_t = \sigma(g^\top S_t)$ con $S_t=(1, z^{VIX}_t, z^{SPY21}_t, |z^{SPY21}_t|)$. El modelo
  aprende cuánto apartarse del índice según el régimen.

**Objetivo** (con $T'$ ventanas):
$$J(\theta) = \frac{1}{T'}\sum_t f(D_t) - \alpha\|\theta\|_2^2.$$

## 5. Gradiente analítico

Sea $\phi_\tau = \partial J/\partial\Delta_\tau = \frac{1}{T'}\sum_{t:\,\tau\in[t-H+1,t]} f'(D_t)$ (convolución de
$f'$ con una ventana de unos). Con $|x|$ suavizado como $\sqrt{x^2+\epsilon}$ y $\mathrm{sg}=x/\sqrt{x^2+\epsilon}$:
$$G_{i,t} \equiv \frac{\partial J}{\partial w_{i,t}} = \phi_t\big(r_{i,t+1} - c\,\mathrm{sg}(w_{i,t}-w_{i,t-1})\big)
+ \phi_{t+1}\,c\,\mathrm{sg}(w_{i,t+1}-w_{i,t}).$$
A través del softmax ($\partial\omega_i/\partial s_j=\omega_i(\delta_{ij}-\omega_j)$):
$$\frac{\partial J}{\partial s_{i,t}} = \gamma_t\,\omega_{i,t}\Big(G_{i,t}-\sum_j\omega_{j,t}G_{j,t}\Big),\qquad
\nabla_\theta J = \sum_{t,i}\frac{\partial J}{\partial s_{i,t}}\tilde x_{i,t} - 2\alpha\theta.$$
Para la MLP se retropropaga por $\tanh$. Para el gate:
$\partial J/\partial\gamma_t = \sum_i G_{i,t}(\omega_{i,t}-\mathbb 1[i=0])$ y $\partial\gamma_t/\partial g = \gamma_t(1-\gamma_t)S_t$.
El gradiente se verifica contra diferencias finitas (error relativo de $10^{-7}$ a $10^{-10}$,
`tests/test_relative.py`). La optimización usa L-BFGS-B (scipy), hasta 300 iteraciones.
Ensemble: media de los pesos de $K$ semillas.

## 6. Protocolo de validación (anti-sobreajuste)

1. **Validación walk-forward**: para cada año $Y\in\{2016,2017,2018\}$ se entrena con $[2005, Y)$ y se predice $Y$.
2. **Búsqueda greedy**: cada ronda cambia una sola dimensión de la mejor configuración; se elige por
   $\bar f$ en validación, exigiendo retorno activo positivo. La búsqueda para si la mejora es menor de 2 pb.
3. **Test** (2019 → hoy): se mira una sola vez, con reentrenamiento anual (cada año se entrena con todo lo anterior).
4. **Registro de todos los intentos** (`reports/*trials*.csv`) para el Sharpe deflactado.
5. **Costes**: 2 pb por unidad de rotación en la evaluación; en el entrenamiento, $c$ es un hiperparámetro
   (penalizar más la rotación regulariza).

## 7. Métricas

- Retorno activo anual $\bar\Delta\cdot252$; tracking error $TE=\mathrm{sd}(\Delta)\sqrt{252}$; information ratio $IR=\bar\Delta/\mathrm{sd}(\Delta)\cdot\sqrt{252}$.
- Caída relativa máxima: drawdown de $V^{modelo}_t/V^{SPY}_t$.
- % de ventanas de 21 días que baten al SPY; recompensa media exacta $\overline{f(D)}$.
- Sharpe probabilístico (Bailey y López de Prado, 2012):
  $\mathrm{PSR}(SR^*)=\Phi\Big(\frac{(\widehat{SR}-SR^*)\sqrt{n-1}}{\sqrt{1-\hat\gamma_3\widehat{SR}+\frac{\hat\gamma_4-1}{4}\widehat{SR}^2}}\Big)$.
- Sharpe deflactado: PSR con $SR^*=\mathbb E[\max_N SR]\approx\sigma_{SR}\big((1-\gamma_E)\Phi^{-1}(1-\tfrac1N)+\gamma_E\Phi^{-1}(1-\tfrac1{Ne})\big)$.
- PBO (CSCV, Bailey et al. 2017): fracción de particiones en las que la mejor configuración dentro de
  muestra queda por debajo de la mediana fuera de muestra.

## 8. Métodos de las fases de investigación (3-4b)

- **Asignación**: inversa de la volatilidad; mínima varianza $\min_w w^\top\hat\Sigma w$ con $\hat\Sigma$ de Ledoit-Wolf
  (encogimiento hacia un objetivo escalado, $\hat\Sigma=\delta F+(1-\delta)S$); ERC (Spinu):
  $\min_y \tfrac12 y^\top\Sigma y-\sum_i b_i\log y_i$, $w=y/\mathbf 1^\top y$; HRP (López de Prado, 2016): clustering
  jerárquico con distancia $d_{ij}=\sqrt{(1-\rho_{ij})/2}$, cuasi-diagonalización y bisección recursiva con
  $\alpha=1-V_1/(V_1+V_2)$.
- **Predicción cross-sectional** (fase 3): Ridge y LightGBM sobre features mensuales; IC de rangos de Spearman.
- **Volatilidad HAR** (Corsi, 2009): $\log RV_{t+1}=\beta_0+\beta_w\log RV^{(w)}_t+\beta_m\log RV^{(m)}_t+\beta_q\log RV^{(q)}_t+\beta_y\log RV^{(y)}_t+\varepsilon$.
  Gestión por volatilidad (Moreira y Muir, 2017): exposición $e_t=\min(L,\ \sigma_{ref}/\hat\sigma_{t+1})$.
- **Régimen**: logística, LightGBM y MLP sobre unas 50 variables de estado (VIX y su estructura temporal,
  VRP, crédito, curva, COT de la CFTC, momentum de AQR, GPR, ACM, Goyal-Welch).

## 9. Resultados y lecciones (resumen)

Ver `reports/PHASE4_SUMMARY.md`, `reports/RELATIVE_SUMMARY.md` y `reports/relative_v*_test.csv`.

1. Seleccionar ETFs por retorno absoluto con ML no da una ventaja robusta: el IC procedía de la beta.
2. La volatilidad sí es predecible (HAR); el riesgo de cola, no (AUC ≈ 0,5).
3. Una recompensa convexa premia la varianza; hay que calibrarla.
4. Entrenar directamente con la recompensa relativa, con núcleo SPY y penalización de rotación, da
   carteras cercanas al índice con exceso de retorno positivo fuera de muestra.
5. Elegir con validación y mirar el test una sola vez. Cuando se consultó el test después de elegir,
   se indica expresamente.

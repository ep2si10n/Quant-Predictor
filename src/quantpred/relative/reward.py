"""Función de recompensa relativa al S&P 500.

Δ = r_cartera − r_SPY (retorno activo de un periodo).

    f(Δ) = Δ + κ·max(Δ, 0) − λ·min(Δ, 0)²

- Δ > 0 (bate al SPY): recompensa (1+κ)·Δ → superar puntúa más que igualar.
- Δ < 0 (queda por debajo): penalización lineal + cuadrática → cuanto más lejos del SPY, mucho peor;
  quedarse cerca del SPY cuesta poco.
- Δ = 0: recompensa 0 (replicar el SPY es el punto neutro).

Para optimizar con gradiente se usa una versión suave (softplus con nitidez `k`):
    max(Δ,0) ≈ sp(Δ) = log(1+e^{kΔ})/k,     min(Δ,0)² ≈ sp(−Δ)²
"""
from __future__ import annotations

import numpy as np

# Calibración: con Δ ruido N(0,σ), E[f] = κσ/√(2π) − λσ²/2. Para que el ruido NO puntúe con σ_TE a 21
# días ≈ 2% hace falta λ > 2κ/(σ√(2π)) ≈ 40κ  →  κ=0.5, λ=50 (sobre rendimiento relativo de 21 días).
DEFAULT = {"kappa": 0.5, "lam": 50.0, "k": 200.0}


def _sp(x, k):
    return np.logaddexp(0.0, k * x) / k


def _sig(x, k):
    return 0.5 * (1 + np.tanh(0.5 * k * x))


def reward(delta, kappa=DEFAULT["kappa"], lam=DEFAULT["lam"], k=DEFAULT["k"]):
    delta = np.asarray(delta, dtype=float)
    return delta + kappa * _sp(delta, k) - lam * _sp(-delta, k) ** 2


def reward_grad(delta, kappa=DEFAULT["kappa"], lam=DEFAULT["lam"], k=DEFAULT["k"]):
    """df/dΔ de la versión suave."""
    delta = np.asarray(delta, dtype=float)
    return 1 + kappa * _sig(delta, k) + 2 * lam * _sp(-delta, k) * _sig(-delta, k)


def reward_exact(delta, kappa=DEFAULT["kappa"], lam=DEFAULT["lam"]):
    delta = np.asarray(delta, dtype=float)
    return delta + kappa * np.maximum(delta, 0) - lam * np.minimum(delta, 0) ** 2

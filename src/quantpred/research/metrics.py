"""Métricas de rendimiento y de sobreajuste (Sharpe, Sortino, drawdown, DSR, PBO)."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

PPY = 12  # periodos por año (mensual)


def perf(r: pd.Series, ppy: int = PPY, rf: pd.Series | None = None) -> dict:
    r = r.dropna()
    if len(r) < 3:
        return {}
    ex = r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    yrs = len(r) / ppy
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * math.sqrt(ppy)
    sharpe = ex.mean() / ex.std() * math.sqrt(ppy) if ex.std() > 0 else np.nan
    down = ex[ex < 0].std() * math.sqrt(ppy)
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe,
            "Sortino": ex.mean() * ppy / down if down > 0 else np.nan,
            "MaxDD": dd.min(), "Calmar": cagr / -dd.min() if dd.min() < 0 else np.nan,
            "HitRate": (r > 0).mean(), "Skew": stats.skew(r), "Kurt": stats.kurtosis(r) + 3, "N": len(r)}


def sharpe_per_period(r: pd.Series) -> float:
    r = r.dropna()
    return r.mean() / r.std() if r.std() > 0 else 0.0


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & López de Prado 2012): P[SR real > SR*] corrigiendo asimetría y curtosis.
    SR en unidades por periodo."""
    r = r.dropna()
    n = len(r)
    sr = sharpe_per_period(r)
    g3, g4 = stats.skew(r), stats.kurtosis(r) + 3
    den = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / den))


def deflated_sharpe(r: pd.Series, trial_sharpes: list[float]) -> float:
    """DSR (Bailey & López de Prado 2014): PSR contra el SR máximo esperado por azar tras N intentos.
    E[max SR] ≈ σ_SR · ((1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))), γ = Euler-Mascheroni."""
    n_trials = max(2, len(trial_sharpes))
    sd = float(np.std(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
    gamma = 0.5772156649
    emax = sd * ((1 - gamma) * stats.norm.ppf(1 - 1 / n_trials) +
                 gamma * stats.norm.ppf(1 - 1 / (n_trials * math.e)))
    return probabilistic_sharpe(r, emax)


def pbo(returns: pd.DataFrame, n_splits: int = 10) -> float:
    """Probability of Backtest Overfitting vía CSCV (Bailey, Borwein, López de Prado & Zhu 2017).
    returns: T x N (una columna por configuración probada). Fracción de combinaciones en las que la
    mejor configuración in-sample queda por debajo de la mediana out-of-sample."""
    df = returns.dropna()
    T = len(df) - len(df) % n_splits
    blocks = np.array_split(df.iloc[:T].values, n_splits)
    logits = []
    for comb in itertools.combinations(range(n_splits), n_splits // 2):
        is_ = np.vstack([blocks[i] for i in comb])
        oos = np.vstack([blocks[i] for i in range(n_splits) if i not in comb])
        sr_is = is_.mean(0) / (is_.std(0) + 1e-12)
        sr_oos = oos.mean(0) / (oos.std(0) + 1e-12)
        best = int(np.argmax(sr_is))
        rank = stats.rankdata(sr_oos)[best] / (len(sr_oos) + 1)
        logits.append(math.log(rank / (1 - rank)))
    return float(np.mean(np.array(logits) <= 0))


def rank_ic(pred: pd.Series, real: pd.Series) -> float:
    m = pred.notna() & real.notna()
    if m.sum() < 5:
        return np.nan
    return float(stats.spearmanr(pred[m], real[m]).statistic)

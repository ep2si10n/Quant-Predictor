"""Esquemas de asignación: 1/N, inversa de volatilidad, ERC (risk parity), mínima varianza
Ledoit-Wolf, HRP (López de Prado 2016) y mean-variance con shrinkage."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


def equal_weight(cols) -> pd.Series:
    return pd.Series(1 / len(cols), index=cols)


def inverse_vol(rets: pd.DataFrame) -> pd.Series:
    v = rets.std().replace(0, np.nan)
    w = (1 / v).fillna(0)
    return w / w.sum()


def lw_cov(rets: pd.DataFrame) -> pd.DataFrame:
    x = rets.fillna(0).to_numpy(dtype="float64")
    # anualizada y en %² para que los optimizadores no se paren por tolerancia (var diaria ~1e-5)
    c = LedoitWolf().fit(x).covariance_ * 252 * 1e4
    return pd.DataFrame(c, index=rets.columns, columns=rets.columns)


def min_variance(rets: pd.DataFrame, max_w: float = 0.25) -> pd.Series:
    cov = lw_cov(rets).values
    n = cov.shape[0]
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), method="SLSQP",
                   bounds=[(0, max_w)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 200})
    return pd.Series(res.x if res.success else np.full(n, 1 / n), index=rets.columns).clip(lower=0)


def risk_parity(rets: pd.DataFrame) -> pd.Series:
    """Equal Risk Contribution (Maillard, Roncalli & Teiletche 2010), formulación convexa de Spinu."""
    cov = lw_cov(rets).values
    n = cov.shape[0]
    b = np.full(n, 1 / n)
    obj = lambda y: 0.5 * y @ cov @ y - b @ np.log(y)  # noqa: E731
    grad = lambda y: cov @ y - b / y  # noqa: E731
    res = minimize(obj, np.full(n, 1.0), jac=grad, method="L-BFGS-B", bounds=[(1e-8, None)] * n)
    w = res.x / res.x.sum()
    return pd.Series(w, index=rets.columns)


def hrp(rets: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity: clustering por correlación + bisección recursiva por varianza."""
    cov = lw_cov(rets)
    corr = rets.corr().fillna(0).to_numpy(copy=True)
    np.fill_diagonal(corr, 1)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
    order = leaves_list(linkage(squareform(dist, checks=False), "single"))
    items = list(rets.columns[order])
    w = pd.Series(1.0, index=items)
    clusters = [items]
    while clusters:
        clusters = [c[j:k] for c in clusters for j, k in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0 = _cluster_var(cov, c0)
            v1 = _cluster_var(cov, c1)
            a = 1 - v0 / (v0 + v1)
            w[c0] *= a
            w[c1] *= 1 - a
    return w.reindex(rets.columns)


def _cluster_var(cov: pd.DataFrame, items) -> float:
    c = cov.loc[items, items].values
    ivp = 1 / np.diag(c)
    ivp /= ivp.sum()
    return float(ivp @ c @ ivp)


def mean_variance(mu: pd.Series, rets: pd.DataFrame, risk_aversion: float = 5.0,
                  max_w: float = 0.2) -> pd.Series:
    """max w'μ - λ/2 w'Σw, long-only, con Σ Ledoit-Wolf."""
    cov = lw_cov(rets).values
    m = mu.reindex(rets.columns).fillna(0).values
    n = len(m)
    res = minimize(lambda w: -(w @ m) + risk_aversion / 2 * w @ cov @ w, np.full(n, 1 / n),
                   method="SLSQP", bounds=[(0, max_w)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 200})
    return pd.Series(res.x if res.success else np.full(n, 1 / n), index=rets.columns).clip(lower=0)


SCHEMES = {"ew": None, "ivol": inverse_vol, "erc": risk_parity, "minvar": min_variance, "hrp": hrp}


def allocate(scheme: str, cols, hist_rets: pd.DataFrame) -> pd.Series:
    if scheme == "ew" or len(cols) < 2:
        return equal_weight(cols)
    h = hist_rets[cols].dropna(how="all", axis=1)
    h = h.loc[:, h.notna().mean() > 0.8].fillna(0)
    if h.shape[1] < 2:
        return equal_weight(cols)
    w = SCHEMES[scheme](h)
    return (w / w.sum()).reindex(cols).fillna(0)

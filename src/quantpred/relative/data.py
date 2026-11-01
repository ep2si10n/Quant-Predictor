"""Dataset diario para la estrategia relativa al SPY.

Tensor X[T, N, F] (fecha, hueco de activo, feature) + máscara de disponibilidad + retornos del día
siguiente R[T, N] y del SPY. Todo con información hasta el cierre de t; la operación se supone al cierre
de t (órdenes MOC) y se gana el retorno cierre(t)→cierre(t+1).

Universo dinámico: cada mes, los N ETFs más líquidos (volumen $ mediano 63d) con ≥1 año de historia y
vol anual ≥3% (sin equivalentes de liquidez); el SPY siempre está (hueco 0).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantpred.research.features import load_daily

FEATURES = ["is_spy", "r1", "r5", "r21", "r63", "r126", "r252", "a5", "a21", "a63", "a252",
            "vol21", "vol63", "beta63", "corr63", "ma50", "ma200", "dd252", "rsi14",
            "beta_x_vix", "beta_x_spy21", "a21_x_vix", "vol_x_vix"]


def feature_frames(P: pd.DataFrame, vix: pd.Series | None = None) -> tuple[dict, pd.DataFrame]:
    """Features crudas (sin normalizar) por ticker y fecha. Reutilizado en entrenamiento y en live."""
    P = P.astype("float64")
    R = P.pct_change(fill_method=None)
    spy = R["SPY"]
    feats = {
        "r1": R, "r5": P / P.shift(5) - 1, "r21": P / P.shift(21) - 1, "r63": P / P.shift(63) - 1,
        "r126": P / P.shift(126) - 1, "r252": P / P.shift(252) - 1,
    }
    sp = P["SPY"]
    for h in (5, 21, 63, 252):
        feats[f"a{h}"] = feats[f"r{h}"].sub(sp / sp.shift(h) - 1, axis=0)
    feats["vol21"] = R.rolling(21, min_periods=15).std() * np.sqrt(252)
    feats["vol63"] = R.rolling(63, min_periods=40).std() * np.sqrt(252)
    cov = R.rolling(63, min_periods=40).cov(spy)
    feats["beta63"] = cov.div(spy.rolling(63, min_periods=40).var(), axis=0)
    feats["corr63"] = R.rolling(63, min_periods=40).corr(spy)
    feats["ma50"] = P / P.rolling(50, min_periods=35).mean() - 1
    feats["ma200"] = P / P.rolling(200, min_periods=150).mean() - 1
    feats["dd252"] = P / P.rolling(252, min_periods=180).max() - 1
    feats["rsi14"] = _rsi(R)
    if vix is None:
        from quantpred.config import PROCESSED
        cb = pd.read_parquet(PROCESSED / "alt" / "cboe.parquet")
        vix = cb[cb["series"].astype(str) == "cboe_vix"].set_index("date")["value"]
    v = vix.reindex(P.index).ffill()
    vz = ((v - v.rolling(252, min_periods=60).mean()) / v.rolling(252, min_periods=60).std()).fillna(0)
    spy21 = (sp / sp.shift(21) - 1)
    spy21z = ((spy21 - spy21.rolling(252, min_periods=60).mean()) / spy21.rolling(252, min_periods=60).std()).fillna(0)
    feats["beta_x_vix"] = feats["beta63"].mul(vz, axis=0)
    feats["beta_x_spy21"] = feats["beta63"].mul(spy21z, axis=0)
    feats["a21_x_vix"] = feats["a21"].mul(vz, axis=0)
    feats["vol_x_vix"] = feats["vol63"].mul(vz, axis=0)

    return feats, R


def cross_section(feats: dict, row: int, tickers: list, mask: np.ndarray | None = None) -> np.ndarray:
    """Matriz [N, F] normalizada (z-score cross-sectional, winsor ±5) para una fecha; hueco 0 = SPY."""
    cols = [feats["r1"].columns.get_loc(t) for t in tickers]
    raw = np.stack([feats[f].iloc[row, cols].to_numpy(dtype="float64") if f != "is_spy"
                    else np.zeros(len(cols)) for f in FEATURES], axis=1)
    raw[0, 0] = 1.0
    m = np.ones(len(cols), bool) if mask is None else mask
    z = raw.copy()
    for j in range(1, len(FEATURES)):
        col = raw[:, j]
        ok = m & np.isfinite(col)
        if ok.sum() > 3:
            mu, sd = np.nanmean(col[ok]), np.nanstd(col[ok]) + 1e-9
            z[:, j] = np.clip((col - mu) / sd, -5, 5)
        z[~np.isfinite(z[:, j]), j] = 0.0
    return z


def select_universe(P: pd.DataFrame, DV: pd.DataFrame, d, n_assets: int, exclude: set = frozenset()) -> list:
    R = P.pct_change(fill_method=None)
    liq = DV.rolling(63, min_periods=40).median().loc[d]
    age = P.notna().cumsum().loc[d]
    vol252 = (R.rolling(252, min_periods=180).std() * np.sqrt(252)).loc[d]
    ok = (age >= 252) & (vol252 >= 0.03) & liq.notna()
    cand = liq[ok].drop(["SPY", *exclude], errors="ignore").sort_values(ascending=False)
    return ["SPY"] + list(cand.index[:n_assets - 1])


@dataclass
class DailyData:
    dates: pd.DatetimeIndex
    tickers: np.ndarray        # [T, N] símbolo en cada hueco ('' si vacío)
    X: np.ndarray              # [T, N, F] float32 (z-score cross-sectional, salvo is_spy)
    mask: np.ndarray           # [T, N] bool
    R: np.ndarray              # [T, N] retorno t→t+1 (0 donde no hay)
    spy: np.ndarray            # [T] retorno SPY t→t+1
    rf: np.ndarray             # [T] tipo libre de riesgo diario
    features: list


def _rsi(r: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = r.clip(lower=0).rolling(n, min_periods=n // 2).mean()
    dn = (-r.clip(upper=0)).rolling(n, min_periods=n // 2).mean()
    return up / (up + dn) - 0.5


def build_daily(n_assets: int = 80, start: str = "2004-01-01", region: str = "us",
                P: pd.DataFrame | None = None, DV: pd.DataFrame | None = None,
                vix: pd.Series | None = None, rf_daily: pd.Series | None = None) -> DailyData:
    if P is None:
        P, DV, _ = load_daily(region, start="2002-01-01")
    P = P.astype("float64")
    feats, R = feature_frames(P, vix)
    # universo mensual
    liq = DV.rolling(63, min_periods=40).median()
    age = P.notna().cumsum()
    vol252 = R.rolling(252, min_periods=180).std() * np.sqrt(252)
    dates = P.index[P.index >= start]
    dates = dates[:-1]  # el último día no tiene retorno siguiente
    month_key = dates.year * 12 + dates.month
    T, N, F = len(dates), n_assets, len(FEATURES)
    X = np.zeros((T, N, F), dtype="float32")
    mask = np.zeros((T, N), dtype=bool)
    tick = np.full((T, N), "", dtype=object)
    Rn = np.zeros((T, N), dtype="float64")
    Rnext = R.shift(-1)
    pos = {d: i for i, d in enumerate(P.index)}
    members = None
    last_key = None
    feat_frames = {k: v for k, v in feats.items()}
    for ti, d in enumerate(dates):
        if month_key[ti] != last_key:  # recalcular universo el primer día hábil del mes (datos hasta d)
            last_key = month_key[ti]
            ok = (age.loc[d] >= 252) & (vol252.loc[d] >= 0.03) & liq.loc[d].notna()
            cand = liq.loc[d][ok].drop("SPY", errors="ignore").sort_values(ascending=False)
            members = ["SPY"] + list(cand.index[:N - 1])
        row = pos[d]
        for ni, tk in enumerate(members):
            rn = Rnext.iat[row, Rnext.columns.get_loc(tk)]
            if not np.isfinite(rn):
                continue
            tick[ti, ni] = tk
            mask[ti, ni] = True
            Rn[ti, ni] = rn
        cols = [P.columns.get_loc(t) for t in members]
        raw = np.stack([feat_frames[f].iloc[row, cols].to_numpy(dtype="float64") if f != "is_spy"
                        else np.zeros(len(cols)) for f in FEATURES], axis=1)
        raw[0, 0] = 1.0
        m = mask[ti, :len(members)]
        # z-score cross-sectional robusto (winsor 5σ) salvo is_spy
        z = raw.copy()
        for j in range(1, F):
            col = raw[:, j]
            ok = m & np.isfinite(col)
            if ok.sum() > 3:
                mu, sd = np.nanmean(col[ok]), np.nanstd(col[ok]) + 1e-9
                z[:, j] = np.clip((col - mu) / sd, -5, 5)
            z[~np.isfinite(z[:, j]), j] = 0.0
        X[ti, :len(members)] = z
    spy_next = Rnext["SPY"].reindex(dates).fillna(0).to_numpy()
    if rf_daily is None:
        from quantpred.config import PROCESSED
        fr = pd.read_parquet(PROCESSED / "macro_fred.parquet")
        tb = fr[fr["series_id"] == "DTB3"].set_index("date")["value"]
        rf_daily = (tb.reindex(P.index).ffill() / 100 / 252).fillna(0)
    return DailyData(dates, tick, X, mask, Rn, spy_next, rf_daily.reindex(dates).fillna(0).to_numpy(), FEATURES)

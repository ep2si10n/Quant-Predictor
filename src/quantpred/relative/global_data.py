"""Dataset diario GLOBAL (ETFs de EEUU + UCITS europeos) para la estrategia relativa al S&P 500.

- Precios UCITS convertidos a USD con el tipo de cambio del día (EURUSD=X, GBPUSD=X, CHF=X, JPY=X) y
  alineados al calendario de EEUU (relleno hacia delante en festivos europeos, máx. 5 días).
- **Horario**: Xetra cierra ~11:30 ET, antes que Wall Street. La señal se calcula al cierre de EEUU en t, así que un
  ETF europeo no puede comprarse al cierre europeo de t (ya pasó). Se asume ejecución en el cierre europeo
  de t+1 y se gana el retorno t+1→t+2 (un día extra de retardo, conservador y sin look-ahead).
- Universo mensual: top-`n_us` ETFs de EEUU (SPY siempre en el hueco 0) + top-`n_eu` UCITS por
  volumen negociado (63 días), con ≥1 año de historia y volatilidad ≥3%; sin cripto.
- Features v2 = v1 + clase de activo (one-hot), indicador UCITS e interacciones clase × régimen (VIX, SPY 21d).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantpred.config import PROCESSED
from quantpred.relative.data import FEATURES as FEATURES_V1
from quantpred.relative.data import DailyData, feature_frames
from quantpred.research.features import load_daily

CLASSES = ["equity", "fixed_income", "commodity", "real_estate", "alternative"]
STATIC = ["is_eu"] + [f"ac_{c}" for c in CLASSES]
REGIME = [f"ac_{c}_x_{s}" for c in ("fixed_income", "commodity", "real_estate") for s in ("vix", "spy21")]
FEATURES_V2 = FEATURES_V1 + STATIC + REGIME
NO_ZSCORE = {"is_spy", *STATIC, *REGIME}
EXCLUDE_CLASSES = {"crypto"}
FX = {"EUR": ("EURUSD=X", False), "GBP": ("GBPUSD=X", False), "GBp": ("GBPUSD=X", False),
      "CHF": ("CHF=X", True), "JPY": ("JPY=X", True), "USD": (None, False)}


def load_eu_usd(us_index: pd.DatetimeIndex):
    P, DV, meta = load_daily("eu", start="2002-01-01")
    ym = pd.read_parquet(PROCESSED / "macro_yahoo.parquet")
    fx = ym.pivot(index="date", columns="ticker", values="close").sort_index().ffill()
    cur = meta["currency"].reindex(P.columns)
    cur = cur.where(cur.notna(), np.where(P.columns.str.endswith(".L"), "GBp", "EUR"))
    keep = [t for t in P.columns if cur[t] in FX]
    P, DV = P[keep].astype("float64"), DV[keep].astype("float64")
    for t in keep:
        sym, invert = FX[cur[t]]
        if sym is None:
            continue
        rate = fx[sym].reindex(P.index).ffill()
        rate = 1 / rate if invert else rate
        scale = rate / 100 if cur[t] == "GBp" else rate
        P[t] = P[t] * scale
        DV[t] = DV[t] * scale
    P = P.reindex(us_index).ffill(limit=5)
    DV = DV.reindex(us_index)
    return P, DV, meta


def _zscore(raw: np.ndarray, m: np.ndarray, feats: list) -> np.ndarray:
    z = raw.copy()
    for j, f in enumerate(feats):
        if f in NO_ZSCORE:
            z[~np.isfinite(z[:, j]), j] = 0.0
            continue
        col = raw[:, j]
        ok = m & np.isfinite(col)
        if ok.sum() > 3:
            mu, sd = np.nanmean(col[ok]), np.nanstd(col[ok]) + 1e-9
            z[:, j] = np.clip((col - mu) / sd, -5, 5)
        z[~np.isfinite(z[:, j]), j] = 0.0
    return z


def regime_series(P: pd.DataFrame, vix: pd.Series) -> tuple[pd.Series, pd.Series]:
    v = vix.reindex(P.index).ffill()
    vz = ((v - v.rolling(252, min_periods=60).mean()) / v.rolling(252, min_periods=60).std()).fillna(0)
    sp = P["SPY"]
    s21 = sp / sp.shift(21) - 1
    sz = ((s21 - s21.rolling(252, min_periods=60).mean()) / s21.rolling(252, min_periods=60).std()).fillna(0)
    return vz, sz


def cross_section_v2(feats: dict, row: int, tickers: list, cls: dict, eu: set, vz: float, sz: float,
                     features: list = FEATURES_V2, mask: np.ndarray | None = None) -> np.ndarray:
    cols = [feats["r1"].columns.get_loc(t) for t in tickers]
    out = []
    for f in features:
        if f == "is_spy":
            v = np.zeros(len(tickers))
            v[0] = 1.0
        elif f == "is_eu":
            v = np.array([t in eu for t in tickers], float)
        elif f.startswith("ac_") and "_x_" not in f:
            v = np.array([cls.get(t) == f[3:] for t in tickers], float)
        elif f.startswith("ac_") and "_x_" in f:
            c, s = f[3:].split("_x_")
            v = np.array([cls.get(t) == c for t in tickers], float) * (vz if s == "vix" else sz)
        else:
            v = feats[f].iloc[row, cols].to_numpy(dtype="float64")
        out.append(v)
    raw = np.stack(out, axis=1)
    return _zscore(raw, np.ones(len(tickers), bool) if mask is None else mask, features)


def monthly_universe(P, DV, dates, n_us, n_eu, cls, eu):
    R = P.pct_change(fill_method=None)
    liq = DV.rolling(63, min_periods=40).median()
    age = P.notna().cumsum()
    vol252 = R.rolling(252, min_periods=180).std() * np.sqrt(252)
    ok_cls = pd.Series({t: cls.get(t) not in EXCLUDE_CLASSES for t in P.columns})
    is_eu = pd.Series({t: t in eu for t in P.columns})
    members, last = {}, None
    for d in dates:
        key = (d.year, d.month)
        if key == last:
            continue
        last = key
        ok = (age.loc[d] >= 252) & (vol252.loc[d] >= 0.03) & liq.loc[d].notna() & ok_cls
        l = liq.loc[d][ok].drop("SPY", errors="ignore")
        us = l[~is_eu.reindex(l.index)].sort_values(ascending=False).index[:n_us - 1]
        ue = l[is_eu.reindex(l.index)].sort_values(ascending=False).index[:n_eu]
        members[key] = ["SPY"] + list(us) + list(ue)
    return members


def build_global(n_us: int = 80, n_eu: int = 40, start: str = "2004-01-01") -> tuple[DailyData, dict]:
    P_us, DV_us, meta_us = load_daily("us", start="2002-01-01")
    P_eu, DV_eu, meta_eu = load_eu_usd(P_us.index)
    P = pd.concat([P_us.astype("float64"), P_eu], axis=1)
    DV = pd.concat([DV_us.astype("float64"), DV_eu], axis=1)
    cls = {**meta_us["asset_class"].to_dict(), **meta_eu["asset_class"].to_dict()}
    cls = {k: ("alternative" if v == "multi_asset" else v) for k, v in cls.items()}
    eu = set(P_eu.columns)
    dates = P.index[P.index >= start][:-2]
    members = monthly_universe(P, DV, dates, n_us, n_eu, cls, eu)
    union = sorted({t for m in members.values() for t in m})
    P = P[union]
    cb = pd.read_parquet(PROCESSED / "alt" / "cboe.parquet")
    vix = cb[cb["series"].astype(str) == "cboe_vix"].set_index("date")["value"]
    feats, R = feature_frames(P, vix)
    vz, sz = regime_series(P, vix)
    Rexec = R.shift(-1)
    eu_cols = [t for t in union if t in eu]
    Rexec[eu_cols] = R[eu_cols].shift(-2)        # retardo extra de ejecución para UCITS
    N, F = n_us + n_eu, len(FEATURES_V2)
    T = len(dates)
    X = np.zeros((T, N, F), dtype="float32")
    mask = np.zeros((T, N), bool)
    tick = np.full((T, N), "", dtype=object)
    Rn = np.zeros((T, N))
    pos = {d: i for i, d in enumerate(P.index)}
    for ti, d in enumerate(dates):
        mem = members[(d.year, d.month)]
        row = pos[d]
        r = Rexec.iloc[row][mem].to_numpy(dtype="float64")
        m = np.isfinite(r)
        k = len(mem)
        tick[ti, :k] = mem
        mask[ti, :k] = m
        Rn[ti, :k] = np.where(m, r, 0.0)
        X[ti, :k] = cross_section_v2(feats, row, mem, cls, eu, vz.iloc[row], sz.iloc[row], mask=m)
    fr = pd.read_parquet(PROCESSED / "macro_fred.parquet")
    tb = fr[fr["series_id"] == "DTB3"].set_index("date")["value"]
    rf = (tb.reindex(P.index).ffill() / 100 / 252).fillna(0).reindex(dates).fillna(0).to_numpy()
    spy = R["SPY"].shift(-1).reindex(dates).fillna(0).to_numpy()
    info = {"cls": cls, "eu": eu, "n_us": n_us, "n_eu": n_eu}
    return DailyData(dates, tick, X, mask, Rn, spy, rf, FEATURES_V2), info

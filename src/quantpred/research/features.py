"""Panel mensual de features + objetivo para el estudio (Fase 3).

Todo se calcula con información disponible en el cierre del último día hábil de cada mes (t):
- Features por ETF a partir de precios/volumen diarios (momentum, volatilidad, tendencia, riesgo,
  liquidez) + versión en rango cross-sectional.
- Features macro comunes a todos los ETFs, desplazadas por su retraso de publicación.
- Objetivo: retorno total del mes siguiente (t -> t+1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantpred.config import META, PROCESSED
from quantpred.ingest.store import load_metadata, load_prices

EXCLUDE_CLASSES = {"leveraged_inverse", "volatility"}
ETF_FEATURES = ["r1m", "r3m", "r6m", "r12m", "r12_1", "vol1m", "vol3m", "vol12m", "mom_radj",
                "ma200", "dd12m", "skew3m", "beta36", "corr36", "amihud", "logdv", "age"]


def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().values)


def load_daily(region: str = "us", start: str = "1997-01-01") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = load_metadata(region)
    ok = meta[~meta["asset_class"].isin(EXCLUDE_CLASSES) & ~meta["is_etn"].fillna(False).astype(bool)]
    # prefiltro de memoria: solo ETFs con volumen $ mediano histórico >= 300k (el filtro dinámico real
    # de liquidez se aplica después, mes a mes, sin look-ahead sobre el universo final)
    qr = pd.read_csv(META / f"quality_report{'' if region == 'us' else '_' + region}.csv")
    liquid = set(qr.loc[qr["median_dollar_vol"] >= 3e5, "ticker"])
    ok = ok[ok["yahoo"].isin(liquid)]
    px = load_prices(columns=["date", "ticker", "adj_close", "close", "volume"], region=region,
                     tickers=ok["yahoo"].tolist())
    px = px[px["date"] >= start]
    P = px.pivot(index="date", columns="ticker", values="adj_close").sort_index().astype("float32")
    DV = (px["close"] * px["volume"]).astype("float32").set_axis(pd.MultiIndex.from_frame(px[["date", "ticker"]])) \
        .unstack("ticker").reindex(P.index)
    return P, DV, ok.set_index("yahoo")


def etf_features(P: pd.DataFrame, DV: pd.DataFrame, bench: str = "SPY") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (features largo indexado (date,ticker), retornos mensuales anchos)."""
    me = _month_ends(P.index)
    R = P.pct_change(fill_method=None)
    f = {}
    Pm = P.reindex(me)
    f["r1m"] = P / P.shift(21) - 1
    f["r3m"] = P / P.shift(63) - 1
    f["r6m"] = P / P.shift(126) - 1
    f["r12m"] = P / P.shift(252) - 1
    f["r12_1"] = P.shift(21) / P.shift(252) - 1
    f["vol1m"] = R.rolling(21, min_periods=15).std() * np.sqrt(252)
    f["vol3m"] = R.rolling(63, min_periods=40).std() * np.sqrt(252)
    f["vol12m"] = R.rolling(252, min_periods=180).std() * np.sqrt(252)
    f["mom_radj"] = f["r12_1"] / f["vol12m"]
    f["ma200"] = P / P.rolling(200, min_periods=150).mean() - 1
    f["dd12m"] = P / P.rolling(252, min_periods=180).max() - 1
    f["skew3m"] = R.rolling(63, min_periods=40).skew()
    f["amihud"] = np.log1p((R.abs() / DV.replace(0, np.nan)).rolling(21, min_periods=10).mean() * 1e9)
    f["logdv"] = np.log1p(DV.rolling(63, min_periods=20).median())
    f["age"] = P.notna().cumsum() / 252
    feats = {k: v.reindex(me) for k, v in f.items()}
    # beta/corr a 36 meses con retornos mensuales
    Rm = Pm.pct_change(fill_method=None)
    Rm.index.name = "date"
    b = Rm[bench] if bench in Rm else Rm.mean(axis=1)
    cov = Rm.rolling(36, min_periods=24).cov(b)
    var = b.rolling(36, min_periods=24).var()
    feats["beta36"] = cov.div(var, axis=0)
    feats["corr36"] = Rm.rolling(36, min_periods=24).corr(b)
    long = pd.concat({k: v.stack() for k, v in feats.items()}, axis=1)
    long.index.names = ["date", "ticker"]
    return long, Rm


def _series_asof(dates: pd.DatetimeIndex, s: pd.Series, lag_days: int) -> pd.Series:
    s = s.dropna().sort_index()
    s.index = s.index + pd.Timedelta(days=lag_days)
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(dates)).ffill().reindex(dates)


def macro_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Features macro/estado de mercado en cada fecha t (con lag de publicación)."""
    out = {}
    fred = pd.read_parquet(PROCESSED / "macro_fred.parquet")
    F = {k: g.set_index("date")["value"] for k, g in fred.groupby("series_id")}
    lag = fred.groupby("series_id")["pub_lag_days"].first().to_dict()
    for sid in ["T10Y3M", "T10Y2Y", "DGS10", "DFF", "NFCI", "UNRATE", "ICSA", "T5YIE", "USEPUINDXD"]:
        if sid in F:
            out[sid.lower()] = _series_asof(dates, F[sid], lag[sid])
    if "DBAA" in F and "DAAA" in F:
        out["baa_aaa"] = _series_asof(dates, (F["DBAA"] - F["DAAA"]).dropna(), 1)
    out["dgs10_chg3m"] = out["dgs10"] - _series_asof(dates - pd.DateOffset(months=3), F["DGS10"], 1).values
    if "CPIAUCSL" in F:
        cpi = F["CPIAUCSL"]
        out["infl_yoy"] = _series_asof(dates, cpi.pct_change(12, fill_method=None), 35)
    if "UNRATE" in F:
        u = F["UNRATE"]
        out["unrate_chg12m"] = _series_asof(dates, u - u.shift(12), 35)

    alt = PROCESSED / "alt"
    cb = pd.read_parquet(alt / "cboe.parquet")
    C = {k: g.set_index("date")["value"] for k, g in cb.groupby("series", observed=True)}
    if "cboe_vix" in C:
        out["vix"] = _series_asof(dates, C["cboe_vix"], 0)
    if "cboe_vix3m" in C:
        out["vix_term"] = _series_asof(dates, (C["cboe_vix"] / C["cboe_vix3m"]).dropna(), 0)
    aa = pd.read_parquet(alt / "aaii.parquet")
    bb = aa[aa["series"] == "aaii_bull_bear"].set_index("date")["value"]
    out["aaii_bb"] = _series_asof(dates, bb.rolling(4, min_periods=1).mean(), 1)
    sh = pd.read_parquet(alt / "shiller.parquet")
    cape = sh[sh["series"] == "shiller_cape"].set_index("date")["value"]
    out["cape"] = _series_asof(dates, cape, 35)

    ym = pd.read_parquet(PROCESSED / "macro_yahoo.parquet") if (PROCESSED / "macro_yahoo.parquet").exists() else None
    if ym is not None:
        Y = ym.pivot(index="date", columns="ticker", values="adj_close").sort_index().ffill()
        for t, name in [("^GSPC", "spx"), ("DX-Y.NYB", "dxy"), ("CL=F", "oil"), ("GC=F", "gold"), ("HG=F", "copper")]:
            if t in Y:
                s = Y[t]
                out[f"{name}_r3m"] = _series_asof(dates, s / s.shift(63) - 1, 0)
                if name == "spx":
                    out["spx_r12m"] = _series_asof(dates, s / s.shift(252) - 1, 0)
                    out["spx_vol1m"] = _series_asof(dates, s.pct_change().rolling(21).std() * np.sqrt(252), 0)
                    out["spx_ma200"] = _series_asof(dates, s / s.rolling(200).mean() - 1, 0)
        if "HG=F" in Y and "GC=F" in Y:
            cg = (Y["HG=F"] / Y["GC=F"]).dropna()
            out["copper_gold_r3m"] = _series_asof(dates, cg / cg.shift(63) - 1, 0)
    return pd.DataFrame(out, index=dates).add_prefix("m_")


def risk_free_monthly(dates: pd.DatetimeIndex) -> pd.Series:
    fred = pd.read_parquet(PROCESSED / "macro_fred.parquet")
    tb = fred[fred["series_id"] == "DTB3"].set_index("date")["value"]
    return (_series_asof(dates, tb, 1) / 100 / 12).fillna(0)


def build_panel(region: str = "us", min_dollar_vol: float = 1e6, top_n: int = 400,
                min_age_years: float = 1.0, cache: bool = True) -> dict:
    f = PROCESSED / f"study_panel_{region}.parquet"
    if cache and f.exists():
        panel = pd.read_parquet(f)
        rets = pd.read_parquet(PROCESSED / f"study_monthly_returns_{region}.parquet")
        daily = pd.read_parquet(PROCESSED / f"study_daily_returns_{region}.parquet")
        return {"panel": panel, "rets": rets, "daily": daily}
    P, DV, meta = load_daily(region)
    feats, Rm = etf_features(P, DV)
    dates = feats.index.get_level_values(0).unique().sort_values()
    # objetivo: retorno del mes siguiente
    fwd = Rm.shift(-1).stack(future_stack=True).rename("fwd_ret")
    fwd.index.names = ["date", "ticker"]
    panel = feats.join(fwd, how="left")
    # universo dinámico: líquido, con historia, top-N por volumen $
    dv = np.expm1(panel["logdv"])
    panel = panel[(dv >= min_dollar_vol) & (panel["age"] >= min_age_years) & panel["r12_1"].notna()]
    panel["dv_rank"] = panel.groupby(level=0)["logdv"].rank(ascending=False)
    panel = panel[panel["dv_rank"] <= top_n].drop(columns="dv_rank")
    # rangos cross-sectional (0-1) de cada feature
    ranks = panel[ETF_FEATURES].groupby(level=0).rank(pct=True).add_suffix("_cs")
    panel = panel.join(ranks)
    # objetivo relativo: exceso sobre la mediana cross-sectional del mes
    panel["fwd_rel"] = panel["fwd_ret"] - panel.groupby(level=0)["fwd_ret"].transform("median")
    # clase de activo
    ac = panel.index.get_level_values(1).map(meta["asset_class"])
    panel["asset_class"] = pd.Categorical(ac)
    macro = macro_features(dates)
    panel = panel.join(macro, on="date")
    rf = risk_free_monthly(dates)
    panel = panel.join(rf.rename("rf"), on="date")
    tick = panel.index.get_level_values(1).unique()
    daily = P[tick].pct_change(fill_method=None).astype("float32")
    if cache:
        panel.to_parquet(f)
        Rm.to_parquet(PROCESSED / f"study_monthly_returns_{region}.parquet")
        daily.to_parquet(PROCESSED / f"study_daily_returns_{region}.parquet")
    return {"panel": panel, "rets": Rm, "daily": daily}

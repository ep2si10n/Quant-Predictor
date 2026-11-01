"""Fase 4b — timing de volatilidad y de régimen (predecir riesgo, no ganadores).

Motivación (F3/F4): la selección de ETFs no mostró ventaja robusta. La literatura indica que el riesgo
es mucho más predecible que el retorno:
- Volatilidad: persistente y de memoria larga → HAR (Corsi 2009) explica ~50-70% de la varianza de la
  volatilidad futura. Carteras gestionadas por volatilidad mejoran el Sharpe (Moreira & Muir 2017).
- Régimen: estrés de crédito, curva invertida, VIX en backwardation, tendencia negativa y
  posicionamiento predicen meses de cola izquierda mejor que la media de retorno.

Todo a frecuencia mensual con información disponible en t (lags de publicación aplicados).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantpred.config import PROCESSED
from quantpred.research.features import _series_asof

ALT = PROCESSED / "alt"
CFTC_CODES = {"13874A": "spx", "043602": "ust10", "088691": "gold", "067651": "wti", "098662": "usd"}


def spy_daily() -> pd.Series:
    d = pd.read_parquet(PROCESSED / "study_daily_returns_us.parquet", columns=["SPY"])["SPY"].astype("float64")
    return d.dropna()


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().values)


def har_features(r: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Componentes HAR de la volatilidad realizada (anualizada, log): 1 semana, 1 mes, 3 meses, 12 meses,
    más semivarianza negativa (Patton & Sheppard 2015) y retorno 1m (efecto apalancamiento)."""
    rv = r.pow(2)
    f = pd.DataFrame({
        "lrv_w": np.log(np.sqrt(rv.rolling(5).mean() * 252)),
        "lrv_m": np.log(np.sqrt(rv.rolling(21).mean() * 252)),
        "lrv_q": np.log(np.sqrt(rv.rolling(63).mean() * 252)),
        "lrv_y": np.log(np.sqrt(rv.rolling(252).mean() * 252)),
        "lrsv_neg_m": np.log(np.sqrt((r.clip(upper=0) ** 2).rolling(21).mean() * 252 * 2) + 1e-4),
        "spy_r1m": (1 + r).rolling(21).apply(np.prod, raw=True) - 1,
        "spy_r12m": (1 + r).rolling(252).apply(np.prod, raw=True) - 1,
        "spy_dd": (1 + r).cumprod() / (1 + r).cumprod().rolling(252).max() - 1,
    })
    return f.reindex(dates)


def targets(r: pd.Series, dates: pd.DatetimeIndex, rf_m: pd.Series) -> pd.DataFrame:
    """Objetivos del mes siguiente: log-vol realizada, retorno en exceso, mes malo (exceso < -3%)."""
    ym = r.groupby([r.index.year, r.index.month])
    vol = ym.apply(lambda x: np.log(np.sqrt((x ** 2).mean() * 252)))
    ret = ym.apply(lambda x: (1 + x).prod() - 1)
    idx = pd.DatetimeIndex([dates[(dates.year == y) & (dates.month == m)][0] for y, m in vol.index
                            if ((dates.year == y) & (dates.month == m)).any()])
    vol = pd.Series(vol.values[:len(idx)], index=idx)
    ret = pd.Series(ret.values[:len(idx)], index=idx)
    out = pd.DataFrame({"y_lvol": vol.shift(-1), "y_ret": ret.shift(-1)}).reindex(dates)
    out["y_exc"] = out["y_ret"] - rf_m.reindex(dates).fillna(0)
    out["y_bad"] = (out["y_exc"] < -0.03).astype(float).where(out["y_exc"].notna())
    return out


def _alt_series(file: str, name: str) -> pd.Series:
    a = pd.read_parquet(ALT / file)
    return a[a["series"].astype(str) == name].set_index("date")["value"].sort_index()


def state_features(dates: pd.DatetimeIndex, macro: pd.DataFrame) -> pd.DataFrame:
    """Variables de estado de mercado/macro/alternativas (además de las m_* del panel)."""
    out = {}
    cb = pd.read_parquet(ALT / "cboe.parquet")
    C = {k: g.set_index("date")["value"].sort_index() for k, g in cb.groupby("series", observed=True)}
    for k in ["cboe_vvix", "cboe_skew", "cboe_vix9d", "cboe_vix6m"]:
        if k in C:
            out[k] = _series_asof(dates, C[k], 0)
    if "cboe_vix9d" in C:
        out["vix9d_vix"] = _series_asof(dates, (C["cboe_vix9d"] / C["cboe_vix"]).dropna(), 0)
    if "cboe_pc_equity" in C:
        out["pc_equity_1m"] = _series_asof(dates, C["cboe_pc_equity"].rolling(21, min_periods=5).mean(), 0)
    # prima de riesgo de varianza (Bollerslev, Tauchen & Zhou 2009): VIX² - RV²
    # (se calcula en build con lrv_m)
    # CFTC: posición neta no comercial % OI y su z-score a 3 años (lag publicación 3 días)
    c = pd.read_parquet(ALT / "cftc_legacy.parquet")
    for code, nm in CFTC_CODES.items():
        d = c[c["code"] == code].set_index("date").sort_index()
        if d.empty:
            continue
        net = (d["noncomm_long"] - d["noncomm_short"]) / d["open_interest"]
        z = (net - net.rolling(156, min_periods=52).mean()) / net.rolling(156, min_periods=52).std()
        out[f"cot_{nm}_net"] = _series_asof(dates, net, 3)
        out[f"cot_{nm}_z"] = _series_asof(dates, z, 3)
    # AQR: momentum multi-activo y macro (mensual, publicado ~1 mes después)
    for nm, key in [("aqr_mom_all", "aqr_century_century_of_factor_premia_all_asset_classes_momentum"),
                    ("aqr_macro_mom", "aqr_century_century_of_factor_premia_all_macro_momentum")]:
        s = _alt_series("aqr.parquet", key)
        if len(s):
            out[nm + "_12m"] = _series_asof(dates, s.rolling(12, min_periods=6).sum(), 35)
    bab = _alt_series("aqr.parquet", "aqr_bab_d_bab_factors_usa")
    if len(bab):
        out["bab_usa_3m"] = _series_asof(dates, bab.rolling(63, min_periods=40).sum(), 35)
    # riesgo geopolítico, sentimiento prensa, actividad, prima por plazo, GSCPI
    for f, key, lag, nm in [("gpr.parquet", "gpr_gprd_ma30", 1, "gpr30"),
                            ("sf_news_sentiment.parquet", "sf_news_sentiment", 7, "news_sent"),
                            ("ads_index.parquet", "ads_index", 7, "ads"),
                            ("acm_term_premium.parquet", "acmtp10", 7, "acm_tp10"),
                            ("gscpi.parquet", "gscpi", 35, "gscpi")]:
        s = _alt_series(f, key)
        if len(s):
            out[nm] = _series_asof(dates, s, lag)
    # Goyal-Welch: d/p, e/p, tms, dfy, svar, ntis (mensual, lag 45 días)
    for key in ["gw_d_p", "gw_e_p", "gw_dfy", "gw_svar", "gw_ntis", "gw_b_m"]:
        s = _alt_series("goyal_welch.parquet", key)
        if len(s):
            out[key] = _series_asof(dates, s, 45)
    return pd.DataFrame(out, index=dates)


def build_regime_frame(panel: pd.DataFrame) -> pd.DataFrame:
    r = spy_daily()
    dates = month_ends(r.index)
    rf = panel.groupby(level=0)["rf"].first().reindex(dates).ffill().fillna(0)
    macro = panel.groupby(level=0)[[c for c in panel.columns if c.startswith("m_")]].first().reindex(dates)
    X = pd.concat([har_features(r, dates), macro, state_features(dates, macro)], axis=1)
    if "m_vix" in X:
        X["vrp"] = (X["m_vix"] / 100) ** 2 - np.exp(X["lrv_m"]) ** 2
    Y = targets(r, dates, rf * 1.0)
    df = pd.concat([X, Y], axis=1)
    df["rf"] = rf
    return df.loc["1994-01-01":]


def walk_forward(df: pd.DataFrame, cols: list, target: str, make_model, first_year: int = 2005,
                 classifier: bool = False) -> pd.Series:
    """Entrena con t < dic(Y-1) (la etiqueta de t es el mes t+1 → embargo de 1 mes) y predice el año Y."""
    preds = []
    for Y in range(first_year, df.index.max().year + 1):
        cut = pd.Timestamp(f"{Y - 1}-12-01")
        tr = df[(df.index < cut)].dropna(subset=[target])
        te = df[df.index.year == Y]
        if len(tr) < 60 or te.empty:
            continue
        m = make_model()
        m.fit(tr[cols], tr[target])
        p = m.predict_proba(te[cols])[:, 1] if classifier else m.predict(te[cols])
        preds.append(pd.Series(p, index=te.index))
    return pd.concat(preds)


def expanding_quantile(p: pd.Series, q: float, min_obs: int = 24) -> pd.Series:
    """Umbral sin look-ahead: cuantil de las predicciones OOS pasadas (excluye la actual)."""
    return p.expanding(min_obs).quantile(q).shift(1)

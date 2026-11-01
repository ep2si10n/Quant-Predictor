"""Motor de backtest mensual con costes y estrategias (reglas + ML walk-forward)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from quantpred.research.features import ETF_FEATURES
from quantpred.research.portfolio import allocate

COST_BPS = 5.0
BORROW_SPREAD = 0.005   # 50 pb/año sobre rf al apalancarse
MAX_LEVERAGE = 2.0      # con vol-target; la financiación se cobra a rf (cash negativo)
CORE = ["SPY", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "TIP", "GLD", "DBC", "VNQ"]


@dataclass
class Result:
    name: str
    returns: pd.Series                     # retorno neto mensual (fecha = fin del mes de tenencia t)
    weights: dict = field(default_factory=dict)
    turnover: pd.Series | None = None
    ic: pd.Series | None = None            # rank-IC de las predicciones (estrategias ML)


def run(name: str, panel: pd.DataFrame, daily: pd.DataFrame,
        selector: Callable[[pd.Timestamp, pd.DataFrame], tuple[list, pd.Series | None]],
        scheme: str = "ivol", start: str = "2008-01-01", end: str | None = None,
        cost_bps: float = COST_BPS, lookback_days: int = 252, vol_target: float | None = None) -> Result:
    """selector(t, cross_section) -> (tickers, pesos opcionales). Si no hay pesos, se usa `scheme`."""
    dates = panel.index.get_level_values(0).unique().sort_values()
    dates = dates[(dates >= start) & ((dates <= end) if end else True)]
    prev = pd.Series(dtype=float)
    rets, tos, W = {}, {}, {}
    for t in dates:
        cs = panel.xs(t, level=0)
        if cs["fwd_ret"].notna().sum() == 0:          # último mes sin retorno futuro conocido
            continue
        tick, w = selector(t, cs)
        tick = [x for x in tick if x in cs.index]
        if not tick:
            w = pd.Series(dtype=float)
        elif w is None:
            hist = daily.loc[:t].iloc[-lookback_days:]
            w = allocate(scheme, tick, hist)
        w = w[w > 0]
        if vol_target and len(w):
            h = daily.loc[:t].iloc[-63:][w.index].fillna(0)
            pv = float(np.sqrt(w @ h.cov().values @ w) * np.sqrt(252))
            w = w * min(MAX_LEVERAGE, vol_target / pv) if pv > 0 else w
        fr = cs["fwd_ret"].reindex(w.index).fillna(0.0)
        rf = float(cs["rf"].iloc[0]) if "rf" in cs else 0.0
        cash = 1 - w.sum()
        fin = rf if cash >= 0 else rf + BORROW_SPREAD / 12      # apalancamiento paga rf + spread
        gross = float((w * fr).sum() + cash * fin)
        to = float(w.subtract(prev, fill_value=0).abs().sum())
        nxt = t + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(1)   # mes de tenencia
        rets[nxt] = gross - to * cost_bps / 1e4
        tos[nxt] = to
        W[t] = w
        # deriva de pesos tras el mes (para turnover realista)
        prev = (w * (1 + fr)) / (1 + gross) if len(w) else w
    return Result(name, pd.Series(rets).sort_index(), W, pd.Series(tos).sort_index())


# ------------------------------- selectores basados en reglas ------------------------------------
def sel_fixed(weights: dict) -> Callable:
    def f(t, cs):
        avail = {k: v for k, v in weights.items() if k in cs.index}
        if not avail:
            return [], None
        w = pd.Series(avail)
        return list(w.index), w / w.sum()
    return f


def sel_all(t, cs):
    return list(cs.index), None


def sel_top(col: str, k: int = 20, ascending: bool = False, require_positive: str | None = None,
            max_per_class: float | None = None) -> Callable:
    def f(t, cs):
        c = cs.dropna(subset=[col])
        if require_positive:
            c = c[c[require_positive] > c["rf"] * 12]
        c = c.sort_values(col, ascending=ascending)
        return _cap_classes(c, k, max_per_class), None
    return f


def _cap_classes(c: pd.DataFrame, k: int, max_per_class: float | None) -> list:
    if not max_per_class:
        return list(c.index[:k])
    cap = max(1, int(np.ceil(k * max_per_class)))
    out, cnt = [], {}
    for tk, ac in zip(c.index, c["asset_class"].astype(str)):
        if cnt.get(ac, 0) < cap:
            out.append(tk)
            cnt[ac] = cnt.get(ac, 0) + 1
        if len(out) == k:
            break
    return out


def sel_core(t, cs):
    return [x for x in CORE if x in cs.index], None


def sel_core_tsmom(t, cs):
    """TSMOM sobre núcleo cross-asset: mantener activos con r12m > rf (Moskowitz-Ooi-Pedersen)."""
    c = cs.reindex([x for x in CORE if x in cs.index])
    return list(c.index[c["r12m"] > c["rf"] * 12]), None


def sel_dual_momentum(t, cs):
    """Global Equity Momentum (Antonacci 2014): SPY vs EFA vs AGG/IEF."""
    bond = "AGG" if "AGG" in cs.index else "IEF"
    if "SPY" not in cs.index:
        return [], None
    if cs.at["SPY", "r12m"] > cs.at["SPY", "rf"] * 12:
        best = "SPY" if "EFA" not in cs.index or cs.at["SPY", "r12m"] >= cs.at["EFA", "r12m"] else "EFA"
        return [best], pd.Series({best: 1.0})
    return [bond], pd.Series({bond: 1.0})


# ------------------------------------ ML walk-forward ------------------------------------------
def feature_cols(panel: pd.DataFrame, macro: bool = True) -> list:
    cols = ETF_FEATURES + [c + "_cs" for c in ETF_FEATURES]
    if macro:
        cols += [c for c in panel.columns if c.startswith("m_")]
    return cols


def walk_forward_predict(panel: pd.DataFrame, make_model: Callable, cols: list, target: str = "fwd_rel",
                         first_year: int = 2008, min_train_year: int = 2000, refit_every: int = 1,
                         with_class: bool = True) -> pd.Series:
    """Predicciones OOS: para cada año Y entrena con fechas t < Y-1-dic (embargo 1 mes: la etiqueta de
    t usa el retorno t->t+1) y predice todos los meses de Y."""
    dates = panel.index.get_level_values(0)
    years = sorted(set(dates.year[dates.year >= first_year]))
    X = panel[cols].astype("float32")
    if with_class:
        X = X.join(pd.get_dummies(panel["asset_class"], prefix="ac", dtype="float32"))
    y = panel[target]
    preds = []
    model = None
    for i, Y in enumerate(years):
        cut = pd.Timestamp(f"{Y - 1}-12-01")          # última etiqueta conocida: nov->dic de Y-1
        tr = (dates < cut) & (dates.year >= min_train_year) & y.notna().values
        te = dates.year == Y
        if tr.sum() < 1000 or te.sum() == 0:
            continue
        if model is None or i % refit_every == 0:
            model = make_model()
            model.fit(X[tr], y[tr])
        preds.append(pd.Series(model.predict(X[te]), index=panel.index[te]))
    return pd.concat(preds).rename("pred")


def sel_pred(pred: pd.Series, k: int = 20, max_per_class: float | None = None) -> Callable:
    def f(t, cs):
        if t not in pred.index.get_level_values(0):
            return [], None
        p = pred.xs(t, level=0).reindex(cs.index).dropna()
        c = cs.loc[p.index].assign(_p=p).sort_values("_p", ascending=False)
        return _cap_classes(c, k, max_per_class), None
    return f


def ic_series(pred: pd.Series, panel: pd.DataFrame, target: str = "fwd_ret") -> pd.Series:
    from quantpred.research.metrics import rank_ic
    df = pd.concat([pred, panel[target]], axis=1, join="inner")
    return df.groupby(level=0).apply(lambda d: rank_ic(d["pred"], d[target]))

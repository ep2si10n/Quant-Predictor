"""Fase 4 — bloques para iterar estrategias tras el análisis de fallos de la Fase 3.

Hallazgo F3: el IC positivo del ML venía de los deciles bajos (activos de beta baja). El modelo aprendía
"beta alta ⇒ más retorno" (prima de mercado de la muestra) en vez de selección; el top-20 concentraba
renta variable de alta beta (MaxDD -45% en 2008) y rotaba 15x/año.

Respuestas implementadas:
- objetivos alternativos: ajustado por riesgo y/o neutral a la clase de activo
- asignación en dos niveles: ERC entre clases (sleeves) + selección ML dentro de cada clase
- filtro de tendencia por clase (TSMOM sobre proxies) y vol-targeting
- suavizado de predicciones (media de los últimos n meses) y ensembles por rango
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantpred.research.portfolio import inverse_vol, risk_parity

MIN_VOL = 0.03   # excluye equivalentes de liquidez (T-bills/ultracorto): la liquidez ya es la rf
SLEEVES = ["equity", "fixed_income", "commodity", "real_estate"]
TREND_PROXY = {"equity": "SPY", "fixed_income": "IEF", "commodity": "DBC", "real_estate": "VNQ"}


def add_targets(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    vol = p["vol3m"].clip(lower=0.02) / np.sqrt(12)             # vol mensual esperada
    cls_med = p.groupby([p.index.get_level_values(0), p["asset_class"]], observed=True)["fwd_ret"] \
        .transform("median")
    p["t_rel"] = p["fwd_rel"]
    p["t_sharpe"] = ((p["fwd_ret"] - p["rf"]) / vol).clip(-4, 4)
    p["t_cls"] = (p["fwd_ret"] - cls_med).clip(-0.3, 0.3)
    p["t_cls_sharpe"] = ((p["fwd_ret"] - cls_med) / vol).clip(-4, 4)
    # rango cross-sectional del objetivo dentro de cada clase (robusto a outliers)
    p["t_cls_rank"] = p.groupby([p.index.get_level_values(0), p["asset_class"]], observed=True)["fwd_ret"] \
        .rank(pct=True) - 0.5
    return p


def smooth(pred: pd.Series, n: int) -> pd.Series:
    if n <= 1:
        return pred
    w = pred.unstack("ticker").rolling(n, min_periods=1).mean()
    out = w.stack()
    out.index.names = pred.index.names
    return out.reindex(pred.index)


def rank_ensemble(preds: list[pd.Series]) -> pd.Series:
    r = [p.groupby(level=0).rank(pct=True) for p in preds]
    return pd.concat(r, axis=1).mean(axis=1).rename("pred")


def sel_two_level(pred: pd.Series | None, daily: pd.DataFrame, k_per: int = 5, trend: bool = False,
                  sleeve_scheme: str = "erc", score_col: str | None = None, lookback: int = 252,
                  sleeves=SLEEVES):
    """Dentro de cada clase: top-k por predicción (o por `score_col` si no hay pred), pesos ivol.
    Entre clases: ERC (o ivol/ew) sobre el retorno histórico de cada sleeve. Con `trend`, una clase
    cuyo proxy tiene r12m < rf pasa a liquidez."""
    def f(t, cs):
        hist = daily.loc[:t].iloc[-lookback:]
        if pred is not None:
            if t not in pred.index.get_level_values(0):
                return [], None
            score = pred.xs(t, level=0).reindex(cs.index)
        else:
            score = cs[score_col]
        sleeve_w, sleeve_ret = {}, {}
        for s in sleeves:
            c = cs[(cs["asset_class"].astype(str) == s) & (cs["vol12m"] >= MIN_VOL)] \
                .assign(_s=score).dropna(subset=["_s"])
            if c.empty:
                continue
            pick = list(c.sort_values("_s", ascending=False).index[:k_per])
            h = hist[pick].fillna(0)
            w = inverse_vol(h) if len(pick) > 1 else pd.Series(1.0, index=pick)
            sleeve_w[s] = w
            sleeve_ret[s] = h @ w
        if not sleeve_w:
            return [], None
        SR = pd.DataFrame(sleeve_ret)
        if sleeve_scheme == "erc" and SR.shape[1] > 1:
            a = risk_parity(SR)
        elif sleeve_scheme == "ivol" and SR.shape[1] > 1:
            a = inverse_vol(SR)
        else:
            a = pd.Series(1 / SR.shape[1], index=SR.columns)
        if trend:
            for s in list(a.index):
                px = TREND_PROXY.get(s)
                if px in cs.index and cs.at[px, "r12m"] < cs.at[px, "rf"] * 12:
                    a[s] = 0.0                                   # a liquidez (rf)
        out = pd.concat([sleeve_w[s] * a[s] for s in a.index if a[s] > 0]) if a.sum() > 0 else pd.Series(dtype=float)
        out = out.groupby(level=0).sum()
        return list(out.index), out
    return f

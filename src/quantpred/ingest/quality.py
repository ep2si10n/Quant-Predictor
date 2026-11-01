"""Control de calidad de precios: limpieza conservadora + flags + informe por ticker.

Reglas (no se interpola nada; solo se eliminan filas imposibles y se marcan anomalías):
  - elimina fechas duplicadas y filas con precios <= 0 o NaN en adj_close
  - corrige high/low incoherentes (high < max(open,close) ...) acotándolos
  - flag `ret_outlier`: |retorno ajustado diario| > 40% (posible mal dato / split no recogido)
  - flag `stale`: precio idéntico al anterior con volumen 0 (sin negociación)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OUTLIER = 0.40


def clean_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    n0 = df.groupby("ticker").size()
    px = ["open", "high", "low", "close", "adj_close"]
    df = df[df["adj_close"].notna() & (df["adj_close"] > 0) & (df["close"] > 0)].copy()
    for c in ["open", "high", "low"]:
        df.loc[~(df[c] > 0), c] = np.nan
    oc_max = df[["open", "close"]].max(axis=1)
    oc_min = df[["open", "close"]].min(axis=1)
    df["high"] = np.fmax(df["high"], oc_max)
    df["low"] = np.fmin(df["low"], oc_min)

    g = df.groupby("ticker", sort=False)
    ret = g["adj_close"].pct_change()
    df["ret_outlier"] = ret.abs() > OUTLIER
    df["stale"] = (g["close"].diff() == 0) & (df["volume"].fillna(0) == 0)

    for c in px:
        df[c] = df[c].astype("float64")
    df["volume"] = df["volume"].fillna(0).astype("int64")
    for c in ["dividends", "splits", "capital_gains"]:
        df[c] = df[c].fillna(0).astype("float32")

    g = df.groupby("ticker", sort=False)
    rep = pd.DataFrame({
        "first_date": g["date"].min(),
        "last_date": g["date"].max(),
        "n_rows_raw": n0,
        "n_rows": g.size(),
        "n_outliers": g["ret_outlier"].sum(),
        "pct_stale": g["stale"].mean().round(4),
        "median_dollar_vol": (df["close"] * df["volume"]).groupby(df["ticker"]).median().round(0),
        "n_dividends": (df["dividends"] > 0).groupby(df["ticker"]).sum(),
        "n_splits": (df["splits"] > 0).groupby(df["ticker"]).sum(),
    })
    rep["years"] = ((rep["last_date"] - rep["first_date"]).dt.days / 365.25).round(2)
    rep = rep.dropna(subset=["first_date"]).reset_index().rename(columns={"index": "ticker"})
    return df.reset_index(drop=True), rep

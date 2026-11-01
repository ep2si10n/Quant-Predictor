"""Escritura/lectura del dataset procesado.

Layout:
  data/processed/etf_prices/asset_class=<clase>/part-XX.parquet   (cada fichero < MAX_PARQUET_MB)
  data/processed/macro_yahoo.parquet
  data/processed/macro_fred.parquet
  data/meta/etf_metadata.parquet, data/meta/quality_report.csv
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quantpred.config import MAX_PARQUET_MB, META, PROCESSED

PRICES_DIR = PROCESSED / "etf_prices"        # EEUU (USD)
REGION_DIRS = {"us": PRICES_DIR, "eu": PROCESSED / "etf_prices_eu"}  # eu = Xetra UCITS (EUR mayoritario)


def _write(df: pd.DataFrame, path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path,
                   compression="zstd", compression_level=9)
    return path.stat().st_size / 2**20


def write_prices(prices: pd.DataFrame, meta: pd.DataFrame, region: str = "us") -> pd.DataFrame:
    PRICES_DIR = REGION_DIRS[region]
    cls = meta.set_index("yahoo")["asset_class"]
    prices = prices.assign(asset_class=prices["ticker"].map(cls).fillna("equity"))
    if PRICES_DIR.exists():
        shutil.rmtree(PRICES_DIR)
    summary = []
    for ac, d in prices.groupby("asset_class"):
        d = d.drop(columns="asset_class").sort_values(["ticker", "date"])
        tickers = sorted(d["ticker"].unique())
        nparts = 1
        while True:
            chunk = math.ceil(len(tickers) / nparts)
            sizes = []
            for p in range(nparts):
                tk = set(tickers[p * chunk:(p + 1) * chunk])
                sizes.append(_write(d[d["ticker"].isin(tk)], PRICES_DIR / f"asset_class={ac}" / f"part-{p:02d}.parquet"))
            if max(sizes) <= MAX_PARQUET_MB:
                break
            shutil.rmtree(PRICES_DIR / f"asset_class={ac}")
            nparts += 1
        summary.append({"asset_class": ac, "tickers": len(tickers), "rows": len(d), "parts": nparts,
                        "mb": round(sum(sizes), 1)})
    return pd.DataFrame(summary)


def load_prices(asset_classes: list[str] | None = None, tickers: list[str] | None = None,
                columns: list[str] | None = None, region: str = "us") -> pd.DataFrame:
    """Carga precios en formato largo. region: 'us', 'eu' o 'all'. Filtra por clase y/o tickers."""
    if region == "all":
        return pd.concat([load_prices(asset_classes, tickers, columns, r) for r in REGION_DIRS],
                         ignore_index=True)
    PRICES_DIR = REGION_DIRS[region]
    filters = []
    if asset_classes:
        filters.append(("asset_class", "in", asset_classes))
    if tickers:
        filters.append(("ticker", "in", tickers))
    df = pd.read_parquet(PRICES_DIR, columns=columns, filters=filters or None)
    if "asset_class" in df:
        df["asset_class"] = df["asset_class"].astype(str)
    return df


def load_panel(field: str = "adj_close", **kw) -> pd.DataFrame:
    """Matriz fecha x ticker de un campo (por defecto adj_close)."""
    df = load_prices(columns=["date", "ticker", field], **kw)
    return df.pivot(index="date", columns="ticker", values=field).sort_index()


def load_metadata(region: str = "us") -> pd.DataFrame:
    if region == "all":
        return pd.concat([load_metadata(r) for r in REGION_DIRS], ignore_index=True)
    return pd.read_parquet(META / ("etf_metadata.parquet" if region == "us" else f"etf_metadata_{region}.parquet"))


def load_macro(source: str = "yahoo") -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"macro_{source}.parquet")

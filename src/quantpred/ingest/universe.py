"""Universo de ETFs/ETNs cotizados en EEUU (NASDAQ Trader symbol directory).

El fichero `nasdaqtraded.txt` lista TODOS los valores negociables en mercados de EEUU
(NASDAQ, NYSE, NYSE Arca, Cboe BZX, IEX...) con un flag `ETF`.
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from quantpred.config import META, NASDAQ_TRADED_URL

EXCHANGES = {"Q": "NASDAQ", "N": "NYSE", "P": "NYSE Arca", "Z": "Cboe BZX",
             "A": "NYSE American", "V": "IEX", "F": "TXSE"}


def yahoo_symbol(sym: str) -> str:
    """Convierte la notación CQS/NASDAQ a Yahoo (BRK.B -> BRK-B, '$' prefs, etc.)."""
    return sym.replace(".", "-").replace("$", "-P")


def fetch_universe(save: bool = True) -> pd.DataFrame:
    txt = requests.get(NASDAQ_TRADED_URL, timeout=60).text
    df = pd.read_csv(io.StringIO(txt), sep="|", dtype=str)
    df = df[~df["Nasdaq Traded"].str.startswith("File Creation", na=False)]
    etf = df[(df["ETF"] == "Y") & (df["Test Issue"] == "N")].copy()
    etf = etf.rename(columns={"Symbol": "symbol", "Security Name": "name",
                              "Listing Exchange": "exchange_code"})
    etf["exchange"] = etf["exchange_code"].map(EXCHANGES).fillna(etf["exchange_code"])
    etf["yahoo"] = etf["symbol"].map(yahoo_symbol)
    etf["name"] = etf["name"].str.strip()
    out = etf[["symbol", "yahoo", "name", "exchange"]].drop_duplicates("yahoo").sort_values("symbol")
    out = out.reset_index(drop=True)
    if save:
        META.mkdir(parents=True, exist_ok=True)
        out.to_csv(META / "universe.csv", index=False)
    return out


if __name__ == "__main__":
    u = fetch_universe()
    print(len(u), "ETFs/ETNs")
    print(u["exchange"].value_counts())

"""Universo europeo UCITS: todos los ETFs/ETCs/ETNs negociables en Xetra (Deutsche Börse).

Xetra es el mayor mercado de ETPs UCITS de Europa: cotizan prácticamente todas las gestoras
(iShares, Xtrackers, Amundi, Vanguard, SPDR, Invesco, WisdomTree, HANetf...). Precios en EUR.
Símbolo Yahoo = Mnemonic + ".DE". Se deduplica por ISIN.
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from quantpred.config import META

XETRA_CSV = ("https://www.cashmarket.deutsche-boerse.com/resource/blob/1528/"
             "d8efea12d8240b569a3cfba0b1a581b7/data/t7-xetr-allTradableInstruments.csv")
TYPES = {"ETF", "ETC", "ETN"}


def fetch_universe_eu(save: bool = True) -> pd.DataFrame:
    r = requests.get(XETRA_CSV, headers={"User-Agent": "Mozilla/5.0 (research; quant-predictor)"}, timeout=120)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text), sep=";", skiprows=2, low_memory=False, dtype=str)
    d = d[d["Instrument Type"].isin(TYPES) & (d["Instrument Status"] == "Active") & d["Mnemonic"].notna()]
    out = pd.DataFrame({
        "symbol": d["Mnemonic"].str.strip(),
        "yahoo": d["Mnemonic"].str.strip() + ".DE",
        "name": d["Instrument"].str.strip(),
        "isin": d["ISIN"].str.strip(),
        "instrument_type": d["Instrument Type"],
        "xetra_group": d["Product Assignment Group Description"].str.strip(),
        "currency": d.get("Currency", pd.Series("EUR", index=d.index)),
        "exchange": "Xetra",
    })
    out = out.drop_duplicates("isin").drop_duplicates("yahoo").sort_values("symbol").reset_index(drop=True)
    if save:
        META.mkdir(parents=True, exist_ok=True)
        out.to_csv(META / "universe_eu.csv", index=False)
    return out


# Preferencia de plazas si el mnemónico Xetra no existe en Yahoo (mismo ISIN = mismo fondo)
SUFFIX_PREF = [".DE", ".F", ".AS", ".PA", ".MI", ".L", ".SW", ".MC", ".BR", ".VI", ".ST", ".IR"]


def resolve_isin(isin: str) -> str | None:
    from quantpred.ingest.yahoo import _get_json
    try:
        js = _get_json("/v1/finance/search", {"q": isin, "quotesCount": 10, "newsCount": 0}, retries=3)
    except Exception:
        return None
    syms = [q["symbol"] for q in js.get("quotes", []) if q.get("symbol") and q.get("quoteType") in ("ETF", "MUTUALFUND", "EQUITY")]
    if not syms:
        return None
    rank = {sfx: i for i, sfx in enumerate(SUFFIX_PREF)}
    return sorted(syms, key=lambda x: rank.get("." + x.split(".")[-1] if "." in x else "", 99))[0]


def resolve_missing(universe: pd.DataFrame, missing: list[str]) -> pd.DataFrame:
    """Sustituye en el universo los símbolos sin datos por el mismo ISIN en otra plaza."""
    from concurrent.futures import ThreadPoolExecutor
    sub = universe[universe["yahoo"].isin(missing)]
    with ThreadPoolExecutor(2) as ex:
        alt = list(ex.map(resolve_isin, sub["isin"]))
    m = dict(zip(sub["yahoo"], alt))
    u = universe.copy()
    u["yahoo_xetra"] = u["yahoo"]
    u["yahoo"] = [m.get(y) or y for y in u["yahoo"]]
    u["yahoo_resolved_by_isin"] = u["yahoo"] != u["yahoo_xetra"]
    u = u.drop_duplicates("yahoo")
    print(f"  ISIN fallback: {sum(1 for v in alt if v)}/{len(sub)} resueltos", flush=True)
    return u


if __name__ == "__main__":
    u = fetch_universe_eu()
    print(len(u))
    print(u["instrument_type"].value_counts())
    print(u["xetra_group"].value_counts().head(10))

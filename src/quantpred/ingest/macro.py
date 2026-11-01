"""Series de referencia y macro: índices/futuros/divisas (Yahoo) + indicadores FRED."""
from __future__ import annotations

import io
import time

import pandas as pd
import requests

from quantpred.config import FRED_CSV_URL, FRED_PUB_LAG, FRED_SERIES, YAHOO_MACRO
from quantpred.ingest.prices import download_batch


def fetch_yahoo_macro() -> pd.DataFrame:
    long = download_batch(list(YAHOO_MACRO))
    long["description"] = long["ticker"].map(YAHOO_MACRO)
    return long


def _fred_one(sid: str) -> pd.Series | None:
    for attempt in range(4):
        try:
            r = requests.get(FRED_CSV_URL.format(sid=sid), timeout=60)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["date", "value"]
            s = pd.to_numeric(df["value"], errors="coerce")
            s.index = pd.to_datetime(df["date"])
            return s.dropna()
        except Exception:
            time.sleep(2 ** attempt)
    return None


def fetch_fred() -> pd.DataFrame:
    """Formato largo: date, series_id, value, freq, description."""
    out = []
    for sid, (freq, desc) in FRED_SERIES.items():
        s = _fred_one(sid)
        if s is None or s.empty:
            print(f"  FRED {sid}: sin datos", flush=True)
            continue
        out.append(pd.DataFrame({"date": s.index, "series_id": sid, "value": s.values,
                                 "freq": freq, "pub_lag_days": FRED_PUB_LAG[freq], "description": desc}))
    return pd.concat(out, ignore_index=True)

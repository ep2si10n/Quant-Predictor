"""Fuentes de precios para el motor live (interfaz común: iterar (timestamp, {ticker: precio})).

- ReplayFeed: reproduce un DataFrame (fecha/minuto x ticker) — pruebas y backtests visuales.
- YahooMinuteFeed: sondeo educado del último precio de 1 minuto (v8/chart) de los tickers en cartera.
  Pensado para demos; para producción real conectar un broker/feed en streaming (Alpaca, IBKR, Polygon)
  implementando el mismo protocolo.
"""
from __future__ import annotations

import time
from typing import Iterator, Protocol

import pandas as pd


class PriceFeed(Protocol):
    def __iter__(self) -> Iterator[tuple[pd.Timestamp, dict]]: ...


class ReplayFeed:
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices

    def __iter__(self):
        for ts, row in self.prices.iterrows():
            yield ts, row.dropna().to_dict()


class YahooMinuteFeed:
    def __init__(self, tickers: list, interval_s: int = 60, max_ticks: int | None = None):
        self.tickers = list(tickers)
        self.interval_s = interval_s
        self.max_ticks = max_ticks

    def last_prices(self) -> tuple[pd.Timestamp, dict]:
        from quantpred.ingest.yahoo import _get_json
        out, ts = {}, pd.Timestamp.utcnow()
        for t in self.tickers:
            try:
                js = _get_json(f"/v8/finance/chart/{t}", {"range": "1d", "interval": "1m"}, retries=2)
                meta = js["chart"]["result"][0]["meta"]
                out[t] = float(meta["regularMarketPrice"])
            except Exception:
                continue
        return ts, out

    def __iter__(self):
        n = 0
        while self.max_ticks is None or n < self.max_ticks:
            t0 = time.time()
            yield self.last_prices()
            n += 1
            time.sleep(max(0.0, self.interval_s - (time.time() - t0)))

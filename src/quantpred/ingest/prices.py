"""Descarga de históricos diarios OHLCV completos (range=max) desde Yahoo `v8/chart`.

- Paralelo (hilos) y reanudable: cada lote se guarda en data/raw/prices/batch_XXXX.parquet
  y los metadatos ligeros del chart (nombre, tipo, exchange, firstTradeDate) en chart_meta_XXXX.
- Guarda precios sin ajustar + `adj_close` (ajustado por splits y dividendos) + eventos
  corporativos (dividendos, splits, capital gains) para reconstruir retorno total.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from quantpred.config import RAW
from quantpred.ingest.yahoo import chart

PRICE_DIR = RAW / "prices"
BATCH = 200
WORKERS = 2
META_KEYS = ["symbol", "longName", "shortName", "instrumentType", "exchangeName", "fullExchangeName",
             "currency", "firstTradeDate", "exchangeTimezoneName", "regularMarketPrice"]


def _one(sym: str) -> tuple[str, pd.DataFrame, dict, str | None]:
    try:
        df, meta = chart(sym)
        m = {k: meta.get(k) for k in META_KEYS}
        if len(df) > 30 and df["date"].diff().dt.days.median() > 4:   # guardia: debe ser diario
            return sym, pd.DataFrame(), m, f"no diario ({meta.get('dataGranularity')})"
        return sym, df, m, None
    except Exception as e:
        return sym, pd.DataFrame(), {"symbol": sym}, str(e)[:200]


def download_symbols(symbols: list[str], workers: int = WORKERS) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, metas = [], []
    with ThreadPoolExecutor(workers) as ex:
        for sym, df, meta, err in ex.map(_one, symbols):
            if not df.empty:
                frames.append(df)
            meta["error"] = err
            meta["yahoo"] = sym
            metas.append(meta)
    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return prices, pd.DataFrame(metas)


def download_all(symbols: list[str], out_dir: Path = PRICE_DIR, batch: int = BATCH,
                 prefix: str = "") -> list[str]:
    """Descarga todos los símbolos. Devuelve la lista de símbolos sin datos."""
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = [symbols[i:i + batch] for i in range(0, len(symbols), batch)]
    for k, tks in enumerate(batches):
        f = out_dir / f"batch_{prefix}{k:04d}.parquet"
        if f.exists():
            continue
        t0 = time.time()
        px, meta = download_symbols(tks)
        px.to_parquet(f, index=False)
        meta.to_parquet(out_dir / f"chart_meta_{prefix}{k:04d}.parquet", index=False)
        n = px["ticker"].nunique() if not px.empty else 0
        print(f"  lote {k + 1}/{len(batches)}: {n}/{len(tks)} tickers, {len(px):,} filas, "
              f"{time.time() - t0:.0f}s", flush=True)
    got = set(load_raw(out_dir, columns=["ticker"])["ticker"].unique())
    return [s for s in symbols if s not in got]


def load_raw(out_dir: Path = PRICE_DIR, columns: list[str] | None = None) -> pd.DataFrame:
    files = sorted(out_dir.glob("batch_*.parquet"))
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


def load_chart_meta(out_dir: Path = PRICE_DIR) -> pd.DataFrame:
    files = sorted(out_dir.glob("chart_meta_*.parquet"))
    m = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    m = m[m["error"].isna() & m["longName"].notna() | ~m["yahoo"].duplicated(keep=False)]
    m["firstTradeDate"] = pd.to_datetime(m["firstTradeDate"], unit="s", errors="coerce")
    return m.drop_duplicates("yahoo", keep="last")


def download_batch(tickers: list[str]) -> pd.DataFrame:
    """Compatibilidad: descarga un conjunto pequeño (p.ej. macro) sin cache."""
    px, _ = download_symbols(tickers)
    return px

"""Fase 1 — pipeline completo de ingesta.

Uso:
    python scripts/run_ingestion.py            # todo
    python scripts/run_ingestion.py --skip-meta --skip-macro
La descarga de precios es reanudable (lotes en data/raw/prices/).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from quantpred.config import META, PROCESSED, RAW  # noqa: E402
from quantpred.ingest import macro, metadata, prices, quality, store, universe, universe_eu  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-meta", action="store_true", help="no llamar a quoteSummary (clasifica por nombre)")
    ap.add_argument("--skip-macro", action="store_true")
    ap.add_argument("--region", choices=["us", "eu"], default="us",
                    help="us = ETFs EEUU (NASDAQ Trader); eu = UCITS en Xetra")
    ap.add_argument("--limit", type=int, default=None, help="solo N primeros ETFs (pruebas)")
    a = ap.parse_args()
    t0 = time.time()

    sfx = "" if a.region == "us" else f"_{a.region}"
    raw_dir = RAW / f"prices{sfx}"
    print(f"[1/5] Universo de ETFs ({'NASDAQ Trader' if a.region == 'us' else 'Xetra UCITS'})…", flush=True)
    uni = universe.fetch_universe() if a.region == "us" else universe_eu.fetch_universe_eu()
    if a.limit:
        uni = uni.head(a.limit)
    print(f"  {len(uni)} ETFs/ETNs", flush=True)

    print("[2/5] Precios diarios históricos (Yahoo v8/chart, range=max)…", flush=True)
    missing = prices.download_all(uni["yahoo"].tolist(), out_dir=raw_dir)
    if a.region == "eu" and missing:
        uni = universe_eu.resolve_missing(uni, missing)
        uni.to_csv(META / "universe_eu.csv", index=False)
        alt = uni.loc[uni["yahoo_resolved_by_isin"], "yahoo"].tolist()
        prices.download_all(alt, out_dir=raw_dir, prefix="isin_")
        got = set(prices.load_raw(raw_dir, columns=["ticker"])["ticker"].unique())
        missing = [s for s in uni["yahoo"] if s not in got]
    (META / f"missing_tickers{sfx}.txt").write_text("\n".join(missing))
    print(f"  sin datos: {len(missing)}", flush=True)

    print("[3/5] Metadatos (chart meta + quoteSummary/Morningstar si Yahoo lo permite)…", flush=True)
    summ = None if a.skip_meta or a.region != "us" else metadata.fetch_summaries(uni["yahoo"].tolist())
    meta = metadata.build_metadata(uni, prices.load_chart_meta(raw_dir), summ, region=a.region)
    print(meta["asset_class"].value_counts().to_string(), flush=True)
    print(meta["class_source"].value_counts().to_string(), flush=True)

    print("[4/5] Limpieza + QA…", flush=True)
    raw = prices.load_raw(raw_dir)
    raw = raw[raw["ticker"].isin(set(uni["yahoo"]))]
    clean, rep = quality.clean_prices(raw)
    rep = rep.merge(meta[["yahoo", "name", "category", "asset_class"] + [c for c in ["fundFamily", "totalAssets"] if c in meta]],
                    left_on="ticker", right_on="yahoo", how="left").drop(columns="yahoo")
    rep.to_csv(META / f"quality_report{sfx}.csv", index=False)
    summary = store.write_prices(clean, meta, region=a.region)
    print(summary.to_string(index=False), flush=True)

    if not a.skip_macro:
        print("[5/5] Macro (Yahoo índices/futuros/FX + FRED)…", flush=True)
        ym = macro.fetch_yahoo_macro()
        ym, _ = quality.clean_prices(ym.drop(columns="description"))
        ym["description"] = ym["ticker"].map(macro.YAHOO_MACRO)
        ym.to_parquet(PROCESSED / "macro_yahoo.parquet", index=False, compression="zstd")
        fr = macro.fetch_fred()
        fr.to_parquet(PROCESSED / "macro_fred.parquet", index=False, compression="zstd")
        print(f"  yahoo macro: {ym['ticker'].nunique()} series, fred: {fr['series_id'].nunique()} series",
              flush=True)

    stats = {
        "n_etfs_universe": int(len(uni)),
        "n_etfs_with_prices": int(clean["ticker"].nunique()),
        "n_rows": int(len(clean)),
        "date_min": str(clean["date"].min().date()),
        "date_max": str(clean["date"].max().date()),
        "median_years": float(rep["years"].median()),
        "n_etfs_10y_plus": int((rep["years"] >= 10).sum()),
        "n_etfs_20y_plus": int((rep["years"] >= 20).sum()),
        "by_asset_class": summary.set_index("asset_class")[["tickers", "rows"]].to_dict("index"),
        "missing_tickers": len(missing),
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    (META / f"dataset_stats{sfx}.json").write_text(json.dumps(stats, indent=2, default=int))
    print(json.dumps(stats, indent=2, default=int))


if __name__ == "__main__":
    main()

"""Demo del motor live: cartera simulada del modelo vs S&P 500, lista para graficar superpuesta.

Modos:
  --replay  : reproduce el periodo de test día a día (rebalanceo diario al cierre) → CSV + PNG
  --yahoo N : arranca hoy con capital simulado y valora cada minuto durante N minutos (precios Yahoo)
Salida (mismo formato en ambos modos): reports/live_<modo>.csv con columnas
  ts, modelo, sp500, modelo_ret, sp500_ret, diff_pct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from quantpred.ingest.store import load_metadata  # noqa: E402
from quantpred.live.engine import LiveEngine, SimulatedPortfolio  # noqa: E402
from quantpred.live.feeds import YahooMinuteFeed  # noqa: E402
from quantpred.research.features import load_daily  # noqa: E402

MODEL = ROOT / "models" / "relative_policy_core_satellite.json"


def crypto_tickers(cols):
    ac = load_metadata("us").set_index("yahoo")["asset_class"]
    return {t for t in cols if ac.get(t) == "crypto"}


def replay(start: str, capital: float):
    P, DV, _ = load_daily("us", start="2002-01-01")
    eng = LiveEngine.from_files(str(MODEL), exclude_tickers=crypto_tickers(P.columns))
    pf = SimulatedPortfolio(capital)
    days = P.index[P.index >= start]
    for i, d in enumerate(days[:-1]):
        eng.prepare(P.loc[:d].iloc[-400:], DV.loc[:d].iloc[-400:])   # 400 días bastan (máx. ventana 252)
        prices = P.loc[d].dropna().to_dict()
        pf.rebalance(eng.target_weights(), prices, d)       # rebalanceo al cierre de d
        if i % 250 == 0:
            print(f"  {d.date()} modelo={pf.history[-1]['modelo']:.0f} sp500={pf.history[-1]['sp500']:.0f}", flush=True)
    last = days[-1]
    pf.on_tick(last, P.loc[last].dropna().to_dict())
    return pf


def live(minutes: int, capital: float):
    P, DV, _ = load_daily("us", start="2002-01-01")
    eng = LiveEngine.from_files(str(MODEL), exclude_tickers=crypto_tickers(P.columns)).prepare(P, DV)
    w = eng.target_weights()
    feed = YahooMinuteFeed(list(w.index) + ["SPY"], interval_s=60, max_ticks=minutes)
    pf = SimulatedPortfolio(capital)
    for k, (ts, prices) in enumerate(feed):
        if k == 0:
            pf.rebalance(w, prices, ts)
        else:
            pt = pf.on_tick(ts, prices)
            print(f"  {ts:%H:%M} modelo={pt['modelo']:.2f} sp500={pt['sp500']:.2f} diff={pt['diff_pct']:+.3f}%",
                  flush=True)
    return pf


def plot(df: pd.DataFrame, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    df[["modelo", "sp500"]].plot(ax=ax[0])
    ax[0].set_title(title)
    ax[0].grid(alpha=.3)
    df["diff_pct"].plot(ax=ax[1], color="tab:green")
    ax[1].axhline(0, color="k", lw=.6)
    ax[1].set_ylabel("modelo − S&P (% capital)")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--start", default="2019-01-02")
    ap.add_argument("--yahoo", type=int, default=0, help="minutos de sondeo en vivo")
    ap.add_argument("--capital", type=float, default=100_000)
    a = ap.parse_args()
    out = ROOT / "reports"
    if a.yahoo:
        pf = live(a.yahoo, a.capital)
        name = "live_yahoo"
    else:
        pf = replay(a.start, a.capital)
        name = "live_replay"
    df = pf.to_frame()
    df.to_csv(out / f"{name}.csv")
    plot(df, out / f"{name}.png", f"Cartera simulada del modelo vs S&P 500 ({name})")
    print(df[["modelo", "sp500", "diff_pct"]].tail(3).round(2).to_string())

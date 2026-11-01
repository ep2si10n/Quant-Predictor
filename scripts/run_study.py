"""Fase 3 — estudio comparativo de estrategias con split temporal.

Periodos: desarrollo 2008-01 → 2018-12 (para iterar en Fase 4) y TEST 2019-01 → hoy (fuera de muestra).
Salida: reports/phase3_results.csv, reports/phase3_results.md, reports/phase3_returns.parquet,
        reports/phase3_equity.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from quantpred.research import backtest as bt  # noqa: E402
from quantpred.research.features import build_panel  # noqa: E402
from quantpred.research.metrics import deflated_sharpe, pbo, perf, sharpe_per_period  # noqa: E402

DEV = ("2008-01-31", "2018-12-31")
TEST = ("2019-01-31", "2099-12-31")
REPORTS = ROOT / "reports"


def models():
    return {
        "ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=50.0)),
        "lgbm": lambda: LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=15,
                                      min_child_samples=300, subsample=0.7, subsample_freq=1,
                                      colsample_bytree=0.6, reg_lambda=10.0, verbose=-1, n_jobs=4),
    }


def main(rebuild: bool = False) -> None:
    t0 = time.time()
    REPORTS.mkdir(exist_ok=True)
    d = build_panel("us", cache=not rebuild)
    panel, daily = d["panel"], d["daily"]
    panel = panel.assign(fwd_rel=panel["fwd_rel"].clip(-0.3, 0.3))
    print(f"panel {panel.shape}, meses {panel.index.get_level_values(0).nunique()}, "
          f"ETFs {panel.index.get_level_values(1).nunique()}", flush=True)

    S = {}
    # Benchmarks
    S["B_SPY"] = bt.run("B_SPY", panel, daily, bt.sel_fixed({"SPY": 1}))
    S["B_60_40"] = bt.run("B_60_40", panel, daily, bt.sel_fixed({"SPY": .6, "AGG": .4}))
    S["B_EW_universo"] = bt.run("B_EW_universo", panel, daily, bt.sel_all, scheme="ew")
    # Reglas clásicas
    S["R_dual_momentum"] = bt.run("R_dual_momentum", panel, daily, bt.sel_dual_momentum)
    S["R_core_ew"] = bt.run("R_core_ew", panel, daily, bt.sel_core, scheme="ew")
    S["R_core_erc"] = bt.run("R_core_erc", panel, daily, bt.sel_core, scheme="erc")
    S["R_core_hrp"] = bt.run("R_core_hrp", panel, daily, bt.sel_core, scheme="hrp")
    S["R_core_minvar"] = bt.run("R_core_minvar", panel, daily, bt.sel_core, scheme="minvar")
    S["R_core_tsmom_ivol"] = bt.run("R_core_tsmom_ivol", panel, daily, bt.sel_core_tsmom, scheme="ivol")
    S["R_xsmom20_ivol"] = bt.run("R_xsmom20_ivol", panel, daily, bt.sel_top("r12_1", 20), scheme="ivol")
    S["R_xsmom20_cap"] = bt.run("R_xsmom20_cap", panel, daily,
                                bt.sel_top("mom_radj", 20, max_per_class=0.4, require_positive="r12_1"),
                                scheme="ivol")
    S["R_lowvol20"] = bt.run("R_lowvol20", panel, daily, bt.sel_top("vol12m", 20, ascending=True), scheme="ew")

    # ML walk-forward
    preds, ics = {}, {}
    for macro in (False, True):
        cols = bt.feature_cols(panel, macro=macro)
        for mname, mk in models().items():
            tag = f"{mname}{'_macro' if macro else ''}"
            t1 = time.time()
            p = bt.walk_forward_predict(panel, mk, cols)
            preds[tag] = p
            ics[tag] = bt.ic_series(p, panel)
            for scheme in ("ivol", "hrp"):
                n = f"ML_{tag}_top20_{scheme}"
                S[n] = bt.run(n, panel, daily, bt.sel_pred(p, 20, max_per_class=0.4), scheme=scheme)
            print(f"  {tag}: IC medio {ics[tag].mean():.3f} ({time.time() - t1:.0f}s)", flush=True)

    R = pd.DataFrame({k: v.returns for k, v in S.items()})
    R.to_parquet(REPORTS / "phase3_returns.parquet")
    pd.concat(preds, axis=1).to_parquet(REPORTS / "phase3_predictions.parquet")

    rf = panel.groupby(level=0)["rf"].first()
    rf.index = rf.index + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(1)   # rf del mes de tenencia
    rows = []
    for period, (a, b) in {"dev": DEV, "test": TEST}.items():
        Rp = R.loc[a:b]
        trial_sr = [sharpe_per_period(Rp[c] - rf.reindex(Rp.index).fillna(0)) for c in Rp]
        for c in Rp:
            m = perf(Rp[c], rf=rf)
            m.update(strategy=c, period=period,
                     Turnover=S[c].turnover.loc[a:b].mean() * 12,
                     DSR=deflated_sharpe(Rp[c] - rf.reindex(Rp.index).fillna(0), trial_sr))
            tag = next((k for k in ics if c.startswith(f"ML_{k}_top")), None)
            m["IC"] = ics[tag].loc[a:b].mean() if tag else np.nan
            rows.append(m)
    res = pd.DataFrame(rows).set_index(["period", "strategy"])
    res.to_csv(REPORTS / "phase3_results.csv")
    pbo_dev = pbo(R.loc[DEV[0]:DEV[1]].dropna(axis=1))
    pbo_all = pbo(R.dropna(axis=1))

    # equity chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = res.loc["test"].sort_values("Sharpe", ascending=False).index[:6].tolist()
        keep = list(dict.fromkeys(top + ["B_SPY", "B_60_40"]))
        fig, ax = plt.subplots(figsize=(11, 6))
        (1 + R.loc[TEST[0]:, keep].fillna(0)).cumprod().plot(ax=ax, logy=True)
        ax.set_title("Fase 3 — Test fuera de muestra (2019→): crecimiento de 1$")
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(REPORTS / "phase3_equity.png", dpi=120)
    except Exception as e:
        print("gráfico:", e)

    fmt = {"CAGR": "{:.1%}", "Vol": "{:.1%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "MaxDD": "{:.1%}",
           "Calmar": "{:.2f}", "Turnover": "{:.1f}", "DSR": "{:.2f}", "IC": "{:.3f}"}
    lines = ["# Fase 3 — Resultados del estudio comparativo", "",
             f"Desarrollo {DEV[0][:7]}→{DEV[1][:7]} · Test {TEST[0][:7]}→{R.index.max():%Y-%m} · "
             f"costes {bt.COST_BPS:.0f} pb/operación · rebalanceo mensual", "",
             f"PBO (CSCV) desarrollo: **{pbo_dev:.2f}** · todo el periodo: **{pbo_all:.2f}**", ""]
    for period in ("test", "dev"):
        t = res.loc[period].sort_values("Sharpe", ascending=False)[list(fmt)]
        t = t.apply(lambda col: col.map(lambda v, f=fmt[col.name]: "" if pd.isna(v) else f.format(v)))
        lines += [f"## {'TEST (fuera de muestra)' if period == 'test' else 'Desarrollo'}", "",
                  t.to_markdown(), ""]
    (REPORTS / "phase3_results.md").write_text("\n".join(lines))
    (REPORTS / "phase3_meta.json").write_text(json.dumps(
        {"pbo_dev": pbo_dev, "pbo_all": pbo_all, "n_strategies": R.shape[1],
         "panel_rows": len(panel), "runtime_min": round((time.time() - t0) / 60, 1)}, indent=2))
    print("\n".join(lines[:6]))
    print(res.loc["test"].sort_values("Sharpe", ascending=False)[["CAGR", "Vol", "Sharpe", "MaxDD", "DSR", "IC"]]
          .round(3).to_string())


if __name__ == "__main__":
    main(rebuild="--rebuild" in sys.argv)

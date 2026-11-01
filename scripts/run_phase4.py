"""Fase 4 — iteración guiada por fallos hasta que no haya mejora.

Protocolo (anti-sobreajuste):
- Cada ronda modifica UNA dimensión de la mejor configuración actual. Se elige SOLO con el Sharpe
  del periodo de DESARROLLO (2008-2018). Parada: mejora < MIN_GAIN en una ronda.
- Se registran TODOS los intentos (reports/phase4_trials.csv) → Deflated Sharpe con N real.
- El TEST (2019→) se evalúa al final, una vez, para la configuración ganadora y los benchmarks.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from quantpred.research import backtest as bt  # noqa: E402
from quantpred.research import phase4 as p4  # noqa: E402
from quantpred.research.features import build_panel  # noqa: E402
from quantpred.research.metrics import deflated_sharpe, pbo, perf, sharpe_per_period  # noqa: E402
from run_study import DEV, TEST, models  # noqa: E402

REPORTS = ROOT / "reports"
MIN_GAIN = 0.03
PRED_CACHE: dict = {}


def get_pred(panel, target, model, macro=True):
    key = (target, model, macro)
    if key not in PRED_CACHE:
        t0 = time.time()
        PRED_CACHE[key] = bt.walk_forward_predict(panel, models()[model], bt.feature_cols(panel, macro), target=target)
        print(f"    pred {key} {time.time() - t0:.0f}s", flush=True)
    return PRED_CACHE[key]


def build_pred(panel, cfg):
    if cfg["model"] == "ens":
        ps = [get_pred(panel, cfg["target"], m) for m in ("ridge", "lgbm")]
        pred = p4.rank_ensemble(ps)
    elif cfg["model"] == "none":
        return None
    else:
        pred = get_pred(panel, cfg["target"], cfg["model"])
    return p4.smooth(pred, cfg["smooth"])


def evaluate(name, cfg, panel, daily, rf, trials):
    pred = build_pred(panel, cfg)
    sel = p4.sel_two_level(pred, daily, k_per=cfg["k"], trend=cfg["trend"], sleeve_scheme=cfg["scheme"],
                           score_col="mom_radj", sleeves=cfg["sleeves"])
    res = bt.run(name, panel, daily, sel, vol_target=cfg["vt"])
    r = res.returns
    dev = perf(r.loc[DEV[0]:DEV[1]], rf=rf)
    trials.append({"name": name, **{k: str(v) for k, v in cfg.items()}, "dev_sharpe": dev["Sharpe"],
                   "dev_cagr": dev["CAGR"], "dev_maxdd": dev["MaxDD"],
                   "dev_turnover": res.turnover.loc[DEV[0]:DEV[1]].mean() * 12, "returns": r})
    print(f"  {name:55s} devSR={dev['Sharpe']:.3f} CAGR={dev['CAGR']:.3f} DD={dev['MaxDD']:.3f}", flush=True)
    return dev["Sharpe"]


def main():
    t0 = time.time()
    d = build_panel("us")
    panel = p4.add_targets(d["panel"])
    daily = d["daily"]
    rf = panel.groupby(level=0)["rf"].first()
    rf.index = rf.index + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(1)
    trials: list = []

    best = {"target": "t_rel", "model": "lgbm", "smooth": 1, "k": 5, "trend": False, "scheme": "erc",
            "vt": None, "sleeves": tuple(p4.SLEEVES)}
    # Ronda 0: referencia (dos niveles con momentum, sin ML) y configuración inicial
    print("Ronda 0 — referencias", flush=True)
    evaluate("R0_two_level_momentum", {**best, "model": "none"}, panel, daily, rf, trials)
    best_sr = evaluate("R0_base", best, panel, daily, rf, trials)

    grid = [
        ("target", ["t_sharpe", "t_cls", "t_cls_sharpe", "t_cls_rank"]),
        ("model", ["ridge", "ens", "none"]),
        ("smooth", [3, 6]),
        ("k", [3, 10]),
        ("trend", [True]),
        ("scheme", ["ivol", "ew"]),
        ("vt", [0.08, 0.12]),
        ("sleeves", [tuple(p4.SLEEVES) + ("alternative",), ("equity", "fixed_income", "commodity")]),
    ]
    history = [{"round": 0, "cfg": dict(best), "dev_sharpe": best_sr}]
    rnd = 0
    while True:
        rnd += 1
        start_sr = best_sr
        print(f"Ronda {rnd} — mejor actual devSR={best_sr:.3f} {best}", flush=True)
        for dim, values in grid:
            for v in values:
                if best[dim] == v:
                    continue
                cfg = {**best, dim: v}
                name = f"R{rnd}_{dim}={v if not isinstance(v, tuple) else '+'.join(x[:3] for x in v)}"
                sr = evaluate(name, cfg, panel, daily, rf, trials)
                if sr > best_sr + 1e-9:
                    best_sr, best_new = sr, cfg
                    best = best_new
        history.append({"round": rnd, "cfg": dict(best), "dev_sharpe": best_sr})
        if best_sr - start_sr < MIN_GAIN or rnd >= 4:
            print(f"Parada: mejora de la ronda {best_sr - start_sr:.3f} < {MIN_GAIN}", flush=True)
            break

    # --- evaluación final -------------------------------------------------------------------------
    T = pd.DataFrame([{k: v for k, v in t.items() if k != "returns"} for t in trials])
    T.to_csv(REPORTS / "phase4_trials.csv", index=False)
    Rall = pd.DataFrame({t["name"]: t["returns"] for t in trials})
    Rall.to_parquet(REPORTS / "phase4_returns.parquet")
    win = T.sort_values("dev_sharpe", ascending=False).iloc[0]["name"]
    final_cfg = history[-1]["cfg"]
    p3 = pd.read_parquet(REPORTS / "phase3_returns.parquet")
    comp = pd.concat([Rall[[win]].rename(columns={win: "GANADORA_F4"}),
                      p3[["B_SPY", "B_60_40", "B_EW_universo", "R_dual_momentum", "R_core_erc",
                          "ML_lgbm_macro_top20_hrp"]]], axis=1)
    n_trials = len(trials) + p3.shape[1]
    rows = []
    for period, (a, b) in {"dev": DEV, "test": TEST}.items():
        Rp = comp.loc[a:b]
        trial_sr = [sharpe_per_period(Rall[c].loc[a:b] - rf.reindex(Rall.loc[a:b].index).fillna(0)) for c in Rall] + \
                   [sharpe_per_period(p3[c].loc[a:b] - rf.reindex(p3.loc[a:b].index).fillna(0)) for c in p3]
        for c in Rp:
            m = perf(Rp[c].dropna(), rf=rf)
            m.update(period=period, strategy=c,
                     DSR=deflated_sharpe(Rp[c].dropna() - rf.reindex(Rp[c].dropna().index).fillna(0), trial_sr))
            rows.append(m)
    res = pd.DataFrame(rows).set_index(["period", "strategy"])
    res.to_csv(REPORTS / "phase4_final.csv")
    pbo4 = pbo(Rall.loc[DEV[0]:DEV[1]].dropna(axis=1))
    meta = {"winner": win, "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in final_cfg.items()},
            "history": [{**h, "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in h["cfg"].items()}}
                        for h in history],
            "n_trials_total": n_trials, "pbo_phase4_dev": pbo4, "runtime_min": round((time.time() - t0) / 60, 1)}
    (REPORTS / "phase4_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (a, b), ttl in zip(axes, (DEV, TEST), ("Desarrollo 2008-2018", "TEST 2019→ (fuera de muestra)")):
            (1 + comp.loc[a:b].fillna(0)).cumprod().plot(ax=ax, logy=True, legend=ax is axes[1])
            ax.set_title(ttl)
            ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(REPORTS / "phase4_equity.png", dpi=120)
    except Exception as e:
        print("gráfico:", e)
    print(json.dumps({k: meta[k] for k in ("winner", "config", "n_trials_total", "pbo_phase4_dev")}, indent=1,
                     default=str))
    print(res[["CAGR", "Vol", "Sharpe", "MaxDD", "Calmar", "DSR"]].round(3).to_string())


if __name__ == "__main__":
    main()

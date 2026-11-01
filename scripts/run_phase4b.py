"""Fase 4b — timing de volatilidad y régimen sobre carteras base (60/40, SPY, núcleo ERC).

1) Modelos de volatilidad del mes siguiente (log-vol SPY): naive, HAR, HAR-X (ridge), LightGBM.
2) Clasificadores de "mes malo" (exceso SPY < -3%): logística, LightGBM, MLP (red neuronal), ensemble.
3) Overlays: gestión por volatilidad (Moreira-Muir), recorte de renta variable por régimen, combinación.
Selección SOLO con desarrollo (2008-2018); test 2019→ una vez. Todos los intentos registrados → DSR.
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
from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from quantpred.config import PROCESSED  # noqa: E402
from quantpred.research import regime as rg  # noqa: E402
from quantpred.research.features import build_panel  # noqa: E402
from quantpred.research.metrics import deflated_sharpe, pbo, perf, sharpe_per_period  # noqa: E402

DEV = ("2008-01-31", "2018-12-31")
TEST = ("2019-01-31", "2099-12-31")
REPORTS = ROOT / "reports"
COST = 5e-4
SPREAD = 0.005 / 12
HAR = ["lrv_w", "lrv_m", "lrv_q", "lrv_y"]


def pipe(model):
    return lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


def calmonth(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = s.index + pd.offsets.MonthEnd(0)
    return s


def main():
    t0 = time.time()
    d = build_panel("us")
    df = rg.build_regime_frame(d["panel"])
    state = [c for c in df.columns if not c.startswith("y_") and c != "rf"]
    print(f"frame {df.shape}, features {len(state)}", flush=True)

    # ---------------- 1. volatilidad ----------------
    vol_models = {
        "naive": None,
        "har": (HAR, lambda: make_pipeline(SimpleImputer(), LinearRegression())),
        "harx_ridge": (state, pipe(Ridge(alpha=30.0))),
        "lgbm": (state, lambda: LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                              min_child_samples=20, subsample=0.8, subsample_freq=1,
                                              colsample_bytree=0.7, verbose=-1)),
    }
    VP, vol_eval = {}, []
    for n, spec in vol_models.items():
        VP[n] = df["lrv_m"].loc["2005":] if spec is None else rg.walk_forward(df, spec[0], "y_lvol", spec[1])
    VP["ens_vol"] = pd.concat([VP["har"], VP["harx_ridge"], VP["lgbm"]], axis=1).mean(axis=1)
    for n, p in VP.items():
        for per, (a, b) in {"dev": DEV, "test": TEST}.items():
            y = df["y_lvol"].loc[a:b]
            pp = p.reindex(y.index)
            m = y.notna() & pp.notna()
            sse = ((y[m] - pp[m]) ** 2).sum()
            sse0 = ((y[m] - df["lrv_m"].reindex(y.index)[m]) ** 2).sum()
            vol_eval.append({"model": n, "period": per, "R2_vs_naive": 1 - sse / sse0,
                             "corr": np.corrcoef(y[m], pp[m])[0, 1]})
    vol_eval = pd.DataFrame(vol_eval)
    print(vol_eval.round(3).to_string(index=False), flush=True)

    # ---------------- 2. mes malo ----------------
    clf_models = {
        "logit": pipe(LogisticRegression(C=0.1, max_iter=2000)),
        "lgbm": lambda: LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=20,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1),
        "mlp": pipe(MLPClassifier(hidden_layer_sizes=(16, 8), alpha=1e-2, max_iter=2000, early_stopping=True,
                                  validation_fraction=0.2, random_state=0)),
    }
    BP, clf_eval = {}, []
    for n, mk in clf_models.items():
        if n == "mlp":   # red neuronal: media de 5 semillas para estabilizar
            ps = []
            for seed in range(5):
                mk_s = pipe(MLPClassifier(hidden_layer_sizes=(16, 8), alpha=1e-2, max_iter=2000,
                                          early_stopping=True, validation_fraction=0.2, random_state=seed))
                ps.append(rg.walk_forward(df, state, "y_bad", mk_s, classifier=True))
            BP[n] = pd.concat(ps, axis=1).mean(axis=1)
        else:
            BP[n] = rg.walk_forward(df, state, "y_bad", mk, classifier=True)
    BP["ens"] = pd.concat([BP[k].rank(pct=True) for k in clf_models], axis=1).mean(axis=1)
    for n, p in BP.items():
        for per, (a, b) in {"dev": DEV, "test": TEST}.items():
            y = df["y_bad"].loc[a:b].dropna()
            pp = p.reindex(y.index)
            m = pp.notna()
            clf_eval.append({"model": n, "period": per, "AUC": roc_auc_score(y[m], pp[m]),
                             "Brier": brier_score_loss(y[m], pp[m].clip(0, 1)) if n != "ens" else np.nan,
                             "base_rate": y.mean()})
    clf_eval = pd.DataFrame(clf_eval)
    print(clf_eval.round(3).to_string(index=False), flush=True)

    # ---------------- 3. carteras y overlays ----------------
    Rm = d["rets"]
    rf = calmonth(df["rf"])
    spy, agg = calmonth(Rm["SPY"]), calmonth(Rm["AGG"].fillna(Rm["IEF"]))
    p3 = pd.read_parquet(REPORTS / "phase3_returns.parquet")
    bases = {"6040": None, "core_erc": p3["R_core_erc"]}
    trials, R = [], {}

    def run_overlay(name, base, expo=None, eq_cut=None):
        """expo: exposición (serie en t, se aplica a t+1); eq_cut: fracción de renta variable a recortar."""
        idx = spy.loc["2005-01-31":].index
        e = (expo.reindex(df.index).pipe(calmonth).reindex(idx).shift(1) if expo is not None
             else pd.Series(1.0, index=idx)).fillna(1.0)
        c = (eq_cut.reindex(df.index).pipe(calmonth).reindex(idx).shift(1) if eq_cut is not None
             else pd.Series(0.0, index=idx)).fillna(0.0)
        if base is None:
            weq = 0.6 * (1 - c)
            br = weq * spy.reindex(idx) + (1 - weq) * agg.reindex(idx)
            to_base = (weq.diff().abs() * 2).fillna(0)
        else:
            br = base.reindex(idx)
            # recorte de régimen en una base genérica: pasar la fracción c a liquidez
            br = (1 - c) * br + c * rf.reindex(idx)
            to_base = c.diff().abs().fillna(0)
        cash = 1 - e
        fin = np.where(cash >= 0, rf.reindex(idx), rf.reindex(idx) + SPREAD)
        r = e * br + cash * fin - (e.diff().abs().fillna(0) + to_base) * COST
        r = r.dropna()
        R[name] = r
        dv = perf(r.loc[DEV[0]:DEV[1]], rf=rf)
        trials.append({"name": name, "dev_sharpe": dv["Sharpe"], "dev_cagr": dv["CAGR"], "dev_maxdd": dv["MaxDD"]})
        print(f"  {name:45s} devSR={dv['Sharpe']:.3f} CAGR={dv['CAGR']:.3f} DD={dv['MaxDD']:.3f}", flush=True)

    for bn, base in bases.items():
        run_overlay(f"{bn}_base", base)
        # gestión por volatilidad: exposición = σ_ref / σ̂ (σ_ref = mediana expansiva de σ̂ pasadas)
        for vm in ["naive", "har", "harx_ridge", "ens_vol"]:
            sig = np.exp(VP[vm])
            ref = sig.expanding(24).median().shift(1)
            for power, lev in [(1, 1.0), (1, 1.5), (2, 1.5)]:
                expo = ((ref / sig) ** power).clip(0, lev)
                run_overlay(f"{bn}_vm_{vm}_p{power}_L{lev}", base, expo=expo)
        # régimen: recortar renta variable cuando p(mes malo) supera su cuantil histórico (sin look-ahead)
        for cm in ["logit", "lgbm", "mlp", "ens"]:
            thr80 = rg.expanding_quantile(BP[cm], 0.8)
            thr70 = rg.expanding_quantile(BP[cm], 0.7)
            for tn, thr in [("q80", thr80), ("q70", thr70)]:
                for cut in [0.5, 1.0]:
                    run_overlay(f"{bn}_reg_{cm}_{tn}_cut{cut}", base, eq_cut=(BP[cm] > thr).astype(float) * cut)

    # combinación: mejor vm + mejor régimen de cada base (elegidos en desarrollo)
    T = pd.DataFrame(trials)
    for bn, base in bases.items():
        bvm = T[T.name.str.startswith(f"{bn}_vm_")].sort_values("dev_sharpe").iloc[-1]["name"]
        brg = T[T.name.str.startswith(f"{bn}_reg_")].sort_values("dev_sharpe").iloc[-1]["name"]
        vm, pw, lv = bvm.split("_vm_")[1].rsplit("_p", 1)[0], int(bvm.split("_p")[-1].split("_")[0]), \
            float(bvm.split("_L")[-1])
        cm, tn, cut = brg.split("_reg_")[1].split("_")[0], brg.split("_")[-2], float(brg.split("cut")[-1])
        sig = np.exp(VP[vm])
        expo = ((sig.expanding(24).median().shift(1) / sig) ** pw).clip(0, lv)
        thr = rg.expanding_quantile(BP[cm], 0.8 if tn == "q80" else 0.7)
        run_overlay(f"{bn}_combo[{bvm.split(bn + '_')[1]}+{brg.split(bn + '_')[1]}]", base, expo=expo,
                    eq_cut=(BP[cm] > thr).astype(float) * cut)

    T = pd.DataFrame(trials)
    T.to_csv(REPORTS / "phase4b_trials.csv", index=False)
    RR = pd.DataFrame(R)
    RR.to_parquet(REPORTS / "phase4b_returns.parquet")
    win = T.sort_values("dev_sharpe", ascending=False).iloc[0]["name"]

    # --------------- evaluación final (test una vez) ---------------
    p4 = pd.read_parquet(REPORTS / "phase4_returns.parquet")
    allR = pd.concat([RR, p3, p4], axis=1)
    comp = pd.concat([RR[[win]].rename(columns={win: f"GANADORA_4b: {win}"}),
                      RR[["6040_base", "core_erc_base"]], p3[["B_SPY", "R_dual_momentum"]]], axis=1)
    rows = []
    for per, (a, b) in {"dev": DEV, "test": TEST}.items():
        trial_sr = [sharpe_per_period(allR[c].loc[a:b].dropna() - rf.reindex(allR[c].loc[a:b].dropna().index).fillna(0))
                    for c in allR]
        for c in comp:
            r = comp[c].loc[a:b].dropna()
            m = perf(r, rf=rf)
            m.update(period=per, strategy=c, DSR=deflated_sharpe(r - rf.reindex(r.index).fillna(0), trial_sr))
            rows.append(m)
    res = pd.DataFrame(rows).set_index(["period", "strategy"])
    res.to_csv(REPORTS / "phase4b_final.csv")
    vol_eval.to_csv(REPORTS / "phase4b_vol_models.csv", index=False)
    clf_eval.to_csv(REPORTS / "phase4b_regime_models.csv", index=False)
    pd.concat({**{f"vol_{k}": v for k, v in VP.items()}, **{f"bad_{k}": v for k, v in BP.items()}}, axis=1) \
        .to_parquet(REPORTS / "phase4b_signals.parquet")
    meta = {"winner": win, "n_trials_total": allR.shape[1],
            "pbo_4b_dev": pbo(RR.loc[DEV[0]:DEV[1]].dropna(axis=1)), "features": state,
            "runtime_min": round((time.time() - t0) / 60, 1)}
    (REPORTS / "phase4b_meta.json").write_text(json.dumps(meta, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (a, b), ttl in zip(axes, (DEV, TEST), ("Desarrollo 2008-2018", "TEST 2019→")):
            (1 + comp.loc[a:b].fillna(0)).cumprod().plot(ax=ax, logy=True, legend=ax is axes[1])
            ax.set_title(ttl)
            ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(REPORTS / "phase4b_equity.png", dpi=120)
    except Exception as e:
        print("gráfico:", e)
    print(json.dumps({k: meta[k] for k in ("winner", "n_trials_total", "pbo_4b_dev")}))
    print(res[["CAGR", "Vol", "Sharpe", "MaxDD", "Calmar", "DSR"]].round(3).to_string())


if __name__ == "__main__":
    main()

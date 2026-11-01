"""Estrategia relativa al S&P 500 — entrenamiento con la recompensa f(Δ) y evaluación.

Splits: TRAIN 2005-2015 · VALIDACIÓN 2016-2018 (selección de hiperparámetros) · TEST 2019→ (una vez).
Modelo final de producción: reentrenado con todo el histórico → models/relative_policy.json
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantpred.config import PROCESSED  # noqa: E402
from quantpred.relative import reward as rw  # noqa: E402
from quantpred.relative.data import FEATURES, build_daily  # noqa: E402
from quantpred.relative.policy import Policy, simulate  # noqa: E402

REPORTS, MODELS = ROOT / "reports", ROOT / "models"
SPLITS = {"train": ("2005-01-01", "2015-12-31"), "val": ("2016-01-01", "2018-12-31"),
          "test": ("2019-01-01", "2099-12-31")}
COST = 2e-4
EXCLUDE = {"crypto"}


def idx_of(d, a, b):
    return np.where((d.dates >= a) & (d.dates <= b))[0]


def metrics(sim, rf, name, period):
    r, s = sim["ret"], sim["spy"]
    delta = r - s
    yrs = len(r) / 252
    eq, eqs = np.cumprod(1 + r), np.cumprod(1 + s)
    rel = eq / eqs
    ex = r - rf
    return {"strategy": name, "period": period,
            "CAGR": eq[-1] ** (1 / yrs) - 1, "CAGR_SPY": eqs[-1] ** (1 / yrs) - 1,
            "Active_ann": delta.mean() * 252, "TE": delta.std() * np.sqrt(252),
            "IR": delta.mean() / (delta.std() + 1e-12) * np.sqrt(252),
            "Sharpe": ex.mean() / (ex.std() + 1e-12) * np.sqrt(252),
            "MaxDD": (eq / np.maximum.accumulate(eq) - 1).min(),
            "MaxRelDD": (rel / np.maximum.accumulate(rel) - 1).min(),
            "HitDays": (delta > 0).mean(),
            "Hit21d": (np.convolve(delta, np.ones(21), "valid") > 0).mean(),
            "Reward21_bp": rw.reward_exact(np.convolve(delta, np.ones(21), "valid")).mean() * 1e4,
            "Turnover_ann": sim["turnover"].mean() * 252, "Final_vs_SPY": rel[-1] - 1}


def hold_every(W, n):
    """Rebalanceo cada n días (mantener pesos objetivo; la deriva la gestiona simulate)."""
    if n <= 1:
        return W
    W2 = W.copy()
    for t in range(len(W)):
        W2[t] = W[t - t % n]
    return W2


def baseline_weights(d, idx, kind):
    M = d.mask[idx]
    W = np.zeros(M.shape)
    if kind == "spy":
        W[:, 0] = 1
    elif kind == "ew":
        W = M / M.sum(1, keepdims=True)
    elif kind.startswith("mom"):
        f = d.features.index("a63")
        sc = np.where(M, d.X[idx][..., f], -np.inf)
        top = np.argsort(-sc, axis=1)[:, :10]
        for t in range(len(idx)):
            W[t, top[t]] = 0.1
    return W


def main():
    t0 = time.time()
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    cache = PROCESSED / "study_relative_daily.pkl"
    d = pickle.load(open(cache, "rb")) if cache.exists() else build_daily(80)
    # excluir cripto (volatilidad extrema, historia corta): se enmascara del universo
    from quantpred.ingest.store import load_metadata
    ac = load_metadata("us").set_index("yahoo")["asset_class"]
    crypto = np.vectorize(lambda t: ac.get(t, "") in EXCLUDE)(d.tickers)
    d.mask &= ~crypto
    d.R = np.where(d.mask, d.R, 0.0)
    I = {k: idx_of(d, *v) for k, v in SPLITS.items()}
    rows, trials = [], []

    # ---------- selección en validación ----------
    grid = []
    for gamma in (0.3, 0.6, 1.0):
        for tcost in (2e-4, 1e-3):
            for alpha in (1e-4, 1e-3):
                grid.append(("linear", alpha, f"g{gamma}_c{tcost:g}", dict(rw.DEFAULT), gamma, tcost))
    best, best_val = None, -np.inf
    for kind, alpha, rp_name, rp, gamma, tcost in grid:
        pol = Policy(FEATURES, kind=kind, alpha=alpha, cost=tcost, reward_params=rp, hidden=16,
                     gamma=gamma).fit(d, I["train"])
        Wv = pol.weights(d.X[I["val"]], d.mask[I["val"]])
        mv = metrics(simulate(Wv, d, I["val"], COST), d.rf[I["val"]], "x", "val")
        name = f"pol_{kind}_a{alpha:g}_{rp_name}"
        trials.append({"name": name, **{k: mv[k] for k in ("Reward21_bp", "Active_ann", "IR", "TE")}})
        print(f"  {name:34s} val reward21={mv['Reward21_bp']:.1f}bp activo={mv['Active_ann']:.3%} "
              f"IR={mv['IR']:.2f} TE={mv['TE']:.2%}", flush=True)
        # criterio = recompensa pedida (asimétrica, exacta, 21d), exigiendo retorno activo > 0
        score = mv["Reward21_bp"] if mv["Active_ann"] > 0 else mv["Reward21_bp"] - 1e6
        if score > best_val:
            best_val, best = score, (kind, alpha, rp_name, rp, gamma, tcost)
    pd.DataFrame(trials).to_csv(REPORTS / "relative_trials.csv", index=False)
    kind, alpha, rp_name, rp, gamma, tcost = best
    print("mejor en validación:", best[:3], flush=True)

    # ---------- reentrenar train+val y evaluar test (una vez) ----------
    trval = np.concatenate([I["train"], I["val"]])
    pol = Policy(FEATURES, kind=kind, alpha=alpha, cost=tcost, reward_params=rp, hidden=16,
                 gamma=gamma).fit(d, trval)
    for period in ("train", "val", "test"):
        ix = I[period]
        Wp = (Policy.from_dict(pol.to_dict()) if period == "test" else pol).weights(d.X[ix], d.mask[ix])
        for nm, W in [("POLITICA_diaria", Wp), ("POLITICA_semanal", hold_every(Wp, 5)),
                      ("SPY", baseline_weights(d, ix, "spy")), ("EW_80", baseline_weights(d, ix, "ew")),
                      ("Momentum_top10", baseline_weights(d, ix, "mom"))]:
            sim = simulate(W, d, ix, COST)
            rows.append(metrics(sim, d.rf[ix], nm, period))
            if period == "test" and nm == "POLITICA_diaria":
                test_sim, test_W = sim, W
    res = pd.DataFrame(rows).set_index(["period", "strategy"])
    res.to_csv(REPORTS / "relative_results.csv")

    # pesos medios en test (interpretación)
    ix = I["test"]
    avg = pd.Series(dtype=float)
    for t in range(len(ix)):
        for i in range(test_W.shape[1]):
            if test_W[t, i] > 1e-4:
                avg[d.tickers[ix][t, i]] = avg.get(d.tickers[ix][t, i], 0) + test_W[t, i] / len(ix)
    avg.sort_values(ascending=False).head(25).to_csv(REPORTS / "relative_test_avg_weights.csv")
    coef = pd.Series(pol.theta[:len(FEATURES)], index=FEATURES) if kind == "linear" else None

    # curva diaria (para gráfico) y figura
    curve = pd.DataFrame({"modelo": np.cumprod(1 + test_sim["ret"]), "SPY": np.cumprod(1 + test_sim["spy"])},
                         index=d.dates[ix])
    curve.to_csv(REPORTS / "relative_test_curve.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        curve.plot(ax=ax[0])
        ax[0].set_title("TEST 2019→: cartera del modelo vs S&P 500 (SPY), 1$ invertido")
        ax[0].grid(alpha=.3)
        (curve["modelo"] / curve["SPY"] - 1).plot(ax=ax[1], color="tab:green")
        ax[1].axhline(0, color="k", lw=.6)
        ax[1].set_title("Diferencia acumulada vs SPY")
        ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(REPORTS / "relative_test.png", dpi=120)
    except Exception as e:
        print("gráfico:", e)

    # ---------- modelo de producción: todo el histórico ----------
    allix = np.arange(len(d.dates))
    prod = Policy(FEATURES, kind=kind, alpha=alpha, cost=tcost, reward_params=rp, hidden=16,
                  gamma=gamma).fit(d, allix)
    prod.save(MODELS / "relative_policy.json")
    meta = {"selected": {"kind": kind, "alpha": alpha, "variant": rp_name, "reward_params": rp,
                         "gamma": gamma, "train_cost": tcost, "excluded_classes": sorted(EXCLUDE)},
            "n_assets": int(d.X.shape[1]), "train_end_production": str(d.dates[-1].date()),
            "n_trials": len(trials), "runtime_min": round((time.time() - t0) / 60, 1),
            "linear_coefficients": coef.round(4).to_dict() if coef is not None else None}
    (REPORTS / "relative_meta.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 200)
    cols = ["CAGR", "CAGR_SPY", "Active_ann", "TE", "IR", "Sharpe", "MaxDD", "MaxRelDD", "HitDays", "Hit21d",
            "Reward21_bp", "Turnover_ann"]
    print(res[cols].round(3).to_string())
    print(json.dumps(meta, indent=1, default=str)[:1500])


if __name__ == "__main__":
    main()

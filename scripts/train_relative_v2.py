"""Mejora del modelo relativo al S&P 500 (universo global EEUU + UCITS) con progreso en tiempo real.

Protocolo:
- VALIDACIÓN walk-forward 2016, 2017, 2018: para cada año Y se entrena con [2005, Y) y se predice Y.
- Búsqueda greedy: cada ronda cambia UNA dimensión de la mejor configuración; criterio = recompensa
  exacta a 21 días en validación (exigiendo retorno activo > 0). Parada si la mejora < MIN_GAIN.
- TEST 2019→ walk-forward con reentrenamiento ANUAL (para cada año, entrenar con todo lo anterior),
  solo para la configuración ganadora y la referencia v1. Se mira una vez.
- Modelo de producción: ganadora reentrenada con todo el histórico → models/relative_policy_v2.json
Progreso: reports/training_progress.json (dashboard: python scripts/dashboard.py).
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
from quantpred.relative.data import FEATURES as F_V1  # noqa: E402
from quantpred.relative.data import DailyData  # noqa: E402
from quantpred.relative.global_data import FEATURES_V2, build_global  # noqa: E402
from quantpred.relative.policy import Policy, simulate  # noqa: E402
from quantpred.relative.progress import Progress, downsample  # noqa: E402

REPORTS, MODELS = ROOT / "reports", ROOT / "models"
COST_EVAL = 2e-4
VAL_YEARS = [2016, 2017, 2018]
TEST_START = 2019
MIN_GAIN = 2.0          # pb de recompensa media a 21 días
BASE = {"universe": "global", "features": "v2", "kind": "linear", "gamma": 0.6, "alpha": 1e-4,
        "tcost": 1e-3, "horizon": 21, "seeds": 1, "band": 0.0}
GRID = [("universe", ["us"]), ("features", ["v1"]), ("gamma", [0.4, 0.8]), ("kind", ["mlp"]),
        ("alpha", [1e-3, 1e-5]), ("tcost", [3e-3]), ("horizon", [63]), ("band", [0.1, 0.25]),
        ("seeds", [5])]


def subset(d: DailyData, info: dict, cfg: dict) -> DailyData:
    feats = F_V1 if cfg["features"] == "v1" else FEATURES_V2
    fi = [FEATURES_V2.index(f) for f in feats]
    mask = d.mask.copy()
    if cfg["universe"] == "us":
        mask[:, info["n_us"]:] = False
    return DailyData(d.dates, d.tickers, d.X[..., fi], mask, np.where(mask, d.R, 0.0), d.spy, d.rf, list(feats))


def make_policy(cfg, feats, seed=0):
    return Policy(feats, kind=cfg["kind"], alpha=cfg["alpha"], cost=cfg["tcost"], horizon=cfg["horizon"],
                  gamma=cfg["gamma"], hidden=16, reward_params=dict(rw.DEFAULT), seed=seed)


class Ensemble:
    def __init__(self, pols):
        self.pols = pols

    def weights(self, X, M):
        return np.mean([p.weights(X, M) for p in self.pols], axis=0)


def fit_cfg(cfg, dd, idx, prog=None, exp_id=None, fold=""):
    pols = []
    for s in range(cfg["seeds"]):
        p = make_policy(cfg, dd.features, seed=s)
        cb = (lambda it, J: prog.iteration(exp_id, it, J, fold)) if prog else None
        p.fit(dd, idx, callback=cb if s == 0 else None)
        pols.append(p)
    return Ensemble(pols) if len(pols) > 1 else pols[0]


def metrics(ret, spy, rf):
    delta = ret - spy
    yrs = len(ret) / 252
    eq, eqs = np.cumprod(1 + ret), np.cumprod(1 + spy)
    rel = eq / eqs
    d21 = np.convolve(delta, np.ones(21), "valid")
    return {"CAGR": eq[-1] ** (1 / yrs) - 1, "CAGR_SPY": eqs[-1] ** (1 / yrs) - 1,
            "Active_ann": delta.mean() * 252, "TE": delta.std() * np.sqrt(252),
            "IR": delta.mean() / (delta.std() + 1e-12) * np.sqrt(252),
            "Sharpe": (ret - rf).mean() / ((ret - rf).std() + 1e-12) * np.sqrt(252),
            "MaxDD": (eq / np.maximum.accumulate(eq) - 1).min(),
            "MaxRelDD": (rel / np.maximum.accumulate(rel) - 1).min(),
            "Hit21d": (d21 > 0).mean(), "Reward21_bp": rw.reward_exact(d21).mean() * 1e4}


def walk_forward(cfg, dd, years, prog=None, exp_id=None):
    """Concatena predicciones OOS año a año (entrena con [2005, Y))."""
    rets, spys, rfs, dates, tos = [], [], [], [], []
    for Y in years:
        tr = np.where((dd.dates >= "2005-01-01") & (dd.dates < f"{Y}-01-01"))[0]
        te = np.where(dd.dates.year == Y)[0]
        if len(te) == 0:
            continue
        model = fit_cfg(cfg, dd, tr, prog, exp_id, fold=str(Y))
        W = model.weights(dd.X[te], dd.mask[te])
        sim = simulate(W, dd, te, COST_EVAL, band=cfg["band"])
        rets.append(sim["ret"]), spys.append(sim["spy"]), rfs.append(dd.rf[te]), dates.append(dd.dates[te])
        tos.append(sim["turnover"])
    r, s = np.concatenate(rets), np.concatenate(spys)
    m = metrics(r, s, np.concatenate(rfs))
    m["Turnover_ann"] = np.concatenate(tos).mean() * 252
    D = pd.DatetimeIndex(np.concatenate(dates))
    return m, r, s, D


def main():
    t0 = time.time()
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    cache = PROCESSED / "study_relative_global.pkl"
    if cache.exists():
        d, info = pickle.load(open(cache, "rb"))
    else:
        d, info = build_global()
        pickle.dump((d, info), open(cache, "wb"))
    prog = Progress(REPORTS / "training_progress.json", n_total=1 + sum(len(v) for _, v in GRID))
    prog.phase("Validación walk-forward 2016-2018 (búsqueda greedy)")
    trials, cache_res = [], {}

    def evaluate(cfg, rnd):
        key = json.dumps(cfg, sort_keys=True)
        if key in cache_res:
            return cache_res[key]
        exp_id = f"E{len(trials) + 1:02d}"
        prog.start(exp_id, rnd, cfg)
        dd = subset(d, info, cfg)
        m, r, s, D = walk_forward(cfg, dd, VAL_YEARS, prog, exp_id)
        score = m["Reward21_bp"] if m["Active_ann"] > 0 else m["Reward21_bp"] - 1e4
        curve = downsample(D, {"rel": np.cumprod(1 + r) / np.cumprod(1 + s)}, every=5)
        trials.append({"id": exp_id, "round": rnd, **cfg, **m, "score": score})
        best = max(trials, key=lambda t: t["score"])
        prog.done(exp_id, m, curve, best={"id": best["id"], "score": best["score"],
                                          "Reward21_bp": best["Reward21_bp"], "Active_ann": best["Active_ann"]})
        print(f"  {exp_id} R{rnd} {cfg} → reward21={m['Reward21_bp']:.1f}bp activo={m['Active_ann']:.2%} "
              f"IR={m['IR']:.2f} TE={m['TE']:.1%}", flush=True)
        cache_res[key] = score
        return score

    best, best_sc = dict(BASE), evaluate(BASE, 0)
    for rnd in range(1, 4):
        start_sc = best_sc
        cand = dict(best)
        for dim, vals in GRID:
            for v in vals:
                if best[dim] == v:
                    continue
                cfg = {**best, dim: v}
                sc = evaluate(cfg, rnd)
                if sc > best_sc:
                    best_sc, cand = sc, cfg
        best = cand
        if best_sc - start_sc < MIN_GAIN:
            break
    pd.DataFrame(trials).to_csv(REPORTS / "relative_v2_trials.csv", index=False)
    print("GANADORA:", best, best_sc, flush=True)

    # ---------------- TEST walk-forward con reentrenamiento anual ----------------
    prog.phase("Test 2019→ (reentrenamiento anual, una sola evaluación)")
    years = list(range(TEST_START, d.dates.max().year + 1))
    ref_v1 = {"universe": "us", "features": "v1", "kind": "linear", "gamma": 0.6, "alpha": 1e-4,
              "tcost": 1e-3, "horizon": 21, "seeds": 1, "band": 0.0}
    out, curves = {}, {}
    for name, cfg in [("Modelo v2 (ganador)", best), ("Referencia v1 (solo EEUU)", ref_v1)]:
        exp_id = "TEST_v2" if "v2" in name else "TEST_v1"
        prog.start(exp_id, 99, cfg)
        m, r, s, D = walk_forward(cfg, subset(d, info, cfg), years, prog, exp_id)
        out[name] = m
        curves[name] = np.cumprod(1 + r)
        prog.done(exp_id, m)
    curves["S&P 500 (SPY)"] = np.cumprod(1 + s)
    out["S&P 500 (SPY)"] = metrics(s, s, d.rf[np.isin(d.dates, D)])
    res = pd.DataFrame(out).T
    res.to_csv(REPORTS / "relative_v2_test.csv")

    # ---------------- producción ----------------
    prog.phase("Entrenando modelo de producción (todo el histórico)")
    dd = subset(d, info, best)
    allix = np.where(dd.dates >= "2005-01-01")[0]
    pols = []
    for s_ in range(best["seeds"]):
        p = make_policy(best, dd.features, seed=s_)
        p.fit(dd, allix)
        pols.append(p.to_dict())
    prod = {"type": "ensemble" if len(pols) > 1 else "single", "members": pols, "config": best,
            "universe": {"n_us": info["n_us"], "n_eu": info["n_eu"] if best["universe"] == "global" else 0},
            "trained_until": str(dd.dates[-1].date()), "band": best["band"]}
    (MODELS / "relative_policy_v2.json").write_text(json.dumps(prod))
    prog.final({"selected": best, "metrics": [{"strategy": k, **v} for k, v in out.items()],
                "curves": downsample(D, curves, every=3), "runtime_min": round((time.time() - t0) / 60, 1)})
    pd.set_option("display.width", 200)
    print(res[["CAGR", "CAGR_SPY", "Active_ann", "TE", "IR", "MaxDD", "MaxRelDD", "Hit21d", "Reward21_bp",
               "Turnover_ann"]].round(3).to_string())


if __name__ == "__main__":
    main()

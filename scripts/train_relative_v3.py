"""Ronda v3: continúa desde la ganadora de v2 y añade γ dinámico por régimen (gate) y más coste de
entrenamiento. Mismo protocolo (validación walk-forward 2016-18, greedy, test anual una vez).
El progreso se AÑADE al mismo reports/training_progress.json (el dashboard sigue mostrando todo).
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import train_relative_v2 as v2  # noqa: E402
from quantpred.config import PROCESSED  # noqa: E402
from quantpred.relative.global_data import regime_series  # noqa: E402
from quantpred.relative.progress import Progress, downsample  # noqa: E402

GRID = [("gate", [True]), ("tcost", [1e-2]), ("gamma", [0.8, 1.0]), ("seeds", [5]), ("band", [0.1])]


def state_matrix(dates):
    r = pd.read_parquet(PROCESSED / "study_daily_returns_us.parquet", columns=["SPY"])["SPY"].astype(float)
    spy = (1 + r.fillna(0)).cumprod().to_frame("SPY")
    cb = pd.read_parquet(PROCESSED / "alt" / "cboe.parquet")
    vix = cb[cb["series"].astype(str) == "cboe_vix"].set_index("date")["value"]
    vz, sz = regime_series(spy, vix)
    vz, sz = vz.reindex(dates).fillna(0).values, sz.reindex(dates).fillna(0).values
    return np.c_[np.ones(len(dates)), vz, sz, np.abs(sz)]


def main():
    t0 = time.time()
    d, info = pickle.load(open(PROCESSED / "study_relative_global.pkl", "rb"))
    d.S = state_matrix(d.dates)
    T2 = pd.read_csv(v2.REPORTS / "relative_v2_trials.csv")
    best_row = T2.sort_values("score", ascending=False).iloc[0]
    best = {k: (best_row[k].item() if hasattr(best_row[k], "item") else best_row[k]) for k in v2.BASE}
    best = {k: (bool(v) if k == "gate" else v) for k, v in best.items()}
    best["gate"] = False
    best_sc = float(best_row["score"])
    prog = Progress(v2.REPORTS / "training_progress.json", n_total=sum(len(v) for _, v in GRID), resume=True)
    prog.phase("Ronda v3: γ dinámico por régimen y coste de entrenamiento (validación 2016-2018)")
    trials, n0 = [], len(prog.state["experiments"])

    orig_subset = v2.subset

    def subset(dd, inf, cfg):
        out = orig_subset(dd, inf, cfg)
        out.S = d.S
        return out
    v2.subset = subset
    orig_make = v2.make_policy

    def make_policy(cfg, feats, seed=0):
        p = orig_make(cfg, feats, seed)
        p.gate = bool(cfg.get("gate", False))
        return p
    v2.make_policy = make_policy

    _orig_weights = v2.Ensemble.weights
    # pesos con estado (gate): parchear Policy.weights vía closure en walk_forward
    from quantpred.relative.policy import Policy
    _pw = Policy.weights

    def pw(self, X, mask, th=None, S=None):
        if self.gate and S is None:
            S = pw.S_current
        return _pw(self, X, mask, th, S)
    Policy.weights = pw

    def walk_forward(cfg, dd, years, prog_=None, exp_id=None):
        rets, spys, rfs, dates, tos = [], [], [], [], []
        for Y in years:
            tr = np.where((dd.dates >= "2005-01-01") & (dd.dates < f"{Y}-01-01"))[0]
            te = np.where(dd.dates.year == Y)[0]
            if len(te) == 0:
                continue
            model = v2.fit_cfg(cfg, dd, tr, prog_, exp_id, fold=str(Y))
            pw.S_current = dd.S[te]
            W = model.weights(dd.X[te], dd.mask[te])
            sim = v2.simulate(W, dd, te, v2.COST_EVAL, band=cfg["band"])
            rets.append(sim["ret"]), spys.append(sim["spy"]), rfs.append(dd.rf[te]), dates.append(dd.dates[te])
            tos.append(sim["turnover"])
        r, s = np.concatenate(rets), np.concatenate(spys)
        m = v2.metrics(r, s, np.concatenate(rfs))
        m["Turnover_ann"] = np.concatenate(tos).mean() * 252
        return m, r, s, pd.DatetimeIndex(np.concatenate(dates))

    def evaluate(cfg, rnd):
        exp_id = f"E{n0 + len(trials) + 1:02d}"
        prog.start(exp_id, rnd, cfg)
        dd = v2.subset(d, info, cfg)
        m, r, s, D = walk_forward(cfg, dd, v2.VAL_YEARS, prog, exp_id)
        score = m["Reward21_bp"] if m["Active_ann"] > 0 else m["Reward21_bp"] - 1e4
        trials.append({"id": exp_id, "round": rnd, **cfg, **m, "score": score})
        allb = max([*trials, {"id": best_row["id"], "score": best_sc, "Reward21_bp": best_row["Reward21_bp"],
                              "Active_ann": best_row["Active_ann"]}], key=lambda t: t["score"])
        prog.done(exp_id, m, downsample(D, {"rel": np.cumprod(1 + r) / np.cumprod(1 + s)}),
                  best={k: allb[k] for k in ("id", "score", "Reward21_bp", "Active_ann")})
        print(f"  {exp_id} R{rnd} {cfg} → reward21={m['Reward21_bp']:.1f}bp activo={m['Active_ann']:.2%} "
              f"IR={m['IR']:.2f} TE={m['TE']:.1%}", flush=True)
        return score

    for rnd in range(10, 13):
        start, cand = best_sc, dict(best)
        for dim, vals in GRID:
            for v in vals:
                if best.get(dim) == v:
                    continue
                cfg = {**best, dim: v}
                sc = evaluate(cfg, rnd)
                if sc > best_sc:
                    best_sc, cand = sc, cfg
        best = cand
        if best_sc - start < v2.MIN_GAIN:
            break
    pd.DataFrame(trials).to_csv(v2.REPORTS / "relative_v3_trials.csv", index=False)
    print("GANADORA v3:", best, best_sc, flush=True)

    prog.phase("Test 2019→ (reentrenamiento anual, una sola evaluación)")
    years = list(range(v2.TEST_START, d.dates.max().year + 1))
    out, curves = {}, {}
    prev = pd.read_csv(v2.REPORTS / "relative_v2_test.csv", index_col=0)
    for name, cfg in [("Modelo v3 (ganador)", best)]:
        prog.start("TEST_v3", 99, cfg)
        m, r, s, D = walk_forward(cfg, v2.subset(d, info, cfg), years, prog, "TEST_v3")
        out[name] = m
        curves[name] = np.cumprod(1 + r)
        prog.done("TEST_v3", m)
    for k in prev.index:
        out[k] = prev.loc[k].to_dict()
    curves["S&P 500 (SPY)"] = np.cumprod(1 + s)
    pd.DataFrame(out).T.to_csv(v2.REPORTS / "relative_v3_test.csv")

    prog.phase("Entrenando modelo de producción (todo el histórico)")
    dd = v2.subset(d, info, best)
    allix = np.where(dd.dates >= "2005-01-01")[0]
    members = []
    for s_ in range(int(best["seeds"])):
        p = v2.make_policy(best, dd.features, seed=s_)
        p.fit(dd, allix)
        members.append(p.to_dict())
    prod = {"type": "ensemble" if len(members) > 1 else "single", "members": members, "config": best,
            "universe": {"n_us": info["n_us"], "n_eu": info["n_eu"] if best["universe"] == "global" else 0},
            "state": ["1", "vix_z", "spy21_z", "abs_spy21_z"], "trained_until": str(dd.dates[-1].date()),
            "band": best["band"]}
    (v2.MODELS / "relative_policy_v3.json").write_text(json.dumps(prod))
    v2curve = None
    prog.final({"selected": best, "metrics": [{"strategy": k, **v} for k, v in out.items()],
                "curves": downsample(D, curves, every=3), "runtime_min": round((time.time() - t0) / 60, 1)})
    print(pd.DataFrame(out).T[["CAGR", "Active_ann", "TE", "IR", "MaxDD", "MaxRelDD", "Hit21d", "Reward21_bp"]]
          .round(3).to_string())


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantpred.live.engine import SimulatedPortfolio  # noqa: E402
from quantpred.relative import reward as rw  # noqa: E402
from quantpred.relative.data import FEATURES, DailyData  # noqa: E402
from quantpred.relative.policy import Policy  # noqa: E402


def test_reward_shape():
    assert rw.reward_exact(0.0) == 0.0
    assert rw.reward_exact(0.02) > 0.02                       # batir puntúa más que el Δ
    assert rw.reward_exact(-0.02) < -0.02                     # quedarse por debajo penaliza extra
    assert rw.reward_exact(-0.04) - rw.reward_exact(-0.02) < -0.02   # penalización creciente (cuadrática)
    z = np.random.default_rng(0).normal(0, 0.02, 200_000)
    assert rw.reward_exact(z).mean() < 0                      # el ruido no se premia (calibración)


def _toy(T=80, N=8, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, N, len(FEATURES)))
    M = rng.random((T, N)) > 0.1
    M[:, 0] = True
    tick = np.array([[f"t{i}" for i in range(N)]] * T, dtype=object)
    return DailyData(pd.date_range("2020", periods=T), tick, X, M, rng.normal(0, .01, (T, N)),
                     rng.normal(0, .01, T), np.zeros(T), FEATURES), rng


def test_policy_gradient_matches_finite_differences():
    d, rng = _toy()
    for kw in ({"kind": "linear", "gamma": 1.0}, {"kind": "linear", "gamma": 0.5}, {"kind": "mlp", "hidden": 3}):
        p = Policy(FEATURES, horizon=21, **kw)
        th = rng.normal(0, .3, p.n_params())
        idx = np.arange(len(d.dates))
        _, g = p.objective(th, d, idx)
        e = 1e-6
        num = np.array([(p.objective(th + e * np.eye(len(th))[j], d, idx)[0] -
                         p.objective(th - e * np.eye(len(th))[j], d, idx)[0]) / (2 * e) for j in range(len(th))])
        assert np.max(np.abs(num - g)) / np.max(np.abs(num)) < 1e-5


def test_weights_long_only_and_core():
    d, _ = _toy()
    p = Policy(FEATURES, gamma=0.6)
    p.theta = np.random.default_rng(2).normal(size=p.n_params())
    w = p.weights(d.X, d.mask)
    assert np.allclose(w.sum(1), 1) and (w >= 0).all() and (w[:, 0] >= 0.4 - 1e-12).all()


def test_simulated_portfolio_vs_benchmark():
    pf = SimulatedPortfolio(capital=1000, cost_bps=0.0)
    pf.rebalance(pd.Series({"SPY": 0.5, "TLT": 0.5}), {"SPY": 100.0, "TLT": 50.0}, "2024-01-02 15:59")
    pt = pf.on_tick("2024-01-02 16:00", {"SPY": 110.0, "TLT": 50.0})
    assert abs(pt["modelo"] - 1050) < 1e-9 and abs(pt["sp500"] - 1100) < 1e-9
    assert abs(pt["diff_pct"] + 5.0) < 1e-9
    assert pf.tickers() == ["SPY", "TLT"]

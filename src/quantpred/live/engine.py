"""Motor de ejecución simulada: modelo diario + valoración minuto a minuto frente al S&P 500.

Flujo en producción (o en un proyecto de gráfico en tiempo real):

    eng = LiveEngine.from_files("models/relative_policy_core_satellite.json")
    eng.prepare(daily_closes, daily_dollar_volume, vix)     # 1 vez al día (histórico diario)
    w = eng.target_weights()                                 # pesos objetivo del día
    pf = SimulatedPortfolio(capital=100_000)
    pf.rebalance(w, precios_actuales, ts)                    # p.ej. cerca del cierre (MOC)
    for ts, precios in feed:                                 # cada minuto
        punto = pf.on_tick(ts, precios)  # {'ts','modelo','sp500','diff_pct',...} -> graficar

`SimulatedPortfolio` mantiene en paralelo un benchmark (mismo capital 100% en SPY), así las dos curvas
se pueden superponer directamente. El modelo se entrenó con datos diarios; la valoración intradía es
mark-to-market de las posiciones (no se reentrena por minuto).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantpred.relative.data import FEATURES, cross_section, feature_frames, select_universe
from quantpred.relative.policy import Policy


class LiveEngine:
    def __init__(self, policy: Policy, n_assets: int = 80, exclude: set = frozenset(), min_weight: float = 0.002):
        self.policy = policy
        self.n_assets = n_assets
        self.exclude = set(exclude)
        self.min_weight = min_weight
        self.universe: list = []
        self.X: np.ndarray | None = None
        self.asof = None

    @classmethod
    def from_files(cls, model_path: str, exclude_tickers: set = frozenset(), **kw) -> "LiveEngine":
        import json
        with open(model_path) as fh:
            pol = Policy.from_dict(json.load(fh))
        return cls(pol, exclude=exclude_tickers, **kw)

    def prepare(self, closes: pd.DataFrame, dollar_volume: pd.DataFrame, vix: pd.Series | None = None):
        """closes/dollar_volume: históricos diarios (fecha x ticker) hasta el último cierre (≥ 300 días)."""
        d = closes.index[-1]
        # como en entrenamiento: universo fijado el primer día hábil del mes en curso
        month = closes.index[(closes.index.year == d.year) & (closes.index.month == d.month)]
        self.universe = select_universe(closes, dollar_volume, month[0], self.n_assets, self.exclude)
        feats, _ = feature_frames(closes[self.universe], vix)
        self.X = cross_section(feats, len(closes) - 1, self.universe)
        self.asof = d
        return self

    def target_weights(self) -> pd.Series:
        mask = np.ones((1, len(self.universe)), bool)
        w = self.policy.weights(self.X[None, :, :], mask)[0]
        w = pd.Series(w, index=self.universe)
        # podar posiciones residuales (< min_weight) y renormalizar: menos órdenes y menos ticks a seguir
        w = w[w >= self.min_weight]
        return w / w.sum()


@dataclass
class SimulatedPortfolio:
    capital: float = 100_000.0
    cost_bps: float = 2.0               # coste por operación (spread + comisión)
    benchmark: str = "SPY"
    cash: float = field(init=False)
    shares: dict = field(default_factory=dict)
    bench_shares: float = 0.0
    history: list = field(default_factory=list)
    last_prices: dict = field(default_factory=dict)

    def __post_init__(self):
        self.cash = self.capital

    def value(self, prices: dict | None = None) -> float:
        p = {**self.last_prices, **(prices or {})}
        return self.cash + sum(q * p[t] for t, q in self.shares.items() if t in p)

    def rebalance(self, weights: pd.Series, prices: dict, ts=None) -> float:
        """Lleva la cartera a los pesos objetivo a los precios dados. Devuelve el coste pagado."""
        self.last_prices.update(prices)
        if self.bench_shares == 0.0 and self.benchmark in prices:
            self.bench_shares = self.capital / prices[self.benchmark]
        nav = self.value()
        target = {t: nav * w / prices[t] for t, w in weights.items() if t in prices and prices[t] > 0}
        traded = sum(abs(target.get(t, 0) - self.shares.get(t, 0)) * prices[t]
                     for t in set(target) | set(self.shares) if t in prices)
        cost = traded * self.cost_bps / 1e4
        self.shares = {t: q for t, q in target.items() if q > 0}
        self.cash = nav - sum(q * prices[t] for t, q in self.shares.items()) - cost
        if ts is not None:
            self.on_tick(ts, prices)
        return cost

    def on_tick(self, ts, prices: dict) -> dict:
        """Valora la cartera y el benchmark con los precios del minuto (dict ticker->precio)."""
        self.last_prices.update(prices)
        v = self.value()
        b = self.bench_shares * self.last_prices.get(self.benchmark, np.nan)
        point = {"ts": pd.Timestamp(ts), "modelo": v, "sp500": b,
                 "modelo_ret": v / self.capital - 1, "sp500_ret": b / self.capital - 1,
                 "diff_pct": (v - b) / self.capital * 100}
        self.history.append(point)
        return point

    def tickers(self) -> list:
        """Tickers a suscribir en el feed de precios (posiciones + benchmark)."""
        return sorted(set(self.shares) | {self.benchmark})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history).set_index("ts")

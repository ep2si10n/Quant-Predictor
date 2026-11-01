"""Política de cartera entrenada directamente con la recompensa relativa al SPY.

Puntuación por activo  s_ti = g_θ(x_ti)  (lineal o MLP de una capa oculta tanh)
Pesos (long-only, 100% invertido)  w_t = softmax_i(s_ti)  sobre los activos disponibles
Retorno activo neto  Δ_t = Σ_i w_ti R_ti − c·Σ_i |w_ti − w_{t−1,i}| − r_SPY,t
Objetivo  J(θ) = mean_t f(D_t) − α‖θ‖²,  D_t = Σ_{τ=t−H+1..t} Δ_τ  (rendimiento relativo a H días)
           (f = reward.reward, versión suave; H=21 evita premiar el ruido diario)

Gradiente analítico (regla de la cadena por el softmax):
  ∂J/∂w_ti = (1/T)[ f'(Δ_t)(R_ti − c·sg_ti) + f'(Δ_{t+1})·c·sg_{t+1,i} ],  sg = d|x|/dx suavizado
  ∂J/∂s_ti = w_ti (∂J/∂w_ti − Σ_j w_tj ∂J/∂w_tj)
y se retropropaga a θ. Optimización L-BFGS (scipy).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from quantpred.relative import reward as rw
from quantpred.relative.data import DailyData

SG_K = 200.0


@dataclass
class Policy:
    features: list
    kind: str = "linear"            # "linear" | "mlp"
    hidden: int = 16
    alpha: float = 1e-3             # L2
    cost: float = 2e-4              # coste por unidad de rotación (2 pb)
    reward_params: dict = field(default_factory=lambda: dict(rw.DEFAULT))
    horizon: int = 21               # días de la ventana de rendimiento relativo
    gamma: float = 1.0              # fracción satélite: w = (1-γ)·SPY + γ·softmax (SPY = hueco 0)
    gate: bool = False              # γ_t = σ(S_t·g) aprendido según el régimen (S = estado de mercado)
    theta: np.ndarray | None = None
    seed: int = 0

    # ---------------- parámetros ----------------
    N_STATE = 4                     # S_t = [1, VIX_z, SPY21_z, |SPY21_z|]

    def n_score(self):
        F = len(self.features)
        return F if self.kind == "linear" else F * self.hidden + self.hidden + self.hidden

    def n_params(self):
        return self.n_score() + (self.N_STATE if self.gate else 0)

    def _split(self, th):
        n = self.n_score()
        return th[:n], (th[n:] if self.gate else None)

    def gammas(self, S, th=None):
        """γ_t por fecha (vector) o escalar fijo."""
        th = self.theta if th is None else th
        if not self.gate:
            return None
        _, gp = self._split(th)
        return 1 / (1 + np.exp(-(S @ gp)))

    def _unpack(self, th):
        F, H = len(self.features), self.hidden
        if self.kind == "linear":
            return th, None, None
        W1 = th[:F * H].reshape(F, H)
        b1 = th[F * H:F * H + H]
        w2 = th[F * H + H:]
        return W1, b1, w2

    # ---------------- forward ----------------
    def scores(self, X, th=None):
        th = self.theta if th is None else th
        th = self._split(th)[0]
        if self.kind == "linear":
            return X @ th, None
        W1, b1, w2 = self._unpack(th)
        Hh = np.tanh(X @ W1 + b1)
        return Hh @ w2, Hh

    @staticmethod
    def softmax(s, mask):
        s = np.where(mask, s, -np.inf)
        s = s - s.max(axis=1, keepdims=True)
        e = np.where(mask, np.exp(s), 0.0)
        return e / e.sum(axis=1, keepdims=True)

    def weights(self, X, mask, th=None, S=None):
        ws = self.softmax(self.scores(X, th)[0], mask)
        return self._core(ws, self.gammas(S, th) if self.gate else None)

    def _core(self, ws, gt=None):
        if gt is None:
            if self.gamma >= 1.0:
                return ws
            gt = np.full(len(ws), self.gamma)
        w = gt[:, None] * ws
        w[:, 0] += 1 - gt
        return w

    # ---------------- objetivo y gradiente ----------------
    def objective(self, th, d: DailyData, idx):
        X, M, R, spy, tick = d.X[idx], d.mask[idx], d.R[idx], d.spy[idx], d.tickers[idx]
        T = len(idx)
        s, Hh = self.scores(X, th)
        ws = self.softmax(s, M)
        gt = self.gammas(d.S[idx], th) if self.gate else None
        w = self._core(ws, gt)
        same = np.zeros_like(M)
        same[1:] = (tick[1:] == tick[:-1]) & M[1:] & M[:-1]
        wprev = np.zeros_like(w)
        wprev[1:] = np.where(same[1:], w[:-1], 0.0)
        diff = w - wprev
        turnover = np.sqrt(diff ** 2 + 1e-10).sum(1)
        delta = (w * R).sum(1) - self.cost * turnover - spy
        H = self.horizon
        if H > 1:   # recompensa sobre la diferencia acumulada en ventanas de H días (suma de Δ)
            D = np.convolve(delta, np.ones(H), mode="valid")            # [T-H+1]
            f = rw.reward(D, **self.reward_params)
            fpD = rw.reward_grad(D, **self.reward_params) / len(D)
            fp = np.convolve(fpD, np.ones(H), mode="full") * T          # ∂/∂Δ_τ = Σ ventanas que contienen τ
        else:
            f = rw.reward(delta, **self.reward_params)
            fp = rw.reward_grad(delta, **self.reward_params)
        sg = diff / np.sqrt(diff ** 2 + 1e-10)          # derivada exacta de |x| suavizado
        g = fp[:, None] * (R - self.cost * sg)
        g[:-1] += np.where(same[1:], fp[1:, None] * self.cost * sg[1:], 0.0)
        g /= T
        gam = gt if gt is not None else np.full(T, min(self.gamma, 1.0))
        ds = gam[:, None] * ws * (g - (ws * g).sum(1, keepdims=True))
        ds = np.where(M, ds, 0.0)
        ths, gp = self._split(th)
        if self.kind == "linear":
            grad = np.einsum("tn,tnf->f", ds, X)
        else:
            W1, b1, w2 = self._unpack(ths)
            g_w2 = np.einsum("tn,tnh->h", ds, Hh)
            dH = ds[..., None] * w2 * (1 - Hh ** 2)
            g_W1 = np.einsum("tnf,tnh->fh", X, dH)
            g_b1 = dH.sum((0, 1))
            grad = np.concatenate([g_W1.ravel(), g_b1, g_w2])
        if self.gate:   # ∂J/∂γ_t = Σ_i g_ti (ws_ti − e0_i);  ∂γ_t/∂g = γ_t(1−γ_t)·S_t
            dgam = (ws * g).sum(1) - g[:, 0]
            grad = np.concatenate([grad, (dgam * gt * (1 - gt)) @ d.S[idx]])
        J = f.mean() - self.alpha * (th ** 2).sum()
        grad = grad - 2 * self.alpha * th
        return -J, -grad

    def fit(self, d: DailyData, idx, maxiter: int = 300, callback=None, theta0=None):
        """callback(it, objective) se llama en cada iteración de L-BFGS (para el dashboard)."""
        rng = np.random.default_rng(self.seed)
        th0 = rng.normal(0, 0.01, self.n_params()) if theta0 is None else np.array(theta0, float)
        if theta0 is None and self.kind == "linear" and self.gamma >= 1.0 and not self.gate:
            th0[self.features.index("is_spy")] = 1.0   # arranque cercano al SPY
        if theta0 is None and self.gate:              # γ inicial = self.gamma
            gm = min(max(self.gamma, 0.05), 0.95)
            th0[self.n_score():] = 0.0
            th0[self.n_score()] = np.log(gm / (1 - gm))
        it = [0]

        def cb(xk):
            it[0] += 1
            if callback is not None:
                callback(it[0], -self.objective(xk, d, idx)[0])

        res = minimize(self.objective, th0, args=(d, idx), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter}, callback=cb if callback else None)
        self.theta = res.x
        return self

    # ---------------- exportación ----------------
    def to_dict(self):
        return {"kind": self.kind, "hidden": self.hidden, "features": self.features,
                "theta": self.theta.tolist(), "alpha": self.alpha, "cost": self.cost,
                "reward_params": self.reward_params, "horizon": self.horizon,
                "gamma": self.gamma, "gate": self.gate}

    @classmethod
    def from_dict(cls, dct):
        p = cls(features=dct["features"], kind=dct["kind"], hidden=dct["hidden"], alpha=dct["alpha"],
                cost=dct["cost"], reward_params=dct["reward_params"], horizon=dct.get("horizon", 1),
                gamma=dct.get("gamma", 1.0), gate=dct.get("gate", False))
        p.theta = np.asarray(dct["theta"])
        return p

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=1)


def simulate(weights: np.ndarray, d: DailyData, idx, cost: float = 2e-4, band: float = 0.0) -> dict:
    """Backtest exacto: deriva de pesos entre días, coste por rotación, recompensa exacta.
    band > 0: banda de no-operación — solo se rebalancea si la distancia L1 entre la cartera actual
    (tras la deriva) y el objetivo supera `band`; si no, se mantienen las posiciones."""
    R, M, tick, spy = d.R[idx], d.mask[idx], d.tickers[idx], d.spy[idx]
    held: dict = {}
    rp, to = np.zeros(len(idx)), np.zeros(len(idx))
    for t in range(len(idx)):
        tgt = {tick[t, i]: weights[t, i] for i in range(R.shape[1]) if M[t, i] and weights[t, i] > 1e-6}
        keys = set(tgt) | set(held)
        dist = sum(abs(tgt.get(k, 0) - held.get(k, 0)) for k in keys)
        avail = {tick[t, i]: i for i in range(R.shape[1]) if M[t, i]}
        if band > 0 and held and dist <= band and all(k in avail for k in held):
            cur = held                                   # dentro de la banda: no se opera
            to[t] = 0.0
        else:
            cur = tgt
            to[t] = dist
        rp[t] = sum(w * R[t, avail[k]] for k, w in cur.items() if k in avail) - cost * to[t]
        # deriva
        new = {k: w * (1 + R[t, avail[k]]) for k, w in cur.items() if k in avail and w > 1e-9}
        tot = sum(new.values())
        held = {k: v / tot for k, v in new.items()} if tot > 0 else {}
    return {"ret": rp, "spy": spy, "turnover": to}

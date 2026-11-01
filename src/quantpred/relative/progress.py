"""Registro de progreso del entrenamiento para el dashboard en tiempo real.

Escribe `reports/training_progress.json` de forma atómica (tmp + rename) tras cada evento. Estructura:
  meta: {status, started, updated, phase, current, n_done, n_total, best}
  experiments: [{id, round, config, status, iters:[[it, J]], val:{...}, val_curve:[[fecha, rel]]}]
  final: {selected, metrics:[...], curves:{fecha:[...], modelo:[...], SPY:[...], ...}}
El dashboard (dashboard/index.html) lo lee en local; desde Claude se sube al artifact.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


class Progress:
    def __init__(self, path: Path, n_total: int = 0, resume: bool = False):
        self.path = Path(path)
        if resume and self.path.exists():
            self.state = json.loads(self.path.read_text())
            self.state["meta"].update(status="running", n_total=self.state["meta"]["n_done"] + n_total)
            self.state["final"] = None
            self._last = 0.0
            self.flush(force=True)
            return
        self.state = {"meta": {"status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "updated": None, "phase": "", "current": None, "n_done": 0,
                               "n_total": n_total, "best": None},
                      "experiments": [], "final": None}
        self._last = 0.0
        self.flush(force=True)

    def flush(self, force: bool = False):
        now = time.time()
        if not force and now - self._last < 1.0:
            return
        self._last = now
        self.state["meta"]["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_clean(self.state), separators=(",", ":")))
        os.replace(tmp, self.path)

    def phase(self, name: str, n_total: int | None = None):
        self.state["meta"]["phase"] = name
        if n_total is not None:
            self.state["meta"]["n_total"] = n_total
        self.flush(force=True)

    def start(self, exp_id: str, rnd: int, config: dict):
        self.state["experiments"].append({"id": exp_id, "round": rnd, "config": config, "status": "training",
                                          "iters": [], "val": None, "val_curve": None})
        self.state["meta"]["current"] = exp_id
        self.flush(force=True)

    def iteration(self, exp_id: str, it: int, J: float, fold: str = ""):
        e = self._get(exp_id)
        e["iters"].append([len(e["iters"]) + 1, J, fold])
        if len(e["iters"]) > 600:          # compactar
            e["iters"] = e["iters"][::2]
        self.flush()

    def done(self, exp_id: str, val: dict, val_curve=None, best=None):
        e = self._get(exp_id)
        e["status"] = "done"
        e["val"] = val
        e["val_curve"] = val_curve
        self.state["meta"]["n_done"] += 1
        if best is not None:
            self.state["meta"]["best"] = best
        self.flush(force=True)

    def final(self, payload: dict):
        self.state["final"] = payload
        self.state["meta"]["status"] = "finished"
        self.state["meta"]["current"] = None
        self.flush(force=True)

    def _get(self, exp_id):
        return next(e for e in self.state["experiments"] if e["id"] == exp_id)


def downsample(dates, values: dict, every: int = 5):
    idx = list(range(0, len(dates), every))
    if idx[-1] != len(dates) - 1:
        idx.append(len(dates) - 1)
    out = {"fecha": [str(dates[i].date()) for i in idx]}
    for k, v in values.items():
        out[k] = [float(v[i]) for i in idx]
    return out

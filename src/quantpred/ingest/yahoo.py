"""Cliente HTTP mínimo para Yahoo Finance.

Motivo: `yfinance` necesita cookie+crumb para todo y ese paso sufre rate-limit (HTTP 429)
con facilidad. El endpoint `v8/finance/chart` NO necesita crumb, así que los precios
se descargan directamente. `quoteSummary` (categoría Morningstar, AUM, TER) sí lo necesita:
se intenta con un crumb propio y, si falla, se recurre a clasificación por nombre.
"""
from __future__ import annotations

import random
import threading
import time

import numpy as np
import pandas as pd
import requests

UA = "Mozilla/5.0 (research; quant-predictor)"
PAUSE = 0.35  # s entre peticiones por hilo (cortesía; Yahoo limita con 429)
PERIOD1_MIN = -2208988800  # 1900-01-01
HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
_local = threading.local()


class RateLimited(Exception):
    pass


def _session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": UA, "Accept": "application/json"})
        _local.s = s
    return s


def _get_json(path: str, params: dict, retries: int = 6) -> dict:
    """GET con cortesía: pausa fija y, ante 429, espera Retry-After o backoff exponencial largo."""
    last = None
    for attempt in range(retries):
        host = random.choice(HOSTS)
        time.sleep(PAUSE)
        try:
            r = _session().get(f"https://{host}{path}", params=params, timeout=30)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After") or min(300, 30 * 2 ** attempt))
                time.sleep(wait)
                raise RateLimited(path)
            if r.status_code == 404:
                return r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            r.raise_for_status()
            return r.json()
        except RateLimited as e:
            last = e
        except (requests.RequestException, ValueError) as e:
            last = e
            time.sleep(min(60, 2 ** attempt + random.random()))
    raise RuntimeError(f"Yahoo {path}: {last}")


# --- Precios -----------------------------------------------------------------------------------
def chart(symbol: str, range_: str = "max") -> tuple[pd.DataFrame, dict]:
    """OHLCV diario + adj_close + eventos. Devuelve (df largo de un ticker, meta)."""
    # range=max degrada a mensual en algunos mercados (p.ej. Xetra) -> period1/period2 explícitos
    params = {"interval": "1d", "events": "div,splits,capitalGains", "includeAdjustedClose": "true"}
    if range_ == "max":
        params.update(period1=PERIOD1_MIN, period2=int(time.time()) + 86400)
    else:
        params["range"] = range_
    js = _get_json(f"/v8/finance/chart/{symbol}", params)
    res = (js.get("chart") or {}).get("result")
    if not res:
        return pd.DataFrame(), {}
    r = res[0]
    meta = r.get("meta", {})
    ts = r.get("timestamp")
    if not ts:
        return pd.DataFrame(), meta
    tz = meta.get("exchangeTimezoneName") or "America/New_York"
    dates = pd.to_datetime(np.asarray(ts, dtype="int64"), unit="s", utc=True).tz_convert(tz)
    dates = dates.tz_localize(None).normalize()
    q = r["indicators"]["quote"][0]
    adj = (r["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    df = pd.DataFrame({
        "date": dates, "ticker": symbol,
        "open": q.get("open"), "high": q.get("high"), "low": q.get("low"), "close": q.get("close"),
        "adj_close": adj if adj is not None else q.get("close"), "volume": q.get("volume"),
    })
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ev = r.get("events", {})

    def _events(key: str, fn) -> pd.Series:
        items = ev.get(key) or {}
        if not items:
            return pd.Series(dtype="float64")
        d = pd.to_datetime([int(v["date"]) for v in items.values()], unit="s", utc=True)
        d = d.tz_convert(tz).tz_localize(None).normalize()
        return pd.Series([fn(v) for v in items.values()], index=d).groupby(level=0).sum()

    div = _events("dividends", lambda v: float(v.get("amount", 0)))
    spl = _events("splits", lambda v: float(v["numerator"]) / float(v["denominator"])
                  if v.get("denominator") else 0.0)
    cg = _events("capitalGains", lambda v: float(v.get("amount", 0)))
    df["dividends"] = df["date"].map(div).fillna(0.0)
    df["splits"] = df["date"].map(spl).fillna(0.0)
    df["capital_gains"] = df["date"].map(cg).fillna(0.0)
    df = df.dropna(subset=["close", "adj_close"], how="all")
    # el último día puede ser una barra intradía duplicada -> nos quedamos con la última
    df = df.drop_duplicates("date", keep="last")
    return df, meta


# --- Metadatos (requiere crumb) -------------------------------------------------------------------
_crumb_lock = threading.Lock()
_crumb: dict = {}


def _get_crumb() -> str | None:
    with _crumb_lock:
        if "v" in _crumb:
            return _crumb["v"]
        s = _session()
        try:
            s.get("https://fc.yahoo.com", timeout=20, allow_redirects=True)
        except requests.RequestException:
            pass
        r = s.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=20)
        if r.status_code != 200 or not r.text or "<" in r.text:
            return None
        _crumb["v"] = r.text.strip()
        _crumb["cookies"] = s.cookies
        return _crumb["v"]


def quote_summary(symbol: str, modules: str = "quoteType,fundProfile,summaryDetail,defaultKeyStatistics") -> dict:
    crumb = _get_crumb()
    if crumb is None:
        raise RateLimited("sin crumb")
    s = _session()
    if "cookies" in _crumb:
        s.cookies.update(_crumb["cookies"])
    r = s.get(f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}",
              params={"modules": modules, "crumb": crumb}, timeout=30)
    if r.status_code == 429:
        raise RateLimited(symbol)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    res = (r.json().get("quoteSummary") or {}).get("result") or [{}]
    return res[0]

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantpred.ingest.metadata import classify  # noqa: E402
from quantpred.ingest.quality import clean_prices  # noqa: E402
from quantpred.ingest.universe import yahoo_symbol  # noqa: E402


def test_yahoo_symbol():
    assert yahoo_symbol("BRK.B") == "BRK-B"
    assert yahoo_symbol("SPY") == "SPY"


def test_classify_by_category():
    assert classify("Large Blend", "SPDR S&P 500") == "equity"
    assert classify("Long Government", "iShares 20+ Year Treasury") == "fixed_income"
    assert classify("Commodities Focused", "SPDR Gold Shares") == "commodity"
    assert classify("Trading--Leveraged Equity", "ProShares UltraPro QQQ") == "leveraged_inverse"
    assert classify("Digital Assets", "ProShares Bitcoin ETF") == "crypto"
    assert classify("Real Estate", "Vanguard Real Estate") == "real_estate"
    assert classify("Derivative Income", "JPMorgan Equity Premium Income") == "alternative"


def test_classify_by_name():
    assert classify(None, "iShares 0-3 Month Treasury Bond ETF") == "fixed_income"
    assert classify(None, "SPDR Gold Shares") == "commodity"
    assert classify(None, "VanEck Gold Miners ETF") == "equity"
    assert classify(None, "Direxion Daily Semiconductor Bull 3X") == "leveraged_inverse"
    assert classify(None, "iShares Short Treasury Bond ETF") == "fixed_income"


def test_clean_prices_flags_and_bounds():
    d = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03", "2020-01-06"]),
        "ticker": "X",
        "open": [10, 10, 10, 20, -1], "high": [9, 11, 11, 21, 21], "low": [9, 9, 9, 19, 19],
        "close": [10, 10.5, 10.5, 20, 20], "adj_close": [10, 10.5, 10.5, 20, 20],
        "volume": [100, 0, 0, 100, 0], "dividends": 0.0, "splits": 0.0, "capital_gains": 0.0,
    })
    c, rep = clean_prices(d)
    assert len(c) == 4                                   # duplicado eliminado
    assert (c["high"] >= c[["open", "close"]].max(axis=1)).all()
    assert c["ret_outlier"].sum() == 1                   # +90% el 03-ene
    assert np.isnan(c.loc[c["date"] == "2020-01-06", "open"]).all()
    assert c.loc[c["date"] == "2020-01-06", "stale"].item()
    assert rep.loc[0, "n_rows"] == 4

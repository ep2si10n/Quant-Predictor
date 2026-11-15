<div align="center">

# Quant-Predictor

**ETF return and volatility prediction with overfitting-aware backtesting.**

[![CI](https://github.com/ep2si10n/Quant-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/ep2si10n/Quant-Predictor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

## Overview

Quant-Predictor is a research pipeline that asks a simple question: can a quant-style model, trained on a broad
ETF universe plus macro and alternative data, build a strategy that beats simple benchmarks (SPY, 60/40) out of
sample after costs?

The repository covers the whole workflow: data ingestion (about 9,400 US and European ETFs, macro and alternative
series), a comparative study of ML and rule-based strategies, an iteration loop guided by failure analysis, and a
live-simulation engine that values a portfolio minute by minute against the SPY.

The main point of the project is methodological. Every attempt is logged, so the Deflated Sharpe Ratio and the
Probability of Backtest Overfitting (PBO) are computed with the real number of trials, and the test period is
looked at once per strategy version. Most of the sophisticated models did not survive that protocol, and the
results below report that as it is.

## Key results

Protocol: development 2008-2018 (all selection), test 2019-2026 (evaluated once), 5 bp transaction costs,
125 configurations registered across phases 3, 4 and 4b. Source: [`reports/phase3_results.md`](reports/phase3_results.md),
[`reports/PHASE4_SUMMARY.md`](reports/PHASE4_SUMMARY.md).

### What did not work

| Approach | Dev Sharpe | Test Sharpe | Diagnosis |
|---|---|---|---|
| ML ETF selection (LightGBM / Ridge, top-20) | -0.29 to 0.12 | 0.14 to 0.61 | Positive IC came from avoiding low-beta assets (it learned the market premium); turnover about 15x per year |
| Two-level: ERC across asset classes + selection (phase 4) | 0.53 | 0.29 | Commodities in 2008-15 and the trend filter hurt in 2019-26 |
| "Bad month" classifier (logit, LightGBM, MLP) | AUC 0.45-0.53 | AUC 0.42-0.59 | No skill: tail risk is not predictable with these variables |
| Volatility + regime combo chosen on dev | **1.16** | 0.35 | Overfit (PBO 0.60) |

The correlation between development and test Sharpe across variants is 0.12, so the development ranking barely
predicts the test result. Simple rules with a theoretical basis were preferred over tuned ones.

### What held up

- **HAR volatility forecasting** (Corsi, 2009): R² versus a naive forecast is +17% on dev and +19% on test.
- **Volatility-managed 60/40** (exposure = sigma_ref / sigma_hat, no leverage). Test 2019-2026:

| Strategy | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|
| 60/40 base | 10.8% | 0.74 | -20.0% | 0.54 |
| **60/40 vol-managed (HAR, L=1)** | 8.4% | 0.68 | -14.3% | 0.59 |
| SPY | 16.7% | 0.87 | -23.9% | 0.70 |

The vol-managed version gives up return and Sharpe in exchange for a shallower drawdown and a slightly better
Calmar. It does not beat SPY. It was chosen as the final model because it has few parameters, a documented
mechanism and the lowest drawdown. The ML selection code stays as a research module, not as a recommended strategy.

Deflated Sharpe and PBO (CSCV) from the phase 3 study: PBO is 0.21 on the development period and 0.07 on the full
period. Test DSR is 0.78 for SPY, 0.66 for 60/40 and 0.54-0.55 for the best ML and momentum strategies
(full table in [`reports/phase3_results.md`](reports/phase3_results.md)).

### Relative-to-S&P-500 policy

A second line of work trains a linear scoring policy over about 80 liquid US ETFs to beat the SPY, by maximising a
reward on the 21-day active return (exact analytic gradient, L-BFGS). Test 2019-2026, from
[`reports/RELATIVE_SUMMARY.md`](reports/RELATIVE_SUMMARY.md) (2 bp costs):

| Strategy | CAGR | Active / yr | IR | MaxDD | Turnover / yr |
|---|---|---|---|---|---|
| Daily policy (selected on validation) | 23.4% | +10.0% | 0.36 | -53% | 224x |
| Core-satellite variant (gamma = 0.6) | 22.4% | +5.7% | 0.57 | -27.5% | 30x |
| SPY | 16.9% | 0 | n/a | -34% | 0 |

Caveats: the aggressive daily policy loses its edge with weekly rebalancing (CAGR 11.3%, -1.3% active), so it is a
fragile short-horizon signal. The core-satellite variant was evaluated on test after the selected one, so it
carries a selection caveat. Survivorship bias, real costs above 2 bp and a single test period also apply.
A reward-design lesson is in the report: with a convex reward the policy first learned to reward volatility
itself (option effect), which was fixed and is covered by a test in `tests/test_relative.py`.

## Methodology

- **Universe** ([`docs/DATA.md`](docs/DATA.md)): about 5,800 US-listed ETFs/ETNs (NASDAQ Trader list), about
  3,600 European UCITS ETFs/ETCs/ETNs (Xetra list, deduplicated by ISIN), about 75 long-history market series
  (indices, futures, FX, rates, volatility), about 74 FRED macro series and about 260 alternative series
  (Fama-French and AQR factors, Goyal-Welch predictors, CAPE, Cboe volatility term structure, CFTC positioning,
  sentiment, geopolitical risk, term premium).
- **Time split**: development 2008-2018 with expanding annual walk-forward; test 2019-2026 untouched during
  selection. The relative-policy study uses train 2005-15, validation 2016-18, test 2019 onward.
- **Costs**: 5 bp per trade (phases 3-4b), 2 bp in the relative-policy study.
- **Look-ahead control**: every macro feature is shifted by its publication lag (`pub_lag_days`).
- **Overfitting controls**: every trial is registered (`reports/*_trials.csv`); Deflated Sharpe uses the real
  trial count (Bailey and Lopez de Prado, 2014); PBO via CSCV (Bailey et al., 2017); one selection criterion
  (development Sharpe) and one test evaluation per version.
- **Benchmarks**: SPY, 60/40 (SPY/AGG), equal-weight universe.

## Installation

```bash
git clone https://github.com/ep2si10n/Quant-Predictor.git
cd Quant-Predictor
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests
```

Python 3.11 is what CI uses. The tests are self-contained and do not need the data.

## Usage

```bash
# Data ingestion (resumable, polite to Yahoo; US + EU UCITS + Yahoo macro)
scripts/run_all.sh
python scripts/run_ingestion.py --region eu           # one region only (us | eu)
PYTHONPATH=src python -m quantpred.ingest.alt_data    # alternative data

# Research phases
python scripts/run_study.py        # phase 3: comparative study
python scripts/run_phase4.py       # phase 4: failure-driven iteration
python scripts/run_phase4b.py      # phase 4b: volatility (HAR) and regime models
python scripts/run_relative.py     # relative-to-SPY policy

# Live simulation vs SPY
python scripts/live_demo.py --replay        # day-by-day historical replay
python scripts/live_demo.py --yahoo 30      # 30 minutes of live Yahoo quotes
python scripts/dashboard.py                 # local dashboard at http://localhost:8765
```

```python
import sys; sys.path.insert(0, "src")
from quantpred.ingest.store import load_panel
px = load_panel("adj_close", asset_classes=["fixed_income", "commodity"])  # date x ticker
```

## Project structure

```
src/quantpred/
  ingest/      Yahoo client, universes (US, EU), metadata, quality checks, macro, alternative data, parquet store
  research/    features, backtest, metrics (DSR, PBO), portfolio construction, regime and volatility models
  relative/    relative-to-SPY policy: data, reward, policy, training progress
  live/        minute-level simulated portfolio engine and price feeds
scripts/       entry points for ingestion, study phases, training and dashboard
tests/         unit tests (ingestion helpers, reward and policy)
docs/          DATA.md, RESEARCH.md, INFORME_MATEMATICO.md, PROJECT_STATE.md
reports/       result tables, plots and summaries
models/        exported relative-policy parameters (JSON)
data/          processed datasets and metadata (see below)
```

## Data

- **Sources**: NASDAQ Trader (US universe), Deutsche Boerse Xetra instrument list (EU universe), Yahoo Finance
  (prices), FRED (macro), and public academic data libraries (Ken French, AQR, Goyal-Welch, Shiller) plus Cboe,
  CFTC, NY Fed, Philadelphia Fed, SF Fed and others. Details and rationale per source: [`docs/DATA.md`](docs/DATA.md).
- **Size**: `data/` is about 250 MB of zstd-compressed parquet, committed to the repository on purpose so results
  are reproducible without re-downloading. Raw downloads (`data/raw/`) are ignored.
- **Regenerate**: run the ingestion commands in [Usage](#usage). Yahoo data changes over time, so a fresh download
  will not match the committed snapshot exactly.
- **Alternatives to committing the data** (not applied): Git LFS, a GitHub release asset, or an external bucket
  with a download script.

## Limitations

- Survivorship bias: only ETFs alive today are available, so cross-sectional selection results are likely
  somewhat optimistic.
- Short history for most ETFs; long-history index and futures proxies partly compensate.
- Revised macro series are used without ALFRED vintages, leaving some residual look-ahead.
- Yahoo `adj_close` is an approximate total return; real costs for illiquid ETFs exceed the 2-5 bp assumed.
- A single test period (2019-2026) with a strong US equity run; over that period SPY out-returned the strategies selected as robust (for example the volatility-managed 60/40), which win on drawdown rather than on return.
- Some reports, docs and script docstrings are written in Spanish.

## Roadmap

- Phase 5: export the final model (60/40 with HAR volatility management) and finish the mathematical report
  ([`docs/INFORME_MATEMATICO.md`](docs/INFORME_MATEMATICO.md)).
- Account for delisted ETFs to reduce survivorship bias.
- Out-of-sample evaluation on the European UCITS universe, deduplicated by benchmark index.

## References

- Corsi, F. (2009). A simple approximate long-memory model of realized volatility (HAR).
- Moreira, A. and Muir, T. (2017). Volatility-managed portfolios.
- Bailey, D. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
- Bailey, D., Borwein, J., Lopez de Prado, M. and Zhu, Q. (2017). The probability of backtest overfitting.
- Lopez de Prado, M. (2018). Advances in Financial Machine Learning (purged walk-forward validation).
- Gu, S., Kelly, B. and Xiu, D. (2020). Empirical asset pricing via machine learning. NBER w25398.
- Goyal, A., Welch, I. and Zafirov, A. (2024). A comprehensive 2022 look at the empirical performance of equity
  premium prediction.

Further factor and data-source references are listed in [`docs/RESEARCH.md`](docs/RESEARCH.md) and
[`docs/DATA.md`](docs/DATA.md).

## Contributing

Issues and pull requests are welcome. Before opening a PR, please run the checks locally:

```bash
pip install -r requirements.txt
ruff check .
python -m pytest -q tests
```

## License

MIT, see [LICENSE](LICENSE).

## Disclaimer

This is a research project, not investment advice. Backtested results are hypothetical, do not account for all
real-world frictions and do not guarantee future performance.

#!/usr/bin/env bash
# Ingesta completa secuencial (cortesía con Yahoo): EEUU -> Europa UCITS -> macro Yahoo.
set -e
cd "$(dirname "$0")/.."
python3 scripts/run_ingestion.py --region us --skip-macro "$@"
python3 scripts/run_ingestion.py --region eu --skip-macro "$@"
PYTHONPATH=src python3 -c "
from quantpred.ingest import macro, quality
from quantpred.config import PROCESSED
ym = macro.fetch_yahoo_macro()
ym, _ = quality.clean_prices(ym.drop(columns='description'))
ym['description'] = ym['ticker'].map(macro.YAHOO_MACRO)
ym.to_parquet(PROCESSED / 'macro_yahoo.parquet', index=False, compression='zstd')
print('macro yahoo:', ym['ticker'].nunique(), 'series', len(ym), 'filas')
"
echo "INGESTA COMPLETA"

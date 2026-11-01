"""Metadatos por ETF (categoría Morningstar, gestora, AUM, TER...) y clasificación por clase de activo."""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from quantpred.config import META

# --- Clasificación ---------------------------------------------------------------------------
# Clases: equity, fixed_income, commodity, real_estate, currency, volatility, crypto,
#         multi_asset, alternative, leveraged_inverse
_FI = re.compile(r"bond|government|treasury|muni|bank loan|inflation|corporate|ultrashort|"
                 r"short-term|intermediate core|preferred|nontraditional|multisector|securitized|"
                 r"money market|high yield|credit|emerging markets bond|long-term|target maturity|"
                 r"short government|intermediate government|long government|world bond", re.I)
_ALT = re.compile(r"derivative income|long-short|macro trading|multistrategy|event driven|"
                  r"systematic trend|market neutral|options trading|defined outcome|"
                  r"relative value|equity hedged|managed futures|multialternative", re.I)


_INV_ISSUERS = re.compile(r"proshares|direxion|microsectors|daily|tuttle|axs|graniteshares|defiance|"
                          r"t-rex|tradr|leverage shares|rex shares|volatility shares")
_SHORT_DURATION = re.compile(r"short[- ](term|duration|maturity|treasury|government|bond|corporate|"
                             r"high yield|muni|inflation|tips|dated|-term)")


def _name_flags(name: str) -> dict:
    n = (name or "").lower()
    lev = bool(re.search(r"\b(ultra|ultrapro|leveraged)\b|\b[1-9](\.\d+)?x\b|\bbull\b", n))
    inv = bool(re.search(r"\b(inverse|bear|bearish|ultrashort)\b|-[1-9]x\b", n)) or (
        bool(re.search(r"\bshort\b", n)) and bool(_INV_ISSUERS.search(n)) and not _SHORT_DURATION.search(n))
    if "proshares short" in n:
        inv = True
    return {"is_leveraged": lev, "is_inverse": inv, "is_etn": " etn" in n or "exchange traded note" in n}


_EQ_WORDS = re.compile(r"miners|mining|producers|equity|equities|stock|companies|explor|equipment|"
                       r"services|industry|e&p|midstream|pipeline|royalt|infrastructure|select sector")


def classify_by_name(name: str | None) -> str:
    n = (name or "").lower()
    f = _name_flags(n)
    if re.search(r"\bvix\b|volatility index futures|vol futures", n):
        return "volatility"
    if f["is_leveraged"] and not re.search(r"buffer|income|premium", n) or f["is_inverse"]:
        return "leveraged_inverse"
    if re.search(r"bitcoin|\bether(eum)?\b|crypto|solana|\bxrp\b|digital asset|blockchain trust|"
                 r"litecoin|dogecoin|cardano|avalanche|chainlink|polkadot", n) and "equity" not in n:
        return "crypto"
    if re.search(r"bond|treasur|\bmuni|municipal|credit|\bloan|t-bill|aggregate|fixed income|"
                 r"\btips\b|inflation[- ]protected|mortgage|securitized|\bclo\b|preferred|"
                 r"floating rate|money market|ultra[- ]short|duration|high yield|investment grade", n):
        return "fixed_income"
    if re.search(r"\b(gold|silver|platinum|palladium|precious metals?|oil|crude|brent|natural gas|"
                 r"commodit\w*|copper|agricultur\w*|wheat|corn|soybean|sugar|coffee|cocoa|uranium|"
                 r"carbon|metals?|livestock|grains?)\b", n) and not _EQ_WORDS.search(n):
        return "commodity"
    if re.search(r"currencyshares|currency trust|\b(dollar|euro|yen|franc|sterling|yuan) (index|trust|bullish|bearish)\b|"
                 r"dollar index", n) and not re.search(r"hedged|neutral", n):
        return "currency"
    if re.search(r"\breits?\b|real estate|realty|homebuilders?", n):
        return "real_estate"
    if re.search(r"buffer|hedged equity|market neutral|long[/ -]short|managed futures|\btrend|"
                 r"option|covered call|premium income|enhanced income|structured|defined outcome|"
                 r"merger|arbitrage|tail risk|absolute return|macro|multi-?strategy|put ?write|"
                 r"floor|accelerat|barrier|autocall|outcome", n):
        return "alternative"
    if re.search(r"allocation|balanced|target date|\b20[2-7]\d\b|multi-?asset|60/40|"
                 r"conservative|moderate|aggressive|all weather|risk parity|diversified", n):
        return "multi_asset"
    return "equity"


def classify(category: str | None, name: str | None) -> str:
    """Clase de activo: categoría Morningstar si existe; si no, reglas por nombre."""
    cat = (category or "").strip()
    n = (name or "").lower()
    if not cat:
        return classify_by_name(name)
    if cat.startswith("Trading--Leveraged") or cat.startswith("Trading--Inverse"):
        return "volatility" if "vix" in n else "leveraged_inverse"
    if "vix" in n:
        return "volatility"
    if cat == "Digital Assets":
        return "crypto"
    if cat.startswith("Commodities"):
        return "commodity"
    if "Currency" in cat:
        return "currency"
    if "Real Estate" in cat:
        return "real_estate"
    if "Allocation" in cat or cat.startswith("Target") or "Tactical" in cat:
        return "multi_asset"
    if _ALT.search(cat):
        return "alternative"
    if _FI.search(cat):
        return "fixed_income"
    if cat.startswith("Trading"):
        return classify_by_name(name)
    return "equity"


def _raw(v):
    return v.get("raw") if isinstance(v, dict) else v


def _parse_summary(sym: str, res: dict) -> dict:
    fp, sd = res.get("fundProfile", {}) or {}, res.get("summaryDetail", {}) or {}
    ks, qt = res.get("defaultKeyStatistics", {}) or {}, res.get("quoteType", {}) or {}
    fees = fp.get("feesExpensesInvestment", {}) or {}
    return {
        "yahoo": sym, "quoteType": qt.get("quoteType"), "longName_qs": qt.get("longName"),
        "category": fp.get("categoryName"), "fundFamily": fp.get("family"), "legalType": fp.get("legalType"),
        "totalAssets": _raw(sd.get("totalAssets")) or _raw(ks.get("totalAssets")),
        "netExpenseRatio": _raw(fees.get("annualReportExpenseRatio")),
        "yield": _raw(sd.get("yield")), "navPrice": _raw(sd.get("navPrice")),
        "beta3Year": _raw(ks.get("beta3Year")), "fundInceptionDate": _raw(ks.get("fundInceptionDate")),
        "ytdReturn": _raw(ks.get("ytdReturn")),
    }


def fetch_summaries(symbols: list[str], workers: int = 2, pause: float = 0.25,
                    max_consecutive_429: int = 20) -> pd.DataFrame:
    """quoteSummary con caché incremental (data/raw/meta_cache.jsonl). Para si Yahoo limita."""
    import json
    from quantpred.config import RAW
    from quantpred.ingest.yahoo import RateLimited, quote_summary
    cache_f = RAW / "meta_cache.jsonl"
    cache_f.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache_f.exists():
        for line in cache_f.read_text().splitlines():
            r = json.loads(line); done[r["yahoo"]] = r
    todo = [s for s in symbols if s not in done]
    print(f"  quoteSummary: {len(done)} en caché, {len(todo)} pendientes", flush=True)
    streak = 0

    def one(sym):
        time.sleep(pause)
        try:
            return _parse_summary(sym, quote_summary(sym)), None
        except RateLimited as e:
            return None, "429"
        except Exception as e:
            return {"yahoo": sym, "error": str(e)[:120]}, None

    with cache_f.open("a") as fh, ThreadPoolExecutor(workers) as ex:
        for i in range(0, len(todo), 50):
            for row, err in ex.map(one, todo[i:i + 50]):
                if err == "429":
                    streak += 1
                    continue
                streak = 0
                done[row["yahoo"]] = row
                fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            if streak >= max_consecutive_429:
                print("  quoteSummary: rate-limit persistente, se para (reanudable)", flush=True)
                break
            if (i // 50) % 10 == 0:
                print(f"  quoteSummary {len(done)}/{len(symbols)}", flush=True)
    return pd.DataFrame([done[s] for s in symbols if s in done])


def build_metadata(universe: pd.DataFrame, chart_meta: pd.DataFrame | None = None,
                   summaries: pd.DataFrame | None = None, save: bool = True) -> pd.DataFrame:
    meta = universe.copy()
    if chart_meta is not None and not chart_meta.empty:
        cm = chart_meta[["yahoo", "longName", "instrumentType", "fullExchangeName", "currency", "firstTradeDate"]]
        meta = meta.merge(cm, on="yahoo", how="left")
    else:
        meta["longName"] = None
    if summaries is not None and not summaries.empty:
        meta = meta.merge(summaries.drop(columns=[c for c in ["error"] if c in summaries]), on="yahoo", how="left")
        meta["longName"] = meta["longName"].fillna(meta.pop("longName_qs"))
        meta["fundInceptionDate"] = pd.to_datetime(meta["fundInceptionDate"], unit="s", errors="coerce")
    else:
        meta["category"] = None
    full_name = meta["longName"].fillna(meta["name"])
    flags = full_name.map(_name_flags).apply(pd.Series)
    meta = pd.concat([meta, flags], axis=1)
    meta["asset_class"] = [classify(c, n) for c, n in zip(meta["category"], full_name)]
    meta["class_source"] = meta["category"].notna().map({True: "morningstar_category", False: "name_rules"})
    cat = meta["category"].fillna("")
    meta.loc[cat.str.startswith("Trading--Leveraged"), "is_leveraged"] = True
    meta.loc[cat.str.startswith("Trading--Inverse"), "is_inverse"] = True
    if save:
        META.mkdir(parents=True, exist_ok=True)
        meta.to_parquet(META / "etf_metadata.parquet", index=False)
        meta.to_csv(META / "etf_metadata.csv", index=False)
    return meta

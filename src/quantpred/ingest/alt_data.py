"""Datos alternativos / features para el predictor, seleccionados según la literatura.

Cada fuente -> data/processed/alt/<fuente>.parquet en formato largo (date, series, value)
+ catálogo data/meta/feature_catalog.csv (frecuencia, lag de publicación, referencia).

IMPORTANTE (look-ahead): `date` es la fecha a la que se refiere el dato, NO la de publicación.
`pub_lag_days` en el catálogo es el retraso típico; la fase de features debe desplazar cada
serie ese número de días antes de usarla.
"""
from __future__ import annotations

import io
import re
import time
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from quantpred.config import META, PROCESSED

ALT = PROCESSED / "alt"
UA = {"User-Agent": "Mozilla/5.0 (research; quant-predictor)"}


def _get(url: str, retries: int = 4) -> bytes:
    last = None
    for a in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=120)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            last = e
            time.sleep(2 ** (a + 1))
    raise RuntimeError(f"{url}: {last}")


def _long(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    out = df.melt(id_vars=date_col, var_name="series", value_name="value")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=[date_col, "value"]).rename(columns={date_col: "date"})


def _header_row(raw: pd.DataFrame, token: str) -> int:
    for i in range(min(60, len(raw))):
        if str(raw.iat[i, 0]).strip().upper() == token.upper():
            return i
    raise ValueError(f"cabecera '{token}' no encontrada")


# --- Ken French Data Library --------------------------------------------------------------------
KF = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip"
KF_FILES = {
    "F-F_Research_Data_5_Factors_2x3_daily": "ff5",
    "F-F_Momentum_Factor_daily": "ffmom",
    "F-F_ST_Reversal_Factor_daily": "ffstrev",
    "F-F_LT_Reversal_Factor_daily": "ffltrev",
    "Developed_5_Factors_Daily": "dev5",
    "Developed_Mom_Factor_Daily": "devmom",
    "12_Industry_Portfolios_daily": "ind12",
}


def ken_french() -> pd.DataFrame:
    out = []
    for fname, tag in KF_FILES.items():
        try:
            z = zipfile.ZipFile(io.BytesIO(_get(KF.format(fname))))
            txt = z.read(z.namelist()[0]).decode("latin-1").splitlines()
        except Exception as e:
            print(f"    KF {fname}: {e}")
            continue
        header, rows = None, []
        for line in txt:
            parts = [p.strip() for p in line.split(",")]
            if parts and re.fullmatch(r"\d{8}", parts[0]):
                rows.append(parts)
            elif rows:
                break  # primera sección (rentabilidades value-weighted diarias)
            elif len(parts) > 1 and parts[0] == "":
                header = parts
        if not rows:
            continue
        cols = ["date"] + [f"{tag}_{c.replace('-', '_').lower()}" for c in header[1:len(rows[0])]]
        df = pd.DataFrame(rows, columns=cols)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        lg = _long(df)
        lg["value"] /= 100.0  # % -> decimal
        out.append(lg)
    return pd.concat(out, ignore_index=True)


# --- AQR data library -----------------------------------------------------------------------------
AQR = "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/{}.xlsx"
AQR_FILES = {
    "Time-Series-Momentum-Factors-Monthly": ("tsmom", None),
    "Betting-Against-Beta-Equity-Factors-Daily": ("bab_d", ["USA", "Global", "Global Ex USA", "Europe", "Pacific"]),
    "Quality-Minus-Junk-Factors-Monthly": ("qmj", ["USA", "Global", "Global Ex USA", "Europe", "Pacific"]),
    "Value-and-Momentum-Everywhere-Factors-Monthly": ("vme", None),
    "Century-of-Factor-Premia-Monthly": ("century", None),
}


def aqr() -> pd.DataFrame:
    out = []
    for fname, (tag, keep) in AQR_FILES.items():
        try:
            xl = pd.ExcelFile(io.BytesIO(_get(AQR.format(fname))))
        except Exception as e:
            print(f"    AQR {fname}: {e}")
            continue
        for sh in xl.sheet_names:
            if sh.lower() in ("disclosures", "definitions", "notes", "readme", "sources"):
                continue
            raw = xl.parse(sh, header=None)
            try:
                h = _header_row(raw, "DATE")
            except ValueError:
                continue
            df = raw.iloc[h + 1:].copy()
            df.columns = [str(c).strip() for c in raw.iloc[h]]
            df = df.loc[:, [c for c in df.columns if c and c != "nan"]]
            if keep:
                df = df[["DATE"] + [c for c in keep if c in df.columns]]
            df = df.rename(columns={"DATE": "date"})
            sname = re.sub(r"\W+", "_", sh.strip().lower())
            df.columns = ["date"] + [f"aqr_{tag}_{sname}_{re.sub(r'[^0-9a-zA-Z]+', '_', c).lower()}"
                                     for c in df.columns[1:]]
            out.append(_long(df))
    return pd.concat(out, ignore_index=True)


# --- Goyal-Welch-Zafirov (2024) predictores de la prima de riesgo ---------------------------------
GW_URL = "https://docs.google.com/spreadsheets/d/17mw_IpaiLFDrGnrPRQ2o1ugV5nJsZuD1/export?format=xlsx"


def goyal_welch() -> pd.DataFrame:
    xl = pd.ExcelFile(io.BytesIO(_get(GW_URL)))
    m = xl.parse("Monthly")
    m["date"] = pd.to_datetime(m["yyyymm"].astype(int).astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    m = m.drop(columns="yyyymm")
    m.columns = ["date" if c == "date" else "gw_" + re.sub(r"\W+", "_", str(c)).lower() for c in m.columns]
    readme = xl.parse("ReadMe")
    readme.to_csv(META / "goyal_welch_readme.csv", index=False)
    return _long(m)


# --- Shiller (CAPE, dividendos, beneficios desde 1871) --------------------------------------------
def shiller() -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(_get("http://www.econ.yale.edu/~shiller/data/ie_data.xls")),
                        sheet_name="Data", header=None)
    h = _header_row(raw, "Date")
    df = raw.iloc[h + 1:].copy()
    cols = [str(c).strip() for c in raw.iloc[h]]
    df.columns = [f"{c}_{i}" if cols.count(c) > 1 else c for i, c in enumerate(cols)]
    df = df[pd.to_numeric(df["Date"], errors="coerce").notna()]
    d = pd.to_numeric(df["Date"]).round(2)
    yr = d.astype(int)
    mo = ((d - yr) * 100).round().astype(int).clip(1, 12)
    df["date"] = pd.to_datetime(dict(year=yr, month=mo, day=1)) + pd.offsets.MonthEnd(0)
    keep = {"P": "shiller_price", "D": "shiller_div", "E": "shiller_earn", "CPI": "shiller_cpi",
            "Rate GS10": "shiller_gs10", "CAPE": "shiller_cape"}
    tr = [c for c in df.columns if c.startswith("TR CAPE")]
    if tr:
        keep[tr[0]] = "shiller_tr_cape"
    df = df[["date"] + [c for c in keep if c in df.columns]].rename(columns=keep)
    return _long(df)


# --- AAII sentiment survey (semanal desde 1987) ---------------------------------------------------
def aaii() -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(_get("https://www.aaii.com/files/surveys/sentiment.xls")),
                        sheet_name=0, header=None)
    dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    df = raw[dates.notna() & (dates.dt.year > 1985)].iloc[:, :4].copy()
    df.columns = ["date", "aaii_bullish", "aaii_neutral", "aaii_bearish"]
    df["aaii_bull_bear"] = pd.to_numeric(df["aaii_bullish"]) - pd.to_numeric(df["aaii_bearish"])
    return _long(df)


# --- Cboe: volatilidad implícita multi-plazo y cross-asset, put/call ------------------------------
CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
CBOE_IDX = ["VIX", "VIX9D", "VIX3M", "VIX6M", "VIX1Y", "VVIX", "SKEW", "VXN", "RVX", "VXD",
            "OVX", "GVZ", "VXEEM", "VXEWZ", "VXAPL", "VXAZN", "VXTLT", "EVZ"]
CBOE_PC = {"totalpc": "cboe_pc_total", "equitypc": "cboe_pc_equity", "indexpc": "cboe_pc_index"}


def cboe() -> pd.DataFrame:
    out = []
    for idx in CBOE_IDX:
        try:
            df = pd.read_csv(io.BytesIO(_get(CBOE.format(idx))))
        except Exception as e:
            print(f"    Cboe {idx}: {e}")
            continue
        df.columns = [c.strip().upper() for c in df.columns]
        val = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
        out.append(_long(pd.DataFrame({"date": df["DATE"], f"cboe_{idx.lower()}": df[val]})))
    for f, name in CBOE_PC.items():
        try:
            txt = _get(f"https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/{f}.csv").decode()
            lines = txt.splitlines()
            h = next(i for i, l in enumerate(lines) if l.upper().startswith("DATE"))
            df = pd.read_csv(io.StringIO("\n".join(lines[h:])))
            df.columns = [c.strip().upper() for c in df.columns]
            pc = [c for c in df.columns if "RATIO" in c][0]
            out.append(_long(pd.DataFrame({"date": df["DATE"], name: df[pc]})))
        except Exception as e:
            print(f"    Cboe {f}: {e}")
    return pd.concat(out, ignore_index=True)


# --- CFTC Commitments of Traders (posicionamiento especulativo) ------------------------------------
CFTC = "https://www.cftc.gov/files/dea/history/{}.zip"


def _cftc_files(kind: str) -> list[str]:
    yr = pd.Timestamp.today().year
    if kind == "legacy":
        return ["deacot1986_2016"] + [f"deacot{y}" for y in range(2017, yr + 1)]
    if kind == "disagg":
        return ["fut_disagg_txt_hist_2006_2016"] + [f"fut_disagg_txt_{y}" for y in range(2017, yr + 1)]
    return ["fin_fut_txt_2006_2016"] + [f"fut_fin_txt_{y}" for y in range(2017, yr + 1)]


CFTC_COLS = {
    "legacy": {"Noncommercial Positions-Long (All)": "noncomm_long",
               "Noncommercial Positions-Short (All)": "noncomm_short",
               "Commercial Positions-Long (All)": "comm_long", "Commercial Positions-Short (All)": "comm_short",
               "Nonreportable Positions-Long (All)": "nonrep_long",
               "Nonreportable Positions-Short (All)": "nonrep_short"},
    "disagg": {"M_Money_Positions_Long_All": "mm_long", "M_Money_Positions_Short_All": "mm_short",
               "Prod_Merc_Positions_Long_All": "prod_long", "Prod_Merc_Positions_Short_All": "prod_short",
               "Swap_Positions_Long_All": "swap_long", "Swap__Positions_Short_All": "swap_short"},
    "tff": {"Lev_Money_Positions_Long_All": "lev_long", "Lev_Money_Positions_Short_All": "lev_short",
            "Asset_Mgr_Positions_Long_All": "am_long", "Asset_Mgr_Positions_Short_All": "am_short",
            "Dealer_Positions_Long_All": "dealer_long", "Dealer_Positions_Short_All": "dealer_short"},
}


def cftc(kind: str) -> pd.DataFrame:
    """Formato ancho por mercado: date, market, code, open_interest, posiciones..."""
    frames = []
    for f in _cftc_files(kind):
        try:
            z = zipfile.ZipFile(io.BytesIO(_get(CFTC.format(f))))
        except Exception as e:
            print(f"    CFTC {f}: {e}")
            continue
        for n in z.namelist():
            df = pd.read_csv(z.open(n), low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            date_c = next(c for c in df.columns if "YYYY-MM-DD" in c or c == "Report_Date_as_YYYY-MM-DD")
            name_c = next(c for c in df.columns if c.startswith("Market") and "Name" in c or
                          c == "Market_and_Exchange_Names")
            code_c = next(c for c in df.columns if "Contract Market Code" in c or
                          c == "CFTC_Contract_Market_Code")
            oi_c = next(c for c in df.columns if c in ("Open Interest (All)", "Open_Interest_All"))
            cols = {date_c: "date", name_c: "market", code_c: "code", oi_c: "open_interest"}
            cols.update({k: v for k, v in CFTC_COLS[kind].items() if k in df.columns})
            frames.append(df[list(cols)].rename(columns=cols))
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["market"] = out["market"].str.strip()
    out["code"] = out["code"].astype(str).str.strip()
    for c in out.columns[3:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["date"]).drop_duplicates(["date", "code"]).sort_values(["code", "date"])


# --- Bancos centrales / académicos -----------------------------------------------------------------
def gpr() -> pd.DataFrame:
    """Geopolitical Risk Index (Caldara & Iacoviello, AER 2022)."""
    out = []
    d = pd.read_excel(io.BytesIO(_get("https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls")))
    d.columns = [str(c).strip().lower() for c in d.columns]
    dc = "date" if "date" in d.columns else d.columns[0]
    keep = [c for c in d.columns if c.startswith("gprd")]
    out.append(_long(d[[dc] + keep].rename(columns={dc: "date"}).rename(columns=lambda c: c if c == "date" else f"gpr_{c}")))
    try:
        m = pd.read_excel(io.BytesIO(_get("https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls")))
        m.columns = [str(c).strip() for c in m.columns]
        mk = [c for c in ["GPR", "GPRT", "GPRA", "GPRH", "GPRHT", "GPRHA"] if c in m.columns]
        mm = m[["month"] + mk].rename(columns={"month": "date"})
        mm["date"] = pd.to_datetime(mm["date"]) + pd.offsets.MonthEnd(0)
        out.append(_long(mm.rename(columns=lambda c: c if c == "date" else f"gpr_m_{c.lower()}")))
    except Exception as e:
        print(f"    GPR mensual: {e}")
    return pd.concat(out, ignore_index=True)


def acm_term_premium() -> pd.DataFrame:
    """Prima por plazo ACM (Adrian, Crump & Moench 2013), NY Fed, diaria."""
    xl = pd.ExcelFile(io.BytesIO(_get(
        "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls")))
    sh = next((s for s in xl.sheet_names if "daily" in s.lower()), xl.sheet_names[0])
    d = xl.parse(sh)
    d = d.rename(columns={d.columns[0]: "date"})
    keep = [c for c in d.columns if re.fullmatch(r"ACM(TP|RNY|Y)(02|05|10)", str(c))]
    d["date"] = pd.to_datetime(d["date"], format="%d-%b-%Y", errors="coerce").fillna(
        pd.to_datetime(d["date"], errors="coerce"))
    return _long(d[["date"] + keep].rename(columns=lambda c: c if c == "date" else c.lower()))


def gscpi() -> pd.DataFrame:
    """Global Supply Chain Pressure Index (NY Fed), mensual."""
    xl = pd.ExcelFile(io.BytesIO(_get(
        "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx")))
    sh = next((s for s in xl.sheet_names if "monthly" in s.lower()), xl.sheet_names[-1])
    raw = xl.parse(sh, header=None)
    dates = pd.to_datetime(raw.iloc[:, 0], format="%d-%b-%Y", errors="coerce")
    df = pd.DataFrame({"date": dates, "gscpi": pd.to_numeric(raw.iloc[:, 1], errors="coerce")}).dropna()
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    return _long(df)


def ads_index() -> pd.DataFrame:
    """Aruoba-Diebold-Scotti business conditions index (Philly Fed), diario."""
    d = pd.read_excel(io.BytesIO(_get("https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
                                      "ads/ads_index_most_current_vintage.xlsx")))
    d.columns = ["date", "ads_index"] + list(d.columns[2:])
    d["date"] = pd.to_datetime(d["date"].astype(str).str.replace(":", "-"), errors="coerce")
    return _long(d[["date", "ads_index"]])


def sf_news_sentiment() -> pd.DataFrame:
    """Daily News Sentiment Index (Shapiro, Sudhof & Wilson 2022), SF Fed."""
    xl = pd.ExcelFile(io.BytesIO(_get("https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx")))
    sh = next((s for s in xl.sheet_names if "data" in s.lower()), xl.sheet_names[-1])
    d = xl.parse(sh)
    d = d.iloc[:, :2]
    d.columns = ["date", "sf_news_sentiment"]
    return _long(d)


def baker_wurgler() -> pd.DataFrame:
    """Investor sentiment index (Baker & Wurgler 2006/2007), mensual."""
    d = pd.read_excel(io.BytesIO(_get(
        "https://pages.stern.nyu.edu/~jwurgler/data/Investor_Sentiment_Data_20190327_POST.xlsx")),
        sheet_name="DATA")
    sh = d
    sh.columns = [str(c).strip() for c in sh.columns]
    ym = sh.columns[0]
    sh = sh[pd.to_numeric(sh[ym], errors="coerce").notna()].copy()
    sh["date"] = pd.to_datetime(sh[ym].astype(int).astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    keep = [c for c in sh.columns if c.upper().startswith("SENT")]
    sh = sh[["date"] + keep]
    # SENT^ = índice ortogonalizado a macro (BW 2006); SENT = sin ortogonalizar
    sh.columns = ["date"] + ["bw_sent_orth" if c.endswith("^") else "bw_sent_raw" for c in keep]
    return _long(sh)


# --- Catálogo ----------------------------------------------------------------------------------------
@dataclass
class Source:
    fn: object
    freq: str
    pub_lag_days: int
    why: str
    ref: str


SOURCES: dict[str, Source] = {
    "ken_french": Source(ken_french, "D", 30, "Factores de riesgo (mercado, tamaño, valor, rentabilidad, "
                         "inversión, momentum, reversión) -> exposiciones y rotación de factores",
                         "Fama & French (1993, 2015); Carhart (1997); Jegadeesh & Titman (1993)"),
    "aqr": Source(aqr, "D/M", 30, "TSMOM multi-activo, BAB, QMJ, value/momentum everywhere: "
                  "primas cross-asset documentadas", "Moskowitz, Ooi & Pedersen (2012); Frazzini & "
                  "Pedersen (2014); Asness, Moskowitz & Pedersen (2013); Asness et al. (2019)"),
    "goyal_welch": Source(goyal_welch, "M", 45, "57 predictores clásicos de la prima de riesgo (d/p, e/p, "
                          "b/m, tms, dfy, infl, ntis, svar, cay, vrp, skew...) — benchmark OOS",
                          "Welch & Goyal (2008); Goyal, Welch & Zafirov (2024, RFS)"),
    "shiller": Source(shiller, "M", 30, "Valoración de largo plazo (CAPE) desde 1871",
                      "Campbell & Shiller (1988, 1998)"),
    "aaii": Source(aaii, "W", 1, "Sentimiento minorista (contrarian en extremos)",
                   "Brown & Cliff (2004, 2005)"),
    "cboe": Source(cboe, "D", 0, "Vol. implícita multi-plazo (estructura temporal VIX), vol cross-asset "
                   "(oro, crudo, bonos, EM), SKEW y put/call", "Bekaert & Hoerova (2014); Bollerslev, "
                   "Tauchen & Zhou (2009) VRP; Pan & Poteshman (2006)"),
    "gpr": Source(gpr, "D/M", 1, "Riesgo geopolítico (texto de prensa)", "Caldara & Iacoviello (2022, AER)"),
    "acm_term_premium": Source(acm_term_premium, "D", 7, "Prima por plazo de Treasuries (predice "
                               "excesos de bonos)", "Adrian, Crump & Moench (2013); Cochrane & Piazzesi (2005)"),
    "gscpi": Source(gscpi, "M", 5, "Presión cadenas de suministro (inflación/commodities)",
                    "Benigno et al. (2022, NY Fed)"),
    "ads_index": Source(ads_index, "D", 7, "Condiciones de negocio en tiempo real", "Aruoba, Diebold & "
                        "Scotti (2009)"),
    "sf_news_sentiment": Source(sf_news_sentiment, "D", 7, "Sentimiento económico de prensa",
                                "Shapiro, Sudhof & Wilson (2022)"),
    "baker_wurgler": Source(baker_wurgler, "M", 365, "Sentimiento inversor compuesto (histórico)",
                            "Baker & Wurgler (2006, 2007)"),
}


def run_all(only: list[str] | None = None) -> pd.DataFrame:
    ALT.mkdir(parents=True, exist_ok=True)
    catalog = []
    for name, src in SOURCES.items():
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            df = src.fn()
        except Exception as e:
            print(f"  [alt] {name}: ERROR {e}", flush=True)
            continue
        df = df.sort_values(["series", "date"]).reset_index(drop=True)
        df["series"] = df["series"].astype("category")
        df.to_parquet(ALT / f"{name}.parquet", index=False, compression="zstd")
        g = df.groupby("series", observed=True)["date"]
        for s, (a, b, n) in pd.DataFrame({"a": g.min(), "b": g.max(), "n": g.size()}).iterrows():
            catalog.append({"source": name, "series": s, "freq": src.freq, "start": a.date(), "end": b.date(),
                            "n_obs": n, "pub_lag_days": src.pub_lag_days, "rationale": src.why,
                            "reference": src.ref})
        print(f"  [alt] {name}: {df['series'].nunique()} series, {len(df):,} obs, {time.time() - t0:.0f}s",
              flush=True)
    # CFTC COT (formato propio, ancho por mercado)
    if not only or "cftc" in only:
        for kind in ["legacy", "disagg", "tff"]:
            t0 = time.time()
            try:
                c = cftc(kind)
                c.to_parquet(ALT / f"cftc_{kind}.parquet", index=False, compression="zstd")
                catalog.append({"source": f"cftc_{kind}", "series": f"{c['code'].nunique()} mercados",
                                "freq": "W", "start": c["date"].min().date(), "end": c["date"].max().date(),
                                "n_obs": len(c), "pub_lag_days": 3,
                                "rationale": "Posicionamiento especulativo/comercial en futuros (hedging "
                                             "pressure, crowding)",
                                "reference": "De Roon, Nijman & Veld (2000); Bessembinder (1992); "
                                             "Moskowitz, Ooi & Pedersen (2012)"})
                print(f"  [alt] cftc_{kind}: {c['code'].nunique()} mercados, {len(c):,} filas, "
                      f"{time.time() - t0:.0f}s", flush=True)
            except Exception as e:
                print(f"  [alt] cftc_{kind}: ERROR {e}", flush=True)
    cat = pd.DataFrame(catalog)
    f = META / "feature_catalog_alt.csv"
    if only and f.exists():
        old = pd.read_csv(f)
        cat = pd.concat([old[~old["source"].isin(cat["source"].unique())], cat], ignore_index=True)
    cat.to_csv(f, index=False)
    return cat


if __name__ == "__main__":
    import sys
    run_all(sys.argv[1:] or None)

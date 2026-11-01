"""Rutas y parámetros globales del proyecto."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"                # descargas crudas (no versionado)
PROCESSED = DATA / "processed"    # parquet limpio (versionado)
META = DATA / "meta"              # universo, metadatos, informes QA

NASDAQ_TRADED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

# Descarga de precios
PRICE_BATCH = 100          # tickers por petición a Yahoo
PRICE_RETRIES = 4
META_WORKERS = 8           # hilos para metadatos

# Ficheros parquet de salida < este tamaño para poder subirlos a GitHub sin LFS
MAX_PARQUET_MB = 90

# Series de referencia de historia larga (índices, futuros, divisas, cripto) en Yahoo
YAHOO_MACRO = {
    # Índices renta variable EEUU
    "^GSPC": "S&P 500", "^DJI": "Dow Jones Industrial", "^IXIC": "Nasdaq Composite",
    "^NDX": "Nasdaq 100", "^RUT": "Russell 2000", "^NYA": "NYSE Composite",
    # Volatilidad / riesgo
    "^VIX": "CBOE VIX", "^VIX3M": "CBOE VIX 3M", "^VVIX": "VIX of VIX",
    "^SKEW": "CBOE SKEW", "^MOVE": "ICE BofA MOVE (vol. bonos)",
    # Tipos (rendimiento x10 en Yahoo)
    "^IRX": "T-Bill 13 semanas", "^FVX": "Treasury 5Y", "^TNX": "Treasury 10Y", "^TYX": "Treasury 30Y",
    # Índices internacionales
    "^FTSE": "FTSE 100", "^GDAXI": "DAX", "^FCHI": "CAC 40", "^STOXX50E": "Euro Stoxx 50",
    "^IBEX": "IBEX 35", "^N225": "Nikkei 225", "^HSI": "Hang Seng", "000001.SS": "Shanghai Composite",
    "^BSESN": "BSE Sensex", "^KS11": "KOSPI", "^AXJO": "ASX 200", "^GSPTSE": "TSX", "^BVSP": "Bovespa",
    "^MXX": "IPC México",
    # Futuros materias primas (continuo front-month)
    "GC=F": "Oro", "SI=F": "Plata", "HG=F": "Cobre", "PL=F": "Platino", "PA=F": "Paladio",
    "CL=F": "Crudo WTI", "BZ=F": "Crudo Brent", "NG=F": "Gas natural", "HO=F": "Heating oil", "RB=F": "Gasolina RBOB",
    "ZC=F": "Maíz", "ZW=F": "Trigo", "ZS=F": "Soja", "KC=F": "Café", "SB=F": "Azúcar", "CC=F": "Cacao",
    "CT=F": "Algodón", "LE=F": "Ganado vivo", "HE=F": "Cerdo magro",
    # Futuros financieros
    "ES=F": "E-mini S&P 500", "NQ=F": "E-mini Nasdaq", "ZN=F": "T-Note 10Y fut", "ZB=F": "T-Bond fut",
    "ZF=F": "T-Note 5Y fut", "ZT=F": "T-Note 2Y fut",
    # Divisas
    "DX-Y.NYB": "Índice dólar DXY", "EURUSD=X": "EUR/USD", "JPY=X": "USD/JPY", "GBPUSD=X": "GBP/USD",
    "AUDUSD=X": "AUD/USD", "CHF=X": "USD/CHF", "CAD=X": "USD/CAD", "CNY=X": "USD/CNY", "MXN=X": "USD/MXN",
    # Cripto
    "BTC-USD": "Bitcoin", "ETH-USD": "Ether",
}

# Series macro de FRED (frecuencia, descripción). OJO: las mensuales se publican con retraso;
# al construir features hay que aplicar el lag de publicación para evitar look-ahead.
FRED_SERIES = {
    "DFF": ("D", "Fed funds efectivo"),
    "DGS3MO": ("D", "Treasury 3M"),
    "DGS2": ("D", "Treasury 2Y"),
    "DGS5": ("D", "Treasury 5Y"),
    "DGS10": ("D", "Treasury 10Y"),
    "DGS30": ("D", "Treasury 30Y"),
    "T10Y2Y": ("D", "Pendiente 10Y-2Y"),
    "T10Y3M": ("D", "Pendiente 10Y-3M"),
    "DFII10": ("D", "TIPS 10Y (tipo real)"),
    "T5YIE": ("D", "Breakeven inflación 5Y"),
    "T10YIE": ("D", "Breakeven inflación 10Y"),
    "BAMLH0A0HYM2": ("D", "Spread OAS High Yield"),
    "BAMLC0A0CM": ("D", "Spread OAS Investment Grade"),
    "BAMLEMCBPIOAS": ("D", "Spread OAS bonos corporativos emergentes"),
    "VIXCLS": ("D", "VIX cierre"),
    "DTWEXBGS": ("D", "Dólar ponderado comercio (broad)"),
    "DCOILWTICO": ("D", "WTI spot"),
    "DCOILBRENTEU": ("D", "Brent spot"),
    "DEXUSEU": ("D", "USD por EUR"),
    "NFCI": ("W", "Chicago Fed National Financial Conditions"),
    "STLFSI4": ("W", "St. Louis Fed Financial Stress"),
    "ICSA": ("W", "Peticiones iniciales desempleo"),
    "WALCL": ("W", "Balance Fed total activos"),
    "CPIAUCSL": ("M", "IPC EEUU"),
    "CPILFESL": ("M", "IPC subyacente EEUU"),
    "PCEPILFE": ("M", "PCE subyacente"),
    "UNRATE": ("M", "Tasa de paro"),
    "PAYEMS": ("M", "Nóminas no agrícolas"),
    "INDPRO": ("M", "Producción industrial"),
    "M2SL": ("M", "Masa monetaria M2"),
    "UMCSENT": ("M", "Confianza consumidor Michigan"),
    "HOUST": ("M", "Viviendas iniciadas"),
    "RSAFS": ("M", "Ventas minoristas"),
    "USREC": ("M", "Indicador recesión NBER"),
    "USEPUINDXD": ("D", "Economic Policy Uncertainty (Baker, Bloom & Davis 2016)"),
    # --- ampliación según literatura (tipos, crédito, actividad, liquidez, expectativas) ---
    "DTB3": ("D", "T-Bill 3M (mercado secundario)"),
    "DGS1": ("D", "Treasury 1Y"),
    "DGS20": ("D", "Treasury 20Y"),
    "T10YFF": ("D", "10Y - Fed funds"),
    "DAAA": ("D", "Moody's Aaa corporativo"),
    "DBAA": ("D", "Moody's Baa corporativo (default spread Baa-Aaa)"),
    "BAMLC0A4CBBB": ("D", "Spread OAS BBB"),
    "BAMLH0A3HYC": ("D", "Spread OAS CCC y peor"),
    "BAMLHE00EHYIOAS": ("D", "Spread OAS High Yield euro"),
    "SOFR": ("D", "SOFR"),
    "RRPONTSYD": ("D", "Reverse repo Fed (liquidez)"),
    "WTREGEN": ("W", "Cuenta del Tesoro en la Fed (liquidez)"),
    "WRESBAL": ("W", "Reservas bancarias"),
    "MORTGAGE30US": ("W", "Hipoteca 30 años"),
    "ANFCI": ("W", "NFCI ajustado (Chicago Fed)"),
    "CCSA": ("W", "Peticiones continuadas desempleo"),
    "GVZCLS": ("D", "Vol. implícita oro"),
    "OVXCLS": ("D", "Vol. implícita crudo"),
    "DEXJPUS": ("D", "JPY por USD"),
    "DEXUSUK": ("D", "USD por GBP"),
    "DEXCHUS": ("D", "CNY por USD"),
    "DEXSZUS": ("D", "CHF por USD"),
    "GASREGW": ("W", "Precio gasolina regular EEUU"),
    "CFNAI": ("M", "Chicago Fed National Activity Index"),
    "SAHMREALTIME": ("M", "Regla de Sahm (tiempo real)"),
    "RECPROUSM156N": ("M", "Probabilidad recesión (Chauvet-Piger)"),
    "TCU": ("M", "Utilización de capacidad"),
    "JTSJOL": ("M", "Ofertas de empleo JOLTS"),
    "PCEPI": ("M", "Deflactor PCE"),
    "PPIACO": ("M", "PPI todas las materias primas"),
    "EXPINF1YR": ("M", "Inflación esperada 1Y (Cleveland Fed)"),
    "EXPINF10YR": ("M", "Inflación esperada 10Y (Cleveland Fed)"),
    "MICH": ("M", "Expectativa inflación Michigan"),
    "BUSLOANS": ("M", "Préstamos comerciales e industriales"),
    "TOTALSL": ("M", "Crédito al consumo"),
    "CSUSHPINSA": ("M", "Case-Shiller precio vivienda"),
    "PERMIT": ("M", "Permisos de construcción"),
    "DRTSCILM": ("Q", "SLOOS: endurecimiento crédito C&I (Lown & Morgan 2006)"),
    "A191RL1Q225SBEA": ("Q", "PIB real EEUU % trimestral anualizado"),
}

# Retraso típico de publicación por frecuencia (días) para evitar look-ahead en features
FRED_PUB_LAG = {"D": 1, "W": 7, "M": 35, "Q": 90}

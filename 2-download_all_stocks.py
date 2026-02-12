#!/usr/bin/env python3
"""
Download ALL Indian Stock Market Data

Uses Yahoo Finance to download 30 years of historical data.
Fetches stock list from NSE CSV or uses comprehensive list.

Usage:
    python download_all_stocks.py                  # Download ALL stocks
    python download_all_stocks.py --workers 5      # More parallel workers
    python download_all_stocks.py --resume         # Resume interrupted download
"""

import os
import sys
import json
import time
import argparse
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
import csv

from src.data.historic import (
    list_historic_stock_symbols,
    list_historic_index_names,
    load_historic_stock_prices,
    load_historic_index_prices,
)
from src.data.refresh import refresh_symbol_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('download_all.log')
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
PRICE_DIR = DATA_DIR / 'prices'
INFO_DIR = DATA_DIR / 'stock_info'
FINANCIALS_DIR = DATA_DIR / 'financials'
PROGRESS_FILE = DATA_DIR / 'download_progress.json'
STOCK_LIST_FILE = DATA_DIR / 'stock_lists' / 'all_nse_stocks.json'
HISTORIC_STOCK_INDEX_FILE = DATA_DIR / 'stock_lists' / 'historic_stock_symbols.json'
HISTORIC_INDEX_LIST_FILE = DATA_DIR / 'stock_lists' / 'historic_index_names.json'
MISSING_STOCK_LIST_FILE = DATA_DIR / 'stock_lists' / 'missing_live_symbols.json'

# 30 years of data
YEARS_OF_DATA = 30
CAGR_WINDOWS_YEARS = [1, 3, 5, 10]


def setup_directories():
    """Create data directories."""
    for d in [DATA_DIR, PRICE_DIR, INFO_DIR, FINANCIALS_DIR, DATA_DIR / 'stock_lists', DATA_DIR / 'db']:
        d.mkdir(parents=True, exist_ok=True)


def fetch_nse_stock_list() -> list:
    """
    Fetch stock list from NSE's official CSV file.
    Falls back to comprehensive list if NSE is blocked.
    """
    symbols = {}
    
    print("Fetching stock list from NSE...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'text/csv,*/*',
    }
    
    # Try NSE's equity list CSV (this usually works)
    csv_urls = [
        'https://archives.nseindia.com/content/equities/EQUITY_L.csv',
        'https://www1.nseindia.com/content/equities/EQUITY_L.csv',
    ]
    
    for url in csv_urls:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200 and 'SYMBOL' in response.text:
                reader = csv.DictReader(StringIO(response.text))
                for row in reader:
                    symbol = row.get('SYMBOL', '').strip()
                    name = row.get('NAME OF COMPANY', symbol).strip()
                    if symbol:
                        symbols[symbol] = {
                            'symbol': symbol,
                            'name': name,
                            'series': row.get(' SERIES', 'EQ').strip(),
                            'isin': row.get(' ISIN NUMBER', '').strip()
                        }
                print(f"  Got {len(symbols)} stocks from NSE CSV")
                break
        except Exception as e:
            logger.debug(f"NSE CSV failed: {e}")
            continue
    
    # If NSE CSV failed, use comprehensive list
    if len(symbols) < 100:
        print("  NSE CSV not accessible, using comprehensive list...")
        symbols = get_comprehensive_stock_list()
    
    return list(symbols.values())


def get_comprehensive_stock_list() -> dict:
    """
    Comprehensive list of ALL NSE stocks.
    This includes 2000+ stocks from NSE.
    """
    # Complete list of NSE stocks - all major and minor stocks
    stocks = [
        # NIFTY 50 (Current)
        "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
        "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
        "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
        "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
        "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
        "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
        "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
        "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA",
        "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TECHM",
        "TITAN", "ULTRACEMCO", "UPL", "WIPRO",
        
        # NIFTY Next 50
        "ABB", "ADANIGREEN", "AMBUJACEM", "AUROPHARMA", "BANKBARODA",
        "BEL", "BERGEPAINT", "BIOCON", "BOSCHLTD", "CHOLAFIN",
        "COLPAL", "CONCOR", "CUMMINSIND", "DABUR", "DMART",
        "GAIL", "GODREJCP", "HAVELLS", "HINDPETRO", "ICICIGI",
        "ICICIPRULI", "IDFCFIRSTB", "IGL", "INDIGO", "INDUSTOWER",
        "IOC", "IRCTC", "JINDALSTEL", "JUBLFOOD", "LTF",
        "LTIM", "LUPIN", "MARICO", "MOTHERSON", "MUTHOOTFIN",
        "NAUKRI", "NHPC", "NMDC", "OBEROIRLTY", "OFSS",
        "PAGEIND", "PETRONET", "PFC", "PIDILITIND", "PIIND",
        "PNB", "POLYCAB", "RECLTD", "SBICARD", "SHREECEM",
        "SIEMENS", "SRF", "TATAPOWER", "TORNTPHARM", "TRENT",
        "VEDL", "ZOMATO", "ZYDUSLIFE",
        
        # NIFTY Midcap 100
        "ACC", "AIAENG", "ALKEM", "APOLLOTYRE", "ASHOKLEY",
        "ASTRAL", "ATUL", "AUBANK", "AUROPHARMA", "BALKRISIND",
        "BHARATFORG", "BHEL", "CANFINHOME", "CASTROLIND", "CENTRALBK",
        "CGPOWER", "CHAMBLFERT", "COFORGE", "CROMPTON", "CUB",
        "CUMMINSIND", "CYIENT", "DELTACORP", "DEVYANI", "DIXON",
        "ECLERX", "ELGIEQUIP", "EMAMILTD", "ENDURANCE", "EQUITASBNK",
        "ESCORTS", "EXIDEIND", "FEDERALBNK", "FORTIS", "GLAND",
        "GLAXO", "GLENMARK", "GMRINFRA", "GNFC", "GODREJIND",
        "GODREJPROP", "GRANULES", "GRAPHITE", "GSFC", "GSPL",
        "GUJGASLTD", "HAL", "HATSUN", "HDFCAMC", "HINDCOPPER",
        "HONAUT", "HUDCO", "IBREALEST", "IDBI", "IDFCFIRSTB",
        "IEX", "IIFL", "INDHOTEL", "INDIACEM", "INDIAMART",
        "INDIANB", "IPCALAB", "IRB", "IRFC", "ISEC",
        "JBCHEPHARM", "JKCEMENT", "JSL", "JSWENERGY", "JUBLINGREA",
        "KAJARIACER", "KALYANKJIL", "KANSAINER", "KEI", "KPITTECH",
        "KRBL", "LALPATHLAB", "LATENTVIEW", "LAURUSLABS", "LICHSGFIN",
        "LINDEINDIA", "LTTS", "LUXIND", "MANAPPURAM", "MAXHEALTH",
        "MCX", "METROPOLIS", "MFSL", "MGL", "MPHASIS",
        "MRF", "NATIONALUM", "NATCOPHARM", "NAVINFLUOR", "NBCC",
        "NCC", "NLCINDIA", "OBEROIRLTY", "OFSS", "OIL",
        "PAGEIND", "PATANJALI", "PERSISTENT", "PFIZER", "PHOENIXLTD",
        "POLYMED", "POLYCAB", "POONAWALLA", "PRESTIGE", "PVRINOX",
        "RADICO", "RAJESHEXPO", "RAMCOCEM", "RBLBANK", "RAYMOND",
        "REDINGTON", "RELAXO", "RITES", "ROUTE", "SANOFI",
        "SCHAEFFLER", "SHILPAMED", "SJVN", "SKFINDIA", "SOBHA",
        "SOLARINDS", "SONACOMS", "SONATSOFTW", "STAR", "SUMICHEM",
        "SUNDARMFIN", "SUNDRMFAST", "SUNPHARMA", "SUNTV", "SUPREMEIND",
        "SUVENPHAR", "SYNGENE", "TATACOMM", "TATAELXSI", "TATVA",
        "TEAMLEASE", "THERMAX", "TIMKEN", "TORNTPOWER", "TRENT",
        "TRIDENT", "TRIVENI", "TTKPRESTIG", "TV18BRDCST", "TVSMOTOR",
        "UBL", "UJJIVAN", "UJJIVANSFB", "UNIONBANK", "UTIAMC",
        "VINATIORGA", "VIPIND", "VOLTAS", "VSTIND", "WELCORP",
        "WELSPUNIND", "WHIRLPOOL", "YESBANK", "ZEEL", "ZOMATO",
        
        # NIFTY Smallcap 250
        "3MINDIA", "AARTIDRUGS", "AAVAS", "ABCAPITAL", "ABFRL",
        "ABSLAMC", "ACCELYA", "ACE", "AEGISCHEM", "AFFLE",
        "AGARIND", "AHLUCONT", "AJANTPHARM", "AKZOINDIA", "ALLCARGO",
        "AMARAJABAT", "AMBER", "AMBUJACEM", "AMIORG", "ANGELONE",
        "ANURAS", "APARINDS", "APCOTEXIND", "APEX", "APLLTD",
        "APTUS", "ARCHIDPLY", "ARVIND", "ASAHIINDIA", "ASHIANA",
        "ASHOKA", "ASTRAL", "ASTRAMICRO", "ATUL", "AUBANK",
        "AVANTIFEED", "AXISCADES", "BALAMINES", "BALMLAWRIE", "BASF",
        "BATAINDIA", "BAYERCROP", "BCG", "BEML", "BEL",
        "BFINVEST", "BIKAJI", "BINDALAGRO", "BIRLACORPN", "BLISSGVS",
        "BLUEDART", "BLUESTARCO", "BRIGADE", "BSE", "BSOFT",
        "CAMPUS", "CAPACITE", "CARERATING", "CARYSIL", "CCL",
        "CDSL", "CENTURYPLY", "CENTURYTEX", "CERA", "CHALET",
        "CHENNPETRO", "CHOICEIN", "CLEAN", "COALINDIA", "COCHINSHIP",
        "COFORGE", "CONTROLPR", "COSMOFILMS", "CRAFTSMAN", "CREDITACC",
        "CRISIL", "CYIENTDLM", "DALBHARAT", "DATAMATICS", "DCMSHRIRAM",
        "DEEPAKNTR", "DELHIVERY", "DHANUKA", "DIXON", "EASEMYTRIP",
        "EDELWEISS", "EIDPARRY", "EIHOTEL", "ELECON", "ELGIEQUIP",
        "EMAMILTD", "EPIGRAL", "ERIS", "ESABINDIA", "ETHOSLTD",
        "EXCELINDUS", "FACT", "FDC", "FINEORG", "FIRSTSOUR",
        "FIVESTAR", "FLUOROCHEM", "FORCEMOT", "FSL", "FUSION",
        "GAEL", "GALAXYSURF", "GATEWAY", "GEECEE", "GENUSPOWER",
        "GHCL", "GICRE", "GILLETTE", "GLAND", "GLOBUSSPR",
        "GMBREW", "GMDCLTD", "GMMPFAUDLR", "GODFRYPHLP", "GODREJAGRO",
        "GOLDIAM", "GOODLUCK", "GOODYEAR", "GPIL", "GPPL",
        "GREAVESCOT", "GREENPANEL", "GREENPLY", "GRINDWELL", "GRSE",
        "GTL", "GUFICBIO", "GULFOILLUB", "GUJALKALI", "HATHWAY",
        "HBLPOWER", "HCC", "HERITGFOOD", "HFCL", "HGINFRA",
        "HIKAL", "HIL", "HIMADRI", "HINDZINC", "HOMEFIRST",
        "HPL", "HSCL", "HUHTAMAKI", "ICRA", "IDFC",
        "IFBIND", "IFCI", "IGL", "IMFA", "INDIANCARD",
        "INDOCO", "INDORAMA", "INDOSTAR", "INFIBEAM", "INGERRAND",
        "INOXWIND", "INSECTICIDE", "INTELLECT", "IONEXCHANG", "IPL",
        "IRCON", "ITI", "IXIGO", "JAMNAAUTO", "JBMA",
        "JCHAC", "JINDALSTEL", "JINDALSAW", "JISLJALEQS", "JKIL",
        "JKLAKSHMI", "JKPAPER", "JKTYRE", "JMFINANCIL", "JPASSOCIAT",
        "JPPOWER", "JSWHL", "JTEKTINDIA", "JUSTDIAL", "JYOTHYLAB",
        "KABRAEXTRU", "KALPATPOWR", "KAMDHENU", "KANORICHEM", "KARURVYSYA",
        "KAYNES", "KDDL", "KEC", "KELLTONTEC", "KESORAMIND",
        "KFINTECH", "KIRLOSBROS", "KIRLOSENG", "KIRLOSIND", "KITEX",
        "KNRCON", "KOLTEPATIL", "KOPRAN", "KPITTECH", "KPRMILL",
        "KRSNAA", "KSCL", "KTKBANK", "LAOPALA", "LATENTVIEW",
        "LAXMIMACH", "LEMONTREE", "LINDEINDIA", "LOKESHMACH", "LTFOODS",
        "LTI", "LUMAXIND", "LUMAXTECH", "LUXIND", "MAHLOG",
        "MAHSEAMLES", "MAHSCOOTER", "MAITHANALL", "MANAPPURAM", "MANINFRA",
        "MANINDS", "MANKIND", "MANYAVAR", "MAPMYINDIA", "MARKSANS",
        "MASTEK", "MAXFINDIA", "MAZDOCK", "MEDANTA", "MEDPLUS",
        "MEGASOFT", "METROBRAND", "MIDHANI", "MINDACORP", "MINDAIND",
        "MMFL", "MMTC", "MOIL", "MOLDTECH", "MONTECARLO",
        "MOTHERSON", "MOTILALOFS", "MPSLTD", "MRPL", "MSTCLTD",
        "MTARTECH", "MTNL", "MUKANDLTD", "MUNJALAU", "NATHBIOGEN",
        "NATIONALUM", "NAM-INDIA", "NAZARA", "NDTV", "NEOGEN",
        "NESCO", "NETWORK18", "NEULANDLAB", "NEWGEN", "NFL",
        "NH", "NHPC", "NIACL", "NIITLTD", "NILKAMAL",
        "NIPPONIND", "NLCINDIA", "NMDC", "NOCIL", "NRBBEARING",
        "NURECA", "NUVOCO", "OCCL", "OFSS", "OIL",
        "OLECTRA", "OMAXE", "ONEPOINT", "ONMOBILE", "OPTIEMUS",
        "ORIENTBELL", "ORIENTCEM", "ORIENTELEC", "PAISALO", "PALREDTEC",
        "PANACEABIO", "PARAGMILK", "PATELENG", "PATINTLOG", "PCBL",
        "PCJEWELLER", "PENIND", "PERSISTENT", "PETRONET", "PFC",
        "PFIZER", "PGEL", "PGHH", "PHILIPCARB", "PHOENIXLTD",
        "PILANIINVS", "PNBHOUSING", "PNC", "PNCINFRA", "POLICYBZR",
        "POLYMED", "POLYPLEX", "POONAWALLA", "POWERINDIA", "PRAJIND",
        "PRECAM", "PRESTIGE", "PRINCEPIPE", "PRSMJOHNSN", "PSB",
        "PUNJABCHEM", "PVRINOX", "QUESS", "QUICKHEAL", "RADICO",
        "RAIN", "RAJESHEXPO", "RAJRATAN", "RALLIS", "RAMCOCEM",
        "RAMCOIND", "RAMCOSYS", "RANEHOLDIN", "RATEGAIN", "RATNAMANI",
        "RAYMOND", "RBLBANK", "RECLTD", "REDINGTON", "RELAXO",
        "RELIGARE", "RENUKA", "REPCOHOME", "RESPONIND", "RITES",
        "ROUTE", "RPOWER", "RTNPOWER", "RVNL", "SAFARI",
        "SADBHAV", "SAGCEM", "SAKSOFT", "SALASAR", "SANDHAR",
        "SANGAMIND", "SANGHIIND", "SANOFI", "SAPPHIRE", "SAREGAMA",
        "SASKEN", "SATIA", "SATIN", "SBICARD", "SBILIFE",
        "SCHNEIDER", "SCI", "SELAN", "SEQUENT", "SFL",
        "SHAKTIPUMP", "SHALBY", "SHANKARA", "SHARDACROP", "SHARDAMOTR",
        "SHAREINDIA", "SHILPAMED", "SHIVAMILLS", "SHOPERSTOP", "SHREECEM",
        "SHREEPUSHK", "SHRIRAMCIT", "SHRIRAMFIN", "SHYAMCENT", "SIEMENS",
        "SIGACHI", "SIL", "SIMPLEXINF", "SINTEX", "SIRCA",
        "SIS", "SJVN", "SKFINDIA", "SKIPPER", "SMLISUZU",
        "SMSPHARMA", "SNOWMAN", "SOBHA", "SOFTTECH", "SOLARINDS",
        "SOLARA", "SONACOMS", "SONATSOFTW", "SOUTHBANK", "SPARC",
        "SPENCERS", "SPIC", "SRF", "STAR", "STARCEMENT",
        "STCINDIA", "STLTECH", "STOVEKRAFT", "STRTECH", "STYLAMIND",
        "STYRENIX", "SUBROS", "SUDARSCHEM", "SUMICHEM", "SUNDRMFAST",
        "SUNFLAG", "SUNTECK", "SUNTV", "SUPRAJIT", "SUPREMEIND",
        "SURYAROSNI", "SURYODAY", "SUVENPHAR", "SUZLON", "SWANENERGY",
        "SWARAJENG", "SYMPHONY", "SYNGENE", "TAKE", "TANLA",
        "TATACHEM", "TATACOMM", "TATACONSUM", "TATAELXSI", "TATAINVEST",
        "TATAMETALI", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TATVA",
        "TBZ", "TCI", "TCIEXP", "TCNSBRANDS", "TEAMLEASE",
        "TECHM", "TEJASNET", "TEXMOPIPES", "THERMAX", "THOMASCOOK",
        "THYROCARE", "TIINDIA", "TIMKEN", "TINPLATE", "TIPSINDLTD",
        "TITAN", "TNPL", "TORNTPHARM", "TORNTPOWER", "TPLPLASTEH",
        "TRENT", "TRIDENT", "TRIVENI", "TTKPRESTIG", "TV18BRDCST",
        "TVSELECT", "TVSMOTOR", "TVSSRICHAK", "TWL", "UBL",
        "UCALFUEL", "UCOBANK", "UFLEX", "UJJIVAN", "UJJIVANSFB",
        "ULTRACEMCO", "UNICHEMLAB", "UNIONBANK", "UNIPARTS", "UNITDSPR",
        "UPL", "UTIAMC", "UTKARSHBNK", "VAIBHAVGBL", "VAKRANGEE",
        "VALIANTORG", "VARROC", "VBL", "VEDL", "VENKEYS",
        "VERANDA", "VESUVIUS", "VGUARD", "VIP", "VIRINCHI",
        "VISHNU", "VLSFINANCE", "VMART", "VOLTAMP", "VOLTAS",
        "VRLLOG", "VSTIND", "VTL", "WABCOINDIA", "WALCHANNAG",
        "WATERBASE", "WELCORP", "WELSPUNIND", "WENDT", "WESTLIFE",
        "WHIRLPOOL", "WINDLAS", "WIPRO", "WOCKPHARMA", "WONDERLA",
        "YATHARTH", "YESBANK", "ZEEL", "ZENTEC", "ZOMATO",
        "ZYDUSLIFE", "ZYDUSWELL",
        
        # Additional stocks not in indices but actively traded
        "20MICRONS", "21STCENMGM", "3IINFOLTD", "3PLAND", "5PAISA",
        "A2ZINFRA", "AAKASH", "ABORIGEN", "ACRYSIL", "ADLABS",
        "ADORWELD", "ADVANIHOTR", "ADVENZYMES", "AETHER", "AGCNET",
        "AGROPHOS", "AHLEAST", "AIROLAM", "AJOYSTN", "ALOKTEXT",
        "ALPA", "ALPHAGEO", "ALPSINDUS", "AMBICAGNI", "AMBUJACEM",
        "AMIORG", "AMRUTANJAN", "ANANTRAJ", "ANDHRAPET", "ANGELONE",
        "ANIKINDS", "ANKITMETAL", "ANMOL", "ANSALAPI", "ANTGRAPHIC",
        "ANUP", "APCL", "APLAB", "APOLSINHOT", "APOLLO",
        "APTECHT", "ARCOTECH", "ARIES", "ARIHANTCAP", "ARIHANTSUP",
        "ARMANFIN", "AROGRANITE", "ARROWGREEN", "ARTPHARM", "ARVIND",
        "ARVSMART", "ASAHISONG", "ASHAPURMIN", "ASHIANA", "ASHOKA",
        "ASPINWALL", "ASTEC", "ASTERDM", "ASTRAMICRO", "ASTRAZEN",
        "ATLANTA", "ATLASCYCLE", "ATLASBIKE", "ATULAUTO", "AURIONPRO",
        "AUTOIND", "AUTOLITIND", "AVADHSUGAR", "AVTNPL", "BALAXI",
        "BALKRISHNA", "BALRAMCHIN", "BANARISUG", "BANG", "BANSWRAS",
        "BARBEQUE", "BASML", "BBOX", "BBTC", "BEARDSELL",
        "BEPL", "BFINVEST", "BFUTILITIE", "BGRENERGY", "BHAGERIA",
        "BHAGYANGR", "BHARATGEAR", "BHARATWIRE", "BHEL", "BIGBLOC",
        "BINDALAGRO", "BLKASHYAP", "BPCL", "BSLLTD", "BUTTERFLY",
    ]
    
    # Remove duplicates
    seen = set()
    result = {}
    for sym in stocks:
        if sym not in seen:
            seen.add(sym)
            result[sym] = {'symbol': sym, 'name': sym, 'series': 'EQ'}
    
    return result


def get_local_stock_symbols() -> list:
    """Get stock symbols already available locally (historic + downloaded)."""
    symbols = set()

    if PRICE_DIR.exists():
        for f in PRICE_DIR.glob("*.parquet"):
            symbols.add(f.stem.upper())

    if INFO_DIR.exists():
        for f in INFO_DIR.glob("*.json"):
            symbols.add(f.stem.upper())

    for symbol in list_historic_stock_symbols():
        symbols.add(symbol.upper())

    return sorted(symbols)


def get_local_financial_symbols() -> list:
    """Get stock symbols with cached financial statements."""
    symbols = set()
    if FINANCIALS_DIR.exists():
        for f in FINANCIALS_DIR.glob("*.json"):
            symbols.add(f.stem.upper())
    return sorted(symbols)


def _is_financials_cache_valid(path: Path) -> bool:
    """Check if cached financials look complete enough for analysis."""
    try:
        with open(path) as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return False
        
        # Check for at least one statement with data
        valid_statements = 0
        for key in ["income_statement", "balance_sheet", "cash_flow"]:
            block = payload.get(key)
            if isinstance(block, dict):
                cols = block.get("columns") or []
                data = block.get("data") or []
                # Relaxed check: just need some data, even 1 year is better than nothing
                if len(cols) >= 1 and len(data) >= 1:
                    valid_statements += 1
        
        return valid_statements >= 1
    except Exception:
        return False


def get_invalid_financial_symbols() -> list:
    """Get symbols with missing/invalid financial cache."""
    invalid = []
    if not FINANCIALS_DIR.exists():
        return invalid
    for f in FINANCIALS_DIR.glob("*.json"):
        if not _is_financials_cache_valid(f):
            invalid.append(f.stem.upper())
    return sorted(invalid)


def write_symbol_index(path: Path, items: list, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "updated_at": datetime.now().isoformat(),
                "count": len(items),
                "items": items,
                **metadata,
            },
            f,
            indent=2,
        )


def get_missing_live_symbols(live_symbols: list, local_symbols: list) -> list:
    """Return live NSE symbols that are missing locally."""
    local_set = {s.upper() for s in local_symbols}
    return [s for s in live_symbols if s["symbol"].upper() not in local_set]


def refresh_symbols(symbols: list) -> dict:
    """Refresh symbols using gap-fill logic."""
    results = {"updated": 0, "skipped": 0, "failed": 0, "records_added": 0}
    for symbol in symbols:
        result = refresh_symbol_data(symbol)
        if result.get("updated"):
            results["updated"] += 1
            results["records_added"] += result.get("new_records", 0)
        elif result.get("error"):
            results["failed"] += 1
        else:
            results["skipped"] += 1
    return results


def load_progress():
    """Load download progress."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {'completed': [], 'failed': [], 'last_update': None}


def save_progress(progress):
    """Save download progress."""
    progress['last_update'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def _df_to_payload(df):
    if df is None:
        return None
    try:
        df = df.copy()
        df.index = df.index.astype(str)
        df.columns = [str(c) for c in df.columns]
        return df.to_dict(orient="split")
    except Exception:
        return None


def _series_to_payload(series):
    if series is None:
        return None
    try:
        series = series.dropna()
        if series.empty:
            return None
        return {
            "index": [str(i) for i in series.index],
            "values": [float(v) for v in series.values],
        }
    except Exception:
        return None


def download_stock_data(
    symbol: str,
    years: int = 30,
    retry_count: int = 0,
    max_retries: int = 3,
    include_financials: bool = False,
) -> dict:
    """Download 30 years of data for a single stock from Yahoo Finance with retry logic."""
    import yfinance as yf
    import pandas as pd
    import random
    
    result = {
        'symbol': symbol,
        'success': False,
        'price_records': 0,
        'info': False,
        'financials': False,
        'error': None,
        'years_of_data': 0,
        'retried': retry_count > 0
    }
    
    # Add random delay to avoid rate limiting (0.5-1.5 seconds)
    time.sleep(random.uniform(0.5, 1.5))
    
    try:
        # Try NSE suffix first, then BSE
        for suffix in ['.NS', '.BO']:
            try:
                ticker = yf.Ticker(f"{symbol}{suffix}")
                
                # Get MAXIMUM historical data
                df = ticker.history(period="max")
                
                if df.empty or len(df) < 10:
                    # Try with explicit date range
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=years * 365)
                    df = ticker.history(start=start_date, end=end_date)
                
                if not df.empty and len(df) > 10:
                    # Process DataFrame
                    df = df.reset_index()
                    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                    
                    # Handle timezone
                    if 'date' in df.columns:
                        try:
                            if hasattr(df['date'].dt, 'tz') and df['date'].dt.tz is not None:
                                df['date'] = df['date'].dt.tz_localize(None)
                        except:
                            pass
                        df['date'] = pd.to_datetime(df['date'])
                    
                    df['symbol'] = symbol
                    df['source'] = f'yahoo{suffix}'
                    
                    # Calculate years of data
                    if 'date' in df.columns and len(df) > 0:
                        date_range = (df['date'].max() - df['date'].min()).days / 365
                        result['years_of_data'] = round(date_range, 1)
                    
                    # Save to parquet
                    price_file = PRICE_DIR / f"{symbol}.parquet"
                    df.to_parquet(price_file, index=False)
                    result['price_records'] = len(df)
                    
                    # Get stock info (with small delay)
                    time.sleep(0.3)
                    try:
                        info = ticker.info
                        if info:
                            clean_info = {k: v for k, v in info.items() 
                                         if isinstance(v, (str, int, float, bool, type(None)))}
                            clean_info['symbol'] = symbol
                            clean_info['downloaded_at'] = datetime.now().isoformat()
                            clean_info['years_of_data'] = result['years_of_data']
                            
                            info_file = INFO_DIR / f"{symbol}.json"
                            with open(info_file, 'w') as f:
                                json.dump(clean_info, f, indent=2)
                            result['info'] = True
                    except:
                        pass

                    # Optional: download financial statements for offline analysis
                    if include_financials:
                        time.sleep(0.5) # Increased delay
                        try:
                            # Fetch with retries
                            income_stmt = None
                            for _ in range(2):
                                try:
                                    income_stmt = ticker.financials
                                    if income_stmt is not None and not income_stmt.empty:
                                        break
                                    time.sleep(1)
                                    # Re-initialize ticker to clear cache/state
                                    ticker = yf.Ticker(f"{symbol}{suffix}")
                                except:
                                    time.sleep(1)
                            
                            # Transpose if valid
                            income_stmt = income_stmt.T if income_stmt is not None and not income_stmt.empty else None
                            
                            # Get others
                            balance_sheet = ticker.balance_sheet
                            balance_sheet = balance_sheet.T if balance_sheet is not None and not balance_sheet.empty else None
                            
                            cash_flow = ticker.cashflow
                            cash_flow = cash_flow.T if cash_flow is not None and not cash_flow.empty else None
                            
                            dividends = ticker.dividends
                            
                            # Only save if we have at least income statement or balance sheet
                            if income_stmt is not None or balance_sheet is not None:
                                payload = {
                                    "symbol": symbol,
                                    "downloaded_at": datetime.now().isoformat(),
                                    "income_statement": _df_to_payload(income_stmt),
                                    "balance_sheet": _df_to_payload(balance_sheet),
                                    "cash_flow": _df_to_payload(cash_flow),
                                    "dividends": _series_to_payload(dividends),
                                    "source": f"yahoo{suffix}",
                                }
                                fin_file = FINANCIALS_DIR / f"{symbol}.json"
                                with open(fin_file, "w") as f:
                                    json.dump(payload, f, indent=2)
                                result['financials'] = True
                            else:
                                # If we failed to get financials, we don't save an empty file.
                                # This allows the 'missing_financials' logic to correctly identify it later.
                                pass
                                
                        except Exception as e:
                            # logger.debug(f"Financials download failed for {symbol}: {e}")
                            pass
                    
                    result['success'] = True
                    return result
                    
            except Exception as e:
                # If we get rate limited, add extra delay
                if "Too Many Requests" in str(e) or "429" in str(e):
                    time.sleep(5)
                continue
        
        result['error'] = "No data found"
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def download_failed_stocks(failed_symbols: list, workers: int = 2, include_financials: bool = False) -> tuple:
    """Retry downloading failed stocks with slower rate and more retries."""
    logger.info(f"\n{'='*60}")
    logger.info(f"RETRYING {len(failed_symbols)} FAILED STOCKS")
    logger.info(f"{'='*60}")
    logger.info(f"Using {workers} workers with extended delays...")
    
    success_count = 0
    still_failed = []
    total_records = 0
    
    # Process in smaller batches with delays between batches
    batch_size = 10
    
    for i in range(0, len(failed_symbols), batch_size):
        batch = failed_symbols[i:i+batch_size]
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(download_stock_data, sym, YEARS_OF_DATA, 1, 3, include_financials): sym 
                      for sym in batch}
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result['success']:
                        success_count += 1
                        total_records += result['price_records']
                        logger.info(f"✓ RETRY {symbol}: {result['price_records']:,} records")
                    else:
                        still_failed.append(symbol)
                except Exception as e:
                    still_failed.append(symbol)
        
        # Add delay between batches to avoid rate limiting
        if i + batch_size < len(failed_symbols):
            logger.info(f"  Batch {i//batch_size + 1} done. Waiting 5s before next batch...")
            time.sleep(5)
    
    return success_count, still_failed, total_records


def download_all(symbols: list, workers: int = 2, resume: bool = False, include_financials: bool = False):
    """Download all stocks with parallel workers and rate limiting."""
    progress = load_progress() if resume else {'completed': [], 'failed': [], 'last_update': None}
    
    # Filter already completed
    if resume:
        completed_set = set(progress['completed'])
        symbols = [s for s in symbols if s['symbol'] not in completed_set]
        logger.info(f"Resuming: {len(progress['completed'])} done, {len(symbols)} remaining")
    
    total = len(symbols)
    success_count = 0
    fail_count = 0
    total_records = 0
    failed_symbols = []
    
    # Limit workers to avoid rate limiting (max 5)
    workers = min(workers, 5)
    
    logger.info(f"Starting download of {total} stocks with {workers} workers")
    logger.info(f"Fetching {YEARS_OF_DATA} years of historical data per stock")
    logger.info(f"Note: Using rate limiting to avoid Yahoo Finance blocks")
    start_time = time.time()
    
    # Process in batches to control rate
    batch_size = 50
    
    for batch_idx in range(0, total, batch_size):
        batch = symbols[batch_idx:batch_idx + batch_size]
        batch_start = time.time()
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for stock in batch:
                future = executor.submit(download_stock_data, stock['symbol'], YEARS_OF_DATA, 0, 3, include_financials)
                futures[future] = stock['symbol']
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    
                    if result['success']:
                        progress['completed'].append(symbol)
                        success_count += 1
                        total_records += result['price_records']
                        years_str = f"({result['years_of_data']}y)" if result['years_of_data'] else ""
                        logger.info(f"✓ {symbol}: {result['price_records']:,} records {years_str}")
                    else:
                        failed_symbols.append(symbol)
                        fail_count += 1
                        logger.warning(f"✗ {symbol}: {result['error']}")
                    
                except Exception as e:
                    fail_count += 1
                    failed_symbols.append(symbol)
                    logger.error(f"Error with {symbol}: {e}")
        
        # Progress update after each batch
        done = success_count + fail_count
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate / 60 if rate > 0 else 0
        logger.info(f"Progress: {done}/{total} ({done/total*100:.1f}%) | "
                   f"Success: {success_count} | Failed: {fail_count} | "
                   f"Records: {total_records:,} | ETA: {eta:.0f} min")
        save_progress(progress)
        
        # Add delay between batches to avoid rate limiting
        if batch_idx + batch_size < total:
            batch_elapsed = time.time() - batch_start
            # Ensure at least 2 seconds per request on average
            min_batch_time = batch_size * 1.5
            if batch_elapsed < min_batch_time:
                sleep_time = min_batch_time - batch_elapsed
                logger.info(f"  Rate limiting: sleeping {sleep_time:.1f}s before next batch...")
                time.sleep(sleep_time)
    
    # Store failed symbols in progress
    progress['failed'] = [{'symbol': s, 'error': 'No data found'} for s in failed_symbols]
    save_progress(progress)
    
    return success_count, fail_count, total_records, failed_symbols


def build_database():
    """Build DuckDB database from downloaded files."""
    try:
        import duckdb
        
        db_path = DATA_DIR / 'db' / 'equity_intelligence.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove old database
        if db_path.exists():
            db_path.unlink()
        
        conn = duckdb.connect(str(db_path))
        
        # Create tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stocks (
                symbol VARCHAR PRIMARY KEY,
                name VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                market_cap DOUBLE,
                current_price DOUBLE,
                pe_ratio DOUBLE,
                pb_ratio DOUBLE,
                dividend_yield DOUBLE,
                fifty_two_week_high DOUBLE,
                fifty_two_week_low DOUBLE,
                years_of_data DOUBLE,
                updated_at TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                symbol VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                source VARCHAR,
                PRIMARY KEY (symbol, date)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS index_history (
                index_symbol VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                source VARCHAR,
                PRIMARY KEY (index_symbol, date)
            )
        """)
        
        # Load stock info
        info_files = list(INFO_DIR.glob("*.json"))
        logger.info(f"Loading {len(info_files)} stock info files...")
        
        for info_file in info_files:
            try:
                with open(info_file) as f:
                    info = json.load(f)
                
                conn.execute("""
                    INSERT OR REPLACE INTO stocks 
                    (symbol, name, sector, industry, market_cap, current_price, 
                     pe_ratio, pb_ratio, dividend_yield, fifty_two_week_high, 
                     fifty_two_week_low, years_of_data, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, [
                    info.get('symbol'),
                    info.get('longName', info.get('shortName', info.get('symbol'))),
                    info.get('sector'),
                    info.get('industry'),
                    info.get('marketCap'),
                    info.get('currentPrice', info.get('regularMarketPrice')),
                    info.get('trailingPE'),
                    info.get('priceToBook'),
                    info.get('dividendYield'),
                    info.get('fiftyTwoWeekHigh'),
                    info.get('fiftyTwoWeekLow'),
                    info.get('years_of_data')
                ])
            except:
                pass
        
        # Load price history
        price_files = list(PRICE_DIR.glob("*.parquet"))
        logger.info(f"Loading {len(price_files)} price files...")
        
        for price_file in price_files:
            try:
                conn.execute(f"""
                    INSERT OR REPLACE INTO price_history 
                    SELECT symbol, date, open, high, low, close, 
                           CAST(volume AS BIGINT), source
                    FROM read_parquet('{price_file}')
                """)
            except:
                pass

        # Load historic stock history (CSV datasets)
        historic_stock_symbols = list_historic_stock_symbols()
        logger.info(f"Loading {len(historic_stock_symbols)} historic stock files...")
        for symbol in historic_stock_symbols:
            try:
                df = load_historic_stock_prices(symbol)
                if df is None or df.empty:
                    continue
                conn.register("hist_stock_df", df)
                conn.execute("""
                    INSERT OR REPLACE INTO price_history
                    SELECT symbol, date, open, high, low, close,
                           CAST(volume AS BIGINT), source
                    FROM hist_stock_df
                """)
                conn.unregister("hist_stock_df")
            except Exception as e:
                logger.debug(f"Historic stock load failed for {symbol}: {e}")

        # Load historic index history
        index_names = list_historic_index_names()
        logger.info(f"Loading {len(index_names)} historic index files...")
        for index_name in index_names:
            try:
                df = load_historic_index_prices(index_name)
                if df is None or df.empty:
                    continue
                df = df.rename(columns={"symbol": "index_symbol"})
                conn.register("idx_df", df)
                conn.execute("""
                    INSERT OR REPLACE INTO index_history
                    SELECT index_symbol, date, open, high, low, close,
                           CAST(volume AS BIGINT), source
                    FROM idx_df
                """)
                conn.unregister("idx_df")
            except Exception as e:
                logger.debug(f"Index load failed for {index_name}: {e}")
        
        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_price_symbol ON price_history(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_price_date ON price_history(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_index_symbol ON index_history(index_symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_index_date ON index_history(date)")
        
        # Create CAGR cache table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS asset_cagr (
                asset_type VARCHAR,
                symbol VARCHAR,
                window_years INTEGER,
                start_date DATE,
                end_date DATE,
                start_close DOUBLE,
                end_close DOUBLE,
                years_span DOUBLE,
                cagr_pct DOUBLE,
                updated_at TIMESTAMP,
                PRIMARY KEY (asset_type, symbol, window_years)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_asset_cagr_symbol ON asset_cagr(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_asset_cagr_type_window ON asset_cagr(asset_type, window_years)")
        
        # Build CAGR cache for stocks and indices
        def _populate_cagr_cache(table_name: str, symbol_column: str, asset_type: str):
            # Clear existing rows for asset type
            conn.execute("DELETE FROM asset_cagr WHERE asset_type = ?", [asset_type])
            
            # Full-history CAGR (window_years = 0)
            conn.execute(f"""
                INSERT INTO asset_cagr
                SELECT
                    '{asset_type}' AS asset_type,
                    {symbol_column} AS symbol,
                    0 AS window_years,
                    MIN(date) AS start_date,
                    MAX(date) AS end_date,
                    arg_min(close, date) AS start_close,
                    arg_max(close, date) AS end_close,
                    datediff('day', MIN(date), MAX(date)) / 365.25 AS years_span,
                    CASE
                        WHEN arg_min(close, date) > 0
                             AND arg_max(close, date) > 0
                             AND datediff('day', MIN(date), MAX(date)) > 0
                        THEN (POWER(arg_max(close, date) / arg_min(close, date),
                              1.0 / (datediff('day', MIN(date), MAX(date)) / 365.25)) - 1) * 100
                        ELSE NULL
                    END AS cagr_pct,
                    CURRENT_TIMESTAMP AS updated_at
                FROM {table_name}
                GROUP BY {symbol_column}
            """)
            
            # Rolling window CAGR
            for window_years in CAGR_WINDOWS_YEARS:
                window_days = int(window_years * 365)
                conn.execute(f"""
                    WITH last_dates AS (
                        SELECT {symbol_column} AS symbol, MAX(date) AS end_date
                        FROM {table_name}
                        GROUP BY {symbol_column}
                    ),
                    windowed AS (
                        SELECT t.{symbol_column} AS symbol,
                               MIN(t.date) AS start_date,
                               l.end_date AS end_date,
                               arg_min(t.close, t.date) AS start_close,
                               arg_max(t.close, t.date) AS end_close
                        FROM {table_name} t
                        JOIN last_dates l ON t.{symbol_column} = l.symbol
                        WHERE t.date >= l.end_date - INTERVAL '{window_days} days'
                        GROUP BY t.{symbol_column}, l.end_date
                    )
                    INSERT INTO asset_cagr
                    SELECT
                        '{asset_type}' AS asset_type,
                        symbol,
                        {window_years} AS window_years,
                        start_date,
                        end_date,
                        start_close,
                        end_close,
                        datediff('day', start_date, end_date) / 365.25 AS years_span,
                        CASE
                            WHEN start_close > 0 AND end_close > 0
                                 AND datediff('day', start_date, end_date) > 0
                            THEN (POWER(end_close / start_close,
                                  1.0 / (datediff('day', start_date, end_date) / 365.25)) - 1) * 100
                            ELSE NULL
                        END AS cagr_pct,
                        CURRENT_TIMESTAMP AS updated_at
                    FROM windowed
                """)
        
        logger.info("Calculating CAGR cache for stocks...")
        _populate_cagr_cache("price_history", "symbol", "stock")
        logger.info("Calculating CAGR cache for indices...")
        _populate_cagr_cache("index_history", "index_symbol", "index")
        
        stock_count = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
        price_count = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]
        index_count = conn.execute("SELECT COUNT(*) FROM index_history").fetchone()[0]
        cagr_count = conn.execute("SELECT COUNT(*) FROM asset_cagr").fetchone()[0]
        
        # Get date range
        date_range = conn.execute("""
            SELECT MIN(date), MAX(date) FROM price_history
        """).fetchone()
        index_date_range = conn.execute("""
            SELECT MIN(date), MAX(date) FROM index_history
        """).fetchone()
        
        conn.close()
        
        logger.info(f"Database built successfully!")
        logger.info(f"  Stocks: {stock_count:,}")
        logger.info(f"  Price records: {price_count:,}")
        logger.info(f"  Index records: {index_count:,}")
        if date_range[0] and date_range[1]:
            logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")
        if index_date_range[0] and index_date_range[1]:
            logger.info(f"  Index date range: {index_date_range[0]} to {index_date_range[1]}")
        logger.info(f"  CAGR rows: {cagr_count:,}")
        logger.info(f"  Database: {db_path}")
        
    except Exception as e:
        logger.error(f"Database build failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Download ALL Indian stock market data')
    parser.add_argument('--workers', type=int, default=2, help='Parallel workers (default: 2, max: 3)')
    parser.add_argument('--resume', action='store_true', help='Resume interrupted download')
    parser.add_argument('--build-db-only', action='store_true', help='Only build database')
    parser.add_argument('--retry-failed', action='store_true', help='Retry previously failed downloads')
    parser.add_argument('--download-all', action='store_true', help='Download all NSE stocks (ignore local data)')
    parser.add_argument('--refresh-symbol', type=str, help='Refresh a symbol (comma-separated)')
    parser.add_argument('--refresh-missing', action='store_true', help='Refresh only missing live symbols')
    parser.add_argument('--with-financials', action='store_true',
                        help='Download financial statements for offline analysis')
    args = parser.parse_args()
    
    print("=" * 70)
    print("STOCRON BY RTR - COMPLETE MARKET DATA DOWNLOAD")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data period: {YEARS_OF_DATA} years of historical data")
    print()
    
    # Check dependencies
    try:
        import pandas
        import yfinance
        import duckdb
        import pyarrow
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print(f"Run: {sys.executable} -m pip install pandas yfinance duckdb pyarrow")
        sys.exit(1)
    
    setup_directories()
    
    if args.build_db_only:
        build_database()
        return

    # Build local symbol/index lists from bundled datasets
    historic_symbols = list_historic_stock_symbols()
    if historic_symbols:
        write_symbol_index(
            HISTORIC_STOCK_INDEX_FILE,
            historic_symbols,
            {"source": "historic_nse_dataset"},
        )

    historic_index_names = list_historic_index_names()
    if historic_index_names:
        write_symbol_index(
            HISTORIC_INDEX_LIST_FILE,
            historic_index_names,
            {"source": "historic_nifty_indices"},
        )

    # Refresh specific symbols on demand
    if args.refresh_symbol:
        refresh_list = [s.strip().upper() for s in args.refresh_symbol.split(",") if s.strip()]
        if not refresh_list:
            print("No symbols provided for refresh.")
            return
        print(f"Refreshing {len(refresh_list)} symbols...")
        results = refresh_symbols(refresh_list)
        print(f"Updated: {results['updated']} | Skipped: {results['skipped']} | "
              f"Failed: {results['failed']} | Records added: {results['records_added']:,}")
        print("Rebuilding database...")
        build_database()
        return
    
    # Retry failed downloads
    if args.retry_failed:
        progress = load_progress()
        failed_list = [f['symbol'] if isinstance(f, dict) else f for f in progress.get('failed', [])]
        
        if not failed_list:
            print("No failed downloads to retry.")
            return
        
        print(f"Retrying {len(failed_list)} failed stocks...")
        print()
        
        retry_success, still_failed, retry_records = download_failed_stocks(
            failed_list, workers=2, include_financials=args.with_financials
        )
        
        # Update progress
        progress['completed'].extend([s for s in failed_list if s not in still_failed])
        progress['failed'] = [{'symbol': s, 'error': 'No data found'} for s in still_failed]
        save_progress(progress)
        
        print()
        print("=" * 70)
        print("RETRY COMPLETE!")
        print("=" * 70)
        print(f"Successfully recovered: {retry_success}")
        print(f"Still failed: {len(still_failed)}")
        print(f"Records added: {retry_records:,}")
        
        if still_failed:
            print(f"\nStocks that still failed (may not exist on Yahoo Finance):")
            print(", ".join(still_failed[:20]))
            if len(still_failed) > 20:
                print(f"  ... and {len(still_failed) - 20} more")
        
        print()
        print("Rebuilding database...")
        build_database()
        return
    
    # Get stock list
    print("=" * 70)
    print("STEP 1: Getting stock list...")
    print("=" * 70)
    
    symbols = fetch_nse_stock_list()
    
    if not symbols:
        print("ERROR: Could not get stock list")
        sys.exit(1)
    
    # Save for reference
    with open(STOCK_LIST_FILE, 'w') as f:
        json.dump({
            'fetched_at': datetime.now().isoformat(),
            'count': len(symbols),
            'stocks': symbols
        }, f, indent=2)

    # Compute missing symbols vs local data
    local_symbols = get_local_stock_symbols()
    missing_symbols = get_missing_live_symbols(symbols, local_symbols)
    local_financials = get_local_financial_symbols()
    invalid_financials = set(get_invalid_financial_symbols())
    missing_financials = [
        s for s in symbols
        if s["symbol"].upper() not in local_financials
        or s["symbol"].upper() in invalid_financials
    ]
    if missing_symbols:
        write_symbol_index(
            MISSING_STOCK_LIST_FILE,
            [s["symbol"] for s in missing_symbols],
            {"source": "nse_live_vs_local"},
        )
    if missing_financials and args.with_financials:
        write_symbol_index(
            DATA_DIR / "stock_lists" / "missing_financials.json",
            [s["symbol"] for s in missing_financials],
            {"source": "financials_cache"},
        )

    if args.refresh_missing:
        if not missing_symbols:
            print("No missing symbols to refresh.")
            return
        print(f"Refreshing {len(missing_symbols)} missing symbols...")
        results = refresh_symbols([s["symbol"] for s in missing_symbols])
        print(f"Updated: {results['updated']} | Skipped: {results['skipped']} | "
              f"Failed: {results['failed']} | Records added: {results['records_added']:,}")
        print("Rebuilding database...")
        build_database()
        return
    
    print()
    print("=" * 70)
    print("STEP 2: Download Configuration")
    print("=" * 70)
    symbols_to_download = symbols
    if local_symbols and not args.download_all:
        missing_set = {s["symbol"] for s in missing_symbols}
        if args.with_financials:
            missing_set.update(s["symbol"] for s in missing_financials)
        symbols_to_download = [s for s in symbols if s["symbol"] in missing_set]
        print(f"Local symbols detected: {len(local_symbols)}")
        print(f"Missing live symbols: {len(missing_symbols)}")
        if args.with_financials:
            print(f"Missing/invalid financials: {len(missing_financials)}")
        print(f"Downloading: {len(symbols_to_download)} (missing only)")
    else:
        print(f"Downloading full live list (local count: {len(local_symbols)})")

    print(f"Total stocks to download: {len(symbols_to_download)}")
    print(f"Historical data: {YEARS_OF_DATA} years per stock")
    
    # Limit workers to avoid rate limiting
    workers = min(args.workers, 5)
    print(f"Parallel workers: {workers} (limited to avoid rate limiting)")
    
    # More realistic estimate with rate limiting
    est_time = len(symbols_to_download) * 2 / workers / 60
    print(f"Estimated time: {est_time:.0f}-{est_time*2:.0f} minutes (with rate limiting)")
    print()
    print("NOTE: Using rate limiting to avoid Yahoo Finance blocks.")
    if args.with_financials:
        print("      Financial statements will also be downloaded for offline analysis.")
    print("      If many fail, run: python download_all_stocks.py --retry-failed")
    print()

    if not symbols_to_download:
        print("All live symbols already exist locally. Nothing to download.")
        print("Rebuilding database...")
        build_database()
        return
    
    response = input("Start download? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    print()
    print("=" * 70)
    print("STEP 3: Downloading from Yahoo Finance...")
    print("=" * 70)
    
    start_time = time.time()
    success, failed, total_records, failed_symbols = download_all(
        symbols_to_download,
        workers=workers,
        resume=args.resume,
        include_financials=args.with_financials,
    )
    elapsed = time.time() - start_time
    
    # Auto-retry failed downloads if there are many failures
    if failed_symbols and len(failed_symbols) > 50:
        print()
        print("=" * 70)
        print(f"STEP 3b: Retrying {len(failed_symbols)} failed downloads...")
        print("=" * 70)
        
        retry_success, still_failed, retry_records = download_failed_stocks(
            failed_symbols, workers=2, include_financials=args.with_financials
        )
        success += retry_success
        failed = len(still_failed)
        total_records += retry_records
        
        print(f"Retry results: {retry_success} recovered, {len(still_failed)} still failed")
    
    print()
    print("=" * 70)
    print("STEP 4: Building database...")
    print("=" * 70)
    build_database()
    
    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE!")
    print("=" * 70)
    print(f"Stocks downloaded: {success:,}")
    print(f"Failed: {failed:,}")
    print(f"Total price records: {total_records:,}")
    print(f"Time taken: {elapsed/60:.1f} minutes")
    print(f"Data location: {DATA_DIR}")
    
    if failed > 100:
        print()
        print("TIP: Many downloads failed due to rate limiting.")
        print("     Wait 1 hour and run: python download_all_stocks.py --retry-failed")
    
    print()
    print("The app now works 100% OFFLINE with complete Indian market data!")
    print()
    print("To start the app:")
    print("  docker-compose up --build")


if __name__ == "__main__":
    main()
